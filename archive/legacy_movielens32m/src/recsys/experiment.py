"""Two explicit milestones, immutable external splits, and a selected-artifact-only Test gate."""
from pathlib import Path
from copy import deepcopy
import json
import numpy as np
import pandas as pd
from .config import default_config
from .data import verify_splits,load_ratings,build_context,relevance,internal_partition
from .runtime import Store,digest,fingerprint,require_memory,profile,environment
from .training import train_channels,load_channels
from .evaluation import evaluate,evaluate_saved_ranker
from .ranking_features import sample_training,distribution,mismatch,FEATURES


class Experiment:
    def __init__(self,project,config=None,code_sha=None,run_id=None,device='cuda',expected=None):
        self.project,self.config,self.device = Path(project),deepcopy(config) if config is not None else default_config(),device
        if expected is None and self.config['retrieval_k'] != 100:
            raise ValueError('The primary six-channel benchmark requires a shared Top-100 retrieval budget')
        self.dataset = self.project/'data'/'ml-32m'
        self.splits = self.project/'data'/'splits'/'v1'
        kwargs = {} if expected is None else {'expected':expected}
        self.manifest,self.audit = verify_splits(self.splits,self.dataset,**kwargs)
        identity = dict(configuration=self.config,split_manifest_sha256=digest(self.splits/'split_manifest.json'),
                        code_sha256=code_sha,version='recsys_v1')
        computed = fingerprint(identity)[:20]
        if run_id is not None and run_id != computed:
            raise ValueError('Selected run ID does not match code/config/split checksums; restore its exact code and configuration')
        self.run_id = computed
        self.store = Store(self.project/'artifacts'/'recsys_v1',self.run_id)
        self.results = Store(self.project/'results'/'recsys_v1',self.run_id)
        self.store.json('experiment',dict(identity=identity,split_audit=self.audit,environment=environment('/tmp')))
        self.train = load_ratings(self.splits/'train.parquet')
        self.movies = pd.read_csv(self.dataset/'movies.csv',usecols=['movieId','title','genres'],dtype={'movieId':'int64'})
        self.ctx = build_context(self.train,self.movies)
        if len(self.train) != self.manifest['row_counts']['train']:
            raise ValueError('Full Train row count differs from manifest; no subsampling permitted')
        self.store.write('full_train/context',self.ctx.save,dict(train_rows=self.ctx.train_rows,cutoff=self.ctx.cutoff))
        self.validation = None
        self.models = None
        print('Run',self.run_id,'all Train rows:',len(self.train),'Policy B items:',int(self.ctx.eligible.sum()))

    def retrieval_milestone(self):
        self.validation = load_ratings(self.splits/'validation.parquet')
        truth, _ = relevance(self.validation,self.ctx)
        self.models = train_channels(self.ctx,self.config,self.store,'full_train',truth,self.device)
        summary,path = evaluate(self.models,self.ctx,self.validation,self.config,self.results,'validation_retrieval')
        self.validation_path = path
        self.results.json('milestone_1',dict(status='complete',channels=list(self.models),summary_stage='validation_retrieval',
                                           train_rows=self.ctx.train_rows,split_manifest_sha256=digest(self.splits/'split_manifest.json')))
        return summary

    def ranker_milestone(self):
        if self.results.load('milestone_1') is None:
            raise RuntimeError('Finish all six full-Train retrieval channels and Validation evaluation first')
        if self.validation is None:
            self.validation = load_ratings(self.splits/'validation.parquet')
        validation_path = self.results.load('validation_retrieval')
        old = self.store.load('ranker/model')
        if old is None:
            prefix,tail,boundary = internal_partition(self.train,self.config['internal_tail_days'])
            snapshot_ctx = build_context(prefix,self.movies)
            self.store.write('ranker_snapshot/context',snapshot_ctx.save,boundary)
            # Internal future labels NEVER drive retriever fitting, stopping, popularity or content profiles.
            # Fixed snapshot epoch counts use config defaults, explicitly independent of the internal labels.
            snapshot_models = train_channels(snapshot_ctx,self.config,self.store,'ranker_snapshot',None,self.device)
            internal_summary,internal_path = evaluate(snapshot_models,snapshot_ctx,tail,self.config,
                                                     self.results,'internal_ranker_candidates')
            rng = np.random.default_rng(self.config['seed'])
            stats = dict(queries=0,effective_queries=0,queries_no_retrieved_positive=0,queries_no_contrast=0,
                         positive_candidate_labels=0,observed_low_candidates=0,sampled_unlabeled_candidates=0)
            sampled_frames = []
            row_count = 0
            for p in sorted((internal_path/'features').glob('*.parquet')):
                sample,counts = sample_training(pd.read_parquet(p),rng,self.config['unlabeled_per_query'])
                for k,v in counts.items():
                    stats[k] += v
                if len(sample):
                    sample = sample[['userId','movieId','label']+FEATURES].copy()
                    row_count += len(sample)
                    require_memory(cpu=row_count*len(FEATURES)*4*4,stage='internal ranker table')
                    sampled_frames.append(sample)
            self.results.json('internal_ranker_sampling',dict(boundary=boundary,counts=stats,
                              sampled_rows=row_count,unlabeled_per_query=self.config['unlabeled_per_query'],
                              candidate_recall_ceiling=internal_summary['union_recall_ceiling']))
            if not sampled_frames:
                raise ValueError('No effective ranker training queries; inspect internal candidate recall and history')
            training = pd.concat(sampled_frames,ignore_index=True)
            del sampled_frames,snapshot_models,prefix,tail,snapshot_ctx
            # A deterministic Validation panel for tree early stopping; all users still receive final Validation metrics.
            val_truth,_ = relevance(self.validation,self.ctx)
            panel = np.array(sorted(val_truth),dtype=np.int64)
            if len(panel) > self.config['early_stop_users']:
                panel = np.sort(rng.choice(panel,self.config['early_stop_users'],replace=False))
            val_frames = []
            for p in sorted((validation_path/'features').glob('*.parquet')):
                f = pd.read_parquet(p,columns=['userId','movieId','label']+FEATURES)
                f = f[f.userId.isin(panel)]
                if len(f):
                    val_frames.append(f)
            validation = pd.concat(val_frames,ignore_index=True)
            from .rankers.lightgbm_ranker import fit_ranker
            with profile() as timing:
                model = fit_ranker(training,validation,self.config)
            snapshot_summary = json.loads((internal_path/'summary.json').read_text())['feature_summary']
            full_summary = json.loads((validation_path/'summary.json').read_text())['feature_summary']
            report = dict(internal_boundary=boundary,training_counts=stats,candidate_rows=len(training),
                          internal_candidate_recall_ceiling=internal_summary['union_recall_ceiling'],
                          internal_eligibility=internal_summary['eligibility'],
                          validation_early_stop_queries=len(panel),validation_early_stop_rows=len(validation),
                          sampling_policy='Keep retrieved observed positives and observed low ratings; at most '
                          f'{self.config["unlabeled_per_query"]} uniformly sampled unobserved candidates per query. '
                          'Drop training queries with no retrieved positive or no contrast. Inference keeps the entire union.',
                          limitation='Unobserved zero labels are weak supervision, not verified negatives. '
                          'Dropping zero-hit training queries causes selection bias. Unknown subsampling changes class balance.',
                          feature_distribution_mismatch=mismatch(snapshot_summary,full_summary),
                          mismatch_limitation='Snapshot and full-Train catalogs, embeddings, score scales and histories differ. '
                          'This measured shift can harm LambdaMART transfer. Diagnostics also reflect population/time shift; '
                          'they do not prove exchangeability. No guarantee of unbiased full-Train inference.',
                          feature_summary_sample_rows=min(len(training),100000),
                          sampled_training_feature_summary=distribution(training.head(100000)),
                          profile=timing,environment=environment('/tmp'),best_iteration=model.best_iteration)
            def write(p):
                model.save_model(str(p/'model.txt'))
                (p/'training_report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
            old = self.store.write('ranker/model',write,dict(snapshot='ranker_snapshot',inference='full_train',boundary=boundary))
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(old/'model.txt'))
        summary = evaluate_saved_ranker(model,validation_path,self.ctx,self.validation,self.config,self.results,'validation_ranker')
        self.results.json('milestone_2',dict(status='complete',ranker='ranker/model',validation='validation_ranker'))
        return summary

    def freeze_selection(self):
        """A deliberate notebook action seals this exact run; immutable and only after both milestones."""
        for marker in ['milestone_1','milestone_2']:
            if self.results.load(marker) is None:
                raise RuntimeError(f'Missing {marker}; no final Test evaluation yet')
        self.store.json('selected',dict(run_id=self.run_id,config=self.config,models='full_train',ranker='ranker/model',
                                       selection_basis='Validation only',test_used_for_selection=False))
        return self.run_id

    def final_evaluation(self,confirm_run_id):
        """Read selected artifacts; never calls fit, tuning, or resplitting."""
        if confirm_run_id != self.run_id or self.store.load('selected') is None:
            raise PermissionError('Set confirm_run_id to the exact frozen selected run ID before Test evaluation')
        models = load_channels(self.ctx,self.store,'full_train',self.device)
        path = self.store.load('ranker/model')
        if path is None:
            raise FileNotFoundError('Missing selected ranker')
        import lightgbm as lgb
        ranker = lgb.Booster(model_file=str(path/'model.txt'))
        test = load_ratings(self.splits/'test.parquet')
        summary,path = evaluate(models,self.ctx,test,self.config,self.results,'final_test',ranker)
        # Measure snapshot -> Test feature shift without changing any selected model.
        internal = self.results.load('internal_ranker_candidates')
        snapshot = json.loads((internal/'summary.json').read_text())['feature_summary']
        self.results.json('test_feature_shift',mismatch(snapshot,summary['feature_summary']))
        return summary

    def artifact_inventory(self):
        rows = []
        for namespace in ['full_train','ranker_snapshot']:
            for name in ['popularity','itemcf','als','genre','lightgcn','two_tower']:
                p = self.store.load(namespace+'/'+name)
                if p:
                    meta = json.loads((p/'COMPLETE.json').read_text())
                    training = json.loads((p/'training.json').read_text())
                    rows.append(dict(snapshot=namespace,channel=name,artifact_bytes=sum(x['bytes'] for x in meta['files'].values()),
                                     train_rows=training['train_rows'],**training['profile']))
        return pd.DataFrame(rows)
