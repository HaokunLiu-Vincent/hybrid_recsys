"""Independent LambdaMART workflow consuming saved fusion features; no retriever training."""
from pathlib import Path
from copy import deepcopy
import json
import shutil
import tempfile
import numpy as np
import pandas as pd
from ..artifacts import read_artifact,artifact_reference,iter_queries,ParquetSink,dump_json,source_hashes
from ..contracts import FRAMEWORK_VERSION,mapping_id,catalog_id
from ..data import FrozenData,EXPECTED_ROWS
from ..features import feature_columns
from ..evaluation import MetricAccumulator
from ..fusion.rrf import rrf
from ..runtime import Store,fingerprint,digest,profile,require_memory,environment
from .features import sample_training,mismatch


class RankingExperiment:
    def __init__(self, project, train_fusion, validation_fusion, config, mode='full_train', checkpoint=None,
                 expected=EXPECTED_ROWS):
        if mode not in ('full_train','load_evaluate'):
            raise ValueError('Ranking supports smoke_test externally, full_train or load_evaluate only')
        self.project,self.config,self.mode = Path(project),deepcopy(config),mode
        self.train_path = Path(train_fusion) if train_fusion else None
        self.validation_path = Path(validation_fusion)
        self.validation_manifest = read_artifact(self.validation_path)
        v = self.validation_manifest
        if v['partition'] != 'validation' or v['training_kind'] != 'full_train':
            raise ValueError('Ranker Validation must use full-Train Validation fusion features')
        self.data = FrozenData(project,expected).prepare()
        for key,value in dict(split_manifest_sha256=self.data.split_sha,evaluation_id=self.data.population.identifier,
                              mapping_id=mapping_id(self.data.ctx),catalog_id=catalog_id(self.data.ctx)).items():
            if v[key] != value:
                raise ValueError(f'Validation features mismatch {key}')
        self.columns = feature_columns()
        if v['feature_columns'] != self.columns:
            raise ValueError('Unexpected feature contract; label-side columns are never model inputs')
        self.checkpoint_path = Path(checkpoint) if checkpoint else None
        self.metadata = dict(framework_version=FRAMEWORK_VERSION,model_name='lightgbm',source_hashes=source_hashes('lightgbm'),
            split_manifest_sha256=self.data.split_sha,config=self.config,feature_columns=self.columns,
            validation_fusion=artifact_reference(self.validation_path),inference_catalog_id=v['catalog_id'],
            inference_mapping_id=v['mapping_id'],inference_training_cutoff=v['training_cutoff'])
        if mode == 'load_evaluate':
            if self.checkpoint_path is None:
                raise ValueError('An explicit LightGBM checkpoint is required')
            saved = read_artifact(self.checkpoint_path)
            for key in self.metadata:
                if saved[key] != self.metadata[key]:
                    raise ValueError(f'Ranker checkpoint incompatible with {key}')
            self.metadata,self.run_id = saved,saved['run_id']
        else:
            if self.train_path is None:
                raise ValueError('Historical snapshot fusion features are required; no in-sample fallback')
            t = read_artifact(self.train_path)
            if t['training_kind'] != 'historical_snapshot' or t['partition'] != 'rank_train':
                raise ValueError('Ranker training must consume historical snapshot rank_train candidates')
            if t['split_manifest_sha256'] != v['split_manifest_sha256'] or t['feature_columns'] != self.columns:
                raise ValueError('Ranker train and Validation split/feature contracts differ')
            boundary = self.data.manifest['boundaries_utc_and_unix']['validation_start_timestamp']
            if not t['training_cutoff'] < t['label_start'] < t['label_end'] <= boundary:
                raise ValueError('Historical labels must follow snapshot fitting and remain inside external Train')
            self.metadata.update(train_fusion=artifact_reference(self.train_path),internal_boundary={
                k:t[k] for k in ['training_cutoff','label_start','label_end','training_rows']})
            self.run_id = fingerprint(self.metadata)[:20]
            self.metadata['run_id'] = self.run_id
        self.store = Store(self.project/'artifacts'/'lightgbm',self.run_id)
        self.results = Store(self.project/'results'/'validation'/'lightgbm',self.run_id)

    def prepare_examples(self):
        if self.mode == 'load_evaluate':
            return None
        old = self.store.load('ranking_examples')
        if old:
            return old
        def write(root):
            training = ParquetSink(root/'train.parquet')
            validation = ParquetSink(root/'validation.parquet')
            rng = np.random.default_rng(self.config['seed'])
            stats = dict(queries=0,effective_queries=0,queries_no_retrieved_positive=0,queries_no_contrast=0,
                         positive_candidate_labels=0,observed_low_candidates=0,sampled_unlabeled_candidates=0)
            for user,frame in iter_queries(self.train_path/'features.parquet'):
                sampled,counts = sample_training(frame,rng,self.config['unlabeled_per_query'])
                for key,value in counts.items(): stats[key] += value
                training.add(sampled[['userId','movieId','label','feedback_kind']+self.columns])
            if not stats['effective_queries']:
                self.store.json('insufficient_examples',dict(stats=stats,internal_boundary=self.metadata['internal_boundary']))
                raise ValueError('No effective historical ranking queries; inspect saved sampling diagnostics and pool recall')
            training.close()
            users = self.data.population.users
            if len(users) > self.config['validation_queries']:
                users = np.sort(np.random.default_rng(self.config['seed']).choice(users,self.config['validation_queries'],replace=False))
            panel = set(map(int,users))
            for user,frame in iter_queries(self.validation_path/'features.parquet'):
                if user in panel:
                    validation.add(frame[['userId','movieId','label','feedback_kind']+self.columns])
            validation.close()
            snapshot_summary = json.loads((self.train_path/'metrics.json').read_text())
            full_summary = json.loads((self.validation_path/'metrics.json').read_text())
            report = dict(counts=stats,training_rows=training.rows,validation_rows=validation.rows,
                validation_panel_queries=len(panel),internal_boundary=self.metadata['internal_boundary'],
                candidate_recall_ceiling=snapshot_summary['union_recall_ceiling'],
                feature_distribution_mismatch=mismatch(snapshot_summary['feature_summary'],full_summary['feature_summary']),
                sampling_policy='All retrieved observed positives and observed low ratings; bounded uniform unknown candidates. '
                    'Drop no-positive/no-contrast training queries. Inference ranks the whole unsampled union.',
                limitation='Unknowns are not verified negatives. Query selection and sampling change the label distribution. '
                    'Historical and full-Train embeddings/catalogs/histories differ; measured shifts do not ensure unbiased transfer.')
            dump_json(root/'sampling_report.json',report)
            dump_json(root/'manifest.json',self.metadata)
        return self.store.write('ranking_examples',write,self.metadata)

    def fit_or_load(self,examples=None):
        path = self.checkpoint_path if self.mode == 'load_evaluate' else self.store.load('checkpoint')
        if path:
            read_artifact(path)
            import lightgbm as lgb
            self.model = lgb.Booster(model_file=str(path/'model.txt'))
            self.checkpoint_path = path
            return path
        if self.mode == 'load_evaluate':
            raise FileNotFoundError('Missing checkpoint; load_evaluate never trains')
        if examples is None:
            raise ValueError('Prepare historical examples first')
        from .lightgbm_ranker import fit_from_shards
        paths = [examples/'train.parquet',examples/'validation.parquet']
        require_memory(disk=sum(p.stat().st_size for p in paths)*2,root='/tmp',stage='local ranking feature staging')
        with tempfile.TemporaryDirectory(prefix='ranking-local-') as tmp:
            local = []
            for path in paths:
                copy = Path(tmp)/path.name
                shutil.copyfile(path,copy)
                if digest(path) != digest(copy): raise IOError('Local feature staging checksum mismatch')
                local.append(copy)
            with profile() as timing:
                self.model,history = fit_from_shards([local[0]],[local[1]],self.columns,self.config)
        def write(root):
            self.model.save_model(str(root/'model.txt'))
            dump_json(root/'manifest.json',dict(self.metadata,examples=artifact_reference(examples)))
            dump_json(root/'training_history.json',dict(history=history,profile=timing,environment=environment('/tmp')))
            pd.DataFrame(dict(feature=self.columns,gain=self.model.feature_importance('gain'),
                              split=self.model.feature_importance('split'))).to_parquet(root/'feature_importance.parquet',index=False)
            shutil.copyfile(examples/'sampling_report.json',root/'sampling_report.json')
        self.checkpoint_path = self.store.write('checkpoint',write,self.metadata)
        return self.checkpoint_path

    def evaluate(self):
        old = self.results.load('evaluation')
        if old: return old
        def write(root):
            metrics = MetricAccumulator(self.data.ctx,self.data.population,self.config)
            with profile() as timing:
                seen = set()
                for user,frame in iter_queries(self.validation_path/'features.parquet'):
                    if user not in self.data.population.truth or user in seen:
                        raise ValueError('Ranking candidates differ from fixed Validation population')
                    seen.add(user)
                    scores = self.model.predict(frame[self.columns])
                    ordered = frame.assign(ranking_score=scores).sort_values(['ranking_score','movieId'],ascending=[False,True])
                    metrics.add('lightgbm',user,ordered.movieId.tolist())
                    metrics.add('rrf',user,rrf(frame,constant=self.validation_manifest['config'].get('rrf_constant',60)))
                for user in set(self.data.population.truth)-seen:
                    metrics.add('lightgbm',user,[])
                    metrics.add('rrf',user,[])
                summary = metrics.finish(root)
            summary.update(profile=timing,checkpoint=artifact_reference(self.checkpoint_path))
            dump_json(root/'metrics.json',summary)
            dump_json(root/'manifest.json',self.metadata)
        return self.results.write('evaluation',write,self.metadata)
