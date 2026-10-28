"""Small CPU retrieval orchestration. Algorithms live only in the retained retriever modules."""
from datetime import datetime,timezone
from pathlib import Path
import copy
import importlib.metadata
import json
import re
import subprocess
import time
import uuid
import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from .artifacts import ParquetSink,candidate_schema,dump_json
from .contracts import CHANNELS,compatible,validate_candidates,mapping_id,catalog_id
from .metrics import query_metrics
from .runtime import Store,digest,fingerprint,profile,environment
from .small_benchmark import load_training,load_validation,selected_manifest
from .small_protocol import POLICY,CROSS_USER_LIMITATION
from .training import fit_retriever
from .retrieval_artifacts import VERSION,critical_sources,save_context,load_checkpoint,checked_manifest

ROOT=Path(__file__).resolve().parents[2]
COHORTS=['cold_0','sparse_1_4','moderate_5_19','warm_20_plus']
SCORES=['recall20','recall50','recall100','ndcg20','hit20','pool_recall']


def identifier(value=None):
    value=value or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,100}',value):raise ValueError('Unsafe run identifier')
    return value


def relative(path):
    path=Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def packages():
    return {p:importlib.metadata.version(p) for p in
            ['numpy','scipy','pandas','pyarrow','psutil','torch','implicit','threadpoolctl','pytz']}


def cpu():
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)


def read_config(path,model):
    config=yaml.safe_load(Path(path).read_text())
    if config.get('model')!=model or model not in CHANNELS:raise ValueError('Model/config identity mismatch')
    if config.get('cpu_threads')!=1 or config.get('retrieval_k')!=100:
        raise ValueError('This benchmark requires one CPU thread and identical Top-100 budgets')
    if not 1<=config['query_batch']<=128 or not 1<=config['item_block']<=4096:
        raise ValueError('Invalid bounded inference configuration')
    if not isinstance(config.get('seed'),int):raise ValueError('Integer seed required')
    if model in ('lightgcn','two_tower'):
        c=config[model]
        if not 1<=c['epochs']<=100 or not 1<=c['dim']<=128 or not 1<=c['batch_size']<=8192:
            raise ValueError('Neural configuration exceeds the initial CPU budget')
    return config


def identity(training):
    m=training.manifest
    return dict(framework_version=VERSION,dataset_id=m['dataset_id'],split_version=m['version'],
        split_manifest_sha256=training.split_manifest_sha256,protocol_name=m['protocol_name'],
        raw_sha256=m['raw_sha256'],training_kind='external_train_per_user',
        training_cutoff='per_user:'+m['files']['user_cutoffs.parquet']['sha256'],
        per_user_cutoff_table_sha256=m['files']['user_cutoffs.parquet']['sha256'],
        cutoff_reference='Frozen user_cutoffs.parquet hash; actual histories are serialized Train only. No global cutoff asserted.',
        mapping_id=mapping_id(training.ctx),catalog_id=catalog_id(training.ctx),
        catalog_policy='train_observed',positive_threshold=4.0,policy=POLICY,
        cross_user_temporal_disclosure=CROSS_USER_LIMITATION)


def score_probe(model,ctx):
    """Train-only sanity probe: trained channels must not conceal failure as popularity."""
    probes=[]
    for u in np.flatnonzero(ctx.positive_count>0)[:12]:
        ids=np.flatnonzero(ctx.eligible)
        ids=ids[~np.isin(ids,ctx.seen(int(u)))][:256]
        if not len(ids) or model.fallback(int(u)):continue
        score=np.asarray(model.scores(int(u),ids))
        if not np.isfinite(score).all():raise FloatingPointError('Nonfinite model-specific score probe')
        probes.append(dict(userId=int(ctx.users[u]),score_std=float(score.std()),
                           differs_from_popularity=not np.allclose(score,ctx.popularity[ids])))
    if not probes or not any(p['score_std']>0 for p in probes):raise ValueError('No varying model-specific scores on Train-only probes')
    if model.name!='popularity' and not any(p['differs_from_popularity'] for p in probes):
        raise ValueError('Non-popularity channel returned only popularity scores')
    return probes


def train(model,config,data_dir=ROOT/'data/splits/small_v1',artifact_root=ROOT/'artifacts',run_id=None):
    config=read_config(config,model);run_id=identifier(run_id);destination=Path(artifact_root)/model/run_id
    if destination.exists():raise FileExistsError('Completed/incomplete run already exists; choose a new run ID')
    cpu();wall=time.perf_counter();training=load_training(data_dir);ctx=training.ctx
    if model in ('lightgcn','two_tower') and not ctx.positive.nnz:
        raise ValueError('BPR model requires historical positive edges; no substitute channel is permitted')
    best=dict(score=-np.inf,epoch=None,state=None)
    def snapshot(epoch,model,logs,state):
        score=logs[-1]['selection_score']
        if not np.isfinite(score):raise FloatingPointError('Nonfinite training selection score')
        if score>best['score']:
            best.update(score=score,epoch=epoch,state=copy.deepcopy(state))
    with threadpool_limits(limits=1),profile() as fit_measurement:
        fitted=fit_retriever(ctx,model,config,selector=None,checkpoint=snapshot,device='cpu')
    probes=score_probe(fitted,ctx)
    metadata=identity(training)
    metadata.update(model_name=model,run_id=run_id,status='completed',device='cpu',
        config=config,config_sha256=fingerprint(config),seed=config['seed'],
        critical_source_hashes=critical_sources(model),workflow_source_sha256=digest(Path(__file__)),
        source_git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True).splitlines(),
        dvc_metadata_sha256={p:digest(ROOT/p) for p in ['dvc.yaml','dvc.lock','data/raw/ml-latest-small.dvc']},
        training_lock_sha256=digest(ROOT/'uv.lock'),packages=packages(),resource_environment=environment(artifact_root if Path(artifact_root).exists() else ROOT),
        train_rows=ctx.train_rows,positive_edges=int(ctx.positive.nnz),train_known_items=int(ctx.eligible.sum()),
        context_fit_seconds=training.context_fit_seconds,algorithm_fit_seconds=fit_measurement['seconds'],
        training_seconds=training.context_fit_seconds+fit_measurement['seconds'],
        preparation_and_fit_wall_seconds=time.perf_counter()-wall,peak_fit_rss_bytes=fit_measurement['peak_rss_bytes'],
        selected_epoch=best['epoch'],selection_rule='Minimum observed Train BPR loss for neural snapshots; fixed configuration, no Validation-based tuning',
        validation_access_during_fit=False,test_target_access=False,train_only_score_probes=probes,
        unknown_sampling='Neural BPR: one uniform Train-catalog unknown per positive, rejecting every observed high/low Train pair; unknown is not a verified dislike',
        als_supervision='Positive rating >=4: preference 1/confidence 1+positive_confidence; observed low: preference 0/confidence 1+low_confidence; unobserved: preference 0/confidence 1')
    def writer(path):
        save_context(ctx,path);fitted.save(path)
        cutoffs=training.train.groupby('userId',sort=True).timestamp.max().rename('train_max_timestamp').reset_index()
        cutoffs.to_parquet(path/'train_cutoffs.parquet',index=False)
        if best['state'] is not None:
            import torch
            torch.save(best['state'],path/'training_state.pt')
            state=torch.load(path/'training_state.pt',map_location='cpu',weights_only=True)
            if model=='lightgcn':
                from .retrievers.lightgcn import graph_tensor,propagation
                with torch.no_grad():z=propagation(graph_tensor(ctx,'cpu'),state['base'],config['lightgcn']['layers']).numpy()
                actual=(z[:len(ctx.users)],z[len(ctx.users):])
            else:
                from .retrievers.two_tower import Towers,export_vectors
                net=Towers(ctx,config['two_tower']);net.load_state_dict(state['network'])
                reconstructed=export_vectors(net,ctx,'cpu');actual=(reconstructed.U,reconstructed.V)
            for a,b in zip(actual,(fitted.U,fitted.V)):
                np.testing.assert_allclose(a,b,rtol=0,atol=1e-7,err_msg='Neural training-state/export mismatch')
            metadata['neural_training_state_export_parity']=True
        dump_json(path/'training_log.json',fitted.logs)
        metadata['serialized_train_cutoffs_sha256']=digest(path/'train_cutoffs.parquet')
        # Checkpoint -> independent load -> identical prediction check before COMPLETE is published.
        from .retrieval_artifacts import load_context
        from .training import load_model
        loaded=load_model(load_context(path),model,path,device='cpu');loaded.name=model
        users=ctx.users[:min(16,len(ctx.users))]
        before=fitted.retrieve_many(users,100,config['item_block'])
        after=loaded.retrieve_many(users,100,config['item_block'])
        if before!=after:raise ValueError('Checkpoint save/reload prediction parity failed')
        metadata['reload_probe_queries']=len(users);metadata['reload_prediction_parity']=True
        dump_json(path/'manifest.json',metadata)
    path=Store(artifact_root,model).write(run_id,writer)
    load_checkpoint(path,model,training.split_manifest_sha256)
    return dict(status='completed',model=model,checkpoint=str(path),training_seconds=metadata['training_seconds'])


def summarize_candidates(frame,population,ctx):
    rows=[];groups={int(u):g for u,g in frame.groupby('userId',sort=False)}
    for user in population.users:
        group=groups.get(int(user));ranked=[] if group is None else group.movieId.tolist()
        rows.append(dict(userId=int(user),cohort=population.cohorts[int(user)],candidate_count=len(ranked),
                         **query_metrics(ranked,population.truth[int(user)])))
    per_query=pd.DataFrame(rows);summaries=[]
    for cohort in ['all',*COHORTS]:
        values=per_query if cohort=='all' else per_query[per_query.cohort.eq(cohort)]
        users=set(map(int,values.userId));selected=frame[frame.userId.isin(users)]
        summaries.append(dict(cohort=cohort,queries=len(values),
            relevant_positives=sum(len(population.truth[u]) for u in users),
            **{s:float(values[s].mean()) if len(values) else None for s in SCORES},
            catalog_coverage100=float(selected.movieId.nunique()/ctx.eligible.sum()) if len(values) else None,
            empty_candidate_queries=int(values.candidate_count.eq(0).sum()),
            undefined_reason=None if len(values) else 'No eligible-positive Validation queries in this cohort'))
    return per_query,summaries


def evaluate(model,checkpoint,split='validation',data_dir=ROOT/'data/splits/small_v1',
             output_root=ROOT/'artifacts/candidates',run_id=None):
    if split!='validation':raise ValueError('Phase 3 never opens Test targets')
    cpu();training=load_training(data_dir)
    fitted,producer=load_checkpoint(checkpoint,model,training.split_manifest_sha256)
    if mapping_id(training.ctx)!=producer['mapping_id'] or catalog_id(training.ctx)!=producer['catalog_id']:
        raise ValueError('Evaluation/model canonical mapping mismatch')
    config=producer['config'];run_id=identifier(run_id)
    root=Path(output_root)/model/producer['run_id']
    if (root/run_id).exists():raise FileExistsError('Evaluation already exists; choose a new run ID')
    primary=load_validation(data_dir,training);diagnostic=load_validation(data_dir,training,'all_observed')
    captured={}
    def writer(path):
        timings={};summaries={};files={}
        # Separate namespaces/identities, not a union of retrieval models.
        for label,data in [('eligible',primary),('all_observed',diagnostic)]:
            filename='candidates.parquet' if label=='eligible' else 'all_observed_candidates.parquet'
            sink=ParquetSink(path/filename,schema=candidate_schema());users=data.population.users
            allowed=set(map(int,users));start=time.perf_counter();scoring_seconds=0.
            with threadpool_limits(limits=1):
                for offset in range(0,len(users),config['query_batch']):
                    chunk=users[offset:offset+config['query_batch']]
                    before=time.perf_counter()
                    output=fitted.retrieve_many(chunk,config['retrieval_k'],config['item_block'])
                    scoring_seconds+=time.perf_counter()-before
                    for user in chunk:
                        frame=pd.DataFrame(output[int(user)],columns=candidate_schema().names)
                        if not frame.empty:frame=frame.astype(dict(userId='int64',movieId='int64',score='float64',rank='int32',source='string',fallback='bool'))
                        validate_candidates(frame,training.ctx,model,100,allowed);sink.add(frame)
            sink.close();timings[label]=time.perf_counter()-start
            candidates=pd.read_parquet(path/filename)
            validate_candidates(candidates,training.ctx,model,100,allowed)
            if not set(map(int,candidates.userId)).issubset(allowed):raise ValueError('Unexpected candidate query')
            per_query,groups=summarize_candidates(candidates,data.population,training.ctx)
            per_query.to_parquet(path/(label+'_per_query_metrics.parquet'),index=False)
            summaries[label]=dict(evaluation_id=data.population.identifier,denominator=label,
                cohorts=groups,candidate_rows=len(candidates),inference_seconds=scoring_seconds,
                candidate_pipeline_seconds=timings[label],
                empty_candidate_queries=int(per_query.candidate_count.eq(0).sum()),
                fallback_queries=int(candidates.groupby('userId').fallback.any().sum()),
                eligibility=data.population.report,
                catalog_ineligible_positive_target_share=data.population.report['policy_B_catalog_excluded']/data.population.report['observed_positive_interactions'])
            files[label]=dict(path=filename,sha256=digest(path/filename),bytes=(path/filename).stat().st_size)
        manifest=identity(training)
        manifest.update(model_name=model,producer_run_id=producer['run_id'],evaluation_run_id=run_id,status='completed',
            source_checkpoint=dict(path=relative(checkpoint),manifest_sha256=digest(Path(checkpoint)/'manifest.json')),
            config=config,config_sha256=producer['config_sha256'],seed=producer['seed'],
            training_lock_sha256=producer['training_lock_sha256'],packages=producer['packages'],
            evaluation_id=primary.population.identifier,top_k=100,phase='validation',
            label_start=dict(per_user_cutoff_table=manifest['per_user_cutoff_table_sha256'],window='validation_start'),
            label_end=dict(per_user_cutoff_table=manifest['per_user_cutoff_table_sha256'],window='validation_end'),
            candidate_files=files,critical_source_hashes=critical_sources(model),workflow_source_sha256=digest(Path(__file__)),
            training_seconds=producer['training_seconds'],measurements=summaries,test_target_access=False)
        dump_json(path/'metrics.json',summaries);dump_json(path/'manifest.json',manifest)
        captured.update(manifest)
    result=Store(root.parent,root.name).write(run_id,writer)
    checked_manifest(result)
    return dict(status='completed',model=model,evaluation=str(result),metrics=captured['measurements']['eligible']['cohorts'][0])


def benchmark(evaluations,split='validation',output_root=ROOT/'outputs/phase3',run_id=None):
    if split!='validation':raise ValueError('Phase 3 benchmark supports Validation only')
    manifests={};records=[]
    for path in evaluations:
        path=Path(path);manifest=checked_manifest(path);model=manifest['model_name']
        if model in manifests:raise ValueError('Exactly one selected evaluation per independent model is required')
        checkpoint=ROOT/manifest['source_checkpoint']['path']
        if digest(checkpoint/'manifest.json')!=manifest['source_checkpoint']['manifest_sha256']:
            raise ValueError('Evaluation/checkpoint manifest mismatch')
        _,producer=load_checkpoint(checkpoint,model,manifest['split_manifest_sha256'])
        if manifest['config_sha256']!=producer['config_sha256'] or manifest['training_lock_sha256']!=producer['training_lock_sha256']:
            raise ValueError('Evaluation/producer configuration or training lock mismatch')
        if manifest['phase']!='validation' or manifest['evaluation_id']!=manifest['measurements']['eligible']['evaluation_id']:
            raise ValueError('Mixed evaluation phase/denominator identities')
        manifests[model]=manifest
        overall=manifest['measurements']['eligible']['cohorts'][0]
        records.append(dict(model=model,**overall,training_seconds=producer['training_seconds'],
            inference_seconds=manifest['measurements']['eligible']['inference_seconds'],
            candidate_rows=manifest['measurements']['eligible']['candidate_rows'],
            catalog_ineligible_positive_target_share=manifest['measurements']['eligible']['catalog_ineligible_positive_target_share'],
            evaluation_path=relative(path),checkpoint_path=relative(checkpoint)))
    compatible(manifests,require_all=True)
    reference=next(iter(manifests.values()))
    if any(m['training_lock_sha256']!=reference['training_lock_sha256'] or m['packages']!=reference['packages'] for m in manifests.values()):
        raise ValueError('Selected model environments/locks differ')
    if any(m['measurements']['eligible']['cohorts'][0]['queries']!=reference['measurements']['eligible']['cohorts'][0]['queries'] for m in manifests.values()):
        raise ValueError('Model evaluation query counts differ')
    run_id=identifier(run_id)
    if (Path(output_root)/run_id).exists():raise FileExistsError('Benchmark output exists')
    def writer(path):
        frame=pd.DataFrame(records).set_index('model').loc[list(CHANNELS)].reset_index()
        frame.to_csv(path/'benchmark.csv',index=False)
        dump_json(path/'benchmark.json',dict(status='completed',scope='Six independent Validation retrievers; no RRF/ranker execution',
            evaluation_id=reference['evaluation_id'],records=frame.to_dict('records'),
            evaluation_artifacts={m:dict(path=relative(p),manifest_sha256=digest(Path(p)/'manifest.json'))
                for p in evaluations for m in [checked_manifest(p)['model_name']]},
            cross_user_temporal_disclosure=CROSS_USER_LIMITATION))
        dump_json(path/'manifest.json',dict(framework_version=VERSION,status='completed',models=list(CHANNELS),
            evaluation_id=reference['evaluation_id'],top_k=100,phase='validation',no_implicit_fitting=True))
    path=Store(Path(output_root).parent,Path(output_root).name).write(run_id,writer)
    return dict(status='completed',benchmark=str(path),records=records)
