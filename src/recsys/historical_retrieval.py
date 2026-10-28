"""Independent historical snapshot fitting/candidates; later labels are not fitting inputs."""
from dataclasses import dataclass
from pathlib import Path
import copy
import importlib.metadata
import json
import subprocess
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .data import build_context
from .training import fit_retriever,load_model
from .contracts import mapping_id,catalog_id,validate_candidates,compatible,CHANNELS
from .runtime import Store,digest,fingerprint,profile
from .artifacts import dump_json,ParquetSink,candidate_schema
from .retrieval_artifacts import save_context,load_context,checked_manifest,critical_sources
from .retrieval_workflow import ROOT,identifier,relative,cpu,packages,read_config,score_probe
from .small_benchmark import selected_manifest
from .small_protocol import POLICY,CROSS_USER_LIMITATION
from .fusion.small_rrf import union_from_frames

VERSION='small_historical_retrieval_v1'


@dataclass
class HistoricalData:
    history: pd.DataFrame
    ctx: object
    manifest: dict
    split_sha256: str
    snapshot_id: str
    context_seconds: float


def historical_file(directory,filename,labels=False):
    allowed={'ranker_snapshot_train.parquet','movies.parquet'}
    if labels:allowed.add('ranker_label_window.parquet')
    if filename not in allowed:raise ValueError('File outside historical stage input allowlist')
    directory=Path(directory);manifest=selected_manifest(directory,[])
    record=manifest['files'][filename];path=directory/filename
    if path.is_symlink() or path.stat().st_size!=record['bytes'] or digest(path)!=record['sha256']:
        raise ValueError('Historical input checksum mismatch: '+filename)
    return manifest


def load_historical_training(directory):
    directory=Path(directory);m=historical_file(directory,'ranker_snapshot_train.parquet')
    historical_file(directory,'movies.parquet')
    history=pd.read_parquet(directory/'ranker_snapshot_train.parquet')
    movies=pd.read_parquet(directory/'movies.parquet')
    start=time.perf_counter();ctx=build_context(history,movies);seconds=time.perf_counter()-start
    if len(history)!=m['files']['ranker_snapshot_train.parquet']['rows']:
        raise ValueError('Historical window row-count mismatch')
    sid=fingerprint(dict(kind='per_user_historical_snapshot_v1',split_manifest_sha256=digest(directory/'split_manifest.json'),
        historical_input_sha256=m['files']['ranker_snapshot_train.parquet']['sha256'],
        historical_cutoffs_sha256=m['files']['ranker_user_cutoffs.parquet']['sha256']))
    return HistoricalData(history,ctx,m,digest(directory/'split_manifest.json'),sid,seconds)


def historical_identity(data):
    query_id=fingerprint(dict(snapshot_id=data.snapshot_id,query_users=data.ctx.users.tolist(),policy=POLICY))
    return dict(framework_version=VERSION,training_kind='historical_snapshot_per_user',
        dataset_id=data.manifest['dataset_id'],split_version=data.manifest['version'],split_manifest_sha256=data.split_sha256,
        snapshot_id=data.snapshot_id,evaluation_id=query_id,query_policy='All independently derived historical Train users, without consulting later labels',
        history_sha256=data.manifest['files']['ranker_snapshot_train.parquet']['sha256'],
        training_cutoff='per_user:'+data.manifest['files']['ranker_user_cutoffs.parquet']['sha256'],
        per_user_cutoff_table_sha256=data.manifest['files']['ranker_user_cutoffs.parquet']['sha256'],
        label_start='per_user_after_historical_prefix',label_end='per_user_before_external_validation',
        mapping_id=mapping_id(data.ctx),catalog_id=catalog_id(data.ctx),catalog_policy='train_observed',
        positive_threshold=4.0,policy=POLICY,top_k=100,cross_user_temporal_disclosure=CROSS_USER_LIMITATION)


def load_historical_checkpoint(path,model=None,expected=None):
    path=Path(path);m=checked_manifest(path)
    if m.get('training_kind')!='historical_snapshot_per_user' or m.get('framework_version')!=VERSION:
        raise ValueError('Historical training rejects Full-Train or incompatible checkpoints')
    if model and m['model_name']!=model:raise ValueError('Historical model identity mismatch')
    if m['critical_source_hashes']!=critical_sources(m['model_name']) or m['historical_source_sha256']!=digest(Path(__file__)):
        raise ValueError('Historical source compatibility mismatch')
    if m['config_sha256']!=fingerprint(m['config']) or m['policy']!=POLICY:
        raise ValueError('Historical model configuration/policy mismatch')
    for package,version in m['packages'].items():
        if importlib.metadata.version(package)!=version:raise ValueError('Historical dependency mismatch: '+package)
    ctx=load_context(path)
    if mapping_id(ctx)!=m['mapping_id'] or catalog_id(ctx)!=m['catalog_id']:
        raise ValueError('Historical mapping/catalog mismatch')
    if expected:
        identity=historical_identity(expected)
        for key in ['snapshot_id','history_sha256','training_cutoff','mapping_id','catalog_id','evaluation_id']:
            if m[key]!=identity[key]:raise ValueError('Wrong historical cutoff/window identity: '+key)
    fitted=load_model(ctx,m['model_name'],path,device='cpu');fitted.name=m['model_name']
    return fitted,m


def train_historical(model,config,data_dir=ROOT/'data/splits/small_v1',output_root=ROOT/'artifacts/historical',run_id=None):
    cpu();config=read_config(config,model);run_id=identifier(run_id)
    if (Path(output_root)/model/run_id).exists():raise FileExistsError('Historical run already exists')
    data=load_historical_training(data_dir);best=dict(score=-np.inf,epoch=None,state=None)
    def checkpoint(epoch,fitted,logs,state):
        if logs[-1]['selection_score']>best['score']:
            best.update(score=logs[-1]['selection_score'],epoch=epoch,state=copy.deepcopy(state))
    with threadpool_limits(limits=1),profile() as measurement:
        fitted=fit_retriever(data.ctx,model,config,selector=None,checkpoint=checkpoint,device='cpu')
    probes=score_probe(fitted,data.ctx);m=historical_identity(data)
    m.update(model_name=model,run_id=run_id,status='completed',config=config,config_sha256=fingerprint(config),seed=config['seed'],
        device='cpu',train_rows=len(data.history),positive_edges=int(data.ctx.positive.nnz),
        train_known_items=int(data.ctx.eligible.sum()),context_fit_seconds=data.context_seconds,
        algorithm_fit_seconds=measurement['seconds'],training_seconds=data.context_seconds+measurement['seconds'],
        peak_rss_bytes=measurement['peak_rss_bytes'],selected_epoch=best['epoch'],
        selection_rule='Fixed Phase 3 configuration; retained minimum observed Train BPR snapshot; no later-label selector',
        packages=packages(),lock_sha256=digest(ROOT/'uv.lock'),critical_source_hashes=critical_sources(model),
        historical_source_sha256=digest(Path(__file__)),source_git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        later_label_access_during_fit=False,full_train_factors_or_context_reused=False,test_access=False,score_probes=probes,
        dvc_metadata_sha256={p:digest(ROOT/p) for p in ['dvc.yaml','dvc.lock']})
    def writer(path):
        save_context(data.ctx,path);fitted.save(path)
        data.history.groupby('userId',sort=True).timestamp.max().rename('history_max_timestamp').reset_index().to_parquet(path/'train_cutoffs.parquet',index=False)
        if best['state'] is not None:
            import torch
            torch.save(best['state'],path/'training_state.pt')
        dump_json(path/'training_log.json',fitted.logs)
        reloaded=load_model(load_context(path),model,path,device='cpu');reloaded.name=model
        users=data.ctx.users[:16]
        if fitted.retrieve_many(users,100,config['item_block'])!=reloaded.retrieve_many(users,100,config['item_block']):
            raise ValueError('Historical checkpoint reload prediction mismatch')
        m['reload_prediction_parity']=True;dump_json(path/'manifest.json',m)
    path=Store(Path(output_root),model).write(run_id,writer)
    load_historical_checkpoint(path,model,data)
    return dict(status='completed',checkpoint=str(path),model=model,training_seconds=m['training_seconds'])


def historical_candidates(checkpoint,data_dir=ROOT/'data/splits/small_v1',output_root=ROOT/'artifacts/historical_candidates',run_id=None):
    cpu();data=load_historical_training(data_dir);fitted,producer=load_historical_checkpoint(checkpoint,expected=data)
    source=producer['model_name'];config=producer['config'];run_id=identifier(run_id)
    root=Path(output_root)/source
    if (root/run_id).exists():raise FileExistsError('Historical candidates already exist')
    result={}
    def writer(path):
        sink=ParquetSink(path/'candidates.parquet',schema=candidate_schema());start=time.perf_counter()
        users=data.ctx.users;allowed=set(map(int,users));scoring_seconds=0.
        with threadpool_limits(limits=1):
            for offset in range(0,len(users),config['query_batch']):
                chunk=users[offset:offset+config['query_batch']];before=time.perf_counter()
                output=fitted.retrieve_many(chunk,100,config['item_block']);scoring_seconds+=time.perf_counter()-before
                for user in chunk:
                    frame=pd.DataFrame(output[int(user)],columns=candidate_schema().names)
                    if len(frame):frame=frame.astype(dict(userId='int64',movieId='int64',score='float64',rank='int32',source='string',fallback='bool'))
                    validate_candidates(frame,data.ctx,source,100,allowed);sink.add(frame)
        sink.close();frame=pd.read_parquet(path/'candidates.parquet')
        validate_candidates(frame,data.ctx,source,100,allowed)
        pd.DataFrame({'userId':users}).to_parquet(path/'query_users.parquet',index=False)
        m=historical_identity(data)
        m.update(status='completed',model_name=source,producer_run_id=producer['run_id'],candidate_run_id=run_id,
            source_checkpoint=dict(path=relative(checkpoint),manifest_sha256=digest(Path(checkpoint)/'manifest.json')),
            config=config,config_sha256=producer['config_sha256'],candidate_sha256=digest(path/'candidates.parquet'),
            candidate_rows=len(frame),queries=len(users),inference_seconds=scoring_seconds,
            pipeline_seconds=time.perf_counter()-start,later_labels_read=False,test_access=False,
            lock_sha256=digest(ROOT/'uv.lock'),source_sha256=digest(Path(__file__)))
        dump_json(path/'manifest.json',m);result.update(m)
    path=Store(root.parent,root.name).write(run_id,writer);checked_manifest(path)
    return dict(status='completed',candidates=str(path),queries=result['queries'],rows=result['candidate_rows'])


def fuse_historical(candidates,data_dir=ROOT/'data/splits/small_v1',output_root=ROOT/'artifacts/fusion',run_id=None):
    data=load_historical_training(data_dir);frames={};manifests={};references={}
    expected=historical_identity(data)
    for path in candidates:
        path=Path(path);m=checked_manifest(path);source=m['model_name']
        if source in frames:raise ValueError('Duplicate historical source')
        for key in ['training_kind','snapshot_id','evaluation_id','training_cutoff','catalog_id','mapping_id']:
            if m.get(key)!=expected[key]:raise ValueError('Historical fusion rejects incompatible/Full-Train stream: '+key)
        checkpoint=ROOT/m['source_checkpoint']['path'];load_historical_checkpoint(checkpoint,source,data)
        if digest(checkpoint/'manifest.json')!=m['source_checkpoint']['manifest_sha256']:raise ValueError('Historical checkpoint reference mismatch')
        if digest(path/'candidates.parquet')!=m['candidate_sha256']:raise ValueError('Historical candidate checksum mismatch')
        frames[source]=pd.read_parquet(path/'candidates.parquet');manifests[source]=m
        references[source]=dict(path=relative(path),manifest_sha256=digest(path/'manifest.json'),checkpoint=m['source_checkpoint'])
    compatible(manifests,require_all=True);frames={s:frames[s] for s in CHANNELS}
    run_id=identifier(run_id)
    if (Path(output_root)/run_id).exists():raise FileExistsError('Historical fusion already exists')
    start=time.perf_counter();pool=union_from_frames(frames,data.ctx,data.ctx.users);seconds=time.perf_counter()-start
    m=historical_identity(data);m.update(kind='historical_fusion',status='completed',run_id=run_id,rrf_constant=60,
        inputs=references,union_rows=len(pool),queries=len(data.ctx.users),fusion_seconds=seconds,later_labels_read=False,
        lock_sha256=digest(ROOT/'uv.lock'),source_sha256=digest(Path(__file__)))
    def writer(path):
        pool.to_parquet(path/'union.parquet',index=False);m['union_sha256']=digest(path/'union.parquet')
        pd.DataFrame({'userId':data.ctx.users}).to_parquet(path/'query_users.parquet',index=False)
        dump_json(path/'manifest.json',m)
    path=Store(Path(output_root).parent,Path(output_root).name).write(run_id,writer);checked_manifest(path)
    return dict(status='completed',fusion=str(path),queries=len(data.ctx.users),rows=len(pool))
