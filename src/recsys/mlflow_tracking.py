"""Retrospective, idempotent imports of verified experiments; never fits a model."""
from datetime import datetime,timezone
from pathlib import Path
import json
import math
import tempfile
import mlflow
from mlflow import MlflowClient
from mlflow.entities import Metric,Param,RunTag
from .contracts import CHANNELS
from .runtime import digest,fingerprint
from .retrieval_artifacts import checked_manifest,load_checkpoint
from .historical_retrieval import load_historical_checkpoint
from .rankers.small_lambdamart import load_ranker
from .retrieval_workflow import ROOT
from .small_protocol import CROSS_USER_LIMITATION
from .mlflow_local import tracking_uri

VERSION='retrospective_import_v1'
RETRIEVAL='MovieLens-Small-Retrieval'
RANKING='MovieLens-Small-Fusion-Ranking'
RELEASE='MovieLens-Small-Release'


def client_for(uri=None):
    uri=uri or tracking_uri()
    return MlflowClient(tracking_uri=uri,registry_uri=uri)


def experiment(client,name):
    found=client.get_experiment_by_name(name)
    return found.experiment_id if found else client.create_experiment(name)


def flatten(value,prefix=''):
    result={}
    for k,v in value.items():
        key=prefix+'.'+k if prefix else k
        if isinstance(v,dict):result.update(flatten(v,key))
        else:result[key]=json.dumps(v) if isinstance(v,(list,bool)) or v is None else str(v)
    return result


def metric_names(row,prefix='validation/policy_b'):
    names=dict(recall20='recall_at_20',recall50='recall_at_50',recall100='recall_at_100',
        ndcg20='ndcg_at_20',hit20='hitrate_at_20',catalog_coverage20='catalog_coverage_at_20',
        catalog_coverage100='catalog_coverage_at_100',queries='queries',relevant_positives='eligible_positives')
    return {prefix+'/'+name:float(row[key]) for key,name in names.items() if row.get(key) is not None}


def import_record(client,experiment_name,key,record,parameters,metrics,files=(),lineage=None):
    """Same original identity with different source/metrics is a conflict, never an overwrite."""
    eid=experiment(client,experiment_name)
    signature=fingerprint(dict(record=record,parameters=parameters,metrics=metrics,lineage=lineage or {}))
    if "'" in key:raise ValueError('Unsafe import key')
    existing=client.search_runs([eid],filter_string=f"tags.`original_key` = '{key}'")
    if existing:
        if len(existing)!=1:raise ValueError('Ambiguous duplicate historical imports')
        old=existing[0]
        if old.data.tags.get('source_fingerprint')!=signature or old.info.status!='FINISHED':
            raise ValueError('Conflicting or incomplete historical source record: '+key)
        if old.data.params!=flatten(parameters) or old.data.metrics!=metrics:
            raise ValueError('Imported metrics/parameters were changed: '+key)
        with tempfile.TemporaryDirectory() as temporary:
            path=client.download_artifacts(old.info.run_id,'provenance/source_record.json',temporary)
            if json.loads(Path(path).read_text())!=record:raise ValueError('Imported provenance differs')
        return old.info.run_id,False
    tags=dict(mlflow_run_name=key,original_key=key,source_fingerprint=signature,retrospective_import='true',
        original_training_outside_mlflow='true',import_timestamp=datetime.now(timezone.utc).isoformat(),
        import_script_version=VERSION,import_script_sha256=digest(Path(__file__)),test_evaluated='false',
        original_phase=str(record['phase']),original_run_id=record['run_id'],
        original_git_revision=record.get('git_revision') or 'not_recorded_in_original_manifest',
        original_execution_timestamp=record.get('execution_timestamp') or 'not_recorded; no timestamp invented',
        dataset='grouplens/movielens-latest-small',split='small_v1',protocol='per_user_70_15_15',
        split_manifest_sha256=record['split_manifest_sha256'],evaluation_id=record.get('evaluation_id','not_applicable'),
        cutoff_identity=record.get('cutoff_identity','see_transitive_source_manifests'),
        temporal_limitation=CROSS_USER_LIMITATION,dirty_source_status=record.get('dirty_source_status','not_recorded'),
        lineage=json.dumps(lineage or {},sort_keys=True))
    tags.update(original_lock_sha256=record.get('original_lock_sha256') or 'not_recorded',
        original_manifest_sha256=record.get('manifest_sha256','not_recorded'),
        original_python=record.get('original_python','not_recorded'),
        dvc_lock_sha256=record.get('dvc_metadata_sha256',{}).get('dvc.lock','not_recorded'),
        catalog_policy='train_observed',seen_policy='all_train_rated')
    tags['mlflow.runName']=tags.pop('mlflow_run_name')
    tags={k:str(v) if v is not None else 'not_recorded' for k,v in tags.items()}
    run=client.create_run(eid,tags=tags);rid=run.info.run_id
    try:
        timestamp=int(datetime.now(timezone.utc).timestamp()*1000)
        if any(not math.isfinite(v) for v in metrics.values()):raise ValueError('Nonfinite import metric')
        client.log_batch(rid,params=[Param(k,v) for k,v in flatten(parameters).items()],
            metrics=[Metric(k,v,timestamp,0) for k,v in metrics.items()])
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'source_record.json';path.write_text(json.dumps(record,indent=2)+'\n')
            client.log_artifact(rid,str(path),'provenance')
        for path in dict.fromkeys(map(Path,files)):
            parent=path.resolve().parent
            folder=str(parent.relative_to(ROOT)) if parent.is_relative_to(ROOT) else parent.name
            client.log_artifact(rid,str(path),'original/'+folder)
        client.set_terminated(rid,'FINISHED')
        actual=client.get_run(rid)
        if actual.data.metrics!=metrics:raise ValueError('Metric round-trip differs')
        return rid,True
    except BaseException:
        client.set_terminated(rid,'FAILED');raise


def backfill(uri=None):
    client=client_for(uri);runs={};created={}
    dvc={p:digest(ROOT/p) for p in ['dvc.yaml','dvc.lock','params.yaml','data/raw/ml-latest-small.dvc']}
    dvc_lock=json.loads(json.dumps(__import__('yaml').safe_load((ROOT/'dvc.lock').read_text())))
    data_objects={o['path']:o['md5'] for stage in dvc_lock['stages'].values() for o in stage.get('outs',[])}
    def add(key,path,phase,params,metrics,parents=(),extra=()):
        path=Path(path);m=checked_manifest(path)
        refs=[path/'manifest.json',path/'COMPLETE.json',*map(Path,extra)]
        record=dict(phase=phase,run_id=m.get('run_id',m.get('producer_run_id',path.name)),
            manifest=m,manifest_sha256=digest(path/'manifest.json'),completion_sha256=digest(path/'COMPLETE.json'),
            source_files={str(p.relative_to(ROOT)):digest(p) for p in refs},
            git_revision=m.get('source_git_revision'),dirty_source_status=json.dumps(m.get('source_git_status','not_recorded')),
            split_manifest_sha256=m['split_manifest_sha256'],evaluation_id=m.get('evaluation_id',m.get('validation_evaluation_id','not_applicable')),
            cutoff_identity=m.get('training_cutoff'),dvc_metadata_sha256=dvc,dvc_data_objects=data_objects,
            original_python=m.get('resource_environment',{}).get('python','see_original_phase_report'),
            original_lock_sha256=m.get('training_lock_sha256',m.get('lock_sha256')))
        rid,new=import_record(client,RETRIEVAL if 'retriever/' in key else RANKING,key,record,params,metrics,refs,
                              {p:runs[p] for p in parents})
        runs[key]=rid;created[key]=new
    for source in CHANNELS:
        path=ROOT/'artifacts'/source/'phase3_v1_seed42_main';_,m=load_checkpoint(path,source)
        cp=ROOT/'artifacts/candidates'/source/'phase3_v1_seed42_main/validation_main';cm=checked_manifest(cp)
        metrics={}
        for label,prefix in [('eligible','validation/policy_b'),('all_observed','validation/all_observed')]:
            part=cm['measurements'][label];metrics.update(metric_names(part['cohorts'][0],prefix))
            metrics[prefix+'/candidate_count']=float(part['candidate_rows'])
            metrics[prefix+'/inference_seconds']=part['inference_seconds']
        metrics['training/seconds']=m['training_seconds']
        add('phase3/retriever/'+source,path,3,dict(model=source,config=m['config'],seed=m['seed'],training_kind=m['training_kind'],
            candidate_sha256=digest(cp/'candidates.parquet'),checkpoint_sha256=digest(path/'COMPLETE.json'),
            all_observed_evaluation_id=cm['measurements']['all_observed']['evaluation_id']),metrics,
            extra=[cp/'manifest.json',cp/'COMPLETE.json',cp/'metrics.json'])
    rr=ROOT/'artifacts/fusion/phase4_v1_validation';m=checked_manifest(rr)
    metrics=metric_names(m['metrics'][0]);metrics.update({'validation/union/macro_recall_ceiling':m['metrics'][0]['pool_recall'],
        'validation/union/micro_positive_coverage':m['diagnostics']['union_positive_micro_coverage'],
        'validation/union/candidate_pairs':float(m['union_rows'])})
    add('phase4/rrf',rr,4,dict(rrf_constant=m['rrf_constant'],source_budget=m['source_budget']),metrics,
        ['phase3/retriever/'+s for s in CHANNELS],extra=[rr/'diagnostics.json',rr/'ablations.json',rr/'metrics.json'])
    for source in CHANNELS:
        path=ROOT/'artifacts/historical'/source/'phase4_v1_snapshot';_,m=load_historical_checkpoint(path,source)
        cp=ROOT/'artifacts/historical_candidates'/source/'phase4_v1_snapshot';cm=checked_manifest(cp)
        add('phase4/historical_retriever/'+source,path,4,dict(model=source,config=m['config'],seed=m['seed'],training_kind=m['training_kind'],
            candidate_sha256=digest(cp/'candidates.parquet')),{'historical_retriever/training_seconds':m['training_seconds'],
            'historical_retriever/inference_seconds':cm['inference_seconds'],'historical_retriever/query_count':float(cm['queries'])},
            extra=[cp/'manifest.json',cp/'COMPLETE.json'])
    add('phase4/historical_union',ROOT/'artifacts/fusion/phase4_v1_rank_train',4,dict(rrf_constant=60),{},
        ['phase4/historical_retriever/'+s for s in CHANNELS])
    add('phase4/features',ROOT/'artifacts/ranking_features/phase4_v1',4,dict(feature_count=53),{},
        ['phase4/historical_union','phase4/rrf'],extra=[ROOT/'artifacts/ranking_features/phase4_v1/feature_schema.json'])
    rp=ROOT/'artifacts/ranking_training/phase4_v1';rm=checked_manifest(rp);stats=rm['statistics']
    add('phase4/ranking_dataset',rp,4,dict(sampling=rm['sampling'],seed=rm['seed']),
        {'historical_ranker/effective_queries':float(stats['effective_training_queries']),
         'historical_ranker/training_rows':float(stats['training_rows']),
         'historical_ranker/union_macro_recall_ceiling':stats['historical_union_macro_recall']},['phase4/features'],
        extra=[rp/'feasibility.json',rp/'feature_shift.json',rp/'historical_metrics.json'])
    lp=ROOT/'artifacts/lightgbm/phase4_v1';_,schema,lm=load_ranker(lp)
    add('phase4/lambdamart',lp,4,dict(config=lm['config'],selected_iteration=lm['best_iteration'],feature_schema_sha256=schema['sha256']),
        metric_names(lm['comparison']['lambdamart'][0])|{'validation/union/macro_recall_ceiling':lm['comparison']['common_union_recall_ceiling'],
            'historical_ranker/effective_queries':float(lm['training_groups']),'historical_ranker/training_rows':float(lm['training_rows'])},
        ['phase4/ranking_dataset','phase4/rrf'],extra=[lp/'feature_importance.csv',lp/'comparison.json',lp/'training_history.json',lp/'feature_schema.json'])
    result=dict(version=VERSION,tracking_uri=client.tracking_uri,runs=runs,created=created,run_count=len(runs),test_evaluated=False)
    return result
