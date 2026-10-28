"""Historical candidate supervision and native LambdaMART with explicit feature/query contracts."""
from pathlib import Path
import json
import subprocess
import time
import warnings
import numpy as np
import pandas as pd
import yaml
from ..artifacts import dump_json
from ..contracts import Population,mapping_id,catalog_id,CHANNELS
from ..data import relevance
from ..runtime import Store,digest,fingerprint,profile,require_memory
from ..small_benchmark import load_training,load_validation
from ..historical_retrieval import load_historical_training,historical_file,historical_identity
from ..retrieval_artifacts import checked_manifest
from ..retrieval_workflow import ROOT,identifier,relative,cpu
from ..fusion.small_rrf import evaluate_rankings,ranked,validate_pool,overlap_diagnostics
from ..small_protocol import CROSS_USER_LIMITATION
from .features import attach_labels,sample_training
from .small_schema import build_features,feature_schema,validate_schema,model_matrix,group_sizes,feature_shift

VERSION='small_lambdamart_v1'


def reference(path):
    return dict(path=relative(path),manifest_sha256=digest(Path(path)/'manifest.json'))


def checked_reference(ref):
    path=ROOT/ref['path']
    if digest(path/'manifest.json')!=ref['manifest_sha256']:raise ValueError('Artifact reference checksum mismatch')
    return path,checked_manifest(path)


def check_fusion(path,kind,identity):
    path=Path(path);m=checked_manifest(path)
    if m.get('kind')!=kind:raise ValueError('Wrong fusion namespace: expected '+kind)
    for key in ['evaluation_id','catalog_id','mapping_id','training_cutoff']:
        if key in identity and m.get(key)!=identity[key]:raise ValueError('Fusion/context identity mismatch: '+key)
    if m['rrf_constant']!=60 or digest(path/'union.parquet')!=m['union_sha256']:
        raise ValueError('Fusion RRF definition/checksum mismatch')
    return m,pd.read_parquet(path/'union.parquet')


def prepare_ranking_features(historical_fusion,validation_fusion,config,
        data_dir=ROOT/'data/splits/small_v1',output_root=ROOT/'artifacts/ranking_features',
        training_output_root=ROOT/'artifacts/ranking_training',run_id=None):
    run_id=identifier(run_id);cfg=yaml.safe_load(Path(config).read_text())
    if (Path(output_root)/run_id).exists() or (Path(training_output_root)/run_id).exists():raise FileExistsError('Ranking data version already exists')
    hist=load_historical_training(data_dir);full=load_training(data_dir);validation=load_validation(data_dir,full)
    hm,hpool=check_fusion(historical_fusion,'historical_fusion',historical_identity(hist))
    vm,vpool=check_fusion(validation_fusion,'validation_fusion',dict(evaluation_id=validation.population.identifier,
        catalog_id=catalog_id(full.ctx),mapping_id=mapping_id(full.ctx),
        training_cutoff='per_user:'+full.manifest['files']['user_cutoffs.parquet']['sha256']))
    validate_pool(hpool,hist.ctx,hist.ctx.users);validate_pool(vpool,full.ctx,validation.population.users)
    require_memory(cpu=(len(hpool)+len(vpool))*60*4*8,stage='Versioned historical and Validation feature frames')
    start=time.perf_counter();hf,schema=build_features(hpool,hist.ctx,hm['evaluation_id']);hist_seconds=time.perf_counter()-start
    start=time.perf_counter();vf,vschema=build_features(vpool,full.ctx,vm['evaluation_id']);val_seconds=time.perf_counter()-start
    if schema!=vschema:raise ValueError('Historical/Validation feature schema differs')
    metadata=dict(kind='ranking_feature_inputs',version=VERSION,status='completed',run_id=run_id,
        historical_fusion=reference(historical_fusion),validation_fusion=reference(validation_fusion),
        feature_schema_sha256=schema['sha256'],historical_evaluation_id=hm['evaluation_id'],validation_evaluation_id=vm['evaluation_id'],
        historical_snapshot_id=hm['snapshot_id'],split_manifest_sha256=hist.split_sha256,
        historical_context=dict(history_sha256=hm['history_sha256'],mapping_id=hm['mapping_id'],catalog_id=hm['catalog_id'],training_cutoff=hm['training_cutoff']),
        validation_context=dict(history_sha256=full.manifest['files']['train.parquet']['sha256'],mapping_id=vm['mapping_id'],catalog_id=vm['catalog_id'],training_cutoff=vm['training_cutoff']),
        historical_rows=len(hf),validation_rows=len(vf),historical_feature_seconds=hist_seconds,validation_feature_seconds=val_seconds,
        future_labels_read_for_features=False,source_sha256=digest(Path(__file__)),schema_source_sha256=digest(Path(__file__).with_name('small_schema.py')),
        lock_sha256=digest(ROOT/'uv.lock'),cross_user_temporal_disclosure=CROSS_USER_LIMITATION,test_access=False)
    def writer(path):
        hf.to_parquet(path/'historical_features.parquet',index=False)
        vf.to_parquet(path/'validation_features.parquet',index=False)
        dump_json(path/'feature_schema.json',schema)
        metadata['historical_features_sha256']=digest(path/'historical_features.parquet')
        metadata['validation_features_sha256']=digest(path/'validation_features.parquet')
        dump_json(path/'manifest.json',metadata)
    feature_path=Store(Path(output_root).parent,Path(output_root).name).write(run_id,writer)
    checked_manifest(feature_path)  # publish label-free features before opening either supervised window
    return prepare_training_table(feature_path,cfg,data_dir,training_output_root,run_id)


def sample_queries(frame,seed,unknown_per_query):
    pieces=[];stats=dict(queries=0,effective_queries=0,queries_no_retrieved_positive=0,queries_no_contrast=0,
        positive_candidate_labels=0,observed_low_candidates=0,sampled_unlabeled_candidates=0)
    for key,group in frame.sort_values(['queryId','movieId'],kind='stable').groupby('queryId',sort=True):
        query_seed=int(fingerprint(dict(seed=seed,queryId=key))[:16],16)
        selected,counts=sample_training(group,np.random.default_rng(query_seed),unknown_per_query)
        for field in stats:stats[field]+=counts[field]
        if len(selected):pieces.append(selected)
    result=pd.concat(pieces,ignore_index=True) if pieces else frame.iloc[:0].copy()
    return result.sort_values(['queryId','movieId'],kind='stable').reset_index(drop=True),stats


def population(ctx,truth,identity):
    names={'cold':'cold_0','sparse':'sparse_1_4','moderate':'moderate_5_19','warm':'warm_20_plus'}
    return Population(truth,{u:names[ctx.cohort(u)] for u in truth},{},identity)


def prepare_training_table(features,config,data_dir,output_root,run_id):
    features=Path(features);fm=checked_manifest(features)
    if fm['kind']!='ranking_feature_inputs':raise ValueError('Expected frozen label-free features')
    hf=pd.read_parquet(features/'historical_features.parquet');vf=pd.read_parquet(features/'validation_features.parquet')
    schema=json.loads((features/'feature_schema.json').read_text());validate_schema(schema)
    model_matrix(hf,schema);model_matrix(vf,schema)
    hist=load_historical_training(data_dir);full=load_training(data_dir);validation=load_validation(data_dir,full)
    hm,hpool=check_fusion(ROOT/fm['historical_fusion']['path'],'historical_fusion',historical_identity(hist))
    _,vpool=check_fusion(ROOT/fm['validation_fusion']['path'],'validation_fusion',dict(evaluation_id=validation.population.identifier))
    historical_file(data_dir,'ranker_label_window.parquet',labels=True)
    labels=pd.read_parquet(Path(data_dir)/'ranker_label_window.parquet')
    if labels.duplicated(['userId','movieId']).any():raise ValueError('Duplicate historical label pair')
    if set(hist.history.interactionId)&set(labels.interactionId):raise ValueError('Ranking labels entered snapshot history')
    combined=pd.concat([hist.history,labels],ignore_index=True).sort_values('interactionId').reset_index(drop=True)
    outer=full.train.sort_values('interactionId').reset_index(drop=True)
    pd.testing.assert_frame_equal(combined,outer,check_dtype=False,check_exact=True)
    cutoff=hist.history.groupby('userId').timestamp.max()
    minimum=labels.groupby('userId').timestamp.min()
    if not (minimum>cutoff.reindex(minimum.index)).all():raise ValueError('Same-user label timestamp is not strictly later than snapshot history')
    truth,eligibility=relevance(labels,hist.ctx)
    expected=hist.manifest['historical_ranker_feasibility']
    if len(labels)!=expected['label_window_rows'] or eligibility['eligible_positive_interactions']!=expected['eligible_positives']:
        raise ValueError('Historical label/candidate eligibility disagrees with frozen manifest')
    labeled=attach_labels(hf,labels)
    selected,sampling=sample_queries(labeled,config['seed'],config['sampling']['unknown_per_query'])
    groups=group_sizes(selected,require_contrast=True)
    original_keys=set(zip(hf.queryId,hf.movieId))
    if not set(zip(selected.queryId,selected.movieId))<=original_keys:raise ValueError('Inserted a non-retrieved training candidate')
    q=labeled.groupby('queryId').label.agg(['sum','size'])
    informative=q.index[q['sum'].gt(0)&q['sum'].lt(q['size'])]
    if int(selected.label.sum())!=int(labeled.loc[labeled.queryId.isin(informative),'label'].sum()):
        raise ValueError('Retrieved positives from informative groups were removed during sampling')
    sizes=selected.groupby('queryId').size()
    pop=population(hist.ctx,truth,hm['evaluation_id']);per_query,history_metrics=evaluate_rankings(hpool,pop,hist.ctx)
    stats=dict(total_historical_queries=int(hf.queryId.nunique()),users_with_future_observed_positive=int(labels.loc[labels.rating>=4,'userId'].nunique()),
        users_with_snapshot_eligible_positive=len(truth),queries_with_retrieved_positive=int(q['sum'].gt(0).sum()),
        queries_with_retrieved_positive_and_contrast=int((q['sum'].gt(0)&q['sum'].lt(q['size'])).sum()),
        queries_only_zero=int(q['sum'].eq(0).sum()),queries_only_positive=int(q['sum'].eq(q['size']).sum()),
        pre_sampling_feedback_counts={str(k):int(v) for k,v in labeled.feedback_kind.value_counts().items()},
        sampled_feedback_counts={str(k):int(v) for k,v in selected.feedback_kind.value_counts().items()},
        sampling=sampling,effective_training_queries=len(groups),training_rows=len(selected),
        query_sizes=dict(min=int(sizes.min()),median=float(sizes.median()),p90=float(sizes.quantile(.9)),max=int(sizes.max())),
        snapshot_rows=len(hist.history),label_rows=len(labels),future_positives=int((labels.rating>=4).sum()),
        eligible_future_positives=eligibility['eligible_positive_interactions'],retrieved_positive_candidates=int(labeled.label.sum()),
        historical_union_macro_recall=history_metrics[0]['pool_recall'],
        eligible_positive_absent_fraction=1-int(labeled.label.sum())/eligibility['eligible_positive_interactions'],
        label_definition='1 observed rating>=4; 0 observed low or sampled unknown proxy. Unknown is not verified dislike.',
        sampling_policy='All retrieved positives and observed lows; <=100 uniform unknowns per composite query, seeded by seed+queryId. Label-conditioned training sampling only, never retrieval.',
        label_conditioned_sampling_disclosure=True)
    if len(groups)<config['sampling']['minimum_effective_queries']:
        raise ValueError('Gate B infeasible: too few informative retrieved-positive/contrast queries: '+str(stats))
    validation_labels=pd.read_parquet(Path(data_dir)/'validation.parquet')
    vl=attach_labels(vf,validation_labels)  # every Validation union row, no label-conditioned selection
    if len(vl)!=len(vf):raise ValueError('Validation pool changed while attaching labels')
    group_sizes(vl)
    shift=feature_shift(hf,selected,vf,schema)
    shift['candidate_distributions']={name:dict(candidate_count=frame.groupby('queryId').size().describe().to_dict(),
        mean_contributing_sources=float(frame.source_count.mean())) for name,frame in [('historical_union',hf),('sampled_training',selected),('validation_union',vf)]}
    manifest=dict(kind='historical_ranking_dataset',version=VERSION,status='completed',run_id=run_id,feature_inputs=reference(features),
        feature_schema_sha256=schema['sha256'],historical_fusion=fm['historical_fusion'],validation_fusion=fm['validation_fusion'],
        historical_evaluation_id=hm['evaluation_id'],validation_evaluation_id=validation.population.identifier,
        snapshot_id=hist.snapshot_id,split_manifest_sha256=hist.split_sha256,sampling=config['sampling'],seed=config['seed'],
        statistics=stats,gate_b_feasible=True,feasibility_rule='At least configured informative groups; each has positive and contrasting labels; all features built/frozen before labels.',
        label_window_sha256=hist.manifest['files']['ranker_label_window.parquet']['sha256'],
        source_sha256=digest(Path(__file__)),lock_sha256=digest(ROOT/'uv.lock'),cross_user_temporal_disclosure=CROSS_USER_LIMITATION,test_access=False)
    def writer(path):
        selected.to_parquet(path/'training_rows.parquet',index=False);vl.to_parquet(path/'validation_rows.parquet',index=False)
        labels.to_parquet(path/'historical_labels.parquet',index=False)  # supervision only, never a feature input
        per_query.to_parquet(path/'historical_per_query_metrics.parquet',index=False)
        dump_json(path/'feature_schema.json',schema);dump_json(path/'feature_shift.json',shift)
        dump_json(path/'historical_metrics.json',history_metrics);dump_json(path/'feasibility.json',stats)
        dump_json(path/'manifest.json',manifest)
    path=Store(Path(output_root).parent,Path(output_root).name).write(run_id,writer);checked_manifest(path)
    return dict(gate='B',status='completed',features=str(features),training_data=str(path),statistics=stats)


def score_ndcg(predictions,frame,truth):
    # full eligible-positive denominator; zero-hit/empty-positive-in-pool queries score zero.
    data=frame[['userId','movieId']].copy();data['ranking_score']=predictions
    data=ranked(data,'ranking_score')
    rows=[]
    for user,g in data.groupby('userId',sort=False):
        from ..metrics import query_metrics
        rows.append(query_metrics(g.movieId.tolist(),truth[int(user)])['ndcg20'])
    return float(np.mean(rows))


def load_ranker(path,expected_schema=None):
    import lightgbm as lgb
    path=Path(path);m=checked_manifest(path);schema=json.loads((path/'feature_schema.json').read_text());validate_schema(schema)
    if m.get('kind')!='lightgbm_lambdamart' or m.get('version')!=VERSION or m['feature_schema_sha256']!=schema['sha256']:
        raise ValueError('Ranker/schema compatibility mismatch')
    if expected_schema is not None and expected_schema!=schema:raise ValueError('Requested feature schema differs')
    if m['lightgbm_version']!=lgb.__version__:raise ValueError('LightGBM runtime version differs')
    if m['schema_source_sha256']!=digest(Path(__file__).with_name('small_schema.py')):
        raise ValueError('Ranker feature implementation differs')
    model=lgb.Booster(model_file=str(path/'model.txt'))
    if model.feature_name()!=schema['names']:raise ValueError('Booster feature order differs from schema')
    return model,schema,m


def train_lambdamart(training_data,config,data_dir=ROOT/'data/splits/small_v1',output_root=ROOT/'artifacts/lightgbm',run_id=None):
    import lightgbm as lgb
    cpu();run_id=identifier(run_id)
    if (Path(output_root)/run_id).exists():raise FileExistsError('Ranker run already exists')
    cfg=yaml.safe_load(Path(config).read_text());training_data=Path(training_data);dm=checked_manifest(training_data)
    if dm.get('kind')!='historical_ranking_dataset' or not dm.get('gate_b_feasible'):raise ValueError('Gate C requires a verified historical Gate B dataset')
    if cfg['sampling']!=dm['sampling'] or cfg['seed']!=dm['seed']:raise ValueError('Training/sampling configuration changed after Gate B')
    schema=json.loads((training_data/'feature_schema.json').read_text());validate_schema(schema)
    if schema['sha256']!=dm['feature_schema_sha256']:raise ValueError('Dataset feature schema identity mismatch')
    train=pd.read_parquet(training_data/'training_rows.parquet');valid=pd.read_parquet(training_data/'validation_rows.parquet')
    gt=group_sizes(train,True);gv=group_sizes(valid);xt=model_matrix(train,schema);xv=model_matrix(valid,schema)
    hist=load_historical_training(data_dir);full=load_training(data_dir);val=load_validation(data_dir,full)
    historical_file(data_dir,'ranker_label_window.parquet',labels=True)
    ht,_=relevance(pd.read_parquet(Path(data_dir)/'ranker_label_window.parquet'),hist.ctx)
    if dm['split_manifest_sha256']!=full.split_manifest_sha256 or dm['validation_evaluation_id']!=val.population.identifier:
        raise ValueError('Ranker dataset/evaluation identity mismatch')
    params=dict(cfg['lightgbm']);params.update(objective='lambdarank',metric='None',label_gain=[0,1],
        seed=cfg['seed'],data_random_seed=cfg['seed'],feature_fraction_seed=cfg['seed'],bagging_seed=cfg['seed'],
        deterministic=True,force_col_wise=True,num_threads=1,verbosity=1)
    require_memory(cpu=(len(train)+len(valid))*len(schema['names'])*4*8,stage='LightGBM bins, features and gradients')
    dtrain=lgb.Dataset(xt,label=train.label,group=gt,feature_name=schema['names'],free_raw_data=True)
    dvalid=lgb.Dataset(xv,label=valid.label,group=gv,reference=dtrain,feature_name=schema['names'],free_raw_data=True)
    def feval(predictions,dataset):
        frame,truth=(valid,val.population.truth) if dataset is dvalid else (train,ht)
        return 'eligible_ndcg20',score_ndcg(predictions,frame,truth),True
    history={}
    with warnings.catch_warnings(record=True) as captured,profile() as measurement:
        booster=lgb.train(params,dtrain,num_boost_round=cfg['boosting_rounds'],valid_sets=[dtrain,dvalid],
            valid_names=['historical_sampled_train','external_validation_full_union'],feval=feval,
            callbacks=[lgb.early_stopping(cfg['early_stopping_rounds'],first_metric_only=True),lgb.record_evaluation(history)])
    start=time.perf_counter();predictions=booster.predict(xv,num_iteration=booster.best_iteration,num_threads=1);prediction_seconds=time.perf_counter()-start
    if not np.isfinite(predictions).all():raise FloatingPointError('Nonfinite LambdaMART predictions')
    vpath,vm=checked_reference(dm['validation_fusion']);pool=pd.read_parquet(vpath/'union.parquet')
    keys=valid[['userId','movieId','queryId']].copy();keys['ranking_score']=predictions
    scored=pool.merge(keys,on=['userId','movieId'],validate='one_to_one')
    if len(scored)!=len(pool) or set(zip(scored.userId,scored.movieId))!=set(zip(pool.userId,pool.movieId)):
        raise ValueError('Ranker changed the Validation candidate pool')
    sort_start=time.perf_counter();scored=ranked(scored,'ranking_score');ranking_seconds=prediction_seconds+time.perf_counter()-sort_start
    scored['model_run_id']=run_id;scored['feature_schema_sha256']=schema['sha256']
    per_query,rank_metrics=evaluate_rankings(scored,val.population,full.ctx,'ranking_score')
    rrf_start=time.perf_counter();rrf_pool=pool.copy()
    rrf_pool['rrf_score']=sum(rrf_pool[s+'_present']/(60+rrf_pool[s+'_rank']) for s in CHANNELS)
    rrf_pool=ranked(rrf_pool);rrf_seconds=time.perf_counter()-rrf_start
    _,rrf_metrics=evaluate_rankings(rrf_pool,val.population,full.ctx)
    best=max(vm['individual_top20'],key=lambda s:vm['individual_top20'][s]['ndcg20'])
    individual=pool.copy();individual['ranking_score']=individual[best+'_present']*(101-individual[best+'_rank'])
    _,best_metrics=evaluate_rankings(individual,val.population,full.ctx,'ranking_score')
    for key in ['recall20','ndcg20','hit20']:
        if abs(best_metrics[0][key]-vm['individual_top20'][best][key])>1e-12:raise ValueError('Embedded individual Top-20 baseline changed')
    comparison=dict(best_individual_source=best,individual=best_metrics,rrf=rrf_metrics,lambdamart=rank_metrics,
        common_union_recall_ceiling=rrf_metrics[0]['pool_recall'],common_queries=len(val.population.users),
        positive_denominator=sum(map(len,val.population.truth.values())),same_candidate_pair_set=True,
        individual_interpretation='Original source ranks first, missing-source union tail by movieId; original Top-20 unchanged; tail has no invented source scores.',
        rrf_ranking_seconds=rrf_seconds,lambdamart_prediction_seconds=prediction_seconds,lambdamart_ranking_seconds=ranking_seconds,
        lambdamart_mean_ranking_ms_per_query=1000*ranking_seconds/len(val.population.users))
    manifest=dict(kind='lightgbm_lambdamart',version=VERSION,status='completed',run_id=run_id,training_data=reference(training_data),
        feature_schema_sha256=schema['sha256'],validation_evaluation_id=val.population.identifier,
        validation_union_sha256=vm['union_sha256'],historical_evaluation_id=dm['historical_evaluation_id'],
        split_manifest_sha256=full.split_manifest_sha256,config=cfg,params=params,lightgbm_version=lgb.__version__,
        best_iteration=booster.best_iteration,trained_iterations=len(history['external_validation_full_union']['eligible_ndcg20']),
        training_groups=len(gt),training_rows=len(train),validation_groups=len(gv),validation_rows=len(valid),
        training_seconds=measurement['seconds'],peak_rss_bytes=measurement['peak_rss_bytes'],
        early_stopping_metric='Custom full-eligible-denominator user-macro NDCG@20 over all 579 full-union Validation queries, retaining zero hits',
        native_candidate_conditional_ndcg_not_used=True,validation_labels_used_for_gradients=False,
        schema_source_sha256=digest(Path(__file__).with_name('small_schema.py')),source_sha256=digest(Path(__file__)),
        source_git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        lock_sha256=digest(ROOT/'uv.lock'),dvc_metadata_sha256={p:digest(ROOT/p) for p in ['dvc.yaml','dvc.lock']},
        python_warnings=[str(w.message) for w in captured],comparison=comparison,
        cross_user_temporal_disclosure=CROSS_USER_LIMITATION,test_access=False)
    def writer(path):
        booster.save_model(str(path/'model.txt'),num_iteration=booster.best_iteration)
        dump_json(path/'feature_schema.json',schema)
        loaded=lgb.Booster(model_file=str(path/'model.txt'))
        replay=loaded.predict(xv,num_threads=1)
        np.testing.assert_array_equal(predictions,replay)
        manifest['save_reload_prediction_parity']=True;manifest['model_bytes']=(path/'model.txt').stat().st_size
        np.save(path/'validation_predictions.npy',predictions)
        scored.to_parquet(path/'validation_rankings.parquet',index=False);per_query.to_parquet(path/'validation_per_query_metrics.parquet',index=False)
        pd.DataFrame(dict(feature=schema['names'],gain=booster.feature_importance('gain'),split=booster.feature_importance('split'))).to_csv(path/'feature_importance.csv',index=False)
        dump_json(path/'training_history.json',history);dump_json(path/'comparison.json',comparison);dump_json(path/'manifest.json',manifest)
    path=Store(Path(output_root).parent,Path(output_root).name).write(run_id,writer);load_ranker(path,schema)
    return dict(gate='C',status='completed',ranker=str(path),best_iteration=booster.best_iteration,comparison=comparison)


def evaluate_lambdamart(checkpoint,training_data,output_root=ROOT/'artifacts/ranking_evaluations',run_id=None):
    run_id=identifier(run_id)
    if (Path(output_root)/run_id).exists():raise FileExistsError('Ranker evaluation already exists')
    model,schema,m=load_ranker(checkpoint);training_data=Path(training_data);dm=checked_manifest(training_data)
    if reference(training_data)!=m['training_data'] or dm['feature_schema_sha256']!=schema['sha256']:
        raise ValueError('Ranker/feature dataset reference mismatch')
    valid=pd.read_parquet(training_data/'validation_rows.parquet');group_sizes(valid)
    prediction=model.predict(model_matrix(valid,schema),num_threads=1)
    np.testing.assert_array_equal(prediction,np.load(Path(checkpoint)/'validation_predictions.npy',allow_pickle=False))
    def writer(path):
        np.save(path/'predictions.npy',prediction)
        dump_json(path/'manifest.json',dict(kind='ranker_reload_parity',status='completed',checkpoint=reference(checkpoint),
            feature_schema_sha256=schema['sha256'],rows=len(valid),exact_prediction_parity=True,test_access=False))
    path=Store(Path(output_root).parent,Path(output_root).name).write(run_id,writer)
    return dict(status='completed',evaluation=str(path),prediction_rows=len(valid),exact_parity=True)
