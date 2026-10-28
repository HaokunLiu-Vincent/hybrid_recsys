"""Verified Small candidate unions, exact RRF, fixed-population metrics and ablations."""
from pathlib import Path
import itertools
import json
import time
import numpy as np
import pandas as pd
from .candidates import fuse
from ..contracts import CHANNELS,compatible,validate_candidates,mapping_id,catalog_id
from ..metrics import query_metrics
from ..runtime import Store,digest,fingerprint,require_memory
from ..artifacts import dump_json
from ..small_benchmark import load_training,load_validation
from ..retrieval_artifacts import checked_manifest,load_checkpoint
from ..retrieval_workflow import ROOT,identifier,relative,COHORTS
from ..small_protocol import CROSS_USER_LIMITATION,POLICY

VERSION='small_rrf_v1'


def ranked(frame,score='rrf_score'):
    result=frame.sort_values(['userId',score,'movieId'],ascending=[True,False,True],kind='stable').copy()
    result['rank']=(result.groupby('userId',sort=False).cumcount()+1).astype('int32')
    return result.reset_index(drop=True)


def validate_pool(frame,ctx,users,channels=CHANNELS):
    if frame.duplicated(['userId','movieId']).any():raise ValueError('Duplicate union pair')
    if not set(frame.userId)<=set(map(int,users)):raise ValueError('Union query outside declared population')
    if not frame.movieId.isin(ctx.items[ctx.eligible]).all():raise ValueError('Union movie outside context catalog')
    if frame.groupby('userId').size().gt(100*len(channels)).any():raise ValueError('Union budget exceeded')
    for user,group in frame.groupby('userId',sort=False):
        seen=ctx.items[ctx.seen(ctx.user_index(user))]
        if group.movieId.isin(seen).any():raise ValueError('Train-seen movie in union')
    expected=sum(frame[s+'_present']/(60+frame[s+'_rank']) for s in channels)
    if not np.isfinite(frame.rrf_score).all() or not np.array_equal(expected.to_numpy(),frame.rrf_score.to_numpy()):
        raise ValueError('RRF must sum only reciprocal present-source ranks with constant 60')
    if not frame.source_count.eq(frame[[s+'_present' for s in channels]].sum(axis=1)).all():
        raise ValueError('Wrong contributing-source count')
    for source in channels:
        present=frame[source+'_present'].eq(1)
        if not frame[source+'_present'].isin([0,1]).all():raise ValueError('Invalid membership flag')
        if not np.isfinite(frame.loc[present,source+'_score']).all():raise ValueError('Nonfinite present source score')
        if not frame.loc[~present,source+'_score'].isna().all():raise ValueError('Absent scores must remain NaN')
        if not frame.loc[present,source+'_rank'].between(1,100).all():raise ValueError('Invalid present source rank')
        if not frame.loc[~present,source+'_rank'].eq(101).all():raise ValueError('Missing rank must be sentinel 101')


def union_from_frames(frames,ctx,users):
    if not frames:raise ValueError('No candidate streams')
    outputs={}
    for source,frame in frames.items():
        validate_candidates(frame,ctx,source,100,set(map(int,users)))
        outputs[source]={int(u):g.to_dict('records') for u,g in frame.groupby('userId',sort=True)}
    require_memory(cpu=sum(len(f) for f in frames.values())*2048,stage='Sparse candidate-union construction')
    result=fuse(outputs,channels=list(frames),k=100,rrf_constant=60)
    validate_pool(result,ctx,users,tuple(frames))
    return ranked(result)


def evaluate_rankings(frame,population,ctx,score='rrf_score'):
    order=ranked(frame,score);groups={int(u):g for u,g in order.groupby('userId',sort=False)}
    rows=[]
    for user in population.users:
        g=groups.get(int(user));items=[] if g is None else g.movieId.tolist()
        rows.append(dict(userId=int(user),cohort=population.cohorts[int(user)],candidates=len(items),
                         **query_metrics(items,population.truth[int(user)])))
    per_query=pd.DataFrame(rows);summary=[]
    for cohort in ['all',*COHORTS]:
        selected=per_query if cohort=='all' else per_query[per_query.cohort.eq(cohort)]
        users=set(map(int,selected.userId));lists=order[order.userId.isin(users)]
        values={key:float(selected[key].mean()) if len(selected) else None for key in
                ['recall20','recall50','recall100','ndcg20','hit20','pool_recall']}
        summary.append(dict(cohort=cohort,queries=len(selected),relevant_positives=sum(len(population.truth[u]) for u in users),
            **values,catalog_coverage20=lists.loc[lists['rank']<=20,'movieId'].nunique()/int(ctx.eligible.sum()) if users else None,
            catalog_coverage100=lists.loc[lists['rank']<=100,'movieId'].nunique()/int(ctx.eligible.sum()) if users else None,
            union_catalog_coverage=lists.movieId.nunique()/int(ctx.eligible.sum()) if users else None,
            zero_union_hit_queries=int(selected.retrieved_positive.eq(0).sum()),
            empty_candidate_queries=int(selected.candidates.eq(0).sum()),
            undefined_reason=None if users else 'No relevance-bearing queries in this cohort'))
    return per_query,summary


def overlap_diagnostics(frames,truth):
    by_user={s:{int(u):set(g.movieId) for u,g in f.groupby('userId',sort=False)} for s,f in frames.items()}
    sets={s:{(int(u),int(i)) for u,i in f[['userId','movieId']].itertuples(index=False,name=None)} for s,f in frames.items()}
    relevant={(int(u),int(i)) for u,items in truth.items() for i in items}
    union=set().union(*sets.values());pairs={}
    for a,b in itertools.combinations(frames,2):
        intersections=[];jaccard=[]
        for user in truth:
            x=by_user[a].get(int(user),set());y=by_user[b].get(int(user),set())
            intersections.append(len(x&y))
            if x|y:jaccard.append(len(x&y)/len(x|y))
        pairs[a+'|'+b]=dict(intersection_pairs=len(sets[a]&sets[b]),mean_overlap_per_query=float(np.mean(intersections)),
            mean_query_jaccard=float(np.mean(jaccard)) if jaccard else None,
            aggregate_pair_jaccard=len(sets[a]&sets[b])/len(sets[a]|sets[b]) if sets[a]|sets[b] else None)
    unique={}
    for source,pool in sets.items():
        other=set().union(*(v for k,v in sets.items() if k!=source))
        only=pool-other
        unique[source]=dict(candidate_pairs=len(only),relevant_pairs=len(only&relevant),
            source_relevant_pairs=len(pool&relevant),source_movie_coverage=len({i for u,i in pool}))
    return dict(before_deduplication={s:len(f) for s,f in frames.items()},union_pairs=len(union),
        retrieved_relevant_pairs=len(union&relevant),relevant_pairs=len(relevant),
        union_positive_micro_coverage=len(union&relevant)/len(relevant) if relevant else None,
        pairwise=pairs,unique_contribution=unique)


def popularity_diagnostics(frame,ctx):
    catalog=np.flatnonzero(ctx.eligible)
    order=np.lexsort((ctx.items[catalog],-ctx.popularity[catalog]))
    head=set(map(int,ctx.items[catalog[order[:int(np.ceil(len(catalog)*.1))]]]))
    result={}
    for label,part in [('full_union',frame),('top20',frame[frame['rank']<=20])]:
        counts=part.movieId.value_counts().reindex(ctx.items[ctx.eligible],fill_value=0).to_numpy()
        sorted_counts=np.sort(counts);n=len(counts);total=int(counts.sum())
        indices=np.searchsorted(ctx.items,part.movieId.to_numpy())
        result[label]=dict(rows=len(part),train_positive_top_decile_share=float(part.movieId.isin(head).mean()) if len(part) else None,
            mean_train_positive_count=float(ctx.popularity[indices].mean()) if len(part) else None,
            recommendation_frequency_gini=float(2*np.dot(np.arange(1,n+1),sorted_counts)/(n*total)-(n+1)/n) if total else None)
    return result


def fuse_validation(evaluations,data_dir=ROOT/'data/splits/small_v1',output_root=ROOT/'artifacts/fusion',run_id=None):
    training=load_training(data_dir);data=load_validation(data_dir,training);frames={};manifests={};references={}
    for path in evaluations:
        path=Path(path);m=checked_manifest(path);source=m['model_name']
        if source in frames:raise ValueError('Duplicate source artifact')
        checkpoint=ROOT/m['source_checkpoint']['path']
        _,producer=load_checkpoint(checkpoint,source,training.split_manifest_sha256)
        if digest(checkpoint/'manifest.json')!=m['source_checkpoint']['manifest_sha256']:
            raise ValueError('Candidate checkpoint reference mismatch')
        if (m['training_kind']!='external_train_per_user' or m['phase']!='validation' or
            m['evaluation_id']!=data.population.identifier or m['top_k']!=100 or
            m['mapping_id']!=mapping_id(training.ctx) or m['catalog_id']!=catalog_id(training.ctx) or
            m['policy']!=POLICY or m['config_sha256']!=producer['config_sha256'] or
            m['per_user_cutoff_table_sha256']!=training.manifest['files']['user_cutoffs.parquet']['sha256']):
            raise ValueError('Incompatible frozen Validation source')
        stats=m['measurements']['eligible']
        if stats['cohorts'][0]['queries']!=len(data.population.users) or stats['cohorts'][0]['relevant_positives']!=sum(map(len,data.population.truth.values())):
            raise ValueError('Source evaluation population/denominator differs')
        frame=pd.read_parquet(path/'candidates.parquet')
        if len(frame)!=stats['candidate_rows']:raise ValueError('Source candidate count differs from manifest')
        if digest(path/'candidates.parquet')!=m['candidate_files']['eligible']['sha256']:
            raise ValueError('Source candidate checksum mismatch')
        frames[source]=frame;manifests[source]=m
        references[source]=dict(path=relative(path),manifest_sha256=digest(path/'manifest.json'),
            candidate_sha256=digest(path/'candidates.parquet'),checkpoint=m['source_checkpoint'],
            config=producer['config'],training_lock_sha256=producer['training_lock_sha256'])
    compatible(manifests,require_all=True)
    frames={s:frames[s] for s in CHANNELS};run_id=identifier(run_id)
    if (Path(output_root)/run_id).exists():raise FileExistsError('Fusion run already exists')
    start=time.perf_counter();pool=union_from_frames(frames,training.ctx,data.population.users)
    fusion_seconds=time.perf_counter()-start
    per_query,metrics=evaluate_rankings(pool,data.population,training.ctx)
    diag=overlap_diagnostics(frames,data.population.truth)
    diag['sources_per_candidate']={str(int(k)):int(v) for k,v in pool.source_count.value_counts().sort_index().items()}
    sizes=pool.groupby('userId').size();diag['query_sizes']=dict(min=int(sizes.min()),median=float(sizes.median()),max=int(sizes.max()))
    diag['popularity_concentration']=popularity_diagnostics(pool,training.ctx)
    individual={s:evaluate_rankings(f, data.population,training.ctx,'score')[1][0] for s,f in frames.items()}
    ablations={}
    for omitted in CHANNELS:
        reduced=union_from_frames({s:f for s,f in frames.items() if s!=omitted},training.ctx,data.population.users)
        pq,m=evaluate_rankings(reduced,data.population,training.ctx)
        removed=diag['retrieved_relevant_pairs']-int(pq.retrieved_positive.sum())
        ablations[omitted]=dict(metrics=m[0],lost_unique_relevant_pairs=removed,
            delta_union_recall=m[0]['pool_recall']-metrics[0]['pool_recall'],
            delta_ndcg20=m[0]['ndcg20']-metrics[0]['ndcg20'],delta_recall20=m[0]['recall20']-metrics[0]['recall20'],
            delta_coverage20=m[0]['catalog_coverage20']-metrics[0]['catalog_coverage20'])
        if removed!=diag['unique_contribution'][omitted]['relevant_pairs']:
            raise ValueError('Ablation/unique-positive contribution disagreement')
    common=next(iter(manifests.values()))
    manifest=dict(framework_version=VERSION,kind='validation_fusion',status='completed',run_id=run_id,
        dataset_id=training.manifest['dataset_id'],split_version=training.manifest['version'],
        split_manifest_sha256=training.split_manifest_sha256,evaluation_id=data.population.identifier,
        training_kind='external_train_per_user',training_cutoff=common['training_cutoff'],
        per_user_cutoff_table_sha256=common['per_user_cutoff_table_sha256'],mapping_id=common['mapping_id'],catalog_id=common['catalog_id'],
        policy=POLICY,rrf_constant=60,source_budget=100,union_budget=600,inputs=references,
        metrics=metrics,individual_top20=individual,diagnostics=diag,ablations=ablations,fusion_seconds=fusion_seconds,
        union_rows=len(pool),queries=len(data.population.users),cross_user_temporal_disclosure=CROSS_USER_LIMITATION,
        lock_sha256=digest(ROOT/'uv.lock'),source_sha256=digest(Path(__file__)),test_access=False)
    def writer(path):
        pool.to_parquet(path/'union.parquet',index=False)
        per_query.to_parquet(path/'per_query_metrics.parquet',index=False)
        manifest['union_sha256']=digest(path/'union.parquet')
        dump_json(path/'metrics.json',dict(metrics=metrics,individual_top20=individual))
        dump_json(path/'diagnostics.json',diag);dump_json(path/'ablations.json',ablations)
        dump_json(path/'manifest.json',manifest)
        reloaded=pd.read_parquet(path/'union.parquet')
        pd.testing.assert_frame_equal(pool,reloaded,check_dtype=False,check_exact=True)
    path=Store(Path(output_root).parent,Path(output_root).name).write(run_id,writer)
    checked_manifest(path)
    return dict(gate='A',status='completed',fusion=str(path),metrics=metrics[0],diagnostics=diag,ablations=ablations)
