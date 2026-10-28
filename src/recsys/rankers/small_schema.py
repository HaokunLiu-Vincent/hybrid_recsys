"""Versioned candidate-time feature allowlist and composite LambdaMART query contract."""
import re
import numpy as np
import pandas as pd
from ..contracts import CHANNELS
from ..features import candidate_features,feature_columns
from ..runtime import fingerprint
from .features import distribution,mismatch

VERSION='small_lambdamart_features_v1'
FORBIDDEN={'rating','label','feedback_kind','timestamp','submitted_at','future_positives','cohort','userId','movieId','queryId','snapshotId'}


def feature_schema(ctx):
    genre={g:'genre_'+re.sub('[^A-Za-z0-9_]','_',g) for g in ctx.genre_names}
    names=feature_columns()+['rrf_score','best_source_rank','mean_source_rank','candidate_count']+list(genre.values())
    if len(names)!=len(set(names)) or set(names)&FORBIDDEN:raise ValueError('Invalid feature allowlist')
    schema=dict(version=VERSION,names=names,dtypes={n:'float32' for n in names},genre_columns=genre,
        missingness='Absent source score = NaN, rank = 101, present/fallback = 0; present scores finite. Other features finite.',
        transformations='Train/snapshot-only log1p count features, L2 genre similarity, binary movie genres; raw source-specific scores, reciprocal-rank score with c=60. No fitted normalization.',
        label_columns_excluded=sorted(FORBIDDEN))
    schema['sha256']=fingerprint(schema)
    return schema


def validate_schema(schema):
    payload={k:v for k,v in schema.items() if k!='sha256'}
    if schema.get('version')!=VERSION or schema.get('sha256')!=fingerprint(payload):
        raise ValueError('Feature schema version/checksum mismatch')
    expected=feature_columns()+['rrf_score','best_source_rank','mean_source_rank','candidate_count']+list(schema['genre_columns'].values())
    if schema['names']!=expected or set(schema['names'])&FORBIDDEN:
        raise ValueError('Feature order/allowlist mismatch or forbidden target feature')
    if any(schema['dtypes'].get(n)!='float32' for n in schema['names']):raise ValueError('Unsupported feature dtype')


def query_keys(frame,namespace):
    result=frame.copy();result['snapshotId']=namespace
    result['queryId']=[namespace+'|'+str(int(u)).zfill(20) for u in result.userId]
    return result


def build_features(pool,ctx,namespace):
    schema=feature_schema(ctx)
    base=candidate_features(pool,ctx)
    present=np.column_stack([base[s+'_present'].to_numpy() for s in CHANNELS])
    ranks=np.column_stack([base[s+'_rank'].to_numpy() for s in CHANNELS])
    base['best_source_rank']=np.where(present>0,ranks,np.inf).min(axis=1)
    base['mean_source_rank']=(ranks*present).sum(axis=1)/present.sum(axis=1)
    base['candidate_count']=base.groupby('userId').userId.transform('size')
    indices=np.searchsorted(ctx.items,base.movieId.to_numpy())
    for j,g in enumerate(ctx.genre_names):base[schema['genre_columns'][g]]=(ctx.genres[indices,j]>0).astype('float32')
    keys=query_keys(base[['userId','movieId']],namespace)
    result=pd.concat([keys,base[schema['names']].astype('float32')],axis=1)
    result=result.sort_values(['queryId','movieId'],kind='stable').reset_index(drop=True)
    model_matrix(result,schema)
    return result,schema


def model_matrix(frame,schema):
    validate_schema(schema)
    missing=set(schema['names'])-set(frame)
    if missing:raise ValueError('Missing ranking features: '+str(sorted(missing)))
    values=frame.loc[:,schema['names']]
    if list(values.columns)!=schema['names'] or any(str(t)!='float32' for t in values.dtypes):
        raise ValueError('Feature order/dtype mismatch')
    for source in CHANNELS:
        present=values[source+'_present'].eq(1)
        if not values[source+'_present'].isin([0.,1.]).all():raise ValueError('Invalid source membership feature')
        score=values[source+'_score']
        if not np.isfinite(score[present]).all() or not score[~present].isna().all():
            raise ValueError('Source score missingness violates the feature schema')
    non_scores=[n for n in schema['names'] if not n.endswith('_score') or n=='rrf_score']
    if not np.isfinite(values[non_scores].to_numpy()).all():raise ValueError('Non-score features must be finite')
    if np.isinf(values.to_numpy()).any():raise ValueError('Infinite ranking feature')
    return values


def group_sizes(frame,require_contrast=False):
    if frame.empty:raise ValueError('Empty ranking table')
    if frame[['queryId','snapshotId','userId','movieId']].isna().any().any():raise ValueError('Missing query key')
    if frame.duplicated(['queryId','movieId']).any():raise ValueError('Duplicate candidate within query')
    segments=frame.loc[frame.queryId.ne(frame.queryId.shift()),'queryId']
    if segments.duplicated().any():raise ValueError('Ranking queries are not contiguous')
    if frame.groupby('queryId').userId.nunique().gt(1).any() or frame.groupby('queryId').snapshotId.nunique().gt(1).any():
        raise ValueError('Composite query identity collides')
    expected=frame.snapshotId+'|'+frame.userId.map(lambda u:str(int(u)).zfill(20))
    if not frame.queryId.eq(expected).all():raise ValueError('Query ID must include snapshot/evaluation namespace')
    groups=frame.groupby('queryId',sort=False).size().to_numpy(dtype='int32')
    if int(groups.sum())!=len(frame):raise ValueError('Group sizes do not sum to candidate rows')
    if 'label' in frame:
        if not frame.label.isin([0,1]).all():raise ValueError('Binary relevance labels required')
        if require_contrast and frame.groupby('queryId').label.nunique().ne(2).any():
            raise ValueError('Training queries require both positive and contrasting labels')
    return groups


def feature_shift(historical,sampled,validation,schema):
    names=schema['names']
    history=distribution(historical,names);train=distribution(sampled,names);valid=distribution(validation,names)
    rows=mismatch(history,valid,names)
    for row in rows:
        a,b=row['snapshot'],row['full_train']
        row['missingness_delta']=b['missing_fraction']-a['missing_fraction']
        row['descriptive_shift_flag']=abs(row['standardized_mean_shift'] or 0)>.5 or abs(row['missingness_delta'])>.05
    return dict(historical_union=history,sampled_training=train,validation_union=valid,
                historical_vs_validation=rows,policy='Descriptive only; no Validation-label-driven normalization or refitting')
