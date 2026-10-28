"""Label-free six-retriever inference using retained RRF/features/native ranker."""
from pathlib import Path
import importlib.metadata
import json
import numbers
import platform
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .contracts import CHANNELS,mapping_id,catalog_id
from .runtime import digest,fingerprint
from .retrieval_artifacts import checked_manifest,load_context,critical_sources
from .training import load_model
from .fusion.small_rrf import union_from_frames,ranked
from .rankers.small_schema import build_features,model_matrix,validate_schema

VERSION='hybrid_release_v1'
OUTPUT=['userId','movieId','rank','score','fallback']


def validate_input(frame):
    if not isinstance(frame,pd.DataFrame) or set(frame.columns)!={'userId','top_k'}:
        raise ValueError('Input requires exactly userId and top_k columns')
    for key in ['userId','top_k']:
        if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,numbers.Integral) for v in frame[key]):
            raise ValueError(key+' must contain integers, not bool/float/string/null')
    if not frame.userId.between(1,np.iinfo(np.int64).max).all():raise ValueError('userId must be a positive int64')
    if not frame.top_k.between(1,100).all():raise ValueError('top_k must be between 1 and 100')
    if frame.userId.duplicated().any():raise ValueError('Duplicate userId in one request is ambiguous')
    return frame.astype(dict(userId='int64',top_k='int64'))


def pipeline_rankings(models,ctx,booster,schema,users,configs,namespace):
    """No data loader or cached candidates: compute every requested user's candidates."""
    frames={}
    with threadpool_limits(limits=1):
        for source in CHANNELS:
            cfg=configs[source];rows=[]
            for start in range(0,len(users),cfg['query_batch']):
                part=users[start:start+cfg['query_batch']]
                result=models[source].retrieve_many(part,100,cfg['item_block'])
                rows.extend(row for u in part for row in result[int(u)])
            frames[source]=pd.DataFrame(rows,columns=['userId','movieId','score','rank','source','fallback'])
        pool=union_from_frames(frames,ctx,users)
        if pool.empty:return pool,pd.DataFrame()
        features,actual_schema=build_features(pool,ctx,namespace)
        if actual_schema!=schema:raise ValueError('Inference feature schema differs from the trained ranker')
        predictions=booster.predict(model_matrix(features,schema),num_threads=1)
    if not np.isfinite(predictions).all():raise ValueError('Nonfinite model predictions')
    keys=features[['userId','movieId']].copy();keys['ranking_score']=predictions
    scored=pool.merge(keys,on=['userId','movieId'],validate='one_to_one')
    return ranked(scored,'ranking_score'),features


class HybridRecommender:
    def __init__(self,ctx,models,booster,schema,configs,namespace):
        self.ctx,self.models,self.booster,self.schema=ctx,models,booster,schema
        self.configs,self.namespace=configs,namespace

    def predict(self,frame):
        frame=validate_input(frame)
        if frame.empty:return pd.DataFrame({c:pd.Series(dtype=t) for c,t in zip(OUTPUT,['int64','int64','int32','float64','str'])})
        users=frame.userId.tolist();known=[u for u in users if self.ctx.user_index(u)>=0]
        if known:
            scored,_=pipeline_rankings(self.models,self.ctx,self.booster,self.schema,known,self.configs,self.namespace)
            groups={int(u):g for u,g in scored.groupby('userId',sort=False)}
        else:groups={}
        pieces=[]
        for user,k in frame[['userId','top_k']].itertuples(index=False,name=None):
            if self.ctx.user_index(user)<0:
                # No invented embedding or personalized statistics; bypass LambdaMART.
                result=pd.DataFrame(self.models['popularity'].retrieve(user,k,self.configs['popularity']['item_block']))
                if result.empty:continue
                result=result[['userId','movieId','rank','score']];result['fallback']='unknown_user_popularity'
            else:
                group=groups.get(user)
                if group is None:continue
                result=group.head(k)[['userId','movieId','rank','ranking_score']].rename(columns={'ranking_score':'score'}).copy()
                result['fallback']='known_zero_positive_history' if self.ctx.positive_count[self.ctx.user_index(user)]==0 else 'none'
            pieces.append(result[OUTPUT])
        if not pieces:return self.predict(frame.iloc[:0])
        result=pd.concat(pieces,ignore_index=True)
        return result.astype(dict(userId='int64',movieId='int64',rank='int32',score='float64',fallback='str'))

    def recommend(self,user_ids,top_k=20):
        if isinstance(user_ids,(str,bytes)) or not hasattr(user_ids,'__iter__'):raise ValueError('user_ids must be a sequence')
        return self.predict(pd.DataFrame({'userId':list(user_ids),'top_k':top_k}))


def load_hybrid_recommender(release_path):
    import lightgbm as lgb
    path=Path(release_path);m=checked_manifest(path)
    if m.get('version')!=VERSION or m.get('kind')!='complete_hybrid_recommender':raise ValueError('Unsupported release kind/version')
    if m['rrf_constant']!=60 or m['retrieval_k']!=100 or set(m['retrievers'])!=set(CHANNELS):raise ValueError('Incomplete six-source inference contract')
    if platform.python_version()!=m['python']:raise ValueError('Release Python version mismatch')
    for package,version in m['runtime_packages'].items():
        if importlib.metadata.version(package)!=version:raise ValueError('Release dependency mismatch: '+package)
    for filename,sha in m['application_source_sha256'].items():
        current=Path(__file__).parent/filename
        if digest(current)!=sha:raise ValueError('Release application source mismatch: '+filename)
    schema=json.loads((path/'ranker/feature_schema.json').read_text());validate_schema(schema)
    if schema['sha256']!=m['feature_schema_sha256']:raise ValueError('Release schema fingerprint mismatch')
    ctx=load_context(path/'context')
    if mapping_id(ctx)!=m['mapping_id'] or catalog_id(ctx)!=m['catalog_id']:raise ValueError('Release context mapping/catalog mismatch')
    models={}
    for source in CHANNELS:
        original=m['retrievers'][source]
        if original['training_kind']!='external_train_per_user' or original['training_cutoff']!=m['training_cutoff']:
            raise ValueError('Release rejects historical/wrong-cutoff models')
        if original['critical_source_hashes']!=critical_sources(source):raise ValueError('Retriever source identity mismatch')
        if original['mapping_id']!=m['mapping_id'] or original['catalog_id']!=m['catalog_id']:
            raise ValueError('Retriever/context compatibility mismatch')
        models[source]=load_model(ctx,source,path/'retrievers'/source,device='cpu');models[source].name=source
    model=lgb.Booster(model_file=str(path/'ranker/model.txt'))
    if model.feature_name()!=schema['names']:raise ValueError('Ranker/feature order mismatch')
    return HybridRecommender(ctx,models,model,schema,{s:m['retrievers'][s]['config'] for s in CHANNELS},m['evaluation_id'])
