"""Bounded real HTTP checks against label-free immutable-model reference predictions."""
from pathlib import Path
import argparse
import json
import time
import httpx
import numpy as np
import pandas as pd


def response_frame(body):
    rows=[]
    for user in body['results']:
        for row in user['recommendations']:
            rows.append(dict(userId=user['user_id'],movieId=row['movie_id'],rank=row['rank'],score=row['score'],fallback=row['fallback']))
    return pd.DataFrame(rows,columns=['userId','movieId','rank','score','fallback']).astype(dict(userId='int64',movieId='int64',rank='int32',score='float64',fallback='str'))


def compare(actual,expected):
    keys=['userId','movieId','rank','fallback']
    if len(actual)!=len(expected):raise AssertionError('Recommendation count differs: '+str((len(actual),len(expected))))
    mismatches=int(actual[keys].ne(expected[keys]).any(axis=1).sum())
    a,b=actual.score.to_numpy(),expected.score.to_numpy()
    absolute=np.abs(a-b);relative=absolute/np.maximum(np.abs(b),np.finfo(float).tiny)
    # No tolerance is relaxed automatically. Any platform difference needs explicit investigation.
    diagnostics=dict(rows=len(actual),ranking_mismatches=mismatches,max_absolute_score_difference=float(absolute.max(initial=0)),
                max_relative_score_difference=float(relative.max(initial=0)),score_tolerance=0.,exact_scores=bool(np.array_equal(a,b)))
    if mismatches or not diagnostics['exact_scores']:raise AssertionError('Parity failure: '+json.dumps(diagnostics))
    return diagnostics


def verify(url,reference_dir,latency=False):
    reference_dir=Path(reference_dir);inputs=pd.read_parquet(reference_dir/'inputs.parquet');expected=pd.read_parquet(reference_dir/'predictions.parquet')
    with httpx.Client(base_url=url,timeout=60,trust_env=False) as client:
        assert client.get('/health/live').json()=={'status':'alive'}
        assert client.get('/health/ready').status_code==200
        info=client.get('/model-info').json()
        assert info['name']=='MovieLensHybridRecommender' and info['version']=='1' and info['feature_schema_fingerprint']=='21b298dc0cef61b0de9514c0db66f4e38cd9a89818bb8fe94694d0251ad48ae0'
        assert info['load_count']==1 and info['release_status']=='validation_candidate' and info['test_evaluated'] is False
        assert '/Users/' not in json.dumps(info) and 'model_path' not in info
        outputs=[]
        for start in range(0,len(inputs),32):
            batch=inputs.iloc[start:start+32];response=client.post('/recommend',json={'user_ids':batch.userId.tolist(),'top_k':20})
            assert response.status_code==200,response.text
            outputs.append(response_frame(response.json()))
        actual=pd.concat(outputs,ignore_index=True);parity=compare(actual,expected)
        invalid=[{}, {'user_ids':[],'top_k':20},{'user_ids':[1,1],'top_k':20},
            *[{'user_ids':[u],'top_k':20} for u in [True,1.5,'1',None,0,-1,2**63]],
            *[{'user_ids':[1],'top_k':k} for k in [True,1.5,'20',None,0,101]],
            {'user_ids':[1],'top_k':20,'model_uri':'client_control_not_allowed'},{'user_ids':[1]}]
        statuses=[]
        for payload in invalid:
            r=client.post('/recommend',json=payload);assert r.status_code==422,(payload,r.text)
            statuses.append(r.status_code)
        for content in ['', '{broken']:
            r=client.post('/recommend',content=content,headers={'Content-Type':'application/json'});assert r.status_code==422
        assert client.post('/recommend',json={'user_ids':list(range(1,34)),'top_k':20}).status_code==413
        assert client.post('/recommend',content=b'x'*65537).status_code==413
        # Top-K prefixes, ordering and unknown/cold paths use genuine reference output.
        cold=int(expected.loc[expected.fallback.eq('known_zero_positive_history'),'userId'].iloc[0])
        unknown=int(expected.loc[expected.fallback.eq('unknown_user_popularity'),'userId'].iloc[0])
        single=client.post('/recommend',json={'user_ids':[cold],'top_k':1});assert single.status_code==200
        assert len(single.json()['results'][0]['recommendations'])==1
        reverse=client.post('/recommend',json={'user_ids':[unknown,42,1],'top_k':10});assert reverse.status_code==200
        assert [r['user_id'] for r in reverse.json()['results']]==[unknown,42,1]
        k100=client.post('/recommend',json={'user_ids':[1],'top_k':100});assert k100.status_code==200
        assert len(k100.json()['results'][0]['recommendations'])==100
        metrics={}
        if latency:
            for label,users,n in [('known_single',[1],10),('unknown_single',[unknown],10),('batch_8',list(range(1,9)),6)]:
                assert client.post('/recommend',json={'user_ids':users,'top_k':20}).status_code==200
                times=[];started=time.perf_counter()
                for _ in range(n):
                    t=time.perf_counter();r=client.post('/recommend',json={'user_ids':users,'top_k':20});elapsed=time.perf_counter()-t
                    assert r.status_code==200;times.append(elapsed*1000)
                metrics[label]=dict(samples=n,p50_ms=float(np.percentile(times,50)),p95_ms=float(np.percentile(times,95)),
                    min_ms=min(times),max_ms=max(times),sequential_requests_per_second=n/(time.perf_counter()-started),raw_ms=times)
        final_info=client.get('/model-info').json();assert final_info['load_count']==1
    return dict(url=url,queries=len(inputs),parity=parity,invalid_validation_cases=len(statuses),
        malformed_json_cases=2,batch_body_limits_verified=True,known_zero_positive_user=cold,unknown_user=unknown,
        startup_and_runtime=final_info,latency=metrics,scope='Local bounded loopback inference checks; no held-out targets or quality metrics computed')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--url',required=True);parser.add_argument('--reference-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--latency',action='store_true')
    args=parser.parse_args();result=verify(args.url,args.reference_dir,args.latency)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))
