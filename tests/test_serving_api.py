"""Pure schema and fake-model ASGI integration tests; no real data/registry reads."""
from pathlib import Path
from types import SimpleNamespace
import asyncio
import json
import tempfile
import threading
import unittest
from unittest.mock import patch
import httpx
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from recsys_serving.app import create_app,response_from_predictions
from recsys_serving.settings import Settings,RELEASE_ID
from recsys_serving.schemas import RecommendRequest
from recsys_serving.model_loader import LoadedModel,relative_file,check_files


def fake_info():
    return dict(name='MovieLensHybridRecommender',version='1',release_status='validation_candidate',
        release_id=RELEASE_ID,protocol='per_user_70_15_15',retriever_count=6,rrf_constant=60,
        ranker='lightgbm_lambdamart',feature_schema_version='synthetic',feature_schema_fingerprint='synthetic',
        test_evaluated=False,source_mode='local')


class FakeModel:
    def predict(self,frame):
        rows=[]
        for user,k in frame[['userId','top_k']].itertuples(index=False,name=None):
            rows.extend(dict(userId=user,movieId=100+j,rank=j+1,score=float(10-j),
                fallback='unknown_user_popularity' if user==99999 else 'none') for j in range(min(k,3)))
        return pd.DataFrame(rows,columns=['userId','movieId','rank','score','fallback'])


class ServingTests(unittest.TestCase):
    def client(self,model=None,**kwargs):
        return TestClient(create_app(Settings(**kwargs),loader=lambda settings:LoadedModel(model or FakeModel(),fake_info())))

    def test_health_readiness_and_load_once(self):
        calls=[]
        app=create_app(Settings(),loader=lambda s:(calls.append(1) or LoadedModel(FakeModel(),fake_info())))
        with TestClient(app) as client:
            for _ in range(3):self.assertEqual(client.post('/recommend',json={'user_ids':[1],'top_k':2}).status_code,200)
            self.assertEqual(client.get('/health/live').json(),{'status':'alive'})
            self.assertEqual(client.get('/health/ready').status_code,200)
            self.assertEqual(client.get('/model-info').json()['load_count'],1)
            self.assertEqual(len(calls),1)
        self.assertFalse(app.state.ready)

    def test_not_ready_and_failed_startup(self):
        app=create_app(Settings(),loader=lambda s:None)
        client=TestClient(app)  # Deliberately do not enter lifespan.
        self.assertEqual(client.get('/health/live').status_code,200)
        self.assertEqual(client.get('/health/ready').status_code,503)

    def test_startup_failure_does_not_fall_back(self):
        def fail(settings):raise ValueError('/sensitive/checkpoint')
        with self.assertRaisesRegex(RuntimeError,'Configured model failed'):
            with TestClient(create_app(Settings(),loader=fail)):pass
        app=create_app(Settings(),loader=fail)
        client=TestClient(app)
        self.assertEqual(client.get('/health/live').status_code,200)
        self.assertEqual(client.get('/health/ready').status_code,503)
        self.assertEqual(client.post('/recommend',json={'user_ids':[1],'top_k':2}).status_code,503)

    def test_strict_invalid_inputs_and_batch_limit(self):
        invalid=[{}, {'user_ids':[],'top_k':20},{'user_ids':[1,1],'top_k':20},
            *[{'user_ids':[u],'top_k':20} for u in [True,1.5,'1',None,0,-1,2**63]],
            *[{'user_ids':[1],'top_k':k} for k in [True,1.5,'20',None,0,101]],
            {'user_ids':[1],'top_k':20,'model_uri':'evil'},{'user_ids':[1]}]
        with self.client() as client:
            for payload in invalid:
                with self.subTest(payload=payload):self.assertEqual(client.post('/recommend',json=payload).status_code,422)
            self.assertEqual(client.post('/recommend',json={'user_ids':list(range(1,34)),'top_k':20}).status_code,413)

    def test_malformed_missing_json_and_body_limit(self):
        with self.client(max_body_bytes=1024) as client:
            for body in ['', '{broken']:
                self.assertEqual(client.post('/recommend',content=body,headers={'Content-Type':'application/json'}).status_code,422)
            self.assertEqual(client.post('/recommend',content='x'*1025).status_code,413)
            self.assertEqual(client.post('/recommend',content=iter([b'x'*700,b'x'*700])).status_code,413)

    def test_user_order_top_k_fallback_and_no_sensitive_metadata(self):
        with self.client() as client:
            result=client.post('/recommend',json={'user_ids':[2,99999,1],'top_k':2}).json()
            self.assertEqual([r['user_id'] for r in result['results']],[2,99999,1])
            self.assertEqual(result['results'][1]['recommendations'][0]['fallback'],'unknown_user_popularity')
            self.assertTrue(all(len(r['recommendations'])==2 for r in result['results']))
            metadata=client.get('/model-info').json()
            self.assertNotIn('path',json.dumps(metadata));self.assertFalse(metadata['test_evaluated'])
            self.assertEqual(metadata['release_status'],'validation_candidate')
            short=client.post('/recommend',json={'user_ids':[1],'top_k':100}).json()
            self.assertEqual(len(short['results'][0]['recommendations']),3)  # Available items only; no padding.

    def test_prediction_failure_nonfinite_and_resource_statuses(self):
        for failure,status in [(RuntimeError('/private/checkpoint/secret'),500),(MemoryError(),503)]:
            model=FakeModel()
            with patch.object(model,'predict',side_effect=failure),self.client(model) as client:
                result=client.post('/recommend',json={'user_ids':[1],'top_k':2})
                self.assertEqual(result.status_code,status);self.assertNotIn('secret',result.text)
        model=FakeModel();original=model.predict
        def nonfinite(frame):
            result=original(frame);result['score']=np.nan;return result
        with patch.object(model,'predict',side_effect=nonfinite),self.client(model) as client:
            self.assertEqual(client.post('/recommend',json={'user_ids':[1],'top_k':2}).status_code,500)

    def test_busy_inference_has_no_unbounded_queue(self):
        app=create_app(Settings(),loader=lambda s:LoadedModel(FakeModel(),fake_info()))
        with TestClient(app) as client:
            app.state.lock.acquire()
            try:self.assertEqual(client.post('/recommend',json={'user_ids':[1],'top_k':2}).status_code,503)
            finally:app.state.lock.release()

    def test_configuration_and_model_file_fail_closed(self):
        for kwargs in [dict(expected_version='2'),dict(model_source='alias'),dict(cpu_threads=8),dict(max_batch_users=100),dict(model_source='registry')]:
            with self.assertRaises(ValueError):Settings(**kwargs)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with self.assertRaises(ValueError):relative_file(root,'../secret')
            (root/'unsafe').symlink_to(root/'secret')
            with self.assertRaises(ValueError):relative_file(root,'unsafe')
            (root/'COMPLETE.json').write_text('{"files": {"missing": {"bytes": 1, "sha256": "none"}}}')
            with self.assertRaises(ValueError):check_files(root,'COMPLETE.json')


if __name__=='__main__':unittest.main(verbosity=2)
