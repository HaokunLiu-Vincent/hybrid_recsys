"""Disposable SQL tracking/registry and synthetic complete-package integration."""
from pathlib import Path
import copy
import importlib.util
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient
from test_retrieval_training import frozen_fixture,configuration,reseal
from recsys.contracts import CHANNELS
from recsys.runtime import Store,digest,fingerprint
from recsys.artifacts import dump_json
from recsys.retrieval_workflow import train
from recsys.retrieval_artifacts import load_checkpoint,checked_manifest
from recsys.rankers.small_schema import feature_schema
from recsys.rankers.small_lambdamart import VERSION as RANKER_VERSION
from recsys.mlflow_tracking import import_record,metric_names,experiment
from recsys.inference import load_hybrid_recommender,validate_input,pipeline_rankings
from recsys.releases import package_release
from recsys.mlflow_registry import register_release,verify_release,candidate_target
from recsys.retrieval_workflow import ROOT


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.client=MlflowClient('sqlite:///'+str(self.root/'tracking.db'))
        self.client.create_experiment('fixture',artifact_location=(self.root/'artifacts').as_uri())

    def tearDown(self):self.temp.cleanup()

    def record(self):return dict(phase=3,run_id='synthetic',split_manifest_sha256='fixture',manifest_sha256='fixture',
                               git_revision=None,cutoff_identity=None)

    def test_import_provenance_artifacts_metrics_and_idempotency(self):
        metrics=metric_names(dict(recall20=.25,ndcg20=.1,hit20=.5))
        first,new=import_record(self.client,'fixture','synthetic/key',self.record(),dict(seed=42),metrics)
        second,again=import_record(self.client,'fixture','synthetic/key',self.record(),dict(seed=42),metrics)
        self.assertTrue(new);self.assertFalse(again);self.assertEqual(first,second)
        run=self.client.get_run(first)
        self.assertEqual(run.data.metrics['validation/policy_b/recall_at_20'],.25)
        self.assertEqual(run.data.tags['retrospective_import'],'true')
        self.assertEqual(run.data.tags['original_training_outside_mlflow'],'true')
        self.assertEqual(run.data.tags['original_execution_timestamp'],'not_recorded; no timestamp invented')
        self.assertNotEqual(run.data.tags['import_timestamp'],'')

    def test_conflicting_source_or_metrics_rejected(self):
        record=self.record();args=(self.client,'fixture','key',record,{}, {'validation/policy_b/recall_at_20':.1})
        import_record(*args)
        bad=dict(record,manifest_sha256='changed')
        with self.assertRaisesRegex(ValueError,'Conflicting'):import_record(self.client,'fixture','key',bad,{},args[-1])
        with self.assertRaisesRegex(ValueError,'Conflicting'):import_record(self.client,'fixture','key',record,{}, {'validation/policy_b/recall_at_20':.2})

    def test_metric_namespaces_and_experiment_discovery(self):
        a=metric_names(dict(recall20=.2),'validation/policy_b');b=metric_names(dict(recall20=.1),'validation/all_observed')
        self.assertFalse(set(a)&set(b));self.assertEqual(experiment(self.client,'fixture'),experiment(self.client,'fixture'))


class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name);data=frozen_fixture(cls.root)
        cls.checkpoints={}
        for source in CHANNELS:
            cfg=cls.root/(source+'.yaml');cfg.write_text(yaml.safe_dump(configuration(source)))
            result=train(source,cfg,data,cls.root/'models','synthetic')
            cls.checkpoints[source]=Path(result['checkpoint'])
        model,m=load_checkpoint(cls.checkpoints['popularity']);schema=feature_schema(model.ctx)
        # Synthetic native booster is a serialization fixture, not an experimental benchmark.
        x=np.random.default_rng(42).normal(size=(24,len(schema['names']))).astype('float32')
        ds=lgb.Dataset(pd.DataFrame(x,columns=schema['names']),label=[1,0,0,0]*6,group=[4]*6)
        booster=lgb.train(dict(objective='lambdarank',num_leaves=3,min_data_in_leaf=2,num_threads=1,verbosity=-1,seed=42),ds,num_boost_round=3)
        def writer(path):
            booster.save_model(str(path/'model.txt'));dump_json(path/'feature_schema.json',schema)
            dump_json(path/'manifest.json',dict(kind='lightgbm_lambdamart',version=RANKER_VERSION,
                feature_schema_sha256=schema['sha256'],lightgbm_version=lgb.__version__,
                schema_source_sha256=digest(ROOT/'src/recsys/rankers/small_schema.py'),source_sha256='synthetic_fixture',
                split_manifest_sha256=m['split_manifest_sha256'],validation_evaluation_id='synthetic_inference',
                comparison={'lambdamart':[dict(recall20=0.,ndcg20=0.,hit20=0.)]}))
        cls.ranker=Store(cls.root,'ranker').write('synthetic',writer)
        cls.packaged=package_release(cls.checkpoints,cls.ranker,cls.root/'releases',cls.root/'parity','synthetic')
        cls.release=Path(cls.packaged['release']);cls.parity=Path(cls.packaged['parity'])

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_complete_package_full_train_shared_context_and_no_targets(self):
        m=checked_manifest(self.release);self.assertEqual(set(m['retrievers']),set(CHANNELS))
        self.assertTrue(all(r['training_kind']=='external_train_per_user' for r in m['retrievers'].values()))
        files=set(json.loads((self.release/'COMPLETE.json').read_text())['files'])
        self.assertFalse(any('validation.parquet' in p or 'test.parquet' in p or 'candidates.parquet' in p or 'labels' in p for p in files))
        self.assertEqual(sum(p.endswith('observed.npz') for p in files),1)
        self.assertIn('src/recsys/inference.py',files);self.assertEqual(m['rrf_constant'],60)

    def test_end_to_end_parity_determinism_and_catalog_seen_rules(self):
        recommender=load_hybrid_recommender(self.release);a=recommender.recommend([2,1],20);b=recommender.recommend([2,1],20)
        pd.testing.assert_frame_equal(a,b,check_exact=True);self.assertEqual(a.userId.drop_duplicates().tolist(),[2,1])
        self.assertFalse(a.duplicated(['userId','movieId']).any());self.assertTrue(np.isfinite(a.score).all())
        for u,g in a.groupby('userId'):
            self.assertLessEqual(len(g),20);self.assertEqual(g['rank'].tolist(),list(range(1,len(g)+1)))
            self.assertFalse(g.movieId.isin(recommender.ctx.items[recommender.ctx.seen(recommender.ctx.user_index(u))]).any())
            self.assertTrue(g.movieId.isin(recommender.ctx.items[recommender.ctx.eligible]).all())
        self.assertTrue(self.packaged['exact_prediction_parity'])

    def test_unknown_popularity_bypasses_ranker(self):
        model=load_hybrid_recommender(self.release)
        with patch.object(model.booster,'predict',side_effect=AssertionError('Unknown user must bypass ranking')):
            result=model.recommend([99999],5)
        expected=model.models['popularity'].retrieve(99999,5,4)
        self.assertEqual(result.movieId.tolist(),[r['movieId'] for r in expected])
        self.assertTrue(result.fallback.eq('unknown_user_popularity').all())

    def test_invalid_ids_top_k_duplicates_and_empty_request(self):
        model=load_hybrid_recommender(self.release)
        for users,k in [([0],20),([-1],20),([True],20),([1.],20),(['1'],20),([1],0),([1],101),([1],True),([1,1],20)]:
            with self.subTest(users=users,k=k),self.assertRaises(ValueError):model.recommend(users,k)
        self.assertTrue(model.recommend([],20).empty)

    def test_relocation_and_corruption_dependency_schema_rejections(self):
        moved=self.root/'relocated';shutil.copytree(self.release,moved)
        model=load_hybrid_recommender(moved);self.assertGreater(len(model.recommend([1],5)),0)
        (moved/'ranker/model.txt').write_text('corrupt fixture')
        with self.assertRaises(OSError):load_hybrid_recommender(moved)
        for key,value in [('rrf_constant',61),('runtime_packages',{'numpy':'0.0.0'}),('mapping_id','wrong')]:
            path=self.root/('invalid_'+key);shutil.copytree(self.release,path)
            m=json.loads((path/'manifest.json').read_text());m[key]=value;dump_json(path/'manifest.json',m);reseal(path)
            with self.assertRaises(ValueError):load_hybrid_recommender(path)

    def test_historical_checkpoint_not_accepted_for_release(self):
        invalid=self.root/'historical';shutil.copytree(self.checkpoints['popularity'],invalid)
        m=json.loads((invalid/'manifest.json').read_text());m['training_kind']='historical_snapshot_per_user'
        dump_json(invalid/'manifest.json',m);reseal(invalid)
        with self.assertRaises(ValueError):package_release(dict(self.checkpoints,popularity=invalid),self.ranker,self.root/'bad_release',self.root/'bad_parity','bad')

    def test_registry_complete_model_alias_idempotency_and_prediction_parity(self):
        uri='sqlite:///'+str(self.root/'registry.db');client=MlflowClient(uri)
        from recsys.mlflow_tracking import RELEASE
        client.create_experiment(RELEASE,artifact_location=(self.root/'registry_artifacts').as_uri())
        eid=experiment(client,'synthetic_lineage');rid=client.create_run(eid).info.run_id;client.set_terminated(rid)
        backfill=self.root/'backfill.json';dump_json(backfill,dict(runs={'phase4/lambdamart':rid}))
        result=register_release(self.release,self.parity,backfill,'SyntheticHybrid',uri)
        again=register_release(self.release,self.parity,backfill,'SyntheticHybrid',uri)
        self.assertTrue(result['created']);self.assertFalse(again['created']);self.assertEqual(result['version'],again['version'])
        self.assertEqual(candidate_target(client,'SyntheticHybrid').tags['release_id'],self.packaged['release_id'])
        checked=verify_release(result['model_uri'],self.parity,uri);self.assertTrue(checked['exact_parity'])
        client.set_model_version_tag('SyntheticHybrid',result['version'],'release_id','different_release_fixture')
        with self.assertRaisesRegex(ValueError,'different release'):register_release(self.release,self.parity,backfill,'SyntheticHybrid',uri)


if __name__=='__main__':unittest.main(verbosity=2)
