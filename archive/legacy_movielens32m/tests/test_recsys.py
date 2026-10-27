"""Small synthetic checks only. Never loads the real MovieLens data."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch
# Bound native CPU sparse-test concurrency; performance is not measured by these tests.
torch.set_num_threads(1)
from scipy import sparse

from recsys.config import default_config
from recsys.data import build_context,relevance,internal_partition,verify_splits
from recsys.metrics import query_metrics,summarize
from recsys.candidate_fusion import fuse,rrf
from recsys.ranking_features import features,attach_labels,sample_training,FEATURES
from recsys.retrievers.popularity import Popularity
from recsys.retrievers.genre import Genre
from recsys.retrievers.itemcf import ItemCF
from recsys.retrievers.base import VectorRetriever,CHANNELS
from recsys.retrievers.neural import PairStream
from recsys.retrievers.lightgcn import propagation,graph_tensor,fit_lightgcn
from recsys.retrievers.two_tower import fit_two_tower
from recsys.runtime import Store,digest
from recsys.training import load_model,train_channels
from recsys.evaluation import evaluate,evaluate_saved_ranker


from helpers import fixture


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.train,self.movies,self.future = fixture()
        self.ctx = build_context(self.train,self.movies)

    def test_metrics_known_answer(self):
        m = query_metrics([9,2,3],{2,3,4})
        self.assertAlmostEqual(m['recall20'],2/3)
        self.assertAlmostEqual(m['ndcg20'],(1/np.log2(3)+1/np.log2(4))/(1+1/np.log2(3)+1/np.log2(4)))
        self.assertEqual(query_metrics([], {2})['ndcg20'],0)
        with self.assertRaises(ValueError): query_metrics([1,1],{1})
        rows = [dict(userId=1,cohort='cold',**query_metrics([1],{2})),
                dict(userId=2,cohort='warm',**query_metrics([2],{2}))]
        self.assertEqual(summarize(rows,bootstrap=10)[0]['recall100'],.5)

    def test_history_catalog_labels_and_time(self):
        u = self.ctx.user_index(1)
        low = int(self.train[(self.train.userId==1)&(self.train.rating<4)].movieId.iloc[0])
        recs = Popularity(self.ctx).retrieve(1,100)
        self.assertNotIn(low,[r['movieId'] for r in recs])
        self.assertNotIn(30,[r['movieId'] for r in recs])
        truth,audit = relevance(self.future,self.ctx)
        self.assertEqual(audit['policy_B_catalog_excluded'],1)
        self.assertNotIn(30,truth[1])
        before,after,boundary = internal_partition(self.train,tail_days=30/86400)
        self.assertLess(before.timestamp.max(),after.timestamp.min())
        self.assertEqual(len(before)+len(after),len(self.train))
        self.assertEqual(self.ctx.cohort(99),'cold')

    def test_retriever_exact_blocks_and_cold(self):
        rng = np.random.default_rng(1)
        model = VectorRetriever(self.ctx,'als',rng.normal(size=(8,4)),rng.normal(size=(21,4)))
        for user in [1,4,99]:
            one = model.retrieve(user,5,item_block=3)
            batch = model.retrieve_many([user],5,item_block=4)[user]
            self.assertEqual([r['movieId'] for r in one],[r['movieId'] for r in batch])
            np.testing.assert_allclose([r['score'] for r in one],[r['score'] for r in batch],atol=1e-6)
        self.assertTrue(model.retrieve(99,1)[0]['fallback'])
        zero = VectorRetriever(self.ctx,'als',np.zeros((8,4)),np.zeros((21,4)))
        self.assertEqual([r['movieId'] for r in zero.retrieve(1,5,2)],
                         [r['movieId'] for r in zero.retrieve(1,5,7)])

    def test_itemcf_exact_cosine_full_history(self):
        model = ItemCF(self.ctx).fit(neighbors_per_item=20,block=3)
        p = self.ctx.positive.toarray()
        norms = np.maximum(np.sqrt(p.sum(0)),1)
        reference = (p.T@p)/norms[:,None]/norms[None,:]
        np.fill_diagonal(reference,0)
        reference[:,~self.ctx.eligible] = 0
        np.testing.assert_allclose(model.neighbors.toarray(),reference,atol=1e-6)
        got = model.scores(0,np.arange(len(self.ctx.items)))
        np.testing.assert_allclose(got,p[0]@reference,atol=1e-6)

    def test_unknown_sampling_rejects_low_and_positive(self):
        stream = PairStream(self.ctx,42)
        users = np.repeat(np.arange(8),100)
        draw = stream.unknown(users)
        self.assertTrue(self.ctx.eligible[draw].all())
        self.assertEqual(self.ctx.observed[users,draw].sum(),0)

    def test_fusion_feature_provenance_and_weak_labels(self):
        outputs = {s:Popularity(self.ctx).retrieve_many([1,2],5) for s in CHANNELS}
        pool = fuse(outputs)
        self.assertEqual(len(pool),10)
        self.assertTrue((pool.source_count==6).all())
        self.assertTrue((pool.popularity_rank==pool.als_rank).all())
        frame = features(pool,self.ctx)
        self.assertTrue(set(FEATURES).issubset(frame))
        labels = pd.DataFrame([(1,int(frame.movieId.iloc[0]),4.5),(2,int(frame[frame.userId==2].movieId.iloc[0]),2.)],
                              columns=['userId','movieId','rating'])
        labeled = attach_labels(frame,labels)
        self.assertIn('observed_low',set(labeled.feedback_kind))
        self.assertNotIn('feedback_kind',FEATURES)
        sampled,stats = sample_training(labeled,np.random.default_rng(2),2)
        self.assertEqual(stats['queries_no_retrieved_positive'],1)
        self.assertEqual(stats['sampled_unlabeled_candidates'],2)
        self.assertEqual(len(rrf(pool[pool.userId==1])),5)

    def test_verified_checkpoint_recovery_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp,'run')
            def writer(p): (p/'x.txt').write_text('original')
            a = store.write('model',writer)
            b = store.write('model',lambda p:self.fail('completed stage rewritten'))
            self.assertEqual(a,b)
            (store.root/'.interrupted').mkdir()
            self.assertIsNone(store.load('missing'))
            (a/'x.txt').write_text('corrupt')
            with self.assertRaises(IOError): store.load('model')

    def test_vector_and_sparse_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            model = Genre(self.ctx)
            model.save(p)
            loaded = load_model(self.ctx,'genre',p,'cpu')
            self.assertEqual(model.retrieve(1,5),loaded.retrieve(1,5))
            item = ItemCF(self.ctx).fit(5,3)
            item.save(p)
            loaded = load_model(self.ctx,'itemcf',p,'cpu')
            self.assertEqual(item.retrieve(2,5),loaded.retrieve(2,5))

    def test_lightgcn_exact_adjoint_gradient(self):
        graph = graph_tensor(self.ctx,'cpu')
        E = torch.randn(graph.shape[0],3,requires_grad=True)
        Z = propagation(graph,E,2)
        loss = Z.square().mean()
        expected, = torch.autograd.grad(loss,E)
        detached = Z.detach().requires_grad_(True)
        # Simulate two loss microbatches, preserving global mean scaling.
        half = len(Z)//2
        (detached[:half].square().sum()/detached.numel()).backward()
        (detached[half:].square().sum()/detached.numel()).backward()
        actual = propagation(graph,detached.grad,2)
        torch.testing.assert_close(expected,actual)

    def test_neural_training_cpu_synthetic(self):
        cfg = default_config()
        for name in ['lightgcn','two_tower']:
            cfg[name].update(dim=4,epochs=2,batch_size=16)
        cfg['two_tower']['hidden']=8
        for fit in [fit_lightgcn,fit_two_tower]:
            model = fit(self.ctx,cfg,device='cpu')
            self.assertTrue(np.isfinite(model.V).all())
            self.assertEqual(len(model.retrieve(1,5)),5)
            self.assertEqual(len(model.logs),2)

    def test_sharded_eval_all_channels_no_future_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = default_config()
            cfg.update(query_batch=3,bootstrap=10,retrieval_k=5,item_block=4,feature_summary_rows=100)
            base = Popularity(self.ctx)
            models = {s:base for s in CHANNELS}
            store = Store(tmp,'test')
            report,path = evaluate(models,self.ctx,self.future,cfg,store,'validation')
            self.assertIn('rrf_without_itemcf',report['metrics'])
            self.assertEqual(report['query_count'],9)
            self.assertEqual(report['budget_per_channel'],5)
            for part in (path/'features').glob('*.parquet'):
                frame = pd.read_parquet(part)
                self.assertTrue((frame.item_observed_log >= 0).all())
            again,_ = evaluate(models,self.ctx,self.future,cfg,store,'validation')
            self.assertEqual(again,report)
            nohit=self.future.iloc[:1].copy()
            nohit['userId']=99
            nohit['movieId']=23  # Train-observed, zero positives, absent from the popularity Top-5
            report,_=evaluate(models,self.ctx,nohit,cfg,store,'zero_hits')
            self.assertEqual(report['metrics']['rrf'][0]['ndcg20'],0.)
            self.assertEqual(report['conditional_on_pool_hit']['rrf'],[])

    def test_frozen_manifest_and_boundary_checks(self):
        from recsys.experiment import Experiment
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root/'data'/'ml-32m'
            splits = root/'data'/'splits'/'v1'
            data.mkdir(parents=True)
            splits.mkdir(parents=True)
            validation = self.future.copy()
            test = self.future.copy()
            test.timestamp += 1000
            test.source_row_id += 1000
            frames = dict(train=self.train,validation=validation,test=test)
            pd.concat(frames.values()).to_csv(data/'ratings.csv',index=False)
            self.movies.to_csv(data/'movies.csv',index=False)
            artifacts = {}
            for name,frame in list(frames.items())+[('user_eval_cohorts',pd.DataFrame({'userId':[1]})),
                                                     ('item_eligibility',pd.DataFrame({'movieId':[11]}))]:
                path = splits/(name+'.parquet')
                frame.to_parquet(path,index=False)
                artifacts[path.name] = dict(bytes=path.stat().st_size,sha256=digest(path),rows=len(frame))
            manifest = dict(protocol_version='v1',main_candidate_policy='B',positive_threshold=4.,
                validation_results=dict(all_passed=True),
                user_cohort_definitions={'cold':[0,0],'sparse':[1,4],'moderate':[5,19],'warm':[20,None]},
                source_files={k:dict(bytes=(data/(k+'.csv')).stat().st_size,sha256=digest(data/(k+'.csv')))
                              for k in ['ratings','movies']},
                artifacts=artifacts,row_counts={**{k:len(v) for k,v in frames.items()},'original':sum(map(len,frames.values()))},
                boundaries_utc_and_unix=dict(dataset_min_timestamp=int(self.train.timestamp.min()),
                  dataset_max_timestamp=int(test.timestamp.max()),validation_start_timestamp=1000,test_start_timestamp=2000))
            manifest_path = splits/'split_manifest.json'
            manifest_path.write_text(json.dumps(manifest))
            expected={k:len(v) for k,v in frames.items()}
            _,audit=verify_splits(splits,data,expected)
            self.assertEqual(audit['train']['rows'],len(self.train))
            cfg=default_config()
            cfg.update(retrieval_k=5,query_batch=3,item_block=4,bootstrap=5,early_stop_users=5)
            cfg['itemcf'].update(neighbors_per_item=5,block=3)
            cfg['als'].update(factors=4,iterations=2)
            cfg['lightgcn'].update(dim=4,epochs=1,batch_size=16)
            cfg['two_tower'].update(dim=4,hidden=8,epochs=1,batch_size=16)
            exp=Experiment(root,cfg,'synthetic-only',device='cpu',expected=expected)
            with self.assertRaises(PermissionError): exp.final_evaluation('unselected')
            with self.assertRaises(RuntimeError): exp.freeze_selection()
            if importlib.util.find_spec('implicit'):
                report=exp.retrieval_milestone()
                self.assertEqual(len(exp.models),6)
                self.assertEqual(report['query_count'],9)
                # Complete models reload without fit, and the result stage is reused.
                exp.models=train_channels(exp.ctx,cfg,exp.store,'full_train',device='cpu')
                self.assertEqual(len(exp.models),6)
                # Control-flow test double only: real LightGBM integration has a separate optional test.
                # No mock score is saved outside this disposable synthetic project.
                import sys
                from types import SimpleNamespace
                from unittest.mock import patch
                class FakeBooster:
                    best_iteration = 1
                    def save_model(self,path): Path(path).write_text('test double, not a trained model')
                    def predict(self,frame): return frame['source_count'].to_numpy(dtype=float)
                old_lgb = sys.modules.get('lightgbm')
                sys.modules['lightgbm'] = SimpleNamespace(Booster=lambda **kwargs:FakeBooster())
                cfg['internal_tail_days'] = 35/86400
                # A new immutable run identity includes the internal-boundary configuration.
                exp = Experiment(root,cfg,'synthetic-only',device='cpu',expected=expected)
                exp.retrieval_milestone()
                try:
                    with patch('recsys.rankers.lightgbm_ranker.fit_ranker',return_value=FakeBooster()):
                        ranker_report=exp.ranker_milestone()
                    training_report=json.loads((exp.store.load('ranker/model')/'training_report.json').read_text())
                    boundary=training_report['internal_boundary']
                    self.assertLess(boundary['retriever_max_timestamp'],boundary['label_min_timestamp'])
                    self.assertGreater(training_report['training_counts']['effective_queries'],0)
                    self.assertIn('unconditional',ranker_report)
                    selected=exp.freeze_selection()
                    test_report=exp.final_evaluation(selected)
                    self.assertIn('lightgbm',test_report['metrics'])
                    self.assertEqual(test_report['query_count'],9)
                    for filename,record in artifacts.items():
                        self.assertEqual(digest(splits/filename),record['sha256'])
                finally:
                    if old_lgb is None: sys.modules.pop('lightgbm',None)
                    else: sys.modules['lightgbm']=old_lgb
            manifest['boundaries_utc_and_unix']['validation_start_timestamp']=150
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError): verify_splits(splits,data,expected)

    @unittest.skipUnless(importlib.util.find_spec('implicit'),'implicit unavailable locally; exercise in Colab')
    def test_optimized_als(self):
        from recsys.retrievers.implicit_als import fit_als
        cfg=default_config()
        cfg['als'].update(factors=4,iterations=2)
        model=fit_als(self.ctx,cfg)
        self.assertEqual(len(model.retrieve(1,5)),5)

    @unittest.skipUnless(importlib.util.find_spec('lightgbm'),'lightgbm unavailable locally; exercise in Colab')
    def test_lightgbm_fit_predict(self):
        from recsys.rankers.lightgbm_ranker import fit_ranker,rank
        outputs={s:Popularity(self.ctx).retrieve_many([1,2,3,4],10) for s in CHANNELS}
        f=features(fuse(outputs),self.ctx)
        f['label']=f.groupby('userId').cumcount().eq(0).astype(int)
        cfg=default_config()
        cfg['ranker'].update(trees=5,patience=2,num_leaves=3)
        cfg['cpu_threads'] = 1  # unit-test CPU budget; avoids native OpenMP pool resizing
        m=fit_ranker(f,f,cfg)
        ranked = rank(m, f[f.userId == 1])
        self.assertEqual(set(ranked), set(f.loc[f.userId == 1, 'movieId']))
        self.assertEqual(len(ranked), f.loc[f.userId == 1, 'movieId'].nunique())
        self.assertLessEqual(len(ranked), 10)


if __name__=='__main__':
    unittest.main(verbosity=2)
