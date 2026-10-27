"""Non-fitting cleanup checks, migrated from legacy tests. These are synthetic contracts, not dataset experiments."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from scipy import sparse
import pyarrow.parquet as pq
from helpers import fixture
from recsys.data import build_context
from recsys.runtime import Store
from recsys.metrics import query_metrics,summarize
from recsys.retrievers.base import VectorRetriever,CHANNELS
from recsys.retrievers.popularity import Popularity
from recsys.candidate_fusion import fuse,rrf
from recsys.ranking_features import features,attach_labels,sample_training,FEATURES
from recsys.contracts import validate_candidates
from recsys.artifacts import source_hashes,iter_queries
from recsys.rankers.dataset import ParquetRows,labels_and_groups
from recsys.training import load_model
try:
    import torch
except ImportError:
    torch=None
else:
    torch.set_num_threads(1)
    from recsys.retrievers.neural import PairStream
    from recsys.retrievers.lightgcn import propagation,graph_tensor


class RetainedContractTests(unittest.TestCase):
    def setUp(self):
        self.train,self.movies,self.future=fixture()
        self.ctx=build_context(self.train,self.movies)

    def test_metrics_known_answer(self):
        m = query_metrics([9,2,3],{2,3,4})
        self.assertAlmostEqual(m['recall20'],2/3)
        self.assertAlmostEqual(m['ndcg20'],(1/np.log2(3)+1/np.log2(4))/(1+1/np.log2(3)+1/np.log2(4)))
        self.assertEqual(query_metrics([], {2})['ndcg20'],0)
        with self.assertRaises(ValueError): query_metrics([1,1],{1})
        rows = [dict(userId=1,cohort='cold',**query_metrics([1],{2})),
                dict(userId=2,cohort='warm',**query_metrics([2],{2}))]
        self.assertEqual(summarize(rows,bootstrap=10)[0]['recall100'],.5)

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

    @unittest.skipIf(torch is None, 'existing PyTorch unavailable; no installation authorized')
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

    @unittest.skipIf(torch is None, 'existing PyTorch unavailable; no installation authorized')
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

    def test_candidate_contract_and_top_k_upper_bound(self):
        from recsys.retrievers.popularity import Popularity
        train,movies,future=fixture();ctx=build_context(train,movies)
        model=Popularity(ctx)
        frame=pd.DataFrame(model.retrieve(1,10))
        validate_candidates(frame,ctx,'popularity',10,{1})
        self.assertLess(len(frame),10)
        expected=set(ctx.items[ctx.eligible])-set(ctx.items[ctx.seen(ctx.user_index(1))])
        self.assertEqual(set(frame.movieId),expected)
        bad=frame.copy();bad.loc[bad.index[0],'movieId']=30
        with self.assertRaises(ValueError):validate_candidates(bad,ctx,'popularity',10)
        bad=frame.copy();bad.loc[bad.index[0],'score']=np.nan
        with self.assertRaises(ValueError):validate_candidates(bad,ctx,'popularity',10)
        bad=frame.copy();bad.loc[bad.index[0],'movieId']=int(train[train.userId==1].movieId.iloc[-1])
        with self.assertRaises(ValueError):validate_candidates(bad,ctx,'popularity',10)
        bad=pd.concat([frame,frame.iloc[:1]])
        with self.assertRaises(ValueError):validate_candidates(bad,ctx,'popularity',10)
        tied=frame.copy();tied['score']=0.;tied=tied.sort_values('movieId');tied['rank']=range(1,len(tied)+1)
        validate_candidates(tied,ctx,'popularity',10)
        with self.assertRaises(ValueError):validate_candidates(tied.iloc[::-1],ctx,'popularity',10)

    def test_source_hashes_are_model_specific(self):
        pop=source_hashes('popularity');graph=source_hashes('lightgcn')
        self.assertIn('retrievers/popularity.py',pop)
        self.assertNotIn('retrievers/lightgcn.py',pop)
        self.assertIn('retrievers/lightgcn.py',graph)
        self.assertNotIn('retrievers/two_tower.py',graph)

    def test_parquet_sequence_preserves_queries_and_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'x.parquet'
            frame=pd.DataFrame(dict(userId=[1,1,2,2,2],label=[1,0,0,1,0],a=[1.,2.,3.,4.,5.],b=[6.,7.,8.,9.,10.]))
            frame.to_parquet(path,index=False,row_group_size=2)
            rows=ParquetRows([path],['a','b'],2)
            np.testing.assert_equal(rows[1:5],frame[['a','b']].to_numpy()[1:5])
            np.testing.assert_equal(rows[[4,0,2]],frame[['a','b']].to_numpy()[[4,0,2]])
            np.testing.assert_equal(rows[3],frame[['a','b']].to_numpy()[3])
            labels,groups=labels_and_groups([path])
            self.assertEqual(groups.tolist(),[2,3]);self.assertEqual(labels.tolist(),frame.label.tolist())
            self.assertEqual([len(f) for u,f in iter_queries(path)],[2,3])

    def test_vector_checkpoint_roundtrip_without_fitting(self):
        rng=np.random.default_rng(7)
        model=VectorRetriever(self.ctx,'implicit_als',rng.normal(size=(len(self.ctx.users),4)),
                              rng.normal(size=(len(self.ctx.items),4)))
        with tempfile.TemporaryDirectory() as tmp:
            model.save(Path(tmp))
            loaded=load_model(self.ctx,'implicit_als',Path(tmp))
            self.assertEqual(model.retrieve(1,10),loaded.retrieve(1,10))

    def test_removed_workflows_are_not_active_imports(self):
        import recsys.data as data
        self.assertFalse(hasattr(data,'FrozenData'))
        self.assertFalse(hasattr(data,'EXPECTED_ROWS'))
        for name in ['recsys.config','recsys.workflows','recsys.experiment','recsys.final_evaluation',
                     'recsys.smoke','recsys.fusion.workflow','recsys.rankers.workflow']:
            self.assertIsNone(importlib.util.find_spec(name),name)

    def test_source_provenance_references_existing_active_files(self):
        for name in ['popularity','genre','itemcf','implicit_als','lightgcn','two_tower','fusion','lightgbm']:
            self.assertTrue(source_hashes(name))


if __name__=='__main__':
    unittest.main(verbosity=2)
