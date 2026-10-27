"""Modular contracts and independent workflows; all inputs are disposable synthetic data."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from helpers import fixture
from recsys.config import default_config
from recsys.contracts import CHANNELS,validate_candidates,compatible
from recsys.artifacts import read_artifact,iter_queries,source_hashes,dump_json
from recsys.data import build_context,relevance,FrozenData
from recsys.workflows import RetrievalExperiment
from recsys.fusion.workflow import FusionExperiment
from recsys.rankers.workflow import RankingExperiment
from recsys.rankers.dataset import ParquetRows,labels_and_groups
from recsys.runtime import digest
from recsys.final_evaluation import freeze_selection,evaluate_selected


def make_project(root):
    train,movies,validation = fixture()
    test = validation.copy()
    test.timestamp += 1000
    test.source_row_id += 1000
    data,splits = root/'data'/'ml-32m',root/'data'/'splits'/'v1'
    data.mkdir(parents=True); splits.mkdir(parents=True)
    frames = dict(train=train,validation=validation,test=test)
    pd.concat(frames.values()).to_csv(data/'ratings.csv',index=False)
    movies.to_csv(data/'movies.csv',index=False)
    ctx = build_context(train,movies)
    truth,_ = relevance(validation,ctx)
    users = sorted(set(train.userId)|set(validation.userId))
    cohorts = pd.DataFrame(dict(userId=users,positive_history_cohort=[ctx.cohort(u) for u in users],
                                validation_eligible_user_B=[u in truth for u in users],
                                test_secret_label=[999]*len(users)))
    items = pd.DataFrame(dict(movieId=ctx.items,eligible_policy_B=ctx.eligible))
    artifacts = {}
    for name,frame in list(frames.items())+[('user_eval_cohorts',cohorts),('item_eligibility',items)]:
        path=splits/(name+'.parquet'); frame.to_parquet(path,index=False)
        artifacts[path.name]=dict(bytes=path.stat().st_size,sha256=digest(path),rows=len(frame))
    manifest = dict(protocol_version='v1',main_candidate_policy='B',positive_threshold=4.,
        validation_results=dict(all_passed=True),
        user_cohort_definitions={'cold':[0,0],'sparse':[1,4],'moderate':[5,19],'warm':[20,None]},
        source_files={k:dict(bytes=(data/(k+'.csv')).stat().st_size,sha256=digest(data/(k+'.csv')))
                      for k in ['ratings','movies']},artifacts=artifacts,
        row_counts={**{k:len(v) for k,v in frames.items()},'original':sum(map(len,frames.values()))},
        boundaries_utc_and_unix=dict(dataset_min_timestamp=int(train.timestamp.min()),
            dataset_max_timestamp=int(test.timestamp.max()),validation_start_timestamp=1000,test_start_timestamp=2000))
    dump_json(splits/'split_manifest.json',manifest)
    return {k:len(v) for k,v in frames.items()},ctx,artifacts


def small_config():
    c=default_config()
    c.update(retrieval_k=10,query_batch=3,item_block=4,bootstrap=5,early_stop_users=5,
             snapshot_cutoff_utc='1970-01-01T00:02:30Z',snapshot_label_end_utc=None)
    c['itemcf'].update(neighbors_per_item=5,block=3)
    c['als'].update(factors=4,iterations=2)
    c['lightgcn'].update(dim=4,epochs=1,batch_size=16)
    c['two_tower'].update(dim=4,hidden=8,epochs=1,batch_size=16)
    return c


def produce(root,name,config,expected,mode='full_train',shared_data=None):
    if importlib.util.find_spec('torch'):
        import torch
        torch.set_num_threads(1)
    exp=RetrievalExperiment(root,name,config,mode=mode,expected=expected,device='cpu')
    exp.prepare(shared_data=shared_data); exp.initialize(); exp.train(); checkpoint=exp.save_checkpoint(); result=exp.evaluate()
    return exp,checkpoint,result


class ModularContracts(unittest.TestCase):
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

    def test_development_does_not_open_test_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);expected,ctx,artifacts=make_project(root)
            original=pq.ParquetFile
            opened=[]
            def guarded(path,*args,**kwargs):
                opened.append(str(path))
                if Path(path).name=='test.parquet':raise AssertionError('Test opened during development')
                return original(path,*args,**kwargs)
            with patch('recsys.data.pq.ParquetFile',side_effect=guarded):
                data=FrozenData(root,expected).prepare()
                self.assertEqual(set(data.population.truth),set(relevance(data.future,data.ctx)[0]))
            with self.assertRaises(PermissionError):data.open_test()
            with self.assertRaises(ValueError):
                FrozenData(root,expected).prepare('historical_snapshot',None)
            with self.assertRaises(ValueError):
                FrozenData(root,expected).prepare('historical_snapshot','1970-01-01T01:00:00Z')

    def test_independent_baseline_checkpoint_and_result_compatibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);expected,ctx,_=make_project(root);config=small_config()
            a,cp,pop=produce(root,'popularity',config,expected)
            b,_,genre=produce(root,'genre',config,expected,shared_data=a.data)
            self.assertIs(a.data,b.data)
            manifests={'popularity':read_artifact(pop),'genre':read_artifact(genre)}
            compatible(manifests,require_all=False)
            bad=deepcopy(manifests);bad['genre']['evaluation_id']='other-users'
            with self.assertRaises(ValueError):compatible(bad,require_all=False)
            loaded=RetrievalExperiment(root,'popularity',config,'load_evaluate',cp,expected=expected)
            loaded.prepare()
            with patch('recsys.workflows.fit_retriever',side_effect=AssertionError('retrained')):
                loaded.train();loaded.save_checkpoint();path=loaded.evaluate()
            self.assertEqual(path,pop)
            changed=deepcopy(config);changed['seed']=7
            with self.assertRaises(ValueError):
                RetrievalExperiment(root,'popularity',changed,'load_evaluate',cp,expected=expected).prepare()
            for user,frame in iter_queries(pop/'candidates.parquet'):
                validate_candidates(frame,ctx,'popularity',10,set(a.data.population.users))
            (cp/'positive_counts.npy').write_bytes(b'corrupt')
            with self.assertRaises(IOError):read_artifact(cp)

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

    @unittest.skipUnless(importlib.util.find_spec('implicit'),'optimized ALS library unavailable')
    def test_six_independent_runs_fusion_and_historical_examples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);expected,ctx,artifacts=make_project(root);config=small_config()
            validation={};historical={}
            for name in CHANNELS:
                _,_,validation[name]=produce(root,name,config,expected)
                _,_,historical[name]=produce(root,name,config,expected,'rank_snapshot')
            # Fusion cannot invoke fitting; consume only saved per-channel Parquet files.
            with patch('recsys.training.fit_retriever',side_effect=AssertionError('fusion retrained')):
                val=FusionExperiment(root,validation,config,expected).run()
                internal=FusionExperiment(root,historical,config,expected).run()
            self.assertIn('rrf_without_lightgcn',json.loads((val/'metrics.json').read_text())['metrics'])
            rank_config=dict(trees=5,num_leaves=3,lr=.05,patience=2,unlabeled_per_query=3,
                             sequence_batch_rows=4,validation_queries=5,seed=42,cpu_threads=1,bootstrap=5)
            with self.assertRaises(ValueError):RankingExperiment(root,val,val,rank_config,expected=expected)
            ranker=RankingExperiment(root,internal,val,rank_config,expected=expected)
            examples=ranker.prepare_examples()
            report=json.loads((examples/'sampling_report.json').read_text())
            self.assertGreater(report['counts']['effective_queries'],0)
            self.assertLess(report['internal_boundary']['training_cutoff'],report['internal_boundary']['label_start'])
            self.assertNotIn('label',ranker.columns);self.assertNotIn('feedback_kind',ranker.columns)
            table=pd.read_parquet(examples/'train.parquet')
            self.assertTrue(set(table.feedback_kind).issubset({'observed_positive','observed_low','unobserved'}))
            if importlib.util.find_spec('lightgbm'):
                checkpoint=ranker.fit_or_load(examples)
                evaluated=ranker.evaluate()
                selection=freeze_selection(root,validation,val,checkpoint,evaluated)
                with self.assertRaises(PermissionError):evaluate_selected(root,selection,expected=expected)
                selection_id=read_artifact(selection)['selection_id']
                final=evaluate_selected(root,selection,enabled=True,confirmation=selection_id,expected=expected)
                final_report=json.loads((final/'metrics.json').read_text())
                self.assertIn('lightgbm',final_report['metrics'])
                self.assertIn('test_feature_distribution_mismatch',final_report)
                again=evaluate_selected(root,selection,enabled=True,confirmation=selection_id,expected=expected)
                self.assertEqual(final,again)
            for filename,record in artifacts.items():self.assertEqual(digest(root/'data'/'splits'/'v1'/filename),record['sha256'])

    @unittest.skipUnless(importlib.util.find_spec('lightgbm'),'LightGBM unavailable locally; run in Colab')
    def test_streaming_lightgbm_sequence_fit(self):
        from recsys.rankers.lightgbm_ranker import fit_from_shards
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'features.parquet'
            frame=pd.DataFrame(dict(userId=np.repeat(np.arange(20),7),label=np.tile([1,0,0,0,0,0,0],20),
                                    x=np.tile(np.arange(7),20).astype(float),y=np.repeat(np.arange(20),7).astype(float)))
            frame.to_parquet(path,index=False,row_group_size=13)
            c=dict(trees=5,lr=.1,num_leaves=3,seed=42,cpu_threads=1,patience=2,sequence_batch_rows=10)
            model,history=fit_from_shards([path],[path],['x','y'],c)
            scores=model.predict(frame[['x','y']])
            self.assertEqual(len(scores),len(frame));self.assertTrue(np.isfinite(scores).all())


if __name__=='__main__':unittest.main(verbosity=2)
