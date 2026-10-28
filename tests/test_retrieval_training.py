"""Synthetic fitting/serialization and strict Train-only/Validation-only integration tests."""
from pathlib import Path
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
import yaml

from helpers import fixture as existing_fixture
from test_small_data import archive_fixture
from recsys.small_data import publish_archive,sha256
from recsys.small_pipeline import ROOT,settings,prepare
from recsys.small_benchmark import load_training,load_validation
from recsys.data import build_context
from recsys.training import fit_retriever
from recsys.contracts import CHANNELS,validate_candidates
from recsys.retrievers.neural import seed_all,PairStream
from recsys.retrievers.lightgcn import graph_tensor,propagation,fit_lightgcn
from recsys.retrievers.two_tower import Towers,export_vectors,fit_two_tower
from recsys.retrieval_artifacts import load_checkpoint,checked_manifest,VERSION
from recsys.retrieval_workflow import train,evaluate,benchmark,summarize_candidates
from recsys.runtime import digest


def configuration(model):
    c=yaml.safe_load((ROOT/'configs'/f'{model}.yaml').read_text())
    c.update(query_batch=2,item_block=4)
    if model=='itemcf':c['itemcf'].update(neighbors_per_item=5,block=4)
    if model=='implicit_als':c['als'].update(factors=4,iterations=2)
    if model in ('lightgcn','two_tower'):c[model].update(dim=4,epochs=2,batch_size=16)
    if model=='two_tower':c[model]['hidden']=8
    return c


def frozen_fixture(root):
    source=root/'source';source.mkdir()
    _,movies,_=existing_fixture()
    ratings=pd.DataFrame([(u,10+(u*2+j)%20,2. if j%4==0 else 4.5,u*100+j)
                          for u in range(1,9) for j in range(12)],
                         columns=['userId','movieId','rating','timestamp'])
    ratings.to_csv(source/'ratings.csv',index=False);movies.to_csv(source/'movies.csv',index=False)
    (source/'README.md').write_text('Synthetic test fixture only: 96 ratings across 21 movies created by 8 users.\n')
    archive_fixture(source,root/'download.zip');raw=root/'raw'
    publish_archive(root/'download.zip',raw,{})
    c=copy.deepcopy(settings())
    c['dataset'].update(provenance_sha256=sha256(raw/'provenance.json'),ratings_sha256=sha256(raw/'ratings.csv'),
        movies_sha256=sha256(raw/'movies.csv'),expected_ratings=96,expected_users=8,expected_metadata_movies=21)
    prepare(raw,root/'splits',c)
    return root/'splits/small_v1'


def reseal(path):
    """Exercise semantic rejection independently of simple byte-corruption detection."""
    marker=json.loads((path/'COMPLETE.json').read_text())
    for filename in marker['files']:
        p=path/filename;marker['files'][filename].update(bytes=p.stat().st_size,sha256=digest(p))
    (path/'COMPLETE.json').write_text(json.dumps(marker))


class RetrievalTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary=tempfile.TemporaryDirectory();cls.root=Path(cls.temporary.name)
        cls.directory=frozen_fixture(cls.root)
        cls.training=load_training(cls.directory)

    @classmethod
    def tearDownClass(cls):cls.temporary.cleanup()

    def test_all_six_fit_independently_and_reject_seen_catalog_candidates(self):
        ctx=self.training.ctx
        for name in CHANNELS:
            with self.subTest(model=name):
                model=fit_retriever(ctx,name,configuration(name),device='cpu')
                for user in [int(ctx.users[0]),999]:
                    records=model.retrieve(user,100,item_block=4)
                    frame=pd.DataFrame(records)
                    validate_candidates(frame,ctx,name,100,{user})
                    self.assertLessEqual(len(frame),int(ctx.eligible.sum()))
                    self.assertTrue(np.isfinite(frame.score).all())
                if name!='popularity':
                    ids=np.flatnonzero(ctx.eligible)
                    self.assertFalse(np.allclose(model.scores(0,ids),ctx.popularity[ids]))

    def test_itemcf_cosine_orientation_neighbor_bound_and_positive_aggregation(self):
        c=configuration('itemcf');c['itemcf']['neighbors_per_item']=100
        ctx=self.training.ctx;model=fit_retriever(ctx,'itemcf',c)
        x=ctx.positive.toarray()  # tiny synthetic matrix only
        norm=np.sqrt((x*x).sum(axis=0));expected=x.T@x/np.maximum(norm[:,None]*norm[None,:],1)
        np.fill_diagonal(expected,0);expected[:,~ctx.eligible]=0
        np.testing.assert_allclose(model.neighbors.toarray(),expected,atol=1e-6)
        ids=np.arange(len(ctx.items))
        np.testing.assert_allclose(model.scores(0,ids),x[0]@expected,atol=1e-6)
        bounded=fit_retriever(ctx,'itemcf',configuration('itemcf'))
        self.assertLessEqual(int(np.diff(bounded.neighbors.indptr).max()),5)

    def test_als_orientation_and_observed_low_confidence_semantics(self):
        ctx=self.training.ctx;captured={}
        class FakeALS:
            def __init__(self,**kwargs):captured['config']=kwargs
            def fit(self,confidence):
                captured['confidence']=confidence.copy()
                self.user_factors=np.ones((len(ctx.users),4),dtype='float32')
                self.item_factors=np.ones((len(ctx.items),4),dtype='float32')
        with patch('implicit.cpu.als.AlternatingLeastSquares',FakeALS):
            model=fit_retriever(ctx,'implicit_als',configuration('implicit_als'))
        confidence=captured['confidence'];self.assertEqual(confidence.shape,ctx.observed.shape)
        self.assertTrue(np.all(confidence.data[ctx.observed.data>=4]==21))
        self.assertTrue(np.all(confidence.data[ctx.observed.data<4]==-2))
        self.assertEqual(captured['config']['num_threads'],1)
        self.assertEqual(model.U.shape,(len(ctx.users),4))

    def test_lightgcn_normalization_propagation_and_actual_optimizer_update(self):
        ctx=self.training.ctx;c=configuration('lightgcn');graph=graph_tensor(ctx,'cpu')
        x=ctx.positive.toarray();adj=np.block([[np.zeros((len(ctx.users),len(ctx.users))),x],
                                             [x.T,np.zeros((len(ctx.items),len(ctx.items)))]]).astype('float32')
        degrees=adj.sum(axis=1);inv=np.divide(1,np.sqrt(degrees),out=np.zeros_like(degrees),where=degrees>0)
        np.testing.assert_allclose(graph.to_dense().numpy(),adj*inv[:,None]*inv[None,:],atol=1e-6)
        seed_all(c['seed']);initial=torch.randn(graph.shape[0],4)*.05
        initial_vectors=propagation(graph,initial,c['lightgcn']['layers']).detach().numpy()
        updates=[]
        model=fit_lightgcn(ctx,c,checkpoint=lambda e,m,l,s:updates.append((e,s['base'].clone(),l[-1].copy())))
        self.assertEqual([r[0] for r in updates],[1,2])
        self.assertFalse(np.array_equal(initial_vectors,np.vstack([model.U,model.V])))
        self.assertFalse(torch.equal(initial,updates[0][1]))
        self.assertTrue(all(np.isfinite(r[2]['mean_bpr']) for r in updates))
        self.assertEqual(updates[-1][2]['optimizer_steps'],2)  # retained one Adam step per epoch

    def test_two_tower_id_genre_architecture_gradients_and_parameter_update(self):
        ctx=self.training.ctx;c=configuration('two_tower');seed_all(c['seed'])
        net=Towers(ctx,c['two_tower']);initial={k:v.clone() for k,v in net.state_dict().items()}
        self.assertEqual(net.user_mlp[0].in_features,4+ctx.genres.shape[1])
        vectors=export_vectors(net,ctx,'cpu');updates=[]
        model=fit_two_tower(ctx,c,checkpoint=lambda e,m,l,s:updates.append(copy.deepcopy(s)))
        self.assertFalse(np.array_equal(vectors.U,model.U))
        self.assertTrue(any(not torch.equal(initial[k],updates[-1]['network'][k]) for k in initial))
        self.assertTrue(all(np.isfinite(row['mean_bpr']) for row in model.logs))
        for name,value in updates[-1]['network'].items():self.assertTrue(torch.isfinite(value).all(),name)

    def test_negative_sampling_excludes_all_observed_low_and_positive_pairs(self):
        ctx=self.training.ctx;users=np.repeat(np.arange(len(ctx.users)),100)
        draws=PairStream(ctx,42).unknown(users)
        self.assertTrue(ctx.eligible[draws].all());self.assertEqual(ctx.observed[users,draws].sum(),0)

    def test_training_stage_does_not_read_either_holdout_or_label_tables(self):
        original=pd.read_parquet;reads=[]
        def guarded(path,*args,**kwargs):
            path=Path(path)
            if path.is_relative_to(self.directory):
                self.assertIn(path.name,{'train.parquet','movies.parquet','user_mapping.parquet','movie_mapping.parquet','item_eligibility.parquet'})
                reads.append(path.name)
            return original(path,*args,**kwargs)
        config=self.root/'pop_guard.yaml';config.write_text(yaml.safe_dump(configuration('popularity')))
        with patch('pandas.read_parquet',side_effect=guarded):
            result=train('popularity',config,self.directory,self.root/'guard_artifacts','guard')
        self.assertEqual(result['status'],'completed');self.assertNotIn('validation.parquet',reads)
        self.assertNotIn('test.parquet',reads)

    def test_checkpoint_parity_portability_config_source_mapping_rejections(self):
        config=self.root/'genre_cfg.yaml';c=configuration('genre');config.write_text(yaml.safe_dump(c))
        result=train('genre',config,self.directory,self.root/'portable_artifacts','portable')
        path=Path(result['checkpoint']);loaded,m=load_checkpoint(path,'genre')
        before=loaded.retrieve_many(self.training.ctx.users,100,4)
        self.assertEqual(before,load_checkpoint(path,'genre')[0].retrieve_many(self.training.ctx.users,100,4))
        with self.assertRaises(ValueError):load_checkpoint(path,'itemcf')
        with self.assertRaises(ValueError):load_checkpoint(path,split_manifest_sha256='0'*64)
        altered=copy.deepcopy(c);altered['seed']=43
        with self.assertRaises(ValueError):load_checkpoint(path,expected_config=altered)
        original=(path/'manifest.json').read_text();m['framework_version']='incompatible'
        (path/'manifest.json').write_text(json.dumps(m));reseal(path)
        with self.assertRaises(ValueError):load_checkpoint(path)
        (path/'manifest.json').write_text(original);reseal(path)
        with np.load(path/'mappings.npz') as x:arrays={k:x[k].copy() for k in x.files}
        arrays['users']+=100;np.savez(path/'mappings.npz',**arrays);reseal(path)
        with self.assertRaises(ValueError):load_checkpoint(path)

    def test_cpu_neural_training_state_and_exported_vector_checkpoint_parity(self):
        for name in ['lightgcn','two_tower']:
            config=self.root/(name+'_state.yaml');config.write_text(yaml.safe_dump(configuration(name)))
            result=train(name,config,self.directory,self.root/'neural_artifacts','state')
            path=Path(result['checkpoint']);_,manifest=load_checkpoint(path,name)
            self.assertTrue(manifest['neural_training_state_export_parity'])
            self.assertTrue((path/'training_state.pt').is_file())

    def test_validation_candidates_metrics_same_population_and_empty_cohort_nulls(self):
        config=self.root/'eval_cfg.yaml';config.write_text(yaml.safe_dump(configuration('popularity')))
        checkpoint=Path(train('popularity',config,self.directory,self.root/'eval_artifacts','eval')['checkpoint'])
        result=evaluate('popularity',checkpoint,data_dir=self.directory,output_root=self.root/'candidates',run_id='eval')
        path=Path(result['evaluation']);manifest=checked_manifest(path)
        population=load_validation(self.directory,self.training).population
        frame=pd.read_parquet(path/'candidates.parquet')
        self.assertEqual(result['metrics']['queries'],len(population.users))
        self.assertNotEqual(manifest['evaluation_id'],manifest['measurements']['all_observed']['evaluation_id'])
        self.assertEqual(set(frame.userId),set(population.users))
        self.assertIsNone(next(g for g in manifest['measurements']['eligible']['cohorts'] if g['cohort']=='cold_0')['recall20'])
        per_query,summaries=summarize_candidates(frame,population,self.training.ctx)
        self.assertAlmostEqual(summaries[0]['recall100'],per_query.recall100.mean())
        empty=frame.iloc[:0]
        pq,summary=summarize_candidates(empty,population,self.training.ctx)
        self.assertEqual(summary[0]['recall100'],0);self.assertEqual(summary[0]['queries'],len(population.users))
        self.assertEqual(summary[0]['empty_candidate_queries'],len(population.users))
        with patch('recsys.retrieval_workflow.fit_retriever',side_effect=AssertionError('benchmark attempted fitting')):
            with self.assertRaises(ValueError):benchmark([path],output_root=self.root/'benchmarks',run_id='missing')
        with self.assertRaises(ValueError):evaluate('popularity',checkpoint,split='test')
        with self.assertRaises(FileExistsError):train('popularity',config,self.directory,self.root/'eval_artifacts','eval')

    def test_candidates_do_not_depend_on_heldout_labels(self):
        ctx=self.training.ctx;model=fit_retriever(ctx,'genre',configuration('genre'))
        users=ctx.users[:3];first=model.retrieve_many(users,100,4)
        unrelated_labels=pd.DataFrame(dict(userId=users,movieId=[10,11,12],rating=[0.5,5.,2.]))
        unrelated_labels['rating']=5-unrelated_labels.rating
        self.assertEqual(first,model.retrieve_many(users,100,4))

    def test_cli_explicitly_rejects_test_evaluation(self):
        result=subprocess.run([sys.executable,'-m','recsys.cli','evaluate','--model','popularity',
            '--split','test','--checkpoint','nonexistent'],capture_output=True,text=True)
        self.assertEqual(result.returncode,2);self.assertIn('invalid choice',result.stderr)


if __name__=='__main__':unittest.main(verbosity=2)
