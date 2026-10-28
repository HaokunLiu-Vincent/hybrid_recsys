"""Historical isolation, versioned features, query sampling and native LambdaMART integration."""
from pathlib import Path
import copy
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import yaml

from test_retrieval_training import frozen_fixture,configuration
from test_small_pipeline import fixture as raw_fixture
from recsys.runtime import Store,digest,fingerprint
from recsys.artifacts import dump_json
from recsys.contracts import CHANNELS,mapping_id,catalog_id
from recsys.small_benchmark import load_training,load_validation
from recsys.historical_retrieval import (load_historical_training,train_historical,load_historical_checkpoint,
                                        historical_candidates,fuse_historical)
from recsys.fusion.small_rrf import union_from_frames,evaluate_rankings
from recsys.rankers.small_schema import feature_schema,validate_schema,build_features,model_matrix,query_keys,group_sizes
from recsys.rankers.small_lambdamart import (sample_queries,prepare_ranking_features,train_lambdamart,
                                           load_ranker,evaluate_lambdamart,score_ndcg)
from recsys.retrieval_artifacts import load_checkpoint,checked_manifest
from recsys.retrieval_workflow import ROOT


class HistoricalRankingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        cls.data=frozen_fixture(cls.root);cls.hist=load_historical_training(cls.data)
        cls.full=load_training(cls.data);cls.val=load_validation(cls.data,cls.full)
        cls.candidates=[];cls.models={}
        original=pd.read_parquet
        def guarded(path,*args,**kwargs):
            path=Path(path)
            if path.is_relative_to(cls.data) and path.name not in ['movies.parquet','ranker_snapshot_train.parquet']:
                raise AssertionError('Historical fitting/candidates read a later/full window: '+str(path))
            return original(path,*args,**kwargs)
        for source in CHANNELS:
            cfg=cls.root/(source+'.yaml');cfg.write_text(yaml.safe_dump(configuration(source)))
            with patch('pandas.read_parquet',side_effect=guarded):
                model=train_historical(source,cfg,cls.data,cls.root/'historical','toy')
                out=historical_candidates(model['checkpoint'],cls.data,cls.root/'historical_candidates','toy')
            cls.models[source]=Path(model['checkpoint']);cls.candidates.append(Path(out['candidates']))
        hf=fuse_historical(cls.candidates,cls.data,cls.root/'fusion','historical')
        cls.hfusion=Path(hf['fusion'])
        frames={}
        for j,source in enumerate(CHANNELS):
            rows=[]
            for user in cls.val.population.users:
                u=cls.full.ctx.user_index(user);ids=cls.full.ctx.items[cls.full.ctx.eligible]
                ids=ids[~np.isin(ids,cls.full.ctx.items[cls.full.ctx.seen(u)])]
                ids=np.roll(ids,j)
                rows.extend(dict(userId=int(user),movieId=int(item),score=float(100-r),rank=r+1,source=source,fallback=False)
                            for r,item in enumerate(ids[:100]))
            frames[source]=pd.DataFrame(rows)
        pool=union_from_frames(frames,cls.full.ctx,cls.val.population.users)
        m=dict(kind='validation_fusion',status='completed',evaluation_id=cls.val.population.identifier,
            catalog_id=catalog_id(cls.full.ctx),mapping_id=mapping_id(cls.full.ctx),
            training_cutoff='per_user:'+cls.full.manifest['files']['user_cutoffs.parquet']['sha256'],
            rrf_constant=60,individual_top20={s:evaluate_rankings(f,cls.val.population,cls.full.ctx,'score')[1][0] for s,f in frames.items()})
        def writer(path):
            pool.to_parquet(path/'union.parquet',index=False);m['union_sha256']=digest(path/'union.parquet')
            dump_json(path/'manifest.json',m)
        cls.vfusion=Store(cls.root,'fusion').write('validation',writer)
        cls.cfg=yaml.safe_load((ROOT/'configs/lambdamart.yaml').read_text())
        cls.cfg.update(boosting_rounds=6,early_stopping_rounds=2)
        cls.cfg['sampling'].update(unknown_per_query=3,minimum_effective_queries=2)
        cls.cfg['lightgbm'].update(num_leaves=3,max_depth=2,min_data_in_leaf=2)
        cls.config=cls.root/'ranker.yaml';cls.config.write_text(yaml.safe_dump(cls.cfg))
        result=prepare_ranking_features(cls.hfusion,cls.vfusion,cls.config,cls.data,cls.root/'features',cls.root/'ranking_data','toy')
        cls.features=Path(result['features']);cls.training_data=Path(result['training_data'])

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_six_historical_models_are_independent_and_snapshot_catalog_only(self):
        self.assertLess(self.hist.ctx.train_rows,self.full.ctx.train_rows)
        self.assertFalse(np.array_equal(self.hist.ctx.popularity,self.full.ctx.popularity))
        for source,path in self.models.items():
            model,m=load_historical_checkpoint(path,source,self.hist)
            self.assertEqual(m['train_rows'],len(self.hist.history));self.assertTrue(m['reload_prediction_parity'])
            self.assertFalse(m['full_train_factors_or_context_reused']);self.assertFalse(m['later_label_access_during_fit'])
            with self.assertRaises(ValueError):load_checkpoint(path,source)
        # Explicitly reject an external-kind manifest, independent of source parity.
        m=checked_manifest(self.models['popularity']);self.assertEqual(m['training_kind'],'historical_snapshot_per_user')

    def test_historical_candidates_precede_labels_and_use_all_snapshot_queries(self):
        for path in self.candidates:
            m=checked_manifest(path);self.assertFalse(m['later_labels_read'])
            self.assertEqual(m['queries'],len(self.hist.ctx.users))
            f=pd.read_parquet(path/'candidates.parquet')
            self.assertTrue(f.movieId.isin(self.hist.ctx.items[self.hist.ctx.eligible]).all())

    def test_historical_loading_and_fusion_reject_external_or_wrong_cutoffs(self):
        original=checked_manifest(self.models['popularity'])
        wrong=dict(original,training_kind='external_train_per_user')
        def model_writer(path):dump_json(path/'manifest.json',wrong)
        external=Store(self.root,'invalid_models').write('external',model_writer)
        with self.assertRaisesRegex(ValueError,'rejects Full-Train'):
            load_historical_checkpoint(external)
        candidate=checked_manifest(self.candidates[0])
        for key in ['training_kind','training_cutoff','mapping_id','catalog_id','evaluation_id']:
            invalid=dict(candidate);invalid[key]='incompatible_fixture'
            def candidate_writer(path):dump_json(path/'manifest.json',invalid)
            path=Store(self.root,'invalid_candidates').write(key,candidate_writer)
            with self.assertRaisesRegex(ValueError,'incompatible/Full-Train'):
                fuse_historical([path,*self.candidates[1:]],self.data,self.root/'invalid_fusion',key)

    def test_feature_allowlist_rrf_schema_order_dtypes_and_label_invariance(self):
        schema=json.loads((self.features/'feature_schema.json').read_text())
        self.assertIn('rrf_score',schema['names']);self.assertNotIn('label',schema['names'])
        self.assertNotIn('feedback_kind',schema['names']);self.assertNotIn('userId',schema['names'])
        frame=pd.read_parquet(self.features/'historical_features.parquet')
        before=model_matrix(frame,schema).copy();frame['label']=np.arange(len(frame))%2;frame['feedback_kind']='fabricated_test_only'
        pd.testing.assert_frame_equal(before,model_matrix(frame,schema))
        self.assertTrue(all(str(t)=='float32' for t in before.dtypes))
        bad=copy.deepcopy(schema);bad['names'].append('label');bad['sha256']=fingerprint({k:v for k,v in bad.items() if k!='sha256'})
        with self.assertRaises(ValueError):validate_schema(bad)
        bad=copy.deepcopy(schema);bad['names']=list(reversed(bad['names']))
        with self.assertRaises(ValueError):model_matrix(frame,bad)

    def test_missing_source_features_keep_nan_and_presence(self):
        pool=pd.read_parquet(self.hfusion/'union.parquet').head(5).copy()
        pool['genre_present']=0;pool['genre_score']=np.nan;pool['genre_rank']=101;pool['genre_fallback']=0
        pool['source_count']-=1
        f,s=build_features(pool,self.hist.ctx,'synthetic_missing')
        self.assertTrue(f.genre_score.isna().all());self.assertTrue(f.genre_present.eq(0).all())
        f.loc[f.index[0],'genre_score']=0
        with self.assertRaises(ValueError):model_matrix(f,s)

    def test_composite_query_groups_do_not_merge_same_user_across_snapshots(self):
        a=query_keys(pd.DataFrame(dict(userId=[1,1],movieId=[10,11],label=[1,0])),'a')
        b=query_keys(pd.DataFrame(dict(userId=[1,1],movieId=[10,12],label=[1,0])),'b')
        frame=pd.concat([a,b],ignore_index=True)
        self.assertEqual(group_sizes(frame,True).tolist(),[2,2])
        with self.assertRaises(ValueError):group_sizes(frame.iloc[[0,2,1,3]])

    def test_sampling_is_deterministic_retains_positive_low_and_adds_nothing(self):
        rows=pd.DataFrame(dict(userId=[1]*12,movieId=np.arange(12),label=[1,0]+[0]*10,
                              feedback_kind=['observed_positive','observed_low']+['unobserved']*10))
        rows=query_keys(rows,'sampling')
        a,stats=sample_queries(rows,42,3);b,_=sample_queries(rows.sample(frac=1,random_state=9),42,3)
        pd.testing.assert_frame_equal(a,b)
        self.assertEqual(stats['positive_candidate_labels'],1);self.assertEqual(stats['observed_low_candidates'],1)
        self.assertEqual(stats['sampled_unlabeled_candidates'],3)
        self.assertTrue(set(a.movieId)<=set(rows.movieId))
        only_zero=rows.assign(label=0,feedback_kind='unobserved')
        selected,z=sample_queries(only_zero,42,3);self.assertTrue(selected.empty);self.assertEqual(z['queries_no_retrieved_positive'],1)
        only_pos=rows.assign(label=1,feedback_kind='observed_positive')
        selected,p=sample_queries(only_pos,42,3);self.assertTrue(selected.empty);self.assertEqual(p['queries_no_contrast'],1)

    def test_actual_gate_b_training_groups_and_window_disjointness(self):
        m=checked_manifest(self.training_data);s=m['statistics']
        self.assertTrue(m['gate_b_feasible']);self.assertGreater(s['effective_training_queries'],1)
        frame=pd.read_parquet(self.training_data/'training_rows.parquet')
        groups=group_sizes(frame,True);self.assertEqual(int(groups.sum()),len(frame))
        features=pd.read_parquet(self.features/'historical_features.parquet')
        self.assertTrue(set(zip(frame.queryId,frame.movieId))<=set(zip(features.queryId,features.movieId)))

    def test_custom_metric_uses_full_truth_and_retains_zero_pool_hit_user(self):
        frame=pd.DataFrame(dict(userId=[1,1,2,2],movieId=[10,11,20,21]))
        value=score_ndcg(np.array([2.,1.,2.,1.]),frame,{1:{10,99},2:{88}})
        self.assertAlmostEqual(value,.5/(1+1/np.log2(3)))

    def test_native_lambdamart_fit_reload_predictions_and_schema_rejection(self):
        result=train_lambdamart(self.training_data,self.config,self.data,self.root/'rankers','toy')
        path=Path(result['ranker']);model,schema,m=load_ranker(path)
        self.assertEqual(m['training_groups'],checked_manifest(self.training_data)['statistics']['effective_training_queries'])
        self.assertFalse(m['validation_labels_used_for_gradients']);self.assertTrue(m['save_reload_prediction_parity'])
        self.assertIn('rrf_score',model.feature_name())
        self.assertTrue(m['comparison']['same_candidate_pair_set'])
        alternate=copy.deepcopy(schema);alternate['version']='wrong'
        with self.assertRaises(ValueError):load_ranker(path,alternate)
        parity=evaluate_lambdamart(path,self.training_data,self.root/'ranking_eval','toy')
        self.assertTrue(parity['exact_parity'])


class FrozenDVCContractTest(unittest.TestCase):
    def test_global_lock_dependency_change_cannot_safely_relabel_frozen_data(self):
        from recsys.small_pipeline import prepare
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);raw,cfg=raw_fixture(root);project=root/'project';project.mkdir()
            (project/'uv.lock').write_text('data environment v1')
            with patch('recsys.small_pipeline.ROOT',project):
                prepare(raw,root/'splits',cfg)
                frozen={str(p):digest(p) for p in (root/'splits').rglob('*') if p.is_file()}
                (project/'uv.lock').write_text('data environment v1 plus unrelated training dependency')
                with self.assertRaises(FileExistsError):prepare(raw,root/'splits',cfg)
                self.assertEqual(frozen,{str(p):digest(p) for p in (root/'splits').rglob('*') if p.is_file()})


if __name__=='__main__':unittest.main(verbosity=2)
