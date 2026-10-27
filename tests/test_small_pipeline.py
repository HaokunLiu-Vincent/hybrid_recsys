"""Synthetic Phase 2 split/contract tests; never fit a model or use the real Test for selection."""
from pathlib import Path
import copy
import importlib
import json
import tempfile
import unittest

import duckdb
import pandas as pd
import yaml

from recsys.small_data import DataValidationError, publish_archive, sha256
from recsys.small_pipeline import (ROOT, SCHEMA, settings, raw_connection, assignment_integrity,
                                   prepare, validate_splits, verify_files)
from recsys.small_protocol import assign, evaluation_tables, cohort_summary
from recsys.small_benchmark import load_benchmark
from recsys.metrics import query_metrics
from test_small_data import write_fixture, archive_fixture


def fixture(root):
    root=Path(root);source=root/'source';write_fixture(source)
    archive_fixture(source,root/'download.zip')
    raw=root/'raw';manifest=publish_archive(root/'download.zip',raw,{})
    config=copy.deepcopy(settings(ROOT/'params.yaml'))
    repin(raw,config)
    return raw,config


def repin(raw,config):
    manifest=json.loads((raw/'provenance.json').read_text())
    manifest['files']['ratings.csv'].update(sha256=sha256(raw/'ratings.csv'),bytes=(raw/'ratings.csv').stat().st_size)
    (raw/'provenance.json').write_text(json.dumps(manifest))
    d=config['dataset']
    d.update(provenance_sha256=sha256(raw/'provenance.json'),ratings_sha256=sha256(raw/'ratings.csv'),
             movies_sha256=sha256(raw/'movies.csv'),expected_ratings=6,expected_users=3,expected_metadata_movies=4)


def table(con,rows):
    frame=pd.DataFrame(rows,columns=['userId','movieId','rating','timestamp'])
    frame.insert(0,'interactionId',range(len(frame)))
    con.register('input_frame',frame)
    con.execute('CREATE OR REPLACE TABLE ratings AS SELECT * FROM input_frame')


class SmallSplitTests(unittest.TestCase):
    def test_nearest_boundary_rounding_ties_and_short_histories(self):
        con=duckdb.connect()
        times=[1]*3+[2]*4+[3]*2+[4]+[5]*2
        rows=[(1,i+1,4.,t) for i,t in enumerate(times)]
        rows += [(2,1,4.,10),(2,2,2.,10),(3,1,4.,10),(3,2,4.,20)]
        table(con,rows);assign(con)
        self.assertEqual(con.sql("SELECT count(*) FROM assigned WHERE userId=1 AND partition='train'").fetchone()[0],7)
        self.assertEqual(con.sql("SELECT DISTINCT partition FROM assigned WHERE userId=2").fetchone()[0],'train')
        self.assertEqual(con.sql("SELECT partition FROM assigned WHERE userId=3 ORDER BY timestamp").fetchall(),[('train',),('test',)])
        assignment_integrity(con)
        first=con.sql('SELECT interactionId,partition FROM assigned ORDER BY interactionId').fetchall()
        con.execute('CREATE OR REPLACE TABLE reordered AS SELECT * FROM ratings ORDER BY interactionId DESC')
        assign(con,'reordered')
        self.assertEqual(first,con.sql('SELECT interactionId,partition FROM assigned ORDER BY interactionId').fetchall())
        con.close()

    def test_global_timestamp_groups_and_short_two_window_snapshot(self):
        con=duckdb.connect();table(con,[(1,1,4.,1),(2,1,4.,1),(1,2,4.,2),(2,2,4.,3),(1,3,4.,4)])
        assign(con,mode='global',ratios=(70,15,15));assignment_integrity(con,mode='global')
        assign(con,mode='per_user',ratios=(80,20));assignment_integrity(con,mode='per_user',inner=True)
        self.assertEqual(con.sql('SELECT count(*) FROM assigned').fetchone()[0],5)
        con.close()

    def test_invalid_ratios_and_identifiers_rejected(self):
        con=duckdb.connect()
        for ratios in [(70,10,10),(100,0),(70.0,15,15)]:
            with self.assertRaises(ValueError):assign(con,ratios=ratios)
        with self.assertRaises(ValueError):assign(con,source='ratings; DROP TABLE movies')
        con.close()

    def test_train_catalog_cold_users_and_frozen_history(self):
        con=duckdb.connect();table(con,[(1,1,2.,1),(1,2,4.,2),(2,1,5.,4),(3,3,4.,4)])
        con.execute("CREATE TABLE movies AS SELECT * FROM (VALUES (1),(2),(3),(4)) m(movieId)")
        con.execute("CREATE TABLE assigned AS SELECT *,CASE WHEN timestamp=1 THEN 'train' WHEN timestamp=2 THEN 'validation' ELSE 'test' END AS partition FROM ratings")
        evaluation_tables(con)
        before=con.sql('SELECT * FROM item_eligibility ORDER BY movieId').fetchall()
        self.assertEqual([r[0] for r in before if r[3]],[1])  # low Train rating still makes item eligible
        rows=con.sql("SELECT * FROM user_eval_cohorts WHERE partition='test' ORDER BY userId").df()
        self.assertEqual(int(rows.no_train_history.sum()),2)
        self.assertTrue(rows.loc[rows.userId.eq(2),'metric_eligible'].item())
        self.assertFalse(rows.loc[rows.userId.eq(3),'metric_eligible'].item())
        self.assertEqual(rows.loc[rows.userId.eq(2),'cohort'].item(),'cold_0')
        con.execute("UPDATE assigned SET rating=0.5 WHERE partition<>'train'")
        evaluation_tables(con)
        self.assertEqual(before,con.sql('SELECT * FROM item_eligibility ORDER BY movieId').fetchall())
        self.assertEqual(sum(r['eligible_positive_users'] for r in cohort_summary(con)),0)
        con.close()

    def test_no_candidate_or_no_relevance_rows_are_retained(self):
        con=duckdb.connect();table(con,[(1,1,2.,1),(1,2,4.,2)])
        con.execute('CREATE TABLE movies AS SELECT 1 AS movieId UNION ALL SELECT 2')
        con.execute("CREATE TABLE assigned AS SELECT *,CASE WHEN timestamp=1 THEN 'train' ELSE 'test' END AS partition FROM ratings")
        evaluation_tables(con)
        row=con.sql("SELECT * FROM user_eval_cohorts WHERE partition='test'").df().iloc[0]
        self.assertTrue(row.no_candidates);self.assertTrue(row.no_retrievable_targets)
        self.assertFalse(row.metric_eligible)
        self.assertEqual(len(con.sql('SELECT * FROM user_eval_cohorts').df()),2)
        con.close()

    def test_exact_coverage_detects_dropped_or_changed_rows(self):
        con=duckdb.connect();table(con,[(1,1,4.,1),(1,2,4.,2),(1,3,4.,3)])
        assign(con);con.execute('UPDATE assigned SET rating=1 WHERE interactionId=0')
        with self.assertRaises(DataValidationError):assignment_integrity(con)
        assign(con);con.execute('DELETE FROM assigned WHERE interactionId=0')
        with self.assertRaises(DataValidationError):assignment_integrity(con)
        con.close()

    def test_raw_pins_schemas_and_sql_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw,config=fixture(tmp)
            con,report=raw_connection(raw,config['dataset']);con.close()
            self.assertEqual(report['observed']['ratings'],6)
            bad=copy.deepcopy(config);bad['dataset']['ratings_sha256']='0'*64
            with self.assertRaises(DataValidationError):raw_connection(raw,bad['dataset'])
            frame=pd.read_csv(raw/'ratings.csv');frame.loc[0,'rating']=3.7;frame.to_csv(raw/'ratings.csv',index=False)
            repin(raw,config)
            with self.assertRaises(DataValidationError):raw_connection(raw,config['dataset'])

    def test_repeated_build_bytes_cohorts_inner_windows_and_compatibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw,config=fixture(root)
            raw_hashes={str(p.relative_to(raw)):sha256(p) for p in raw.rglob('*') if p.is_file()}
            one,_=prepare(raw,root/'one',config)
            two,_=prepare(raw,root/'two',config)
            self.assertEqual(one,two)
            for version in one:
                directory=root/'one'/version
                validated=validate_splits(raw,directory,config)
                self.assertTrue(validated['recomputed_from_raw'])
                self.assertEqual(one[version]['files']['train.parquet']['schema'],SCHEMA)
                self.assertEqual({p.name:sha256(p) for p in directory.iterdir()},
                                 {p.name:sha256(p) for p in (root/'two'/version).iterdir()})
            # Reuse is read-only, even after an interrupted/partial directory: never replace.
            self.assertEqual(prepare(raw,root/'one',config)[0],one)
            b=load_benchmark(root/'one/small_v1',raw,phase='test')
            alternative=load_benchmark(root/'one/small_v1',raw,phase='test',denominator='all_observed')
            self.assertNotEqual(b.population.identifier,alternative.population.identifier)
            self.assertEqual(set(b.ctx.items[b.ctx.eligible]),{10,11})
            self.assertTrue(set(alternative.population.truth)>=set(b.population.truth))
            for user in b.population.users:
                seen=set(b.ctx.items[b.ctx.seen(b.ctx.user_index(user))])
                self.assertFalse(seen&b.population.truth[int(user)])
            self.assertEqual(raw_hashes,{str(p.relative_to(raw)):sha256(p) for p in raw.rglob('*') if p.is_file()})
            path=root/'one/small_v1/train.parquet';path.write_bytes(path.read_bytes()+b'tamper')
            with self.assertRaises(DataValidationError):verify_files(path.parent)

    def test_same_user_external_and_ranker_label_ids_never_in_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw,config=fixture(root);prepare(raw,root/'splits',config)
            directory=root/'splits/small_v1'
            train=pd.read_parquet(directory/'train.parquet');val=pd.read_parquet(directory/'validation.parquet');test=pd.read_parquet(directory/'test.parquet')
            self.assertFalse(set(train.interactionId)&set(val.interactionId))
            self.assertFalse(set(train.interactionId)&set(test.interactionId))
            history=pd.read_parquet(directory/'ranker_snapshot_train.parquet');labels=pd.read_parquet(directory/'ranker_label_window.parquet')
            self.assertFalse(set(history.interactionId)&set(labels.interactionId))
            self.assertEqual(set(history.interactionId)|set(labels.interactionId),set(train.interactionId))
            for user,group in labels.groupby('userId'):
                prior=history[history.userId.eq(user)]
                self.assertTrue(prior.empty or prior.timestamp.max()<group.timestamp.min())

    def test_metric_denominators_and_undefined_empty_relevance(self):
        self.assertEqual(query_metrics([1],{1})['recall20'],1)
        self.assertEqual(query_metrics([1],{1,99})['recall20'],.5)
        self.assertEqual(query_metrics([],{1})['hit20'],0)
        with self.assertRaises(ValueError):query_metrics([],set())

    def test_all_retained_retrievers_fusion_ranker_import_without_fitting(self):
        modules=['popularity','genre','itemcf','implicit_als','lightgcn','two_tower']
        for name in modules:importlib.import_module('recsys.retrievers.'+name)
        for name in ['fusion.rrf','fusion.candidates','rankers.features','rankers.dataset','rankers.lightgbm_ranker']:
            importlib.import_module('recsys.'+name)

    def test_dvc_dependencies_and_disjoint_output_ownership(self):
        pipeline=yaml.safe_load((ROOT/'dvc.yaml').read_text())
        stages=pipeline['stages'];self.assertEqual(set(stages),{'validate_raw','prepare_splits','validate_splits'})
        paths=[]
        for stage in stages.values():
            self.assertIn('params.yaml',str(stage.get('params',[])))
            for output in stage.get('outs',[]):paths.append(output if isinstance(output,str) else next(iter(output)))
        self.assertEqual(len(paths),len(set(paths)))
        self.assertFalse(any(a!=b and Path(a) in Path(b).parents for a in paths for b in paths))
        self.assertIn('data/raw/ml-latest-small',stages['validate_raw']['deps'])
        self.assertIn('data/validation/small_raw_v1.json',stages['prepare_splits']['deps'])
        self.assertIn('data/splits/small_v1',stages['validate_splits']['deps'])
        self.assertIn('uv.lock',stages['prepare_splits']['deps'])


if __name__=='__main__':unittest.main(verbosity=2)
