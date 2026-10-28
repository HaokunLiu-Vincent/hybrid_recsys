"""Synthetic final-assessment regression tests; no real held-out targets or fitting."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import unittest
import numpy as np
import pandas as pd
from recsys.data import build_context
from recsys.contracts import CHANNELS
from recsys_evaluation.core import (population, evaluate, paired_bootstrap, rankings,
    reject_labels, generalization, mlflow_metrics)
from recsys_evaluation.workflow import (CONTRACT, LabelReadGuard, complete, checked,
    frozen, write_json, sha, log_final)


def fixture():
    movies = pd.DataFrame(dict(movieId=[1, 2, 3, 4, 5], title=['a']*5, genres=['Drama']*5))
    train = pd.DataFrame(dict(userId=[1, 1, 2, 2, 3], movieId=[1, 2, 3, 4, 1],
                              rating=[5., 2., 5., 2., 2.], timestamp=[1]*5))
    ctx = build_context(train, movies)
    future = pd.DataFrame(dict(userId=[1, 1, 1, 2, 3, 3], movieId=[3, 5, 2, 1, 4, 2],
                               rating=[5., 5., 5., 5., 5., 2.], timestamp=[2]*6))
    pool = pd.DataFrame(dict(userId=[1, 1, 2, 3], movieId=[4, 3, 2, 2],
        implicit_als_present=[1, 1, 1, 1], implicit_als_rank=[1, 2, 1, 1],
        rrf_score=[.02, .03, .01, .01], ranking_score=[.2, .3, .1, .1]))
    return ctx, future, pool


class FinalEvaluationTests(unittest.TestCase):
    def test_policy_b_population_and_seen_low_exclusion(self):
        ctx, future, _ = fixture()
        p = population(future, ctx, 'hash', 'test')
        self.assertEqual(p.truth, {1:{3}, 2:{1}, 3:{4}})
        self.assertEqual(p.report['policy_B_catalog_excluded'], 1)
        self.assertEqual(p.report['train_seen_positive_excluded'], 1)

    def test_all_observed_keeps_impossible_targets_and_distinct_identity(self):
        ctx, future, _ = fixture()
        a = population(future, ctx, 'hash', 'test')
        b = population(future, ctx, 'hash', 'test', 'all_observed')
        self.assertEqual(b.truth[1], {2, 3, 5})
        self.assertNotEqual(a.identifier, b.identifier)

    def test_known_answer_zero_hits_and_full_population(self):
        ctx, future, pool = fixture()
        result, pq = evaluate(pool, population(future, ctx, 'h', 'test'), ctx)
        self.assertEqual(result['methods']['rrf'][0]['recall20'], 1/3)
        self.assertEqual(result['methods']['rrf'][0]['ndcg20'], 1/3)
        self.assertEqual(result['methods']['als'][0]['ndcg20'], (1/np.log2(3))/3)
        self.assertEqual(result['union']['zero_union_hit_queries'], 2)
        self.assertEqual(len(pq), 9)

    def test_empty_cohort_is_undefined(self):
        ctx, future, pool = fixture()
        result, _ = evaluate(pool, population(future, ctx, 'h', 'test'), ctx)
        empty = next(g for g in result['methods']['rrf'] if g['cohort']=='warm_20_plus')
        self.assertEqual(empty['queries'], 0)
        self.assertIsNone(empty['recall20'])
        self.assertIsNone(empty['hit100'])
        self.assertIsNotNone(empty['undefined_reason'])

    def test_empty_predictions_not_dropped(self):
        ctx, future, pool = fixture()
        result, pq = evaluate(pool.iloc[:0], population(future, ctx, 'h', 'test'), ctx)
        self.assertEqual(result['methods']['als'][0]['recall20'], 0)
        self.assertEqual(result['methods']['als'][0]['empty_candidate_queries'], 3)

    def test_union_ceiling_differs_from_top_k(self):
        ctx, future, _ = fixture()
        pool = pd.DataFrame(dict(userId=[1]*101, movieId=list(range(1000, 1100))+[3],
            implicit_als_present=[0]*101, implicit_als_rank=[101]*101,
            rrf_score=list(range(101, 0, -1)), ranking_score=list(range(101, 0, -1))))
        pop = population(future[future.userId.eq(1)], ctx, 'h', 'test')
        result, _ = evaluate(pool, pop, ctx)
        self.assertEqual(result['union']['macro_recall_ceiling'], 1)
        self.assertEqual(result['methods']['rrf'][0]['recall100'], 0)

    def test_als_order_and_deterministic_tail(self):
        _, _, pool = fixture()
        pool.loc[0, 'implicit_als_present'] = 0
        pool.loc[0, 'implicit_als_rank'] = 101
        self.assertEqual(rankings(pool)['als'].movieId.tolist()[:2], [3, 4])
        pool['ranking_score'] = 1.
        self.assertEqual(rankings(pool)['lambdamart'].movieId.tolist()[:2], [3, 4])

    def test_all_methods_same_union(self):
        _, _, pool = fixture()
        sets = [set(zip(f.userId, f.movieId)) for f in rankings(pool).values()]
        self.assertTrue(all(s == sets[0] for s in sets))

    def test_frozen_history_ignores_validation(self):
        ctx, future, _ = fixture()
        original = ctx.observed.copy()
        population(future, ctx, 'h', 'test')
        self.assertEqual((ctx.observed != original).nnz, 0)

    def test_inference_rejects_label_fields(self):
        _, _, pool = fixture()
        for name in ['label', 'rating', 'cohort', 'future_positives', 'timestamp']:
            with self.assertRaises(ValueError):
                reject_labels(pool.assign(**{name: 1}))

    def test_bootstrap_pairing_reproducibility_and_identity_difference(self):
        ctx, future, pool = fixture()
        _, pq = evaluate(pool, population(future, ctx, 'h', 'test'), ctx)
        a = paired_bootstrap(pq, 100)
        self.assertEqual(a, paired_bootstrap(pq.sample(frac=1, random_state=2), 100))
        self.assertEqual(a['differences']['lambdamart_minus_rrf/recall20']['ci95'], [0., 0.])
        with self.assertRaises(ValueError):
            paired_bootstrap(pq.iloc[1:], 100)

    def test_generalization_and_namespace(self):
        ctx, future, pool = fixture()
        r, pq = evaluate(pool, population(future, ctx, 'h', 'test'), ctx)
        gaps = generalization(r, r)
        self.assertEqual(gaps['als']['all']['metrics']['recall20']['absolute'], 0)
        metrics = mlflow_metrics(dict(primary=r, all_observed=r, bootstrap=paired_bootstrap(pq, 20), generalization=gaps))
        self.assertIn('test/policy_b/als/recall_at_20', metrics)
        self.assertIn('test/all_observed/lambdamart/ndcg_at_20', metrics)
        self.assertIn('test/union/macro_recall_ceiling', metrics)
        self.assertFalse(any(k.startswith('validation/policy_b') for k in metrics))

    def test_guard_denies_exact_target_paths(self):
        with TemporaryDirectory() as tmp:
            guard = LabelReadGuard(Path(tmp))
            for filename in ['test.parquet', 'evaluation_targets.parquet', 'user_eval_cohorts.parquet']:
                with self.assertRaises(PermissionError):
                    guard('open', (str(Path(tmp)/'data/splits/small_v1'/filename), 'r', 0))
            guard('open', (str(Path(tmp)/'data/splits/small_v1/train.parquet'), 'r', 0))
            self.assertEqual(len(guard.reads), 1)

    def test_process_audit_hook_actually_blocks_payload_open(self):
        import subprocess, sys
        with TemporaryDirectory() as tmp:
            root=Path(tmp);target=root/'data/splits/small_v1/test.parquet'
            target.parent.mkdir(parents=True);target.write_bytes(b'secret synthetic targets')
            code = ('import sys; from pathlib import Path; '
                    'from recsys_evaluation.workflow import LabelReadGuard; '
                    'sys.addaudithook(LabelReadGuard(Path(sys.argv[1]))); '
                    'Path(sys.argv[2]).read_bytes()')
            result=subprocess.run([sys.executable, '-B', '-c', code, str(root), str(target)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Pre-Test gate denied', result.stderr)

    def test_seal_corruption_and_extra_files_rejected(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            write_json(path/'report.json', dict(passed=True))
            complete(path)
            self.assertTrue(checked(path)['passed'])
            (path/'extra').write_text('unexpected')
            with self.assertRaises(ValueError):
                checked(path)

    def test_freeze_rejects_model_protocol_source_mismatch(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); path=root/'preflight';path.mkdir()
            for change in ['model_uri', 'split_manifest_sha256', 'release_id']:
                bad=dict(CONTRACT);bad[change]='wrong'
                with patch('recsys_evaluation.workflow.checked', return_value=dict(contract=bad)):
                    with self.assertRaises(ValueError):
                        frozen(root, root)

    def test_completed_outputs_cannot_overwrite(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'report.json';write_json(path, {'first':1})
            with self.assertRaises(FileExistsError):
                write_json(path, {'second':2})
            self.assertEqual(json.loads(path.read_text()), {'first':1})

    def test_no_training_or_registry_mutation_api_in_evaluator(self):
        import ast
        import recsys_evaluation.workflow as workflow
        tree=ast.parse(Path(workflow.__file__).read_text())
        calls={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
        self.assertFalse(calls & {'fit','train','set_registered_model_alias','create_model_version',
                                 'set_model_version_tag','delete_run','update_model_version'})

    def test_real_local_mlflow_reporting_and_read_only_reuse(self):
        from mlflow import MlflowClient
        ctx, future, pool = fixture()
        report, pq = evaluate(pool, population(future, ctx, 'h', 'test'), ctx)
        result = dict(primary=report, all_observed=report, bootstrap=paired_bootstrap(pq, 20),
                      generalization=generalization(report, report))
        with TemporaryDirectory() as tmp:
            root = Path(tmp); output=root/'outputs';output.mkdir()
            for name in ['preflight', 'sealed_predictions', 'final_evaluation']:
                (output/name).mkdir()
                write_json(output/name/'COMPLETE.json', dict(files={}))
            write_json(output/'preflight/registry.json', dict(version='1', aliases={'candidate':'1'}))
            write_json(output/'preflight/historical_mlflow_records.json', {})
            write_json(output/'sealed_predictions/prediction_manifest.json', dict(synthetic=True))
            write_json(output/'final_evaluation/results.json', result)
            uri='sqlite:///'+str(root/'mlflow.db')
            c=MlflowClient(tracking_uri=uri, registry_uri=uri)
            experiment=c.create_experiment('MovieLens-Small-Final-Evaluation', artifact_location=str(root/'store'))
            with patch('recsys_evaluation.workflow.frozen'), \
                 patch('recsys_evaluation.workflow.checked', return_value={'current_head':'synthetic'}), \
                 patch('recsys_evaluation.workflow.sources', return_value={}), \
                 patch('recsys_evaluation.workflow.registry_snapshot', return_value=dict(version='1', aliases={'candidate':'1'})), \
                 patch('recsys_evaluation.workflow.runs_snapshot', return_value={}):
                first=log_final(root, output, uri)
                second=log_final(root, output, uri)
            self.assertTrue(first['created']);self.assertFalse(second['created'])
            self.assertEqual(first['run_id'], second['run_id'])
            self.assertEqual(len(c.search_runs([experiment])), 1)
            run=c.get_run(first['run_id'])
            self.assertEqual(run.data.metrics, mlflow_metrics(result))
            self.assertEqual(run.data.tags['model_version'], '1')
            self.assertEqual(run.data.tags['evaluation_type'], 'final_heldout_test')


if __name__ == '__main__':
    unittest.main()
