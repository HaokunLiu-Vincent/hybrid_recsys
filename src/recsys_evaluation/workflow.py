"""Four sequential final-evaluation gates. No fitting, alias mutation or split writing."""
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import numpy as np
import pandas as pd
from .core import (METHODS, COHORTS, METRICS, reject_labels, rankings, population,
                   evaluate, paired_bootstrap, generalization, mlflow_metrics)

SPLIT_SHA = '905f1370e919f06da3be728f95010d706b078c05f5c5e455c28da398ee41e799'
RELEASE_ID = 'e4b38328a35c992114eb3e6acda90803f0aa16c61a027686d9e86b06bf5e475d'
SCHEMA_ID = '21b298dc0cef61b0de9514c0db66f4e38cd9a89818bb8fe94694d0251ad48ae0'
MODEL_URI = 'models:/MovieLensHybridRecommender/1'
EXPERIMENT = 'MovieLens-Small-Final-Evaluation'
CONTRACT = dict(model_uri=MODEL_URI, release_id=RELEASE_ID, feature_schema_sha256=SCHEMA_ID,
    split_manifest_sha256=SPLIT_SHA, protocol='per_user_70_15_15', split='small_v1',
    methods=list(METHODS), source_budget=100, rrf_constant=60, history='frozen_train',
    catalog='train_observed', seen='all_train_rated', positive_threshold=4.0,
    primary_denominator='eligible', secondary_denominator='all_observed',
    metrics=list(METRICS)+['catalog_coverage20', 'catalog_coverage100'],
    primary_comparison=['recall20', 'ndcg20'], selection='original Validation NDCG@20; 31 trees',
    bootstrap=dict(seed=42, replicates=2000, unit='paired eligible user', interval='95% percentile'),
    ordering='userId ascending, score descending, movieId ascending; ALS original source ranks then movie-ID tail',
    cohorts=list(COHORTS), no_test_tuning=True)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def identity(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def write_json(path, obj):
    with Path(path).open('x') as f:
        f.write(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False, default=str)+'\n')


def complete(path):
    from recsys_serving.model_loader import check_files
    files = {str(p.relative_to(path)): dict(bytes=p.stat().st_size, sha256=sha(p))
             for p in sorted(path.rglob('*')) if p.is_file()}
    write_json(path/'COMPLETE.json', dict(files=files))
    check_files(path, 'COMPLETE.json')


def checked(path):
    from recsys_serving.model_loader import check_files
    check_files(path, 'COMPLETE.json')
    return json.loads((path/'report.json').read_text())


def sources(root):
    paths = list((root/'src/recsys_evaluation').glob('*.py'))
    paths += [root/'tools/final_evaluation.py', root/'tests/test_final_evaluation.py']
    return {str(p.relative_to(root)): sha(p) for p in sorted(paths)}


class LabelReadGuard:
    """Process audit hook: deny Test, mixed target/cohort tables and full raw ratings."""
    def __init__(self, root):
        self.root = root.resolve()
        self.reads = set()

    def __call__(self, event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        try:
            relative = path.relative_to(self.root)
        except ValueError:
            return
        if relative.parts and relative.parts[0] == 'data':
            if (relative.name in {'test.parquet', 'evaluation_targets.parquet', 'user_eval_cohorts.parquet',
                                  'ranker_label_window.parquet', 'ranker_query_diagnostics.parquet'}
                    or 'raw' in relative.parts or 'small_global_v1' in relative.parts):
                raise PermissionError('Pre-Test gate denied label-side data read: '+str(relative))
            self.reads.add(str(relative))


def registry_snapshot(uri):
    from mlflow import MlflowClient
    c = MlflowClient(tracking_uri=uri, registry_uri=uri)
    version = c.get_model_version('MovieLensHybridRecommender', '1')
    registered = c.get_registered_model('MovieLensHybridRecommender')
    return dict(name=version.name, version=version.version, source=version.source,
                run_id=version.run_id, tags=dict(version.tags), aliases=dict(registered.aliases),
                versions=sorted(str(v.version) for v in c.search_model_versions("name='MovieLensHybridRecommender'")))


def runs_snapshot(uri):
    from mlflow import MlflowClient
    c = MlflowClient(tracking_uri=uri)
    result = {}
    for e in c.search_experiments():
        for run in c.search_runs([e.experiment_id], max_results=1000):
            result[run.info.run_id] = dict(experiment_id=e.experiment_id, status=run.info.status,
                params=dict(run.data.params), metrics=dict(run.data.metrics), tags=dict(run.data.tags))
    return result


def load_release(root):
    from recsys_serving.model_loader import load_model, verify_export
    from recsys_serving.settings import Settings
    path = root/'outputs/deployment_models/movielens_hybrid_v1'
    manifest, _ = verify_export(path)
    if manifest['release_id'] != RELEASE_ID or manifest['split_manifest_sha256'] != SPLIT_SHA:
        raise ValueError('Wrong frozen release/protocol')
    loaded = load_model(Settings(model_source='local', model_path=path))
    rec = loaded.model.unwrap_python_model().recommender
    if rec.booster.num_trees() != 31 or rec.schema['sha256'] != SCHEMA_ID:
        raise ValueError('Wrong fixed ranker/schema')
    return loaded.model, rec, manifest


def split_manifest(root):
    path = root/'data/splits/small_v1/split_manifest.json'
    if sha(path) != SPLIT_SHA:
        raise ValueError('Frozen split manifest mismatch')
    m = json.loads(path.read_text())
    if m['version'] != 'small_v1' or m['protocol_name'] != 'per_user_chronological_70_15_15':
        raise ValueError('Protocol mismatch')
    return m


def verified_frame(root, manifest, filename):
    path = root/'data/splits/small_v1'/filename
    record = manifest['files'][filename]
    # One payload read; hash and Parquet reader consume the identical byte buffer.
    payload = path.read_bytes()
    if path.is_symlink() or len(payload) != record['bytes'] or hashlib.sha256(payload).hexdigest() != record['sha256']:
        raise ValueError('Frozen data checksum mismatch: '+filename)
    return pd.read_parquet(BytesIO(payload))


def extract_sources(pool):
    from recsys.contracts import CHANNELS
    result = {}
    for source in CHANNELS:
        columns = ['userId', 'movieId']+[source+'_'+n for n in ['score', 'rank', 'fallback']]
        frame = pool.loc[pool[source+'_present'].eq(1), columns].copy()
        frame.columns = ['userId', 'movieId', 'score', 'rank', 'fallback']
        frame['source'] = source
        frame['rank'] = frame['rank'].astype('int32')
        result[source] = frame.sort_values(['userId', 'rank'], kind='stable').reset_index(drop=True)
    return result


def infer(model, rec, users):
    from recsys.inference import pipeline_rankings
    from recsys.contracts import validate_candidates
    from recsys.fusion.small_rrf import union_from_frames
    pool, features = pipeline_rankings(rec.models, rec.ctx, rec.booster, rec.schema,
                                      list(map(int, users)), rec.configs, rec.namespace)
    reject_labels(features)
    reject_labels(pool)
    streams = extract_sources(pool)
    for source, frame in streams.items():
        validate_candidates(frame, rec.ctx, source, 100, set(map(int, users)))
    rebuilt = union_from_frames(streams, rec.ctx, users)
    cols = [c for c in rebuilt.columns if c != 'rank']
    a = rebuilt[cols].sort_values(['userId', 'movieId']).reset_index(drop=True)
    b = pool[cols].sort_values(['userId', 'movieId']).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False, check_exact=True)
    request = pd.DataFrame(dict(userId=users, top_k=20), dtype='int64')
    public = model.predict(request)
    actual = pool.loc[pool['rank'] <= 20, ['userId', 'movieId', 'rank', 'ranking_score']].rename(columns={'ranking_score': 'score'}).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, public[['userId', 'movieId', 'rank', 'score']], check_dtype=False, check_exact=True)
    return pool, features, streams


def preflight(root, output, uri):
    guard = LabelReadGuard(root)
    sys.addaudithook(guard)
    path = output/'preflight'
    path.mkdir(exist_ok=False)
    model, rec, release = load_release(root)
    from recsys.small_benchmark import load_training, load_validation
    from recsys.contracts import catalog_id, mapping_id
    from recsys.retrieval_artifacts import checked_manifest
    m = split_manifest(root)
    training = load_training(root/'data/splits/small_v1')
    val = load_validation(root/'data/splits/small_v1', training)
    if (catalog_id(rec.ctx) != release['catalog_id'] or mapping_id(rec.ctx) != release['mapping_id']
            or len(rec.ctx.users) != 610 or rec.ctx.train_rows != 70186 or int(rec.ctx.eligible.sum()) != 7525):
        raise ValueError('Train context identity mismatch')
    if rec.ctx.observed.shape != training.ctx.observed.shape or (rec.ctx.observed != training.ctx.observed).nnz:
        raise ValueError('Packaged histories differ from frozen Train')
    registry = registry_snapshot(uri)
    deployment = json.loads((root/'outputs/deployment_models/movielens_hybrid_v1/deployment_manifest.json').read_text())
    original = deployment['registered_model']
    if any(registry[k] != original[k] for k in ['name', 'version', 'source', 'run_id', 'tags']):
        raise ValueError('Registry differs from verified Version-1 export')
    for source, expected in release['original_artifact_checksums'].items():
        checkpoint = root/'artifacts'/source/'phase3_v1_seed42_main'
        cm = checked_manifest(checkpoint)
        if (sha(checkpoint/'manifest.json') != expected['manifest_sha256'] or
                sha(checkpoint/'COMPLETE.json') != expected['completion_sha256'] or
                cm != release['retrievers'][source] or cm['training_kind'] != 'external_train_per_user'):
            raise ValueError('Wrong Full-Train retriever: '+source)
    fusion = root/'artifacts/fusion/phase4_v1_validation'
    ranker = root/'artifacts/lightgbm/phase4_v1'
    checked_manifest(fusion)
    checked_manifest(ranker)
    old = json.loads((ranker/'comparison.json').read_text())
    saved = pd.read_parquet(ranker/'validation_rankings.parquet')
    fresh, _, streams = infer(model, rec, val.population.users)
    keys = ['userId', 'movieId', 'rank', 'ranking_score']
    pd.testing.assert_frame_equal(saved[keys].reset_index(drop=True), fresh[keys], check_dtype=False, check_exact=True)
    saved_union = pd.read_parquet(fusion/'union.parquet')
    for source, frame in streams.items():
        reference = root/'artifacts/candidates'/source/'phase3_v1_seed42_main/validation_main'
        checked_manifest(reference)
        previous = pd.read_parquet(reference/'candidates.parquet')
        cols = ['userId', 'movieId', 'rank', 'score']
        pd.testing.assert_frame_equal(frame[cols], previous[cols].reset_index(drop=True), check_dtype=False, check_exact=True)
    cols = [c for c in saved_union.columns if c != 'rank']
    pd.testing.assert_frame_equal(saved_union[cols].sort_values(['userId', 'movieId']).reset_index(drop=True),
        fresh[cols].sort_values(['userId', 'movieId']).reset_index(drop=True), check_dtype=False, check_exact=True)
    measured, pq = evaluate(fresh, val.population, rec.ctx)
    for method, oldkey in [('als', 'individual'), ('rrf', 'rrf'), ('lambdamart', 'lambdamart')]:
        for group, expected in zip(measured['methods'][method], old[oldkey]):
            for key, value in expected.items():
                actual = group[key]
                if isinstance(value, float):
                    if actual != value:
                        raise ValueError(f'Validation metric parity failed: {method}/{key}')
                elif actual != value:
                    raise ValueError(f'Validation cohort/denominator parity failed: {method}/{key}')
    # Exercise secondary denominator, gaps, bootstrap and reporting before Test opens.
    secondary = load_validation(root/'data/splits/small_v1', training, 'all_observed').population
    secondary_report, _ = evaluate(fresh, secondary, rec.ctx)
    rehearsal = dict(primary=measured, all_observed=secondary_report,
                     bootstrap=paired_bootstrap(pq), generalization=generalization(measured, measured))
    if not mlflow_metrics(rehearsal):
        raise ValueError('Empty reporting rehearsal')
    pq.to_parquet(path/'validation_per_query.parquet', index=False)
    write_json(path/'validation_metrics.json', measured)
    write_json(path/'historical_selection.json', old)
    write_json(path/'historical_mlflow_records.json', runs_snapshot(uri))
    write_json(path/'registry.json', registry)
    write_json(path/'contract.json', CONTRACT)
    runtime = {name: importlib.metadata.version(name) for name in release['runtime_packages']}
    report = dict(gate='A', passed=True, completed_at=now(), contract=CONTRACT,
        evaluator_source_sha256=sources(root), current_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        source_status=subprocess.check_output(['git', 'status', '--short'], cwd=root, text=True),
        original_release=release, registry=registry, python=platform.python_version(), packages=runtime,
        serving_lock_sha256=sha(root/'uv.lock'), dvc_object='455595aeb5f24864381e436f11093797.dir',
        deployment_sha256=sha(root/'outputs/deployment_models/movielens_hybrid_v1/DEPLOYMENT_COMPLETE.json'),
        validation=dict(queries=len(val.population.users), positives=sum(map(len, val.population.truth.values())),
                        union_pairs=len(fresh), exact_metrics=True, exact_rankings=True, exact_sources=True, exact_union=True),
        data_reads=sorted(guard.reads), test_access=False)
    write_json(path/'report.json', report)
    complete(path)
    return report['validation']


def frozen(root, output):
    a = checked(output/'preflight')
    if a['contract'] != CONTRACT or a['evaluator_source_sha256'] != sources(root):
        raise ValueError('Frozen contract/evaluator source changed; Test is blocked')
    split_manifest(root)
    if sha(root/'uv.lock') != a['serving_lock_sha256']:
        raise ValueError('Runtime lock changed')
    return a


def seal(root, output):
    guard = LabelReadGuard(root)
    sys.addaudithook(guard)
    a = frozen(root, output)
    path = output/'sealed_predictions'
    path.mkdir(exist_ok=False)
    model, rec, _ = load_release(root)
    users = rec.ctx.users
    pool, features, streams = infer(model, rec, users)
    if set(map(int, pool.userId.unique())) != set(map(int, users)):
        raise ValueError('Not all frozen Train users processed')
    for source, frame in streams.items():
        folder = path/'six_sources'/source
        folder.mkdir(parents=True)
        frame.to_parquet(folder/'candidates.parquet', index=False)
    features.to_parquet(path/'features.parquet', index=False)
    pool.to_parquet(path/'union.parquet', index=False)
    for method, frame in rankings(pool).items():
        frame.to_parquet(path/(method+'.parquet'), index=False)
    report = dict(gate='B', passed=True, sealed_at=now(), contract=CONTRACT,
        preflight_complete_sha256=sha(output/'preflight/COMPLETE.json'),
        source_sha256=sources(root), users=len(users), user_population_sha256=identity(users.tolist()),
        union_rows=len(pool), sources={s:len(f) for s,f in streams.items()},
        model_uri=MODEL_URI, release_id=RELEASE_ID,
        training_cutoff=a['original_release']['training_cutoff'],
        mapping_id=a['original_release']['mapping_id'], catalog_id=a['original_release']['catalog_id'],
        test_access=False, data_reads=sorted(guard.reads), exact_pyfunc_top20=True,
        parity_rows=len(users)*20, runtime=dict(python=a['python'], packages=a['packages']),
        code_identity=a['evaluator_source_sha256'])
    write_json(path/'report.json', report)
    write_json(path/'prediction_manifest.json', report)
    complete(path)
    return report


def gate_c(root, output):
    a = frozen(root, output)
    b = checked(output/'sealed_predictions')
    if (not a['passed'] or not b['passed'] or b['contract'] != CONTRACT or b['users'] != 610
            or b['preflight_complete_sha256'] != sha(output/'preflight/COMPLETE.json')
            or b['source_sha256'] != sources(root)):
        raise ValueError('Incomplete/mismatched pre-Test gates')
    path = output/'final_evaluation'
    path.mkdir(exist_ok=False)  # Failure/interruption is retained; no silent re-evaluation.
    model, rec, _ = load_release(root)
    m = split_manifest(root)
    seal_hash = sha(output/'sealed_predictions/COMPLETE.json')
    unsealing = dict(unsealed_at=now(), authorization='User Phase 7 Gate C explicit final Test authorization',
        gate_a_complete_sha256=sha(output/'preflight/COMPLETE.json'),
        predictions_complete_sha256=seal_hash, prediction_manifest_sha256=sha(output/'sealed_predictions/prediction_manifest.json'),
        model_uri=MODEL_URI, release_id=RELEASE_ID, source_sha256=sources(root),
        target='data/splits/small_v1/test.parquet', expected_target_sha256=m['files']['test.parquet']['sha256'])
    write_json(path/'unsealing.json', unsealing)
    future = verified_frame(root, m, 'test.parquet')  # First authorized Test payload read.
    primary = population(future, rec.ctx, SPLIT_SHA, 'test')
    all_observed = population(future, rec.ctx, SPLIT_SHA, 'test', 'all_observed')
    expected = next(r for r in m['evaluation_cohorts'] if r['partition'] == 'test')
    counts = (len(future), len(primary.users), sum(map(len, primary.truth.values())),
              len(all_observed.users), sum(map(len, all_observed.truth.values())))
    if counts != (15507, 583, 6725, 584, 7309) or counts[1:3] != (expected['eligible_positive_users'], expected['eligible_positives']):
        raise ValueError('Test manifest/expected population discrepancy; preserve evidence and stop')
    statistics = next(r for r in m['partition_statistics'] if r['partition'] == 'test')
    actual_statistics = dict(interactions=len(future), users=int(future.userId.nunique()),
        movies=int(future.movieId.nunique()), positives=int(future.rating.ge(4).sum()),
        min_timestamp=int(future.timestamp.min()), max_timestamp=int(future.timestamp.max()))
    if any(statistics[k] != v for k,v in actual_statistics.items()):
        raise ValueError('Frozen Test boundaries/counts differ from manifest')
    cutoffs = verified_frame(root, m, 'user_cutoffs.parquet').set_index('userId')
    observed = future.groupby('userId').timestamp.min()
    if not observed.eq(cutoffs.loc[observed.index, 'test_min_timestamp']).all():
        raise ValueError('Per-user Test boundaries differ from frozen cutoffs')
    if not observed.gt(cutoffs.loc[observed.index, 'train_max_timestamp']).all():
        raise ValueError('Same-user Test chronology violated')
    pool = pd.read_parquet(output/'sealed_predictions/union.parquet')
    report, pq = evaluate(pool, primary, rec.ctx)
    diagnostic, dq = evaluate(pool, all_observed, rec.ctx)
    diagnostic['catalog_micro_ceiling'] = counts[2]/counts[4]
    diagnostic['catalog_macro_ceiling'] = float(np.mean([
        len(primary.truth.get(u, set()))/len(all_observed.truth[u]) for u in all_observed.users]))
    from recsys.fusion.small_rrf import overlap_diagnostics
    frames = {s: f[f.userId.isin(primary.users)].copy() for s, f in extract_sources(pool).items()}
    validation = json.loads((output/'preflight/validation_metrics.json').read_text())
    results = dict(evaluation_type='final_heldout_test', model_uri=MODEL_URI, release_id=RELEASE_ID,
        contract=CONTRACT, completed_at=now(), primary=report, all_observed=diagnostic,
        bootstrap=paired_bootstrap(pq), generalization=generalization(validation, report),
        overlap=overlap_diagnostics(frames, primary.truth), unsealing=unsealing,
        corrections=[], retraining=False, test_tuning=False, model_reselection=False,
        test_statistics=actual_statistics, per_user_boundaries_verified=True)
    pq.to_parquet(path/'per_query_policy_b.parquet', index=False)
    dq.to_parquet(path/'per_query_all_observed.parquet', index=False)
    write_json(path/'results.json', results)
    write_json(path/'metric_namespaces.json', mlflow_metrics(results))
    write_json(path/'report.json', dict(gate='C', passed=True, evaluation_id=primary.identifier,
        all_observed_evaluation_id=all_observed.identifier, completed_at=now(), counts=counts,
        prediction_seal_sha256=seal_hash, test_payload_reads=1, results_sha256=sha(path/'results.json')))
    if sha(output/'sealed_predictions/COMPLETE.json') != seal_hash:
        raise ValueError('Prediction seal changed')
    checked(output/'sealed_predictions')
    complete(path)
    return dict(evaluation_id=primary.identifier, methods={s:v[0] for s,v in report['methods'].items()}, union=report['union'])


def log_final(root, output, uri):
    """Create once, or read-only verify an identical finished evaluation run."""
    from mlflow import MlflowClient
    from mlflow.entities import Metric
    import time
    frozen(root, output)
    checked(output/'sealed_predictions')
    checked(output/'final_evaluation')
    results = json.loads((output/'final_evaluation/results.json').read_text())
    before = json.loads((output/'preflight/registry.json').read_text())
    if registry_snapshot(uri) != before:
        raise ValueError('Registry mutated since freeze')
    c = MlflowClient(tracking_uri=uri, registry_uri=uri)
    experiment = c.get_experiment_by_name(EXPERIMENT)
    experiment_id = experiment.experiment_id if experiment else c.create_experiment(EXPERIMENT)
    fingerprint = sha(output/'final_evaluation/COMPLETE.json')
    tags = dict(evaluation_type='final_heldout_test', model_name='MovieLensHybridRecommender',
        model_version='1', release_id=RELEASE_ID, protocol='per_user_70_15_15', split='small_v1',
        model_frozen_before_test='true', predictions_sealed_before_labels='true',
        hyperparameter_tuning_on_test='false', test_unsealed='true',
        source_git_sha=checked(output/'preflight')['current_head'],
        evaluator_source_identity=identity(sources(root)), split_manifest_sha256=SPLIT_SHA,
        dvc_object='455595aeb5f24864381e436f11093797.dir', feature_schema_sha256=SCHEMA_ID,
        label_policy='observed_rating_ge4', candidate_policy='train_observed_exclude_all_train_rated',
        evaluation_id=results['primary']['evaluation_id'], evaluation_fingerprint=fingerprint,
        prediction_manifest_sha256=sha(output/'sealed_predictions/prediction_manifest.json'),
        model_uri=MODEL_URI, **{'mlflow.runName':'Version 1 frozen final held-out Test'})
    metrics = mlflow_metrics(results)
    found = c.search_runs([experiment_id], filter_string=f"tags.evaluation_fingerprint = '{fingerprint}'")
    created = not found
    if len(found) > 1:
        raise ValueError('Duplicate final evaluation records')
    if found:
        run_id = found[0].info.run_id
    else:
        run_id = c.create_run(experiment_id, tags=tags).info.run_id
        entries = [Metric(k, v, int(time.time()*1000), 0) for k,v in metrics.items()]
        for start in range(0, len(entries), 500):
            c.log_batch(run_id, metrics=entries[start:start+500])
        for folder in ['preflight', 'sealed_predictions', 'final_evaluation']:
            for p in sorted((output/folder).glob('*.json')):
                c.log_artifact(run_id, str(p), artifact_path=folder)
        # Label-containing per-query diagnostics stay in this local MLflow store only.
        for p in (output/'final_evaluation').glob('*.parquet'):
            c.log_artifact(run_id, str(p), artifact_path='local_private_per_query')
        c.set_terminated(run_id, status='FINISHED')
    run = c.get_run(run_id)
    if run.info.status != 'FINISHED' or any(run.data.tags.get(k) != str(v) for k,v in tags.items()) or run.data.metrics != metrics:
        raise ValueError('Final MLflow run incomplete/conflicting; no automatic replacement')
    if registry_snapshot(uri) != before:
        raise ValueError('Registry unexpectedly mutated')
    baseline = json.loads((output/'preflight/historical_mlflow_records.json').read_text())
    current = runs_snapshot(uri)
    if any(current.get(k) != v for k,v in baseline.items()):
        raise ValueError('Historical MLflow record changed')
    record = dict(gate='D', passed=True, created=created, run_id=run_id, experiment=EXPERIMENT,
        experiment_id=experiment_id, model_uri=MODEL_URI, metrics=len(metrics),
        historical_runs_verified=len(baseline), registry_unchanged=True, fingerprint=fingerprint)
    destination = output/('mlflow_created.json' if created else 'mlflow_verified.json')
    if not destination.exists():
        write_json(destination, record)
    return record
