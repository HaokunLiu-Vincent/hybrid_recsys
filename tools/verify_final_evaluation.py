"""Read-only verification of saved final evidence; never regenerates predictions or labels."""
from pathlib import Path
import argparse
import ast
import json
import re
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/'src'))
from recsys_evaluation.workflow import (checked, frozen, sha, registry_snapshot,
    runs_snapshot, sources, mlflow_metrics, write_json, now)


def verify(output, uri=None, planned_report=None):
    a = frozen(root, output)
    b = checked(output/'sealed_predictions')
    c = checked(output/'final_evaluation')
    results = json.loads((output/'final_evaluation/results.json').read_text())
    initial = json.loads((output/'initial_audit.json').read_text())
    errors = [name for name, expected in initial['protected'].items()
              if name != 'README.md' and (not (root/name).is_file() or sha(root/name) != expected)]
    if errors:
        raise ValueError('Protected file changed: '+str(errors))
    data_errors = [name for name, record in initial['data_metadata'].items()
        if not (root/name).is_file() or (root/name).stat().st_size != record['bytes']
        or (root/name).stat().st_mtime_ns != record['mtime_ns']]
    if data_errors:
        raise ValueError('Frozen data metadata changed: '+str(data_errors))
    # Verify restored data objects against frozen manifests without evaluating labels again.
    data_hashes = 0
    for directory in ['small_v1', 'small_global_v1']:
        folder = root/'data/splits'/directory
        manifest = json.loads((folder/'split_manifest.json').read_text())
        for filename, info in manifest['files'].items():
            if sha(folder/filename) != info['sha256']:
                raise ValueError('Frozen data hash mismatch: '+filename)
            data_hashes += 1
    from recsys.small_data import verify_provenance
    import yaml
    raw = root/'data/raw/ml-latest-small'
    pins = yaml.safe_load((root/'params.yaml').read_text())['dataset']
    if sha(raw/'provenance.json') != pins['provenance_sha256']:
        raise ValueError('Pinned raw provenance differs')
    raw_manifest = verify_provenance(raw)
    import pandas as pd
    from recsys_evaluation.core import METRICS
    pq = pd.read_parquet(output/'final_evaluation/per_query_policy_b.parquet')
    for method, groups in results['primary']['methods'].items():
        for group in groups:
            rows = pq[pq.method.eq(method)]
            if group['cohort'] != 'all':
                rows = rows[rows.cohort.eq(group['cohort'])]
            if len(rows) != group['queries']:
                raise ValueError('Saved query count mismatch')
            for metric in METRICS:
                measured = float(rows[metric].mean()) if len(rows) else None
                if measured != group[metric]:
                    raise ValueError('Saved per-query aggregation mismatch')
    if c['prediction_seal_sha256'] != sha(output/'sealed_predictions/COMPLETE.json'):
        raise ValueError('Prediction seal changed after evaluation')
    if not (a['completed_at'] < b['sealed_at'] < results['unsealing']['unsealed_at'] < results['completed_at']):
        raise ValueError('Gate chronology mismatch')
    python_files = sorted(p for folder in ['src', 'tests', 'tools'] for p in (root/folder).rglob('*.py'))
    for p in python_files:
        ast.parse(p.read_text(), filename=str(p))
    link_errors = []
    for path in [root/'README.md', root/'docs/system_architecture.md', root/'docs/phase7_final_evaluation.md']:
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if '://' in target or target.startswith('#'):
                continue
            target = target.split('#')[0]
            destination = (path.parent/target).resolve()
            if not destination.exists() and destination != planned_report:
                link_errors.append(str(path)+': '+target)
    if link_errors:
        raise ValueError('Broken local documentation link: '+str(link_errors))
    git = lambda *args: subprocess.check_output(['git', *args], cwd=root, text=True).strip()
    if git('rev-parse', 'HEAD') != initial['head'] or git('diff', '--cached', '--name-only'):
        raise ValueError('HEAD/index changed')
    subprocess.run(['git', 'diff', '--check'], cwd=root, check=True)
    for name in ['outputs/phase7/final_evaluation/per_query_policy_b.parquet',
                 'outputs/phase7/sealed_predictions/union.parquet']:
        subprocess.run(['git', 'check-ignore', '-q', name], cwd=root, check=True)
    record = dict(checked_at=now(), passed=True, head=initial['head'], branch=git('branch', '--show-current'),
        verification_command=[sys.executable, *sys.argv], verification_source_sha256=sha(Path(__file__)),
        protected_hashes_verified=len(initial['protected'])-1, protected_hash_errors=[],
        intentional_existing_edits=['README.md'], frozen_data_metadata_verified=len(initial['data_metadata']),
        frozen_parquet_hashes_verified=data_hashes, original_python_files_unchanged=True,
        raw_acquisition_hashes_verified=len(raw_manifest['files'])+1,
        active_python_ast=len(python_files), documentation_links_valid=True, git_diff_check=True,
        evaluator_source_sha256=sources(root), gate_order_valid=True,
        prediction_complete_sha256=sha(output/'sealed_predictions/COMPLETE.json'),
        prediction_manifest_sha256=sha(output/'sealed_predictions/prediction_manifest.json'),
        final_complete_sha256=sha(output/'final_evaluation/COMPLETE.json'),
        test_target_access='Gate C one buffered payload read; final read-only integrity verification hashes both frozen protocols without recalculating Test scores.',
        per_query_aggregation_exact=True, git_status=git('status', '--short'), staged_paths=[],
        git_status_scope='Captured before writing this optional verification record itself',
        no_commit_or_push=True, no_training_tuning_or_promotion=True)
    record['tests'] = {}
    for environment in ['working', 'isolated']:
        log = output/f'tests_final_{environment}.log'
        content = log.read_text()
        match = re.search(r'Ran (\d+) tests in ([\d.]+)s', content)
        if not match or '\nOK\n' not in content:
            raise ValueError('Final test evidence missing or not passing')
        record['tests'][environment] = dict(count=int(match[1]), seconds=float(match[2]),
            failures=0, skips=0, log_sha256=sha(log))
    if uri:
        from mlflow import MlflowClient
        before = json.loads((output/'preflight/registry.json').read_text())
        if registry_snapshot(uri) != before:
            raise ValueError('Version/aliases changed')
        baseline = json.loads((output/'preflight/historical_mlflow_records.json').read_text())
        current = runs_snapshot(uri)
        if any(current.get(k) != v for k,v in baseline.items()):
            raise ValueError('Historical MLflow run changed')
        final = json.loads((output/'mlflow_created.json').read_text())
        client = MlflowClient(tracking_uri=uri, registry_uri=uri)
        run = client.get_run(final['run_id'])
        if run.info.status != 'FINISHED' or run.data.metrics != mlflow_metrics(results):
            raise ValueError('Final run metric/status mismatch')
        # Downloads verify actual uploaded artifact bytes, not just local logging intent.
        import tempfile
        with tempfile.TemporaryDirectory(prefix='phase7-log-verify-') as temporary:
            downloaded = Path(client.download_artifacts(run.info.run_id, 'final_evaluation/results.json', temporary))
            if sha(downloaded) != sha(output/'final_evaluation/results.json'):
                raise ValueError('MLflow uploaded result bytes differ')
        record.update(registry_unchanged=True, historical_runs_verified=len(baseline),
                      final_run_id=run.info.run_id, metric_namespaces_verified=len(run.data.metrics),
                      uploaded_results_sha256_verified=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=root/'outputs/phase7')
    parser.add_argument('--tracking-uri')
    parser.add_argument('--save', type=Path, help='Optional new verification JSON; existing files never overwritten')
    args = parser.parse_args()
    report = verify(args.output_dir, args.tracking_uri, args.save.resolve() if args.save else None)
    if args.save:
        write_json(args.save, report)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
