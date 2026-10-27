"""Read frozen Small artifacts into retained model/metric interfaces; no fit or selection."""
from dataclasses import dataclass
from pathlib import Path
import json
import pandas as pd
from .contracts import Population
from .data import build_context, relevance
from .small_data import DATASET_ID, verify_provenance, sha256
from .small_pipeline import verify_files, digest
from .small_protocol import POLICY


@dataclass
class BenchmarkData:
    train: pd.DataFrame
    movies: pd.DataFrame
    ctx: object
    population: Population
    cohorts: pd.DataFrame
    manifest: dict


def load_benchmark(directory,raw_dir,phase='validation',denominator='eligible'):
    """Default Policy-B truth; all_observed is a separately identified diagnostic.

    Frozen Train only: never append Validation before Test. Do not call Test during
    tuning. The Context.cutoff scalar is max(Train timestamp), not a global cutoff
    under the primary per-user protocol; use user_cutoffs/manifest for provenance.
    """
    if phase not in ('validation','test') or denominator not in ('eligible','all_observed'):
        raise ValueError('Unsupported evaluation window/denominator')
    directory,raw_dir=Path(directory),Path(raw_dir)
    manifest=verify_files(directory);verify_provenance(raw_dir)
    if manifest.get('dataset_id')!=DATASET_ID or manifest.get('policy')!=POLICY:
        raise ValueError('Frozen artifacts do not implement the Small v1 evaluation contract')
    for filename,expected in manifest['raw_sha256'].items():
        if sha256(raw_dir/filename)!=expected:raise ValueError('Raw acquisition disagrees with split')
    train=pd.read_parquet(directory/'train.parquet')
    movies=pd.read_parquet(directory/'movies.parquet')
    ctx=build_context(train,movies)
    future=pd.read_parquet(directory/(phase+'.parquet'))
    truth,report=relevance(future,ctx)
    if denominator=='all_observed':
        truth={int(u):set(map(int,g.movieId)) for u,g in future[future.rating>=4].groupby('userId',sort=True)}
    cohorts=pd.read_parquet(directory/'user_eval_cohorts.parquet')
    cohorts=cohorts[cohorts.partition.eq(phase)].copy()
    cohort_by_user=cohorts.set_index('userId').cohort.to_dict()
    report.update(phase=phase,denominator=denominator,source_evaluation_id=manifest['evaluation_id'],
                  users_without_positive=int(cohorts.no_positive_targets.sum()),
                  users_without_retrievable_positive=int(cohorts.no_retrievable_targets.sum()),
                  users_without_candidates=int(cohorts.no_candidates.sum()),
                  no_test_selection='Call Test only after a recorded Validation-only selection; this helper cannot enforce human experiment discipline.')
    identity=digest(dict(split_manifest_sha256=sha256(directory/'split_manifest.json'),
                         phase=phase,denominator=denominator,users=sorted(truth)))
    population=Population(truth,{u:cohort_by_user[u] for u in truth},report,identity)
    return BenchmarkData(train,movies,ctx,population,cohorts,manifest)
