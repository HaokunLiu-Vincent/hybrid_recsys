"""Read frozen Small artifacts into retained model/metric interfaces; no fit or selection."""
from dataclasses import dataclass
from pathlib import Path
import json
import time
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


@dataclass
class TrainingData:
    train: pd.DataFrame
    movies: pd.DataFrame
    ctx: object
    manifest: dict
    split_manifest_sha256: str
    context_fit_seconds: float


def selected_manifest(directory,filenames):
    """Verify only stage-required inputs. Never read Test or mixed label-side tables."""
    directory=Path(directory)
    manifest=json.loads((directory/'split_manifest.json').read_text())
    if (manifest.get('dataset_id')!=DATASET_ID or manifest.get('policy')!=POLICY or
        manifest.get('protocol_name')!='per_user_chronological_70_15_15' or manifest.get('version')!='small_v1'):
        raise ValueError('Phase 3 requires the frozen Small primary v1 contract')
    for filename in filenames:
        if filename not in {'train.parquet','movies.parquet','user_mapping.parquet',
                            'movie_mapping.parquet','item_eligibility.parquet','validation.parquet'}:
            raise ValueError('Input is outside the Train/Validation stage allowlist')
        record=manifest['files'][filename];path=directory/filename
        if path.is_symlink() or path.stat().st_size!=record['bytes'] or sha256(path)!=record['sha256']:
            raise ValueError('Frozen input checksum mismatch: '+filename)
    return manifest


def load_training(directory):
    """Train-only fitting interface: no raw full ratings, Validation, Test or evaluation targets."""
    directory=Path(directory)
    manifest=selected_manifest(directory,['train.parquet','movies.parquet','user_mapping.parquet',
                                          'movie_mapping.parquet','item_eligibility.parquet'])
    train=pd.read_parquet(directory/'train.parquet')
    movies=pd.read_parquet(directory/'movies.parquet')
    start=time.perf_counter();ctx=build_context(train,movies);elapsed=time.perf_counter()-start
    users=pd.read_parquet(directory/'user_mapping.parquet')
    items=pd.read_parquet(directory/'movie_mapping.parquet')
    eligibility=pd.read_parquet(directory/'item_eligibility.parquet')
    if ctx.users.tolist()!=users.userId.tolist() or ctx.items.tolist()!=items.movieId.tolist():
        raise ValueError('Canonical Train/metadata mappings disagree with frozen artifacts')
    if ctx.items[ctx.eligible].tolist()!=eligibility.loc[eligibility.eligible,'movieId'].tolist():
        raise ValueError('Train-only catalog disagrees with frozen eligibility')
    return TrainingData(train,movies,ctx,manifest,sha256(directory/'split_manifest.json'),elapsed)


def load_validation(directory,training=None,denominator='eligible'):
    """Read Validation only; cohort features are recomputed from frozen Train, not label tables."""
    if denominator not in ('eligible','all_observed'):raise ValueError('Unknown relevance denominator')
    directory=Path(directory);training=training if training is not None else load_training(directory)
    manifest=selected_manifest(directory,['validation.parquet'])
    if training.split_manifest_sha256!=sha256(directory/'split_manifest.json'):
        raise ValueError('Training/evaluation split identity mismatch')
    future=pd.read_parquet(directory/'validation.parquet')
    truth,report=relevance(future,training.ctx)
    if denominator=='all_observed':
        truth={int(u):set(map(int,g.movieId)) for u,g in future[future.rating>=4].groupby('userId',sort=True)}
    names={'cold':'cold_0','sparse':'sparse_1_4','moderate':'moderate_5_19','warm':'warm_20_plus'}
    cohorts={u:names[training.ctx.cohort(u)] for u in truth}
    identity=digest(dict(split_manifest_sha256=training.split_manifest_sha256,phase='validation',
                         denominator=denominator,users=sorted(truth)))
    report.update(phase='validation',denominator=denominator,source_evaluation_id=manifest['evaluation_id'])
    if denominator=='eligible':
        expected=next(r for r in manifest['evaluation_cohorts'] if r['partition']=='validation')
        if len(truth)!=expected['eligible_positive_users'] or sum(map(len,truth.values()))!=expected['eligible_positives']:
            raise ValueError('Validation relevance disagrees with frozen manifest')
    return BenchmarkData(training.train,training.movies,training.ctx,
                         Population(truth,cohorts,report,identity),pd.DataFrame(),manifest)
