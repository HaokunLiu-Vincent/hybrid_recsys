"""Read-only frozen split contract and training-cutoff feature construction."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from .runtime import digest, require_memory

REQUIRED = ['userId', 'movieId', 'rating', 'timestamp', 'source_row_id']
EXPECTED_ROWS = {'train': 31_414_033, 'validation': 300_000, 'test': 286_171}


def verify_splits(split_dir, dataset_dir, expected=EXPECTED_ROWS, include_test=False):
    """Verify development partitions without deserializing Test; include_test needs the final gate."""
    split_dir, dataset_dir = Path(split_dir), Path(dataset_dir)
    manifest = json.loads((split_dir/'split_manifest.json').read_text())
    if manifest['protocol_version'] != 'v1' or manifest['main_candidate_policy'] != 'B':
        raise ValueError('Requires frozen protocol v1 with primary Policy B')
    required_artifacts = {f'{part}.parquet' for part in ['train','validation','test']} | {'user_eval_cohorts.parquet','item_eligibility.parquet'}
    if set(manifest['artifacts']) != required_artifacts:
        raise ValueError('Frozen split artifact inventory is incomplete or unexpected')
    cohorts = {'cold':[0,0], 'sparse':[1,4], 'moderate':[5,19], 'warm':[20,None]}
    if manifest['user_cohort_definitions'] != cohorts:
        raise ValueError('Frozen cohort definitions differ from the benchmark protocol')
    if manifest['positive_threshold'] != 4.0 or not manifest['validation_results']['all_passed']:
        raise ValueError('Split protocol failed validation or changed positive threshold')
    for kind, filename in [('ratings', 'ratings.csv'), ('movies', 'movies.csv')]:
        p, record = dataset_dir/filename, manifest['source_files'][kind]
        if p.stat().st_size != record['bytes'] or digest(p) != record['sha256']:
            raise IOError(f'Raw source differs from the frozen split manifest: {p}')
    for filename, record in manifest['artifacts'].items():
        if filename == 'test.parquet' and not include_test:
            continue
        p = split_dir/filename
        if p.stat().st_size != record['bytes'] or digest(p) != record['sha256']:
            raise IOError(f'Frozen split checksum mismatch: {p}')
        if pq.ParquetFile(p).metadata.num_rows != record['rows']:
            raise ValueError(f'Parquet row count mismatch: {p}')
    if sum(manifest['row_counts'][p] for p in expected) != manifest['row_counts']['original']:
        raise ValueError('Manifest partition counts do not cover the original source')
    for part, count in expected.items():
        if manifest['row_counts'][part] != count:
            raise ValueError(f'{part}: manifest row count differs from integrity expectation {count}')
    b = manifest['boundaries_utc_and_unix']
    if not b['dataset_min_timestamp'] < b['validation_start_timestamp'] < b['test_start_timestamp'] <= b['dataset_max_timestamp']:
        raise ValueError('Invalid frozen timestamp boundaries')
    bounds = {'train': (b['dataset_min_timestamp'], b['validation_start_timestamp']),
              'validation': (b['validation_start_timestamp'], b['test_start_timestamp']),
              'test': (b['test_start_timestamp'], b['dataset_max_timestamp']+1)}
    audit = {}
    for part, (lo, hi) in bounds.items():
        if part == 'test' and not include_test:
            continue
        pf = pq.ParquetFile(split_dir/f'{part}.parquet')
        if not set(REQUIRED).issubset(pf.schema_arrow.names):
            raise ValueError(f'{part}: missing required schema {REQUIRED}')
        n, mn, mx, positives = 0, None, None, 0
        for batch in pf.iter_batches(batch_size=500_000, columns=REQUIRED):
            frame = batch.to_pandas()
            if frame.isna().any().any():
                raise ValueError(f'{part}: null values')
            for col in ['userId', 'movieId', 'timestamp', 'source_row_id']:
                if not pd.api.types.is_integer_dtype(frame[col]):
                    raise TypeError(f'{part}.{col} must be integral')
            ts, rating = frame.timestamp.to_numpy(), frame.rating.to_numpy()
            if np.any((ts < lo) | (ts >= hi)):
                raise ValueError(f'{part}: timestamp outside manifest boundary')
            if not np.isfinite(rating).all() or np.any((rating < .5) | (rating > 5) | (rating*2 % 1 != 0)):
                raise ValueError(f'{part}: invalid rating values')
            n += len(frame)
            positives += int((rating >= 4).sum())
            mn = min(mn if mn is not None else ts.min(), int(ts.min()))
            mx = max(mx if mx is not None else ts.max(), int(ts.max()))
        if n != manifest['row_counts'][part] or n != expected[part]:
            raise ValueError(f'{part}: {n} rows, manifest={manifest["row_counts"][part]}, expected={expected[part]}')
        if (part == 'train' and mn != b['dataset_min_timestamp']) or (part == 'test' and mx != b['dataset_max_timestamp']):
            raise ValueError(f'{part}: dataset endpoint differs from manifest')
        audit[part] = dict(rows=n, min_timestamp=mn, max_timestamp=mx, positives=positives)
    return manifest, audit


def load_ratings(path):
    pf = pq.ParquetFile(path)
    require_memory(cpu=pf.metadata.num_rows*80, stage=f'load {Path(path).name}')
    return pd.read_parquet(path, columns=REQUIRED).astype(
        {'userId':'int64', 'movieId':'int64', 'rating':'float32', 'timestamp':'int64', 'source_row_id':'int64'})


@dataclass
class Context:
    users: np.ndarray
    items: np.ndarray
    observed: sparse.csr_matrix
    positive: sparse.csr_matrix
    genres: np.ndarray
    genre_names: list
    user_genres: np.ndarray
    item_count: np.ndarray
    popularity: np.ndarray
    user_count: np.ndarray
    positive_count: np.ndarray
    eligible: np.ndarray
    cutoff: int
    train_rows: int

    def user_index(self, user_id):
        p = int(np.searchsorted(self.users, user_id))
        return p if p < len(self.users) and self.users[p] == user_id else -1

    def seen(self, user):
        return self.observed.indices[self.observed.indptr[user]:self.observed.indptr[user+1]] if user >= 0 else np.empty(0, dtype=np.int32)

    def cohort(self, user_id):
        u = self.user_index(user_id)
        n = self.positive_count[u] if u >= 0 else 0
        return 'cold' if n == 0 else 'sparse' if n < 5 else 'moderate' if n < 20 else 'warm'

    def save(self, path):
        # Mappings and provenance only; histories are rebuilt from verified Train on resume.
        np.savez(path/'mappings.npz', users=self.users, items=self.items, eligible=self.eligible)
        (path/'context.json').write_text(json.dumps(dict(cutoff=self.cutoff, train_rows=self.train_rows,
                                                       genre_names=self.genre_names)))


def build_context(train, movies):
    if not {'movieId','title','genres'}.issubset(movies.columns) or movies.movieId.duplicated().any():
        raise ValueError('movies.csv requires unique movieId, title, genres')
    users = np.sort(train.userId.unique())
    items = np.sort(movies.movieId.to_numpy(dtype=np.int64))
    require_memory(cpu=len(train)*70 + (len(users)+len(items))*32*4, stage='CSR and cutoff features')
    ui = np.searchsorted(users, train.userId.to_numpy())
    ii = np.searchsorted(items, train.movieId.to_numpy())
    if np.any(ii >= len(items)) or not np.array_equal(items[ii], train.movieId.to_numpy()):
        raise ValueError('Training movie missing from movies.csv; no automatic data repair')
    shape = (len(users), len(items))
    values = train.rating.to_numpy(dtype=np.float32)
    observed = sparse.csr_matrix((values, (ui, ii)), shape=shape)
    if observed.nnz != len(train):
        raise ValueError('Duplicate training user/movie pairs; protocol must be inspected')
    positive = observed.copy()
    positive.data = (positive.data >= 4).astype(np.float32)
    positive.eliminate_zeros()
    positive.sort_indices()
    observed.sort_indices()
    metadata = movies.set_index('movieId').loc[items]
    sets = [set(str(x).split('|')) - {'(no genres listed)', 'nan', ''} for x in metadata.genres]
    names = sorted(set().union(*sets))
    genres = np.array([[float(g in gs) for g in names] for gs in sets], dtype=np.float32)
    genres /= np.maximum(np.linalg.norm(genres, axis=1, keepdims=True), 1e-12)
    user_genres = positive @ genres
    user_genres /= np.maximum(np.linalg.norm(user_genres, axis=1, keepdims=True), 1e-12)
    item_count = np.asarray((observed != 0).sum(axis=0)).ravel().astype(np.int64)
    return Context(users, items, observed, positive, genres, names, user_genres,
                   item_count, np.asarray(positive.sum(axis=0)).ravel(),
                   np.diff(observed.indptr), np.diff(positive.indptr), item_count > 0,
                   int(train.timestamp.max()), len(train))


def relevance(frame, ctx):
    """Policy B and Train-seen filtering; catalog comes exclusively from this snapshot."""
    positive = frame.loc[frame.rating >= 4, ['userId', 'movieId']]
    eligible_ids = ctx.items[ctx.eligible]
    in_catalog = positive.movieId.isin(eligible_ids)
    result, seen_excluded = {}, 0
    for user, group in positive.loc[in_catalog].groupby('userId', sort=True):
        u = ctx.user_index(user)
        seen_ids = ctx.items[ctx.seen(u)]
        kept = set(map(int, group.movieId)) - set(map(int, seen_ids))
        seen_excluded += len(group)-len(kept)
        if kept:
            result[int(user)] = kept
    return result, dict(observed_positive_interactions=len(positive),
                        policy_B_catalog_excluded=int((~in_catalog).sum()),
                        train_seen_positive_excluded=seen_excluded,
                        eligible_positive_interactions=sum(map(len, result.values())),
                        eligible_queries=len(result),
                        definition='future observed rating >= 4; Train-observed catalog; exclude all Train-rated items')


def internal_partition(train, tail_days=30):
    """Strict timestamp boundary inside Train; ties stay together, external data untouched."""
    cutoff = int(train.timestamp.max()) - int(tail_days*86400)
    before = train.loc[train.timestamp <= cutoff].copy()
    after = train.loc[train.timestamp > cutoff].copy()
    if before.empty or after.empty or before.timestamp.max() >= after.timestamp.min():
        raise ValueError('Internal chronological portion is empty or non-strict; select a supported tail_days')
    report = dict(cutoff_inclusive=cutoff, retriever_rows=len(before), label_rows=len(after),
                  retriever_max_timestamp=int(before.timestamp.max()), label_min_timestamp=int(after.timestamp.min()),
                  label_max_timestamp=int(after.timestamp.max()), tail_days=tail_days,
                  cutoff_inclusive_utc=pd.Timestamp(cutoff,unit='s',tz='UTC').isoformat(),
                  retriever_min_utc=pd.Timestamp(int(before.timestamp.min()),unit='s',tz='UTC').isoformat(),
                  retriever_max_utc=pd.Timestamp(int(before.timestamp.max()),unit='s',tz='UTC').isoformat(),
                  label_min_utc=pd.Timestamp(int(after.timestamp.min()),unit='s',tz='UTC').isoformat(),
                  label_max_utc=pd.Timestamp(int(after.timestamp.max()),unit='s',tz='UTC').isoformat())
    return before, after, report


class FrozenData:
    """Shared read-only data preparation. Development never reads Test label columns."""
    def __init__(self, project, expected=EXPECTED_ROWS):
        self.project = Path(project)
        self.split_dir = self.project/'data'/'splits'/'v1'
        self.dataset_dir = self.project/'data'/'ml-32m'
        self.manifest, self.audit = verify_splits(self.split_dir, self.dataset_dir, expected)
        self.expected = expected
        self.split_sha = digest(self.split_dir/'split_manifest.json')
        self.train = load_ratings(self.split_dir/'train.parquet')
        self.movies = pd.read_csv(self.dataset_dir/'movies.csv', usecols=['movieId','title','genres'],
                                  dtype={'movieId':'int64'})

    def prepare(self, training_kind='full_train', cutoff_utc=None, label_end_utc=None, partition='validation'):
        from .contracts import Population
        from .runtime import fingerprint
        if partition == 'test':
            raise PermissionError('Test labels require the separate final-evaluation gate')
        boundary = self.manifest['boundaries_utc_and_unix']['validation_start_timestamp']
        if training_kind == 'historical_snapshot':
            if cutoff_utc is None:
                raise ValueError('rank_snapshot requires an explicit snapshot_cutoff_utc; no automatic snapshot training')
            def timestamp(value):
                stamp = pd.Timestamp(value)
                if stamp.tzinfo is None:
                    raise ValueError('Specify historical boundaries with UTC timezone, e.g. 2023-01-01T00:00:00Z')
                return int(stamp.timestamp())
            cutoff = timestamp(cutoff_utc)
            end = timestamp(label_end_utc) if label_end_utc else boundary
            if not self.train.timestamp.min() <= cutoff < end <= boundary:
                raise ValueError('Internal snapshot and label window must lie strictly inside external Train')
            history = self.train[self.train.timestamp <= cutoff]
            future = self.train[(self.train.timestamp > cutoff) & (self.train.timestamp < end)].copy()
            if history.empty or future.empty:
                raise ValueError('Historical prefix or later internal label window is empty')
            self.label_start, self.label_end = cutoff+1, end
            self.partition = 'rank_train'
        elif training_kind == 'full_train':
            if cutoff_utc is not None or label_end_utc is not None:
                raise ValueError('Historical boundaries are supported only in rank_snapshot mode')
            history = self.train
            future = load_ratings(self.split_dir/'validation.parquet')
            self.label_start = boundary
            self.label_end = self.manifest['boundaries_utc_and_unix']['test_start_timestamp']
            self.partition = 'validation'
        else:
            raise ValueError(f'Unknown training kind {training_kind}')
        self.ctx = build_context(history, self.movies)
        self.training_kind = training_kind
        self.future = future
        self.population = self._population(future, verify_cohorts=self.partition == 'validation')
        return self

    def _population(self, future, verify_cohorts=False):
        from .contracts import Population
        from .runtime import fingerprint
        truth, report = relevance(future, self.ctx)
        cohorts = {u:self.ctx.cohort(u) for u in truth}
        if verify_cohorts:
            # Read ONLY evaluation columns, not Test labels or future-derived model inputs.
            table = pd.read_parquet(self.split_dir/'user_eval_cohorts.parquet', columns=[
                'userId','positive_history_cohort','validation_eligible_user_B'])
            expected_users = set(table.loc[table.validation_eligible_user_B, 'userId'].astype(int))
            if set(truth) != expected_users:
                raise ValueError('Computed Policy B Validation population differs from the frozen cohort table')
            saved = table.set_index('userId').positive_history_cohort
            if any(str(saved.loc[u]) != cohorts[u] for u in truth):
                raise ValueError('Train-based cohort definitions differ from frozen evaluation reporting')
        identifier = fingerprint(dict(split=self.split_sha, partition=self.partition,
            labels=[[u,sorted(truth[u])] for u in sorted(truth)], label_start=self.label_start, label_end=self.label_end))
        return Population(truth, cohorts, report, identifier)

    def open_test(self, explicitly_enabled=False):
        if not explicitly_enabled or self.training_kind != 'full_train':
            raise PermissionError('Explicit final Test gate and full-Train history required')
        verify_splits(self.split_dir, self.dataset_dir, self.expected, include_test=True)
        self.future = load_ratings(self.split_dir/'test.parquet')
        self.partition = 'test'
        b = self.manifest['boundaries_utc_and_unix']
        self.label_start, self.label_end = b['test_start_timestamp'], b['dataset_max_timestamp']+1
        self.population = self._population(self.future)
        return self
