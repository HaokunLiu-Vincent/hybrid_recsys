"""Public contracts shared by independent model, fusion and ranking experiments."""
from dataclasses import dataclass
from typing import Protocol
import numpy as np
import pandas as pd
from .runtime import fingerprint

CHANNELS = ('popularity', 'genre', 'itemcf', 'implicit_als', 'lightgcn', 'two_tower')
CANDIDATE_COLUMNS = ('userId', 'movieId', 'score', 'rank', 'source')
FRAMEWORK_VERSION = 'modular_v2'
COMPATIBILITY_KEYS = ('framework_version', 'split_manifest_sha256', 'training_kind',
                      'training_cutoff', 'label_start', 'label_end', 'catalog_policy',
                      'catalog_id', 'mapping_id', 'evaluation_id', 'top_k', 'positive_threshold')


class Retriever(Protocol):
    """Existing scalar API plus bounded batched retrieval; factory/load live in training.py."""
    def retrieve(self, user_context: int, k: int = 100, item_block: int = 4096): ...
    def retrieve_many(self, users, k: int = 100, item_block: int = 4096): ...
    def save(self, path): ...


@dataclass
class Population:
    truth: dict
    cohorts: dict
    report: dict
    identifier: str

    @property
    def users(self):
        return np.array(sorted(self.truth), dtype=np.int64)


def mapping_id(ctx):
    return fingerprint({'users': ctx.users.tolist(), 'movies': ctx.items.tolist()})


def catalog_id(ctx):
    return fingerprint(ctx.items[ctx.eligible].tolist())


def validate_candidates(frame, context, source, k, allowed_users=None):
    """Check a whole query batch; query fragments must not be passed separately."""
    missing = set(CANDIDATE_COLUMNS)-set(frame.columns)
    if missing:
        raise ValueError(f'Missing candidate columns: {sorted(missing)}')
    if frame.empty:
        return
    for c in ('userId', 'movieId', 'rank'):
        if not pd.api.types.is_integer_dtype(frame[c]) or frame[c].isna().any():
            raise TypeError(f'{c} must contain finite integer values')
    if not np.isfinite(frame.score.to_numpy(dtype=float)).all():
        raise ValueError('Candidate scores must be finite')
    if set(frame.source) != {source} or frame.duplicated(['userId', 'movieId', 'source']).any():
        raise ValueError('Wrong source or duplicate candidate rows')
    if allowed_users is not None and not set(frame.userId).issubset(allowed_users):
        raise ValueError('Candidates contain users outside the frozen evaluation population')
    if not frame.movieId.isin(context.items[context.eligible]).all():
        raise ValueError('Movie outside snapshot Train-observed catalog')
    for user, group in frame.groupby('userId', sort=False):
        if len(group) > k or not np.array_equal(group['rank'].to_numpy(), np.arange(1, len(group)+1)):
            raise ValueError('Ranks must be contiguous from 1 and no larger than K')
        expected = group.sort_values(['score', 'movieId'], ascending=[False, True], kind='stable')
        if group.movieId.tolist() != expected.movieId.tolist():
            raise ValueError('Rank disagrees with score ordering / deterministic movie-ID ties')
        u = context.user_index(user)
        if np.isin(group.movieId, context.items[context.seen(u)]).any():
            raise ValueError('Previously rated movie leaked into candidates')


def compatible(manifests, require_all=True):
    """Hyperparameters may differ; data, cutoff, catalog and query/label population may not."""
    if require_all and set(manifests) != set(CHANNELS):
        raise ValueError(f'Exactly six channel artifacts required: {CHANNELS}')
    if not manifests:
        raise ValueError('No candidate manifests supplied')
    reference = next(iter(manifests.values()))
    for source, manifest in manifests.items():
        if manifest['model_name'] != source:
            raise ValueError(f'Artifact source/name mismatch: {source}')
        for key in COMPATIBILITY_KEYS:
            if manifest.get(key) != reference.get(key) or key not in manifest:
                raise ValueError(f'Incompatible {key} for {source}')
    return reference
