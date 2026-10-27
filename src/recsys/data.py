"""Reusable in-memory history features. Dataset acquisition and split policy are not implemented here."""
from dataclasses import dataclass
import json
import numpy as np
import pandas as pd
from scipy import sparse
from .runtime import require_memory

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
