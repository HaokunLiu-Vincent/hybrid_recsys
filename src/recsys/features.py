"""Cutoff-only candidate features. Evaluation labels never enter this module."""
import numpy as np
from .contracts import CHANNELS


def feature_columns(channels=CHANNELS):
    return [s+'_'+t for s in channels for t in ('score','rank','present','fallback')]+[
        'source_count','item_positive_log','item_observed_log','user_positive_log','user_observed_log','genre_similarity']


def candidate_features(pool, ctx, channels=CHANNELS):
    frame = pool.copy()
    ii = np.searchsorted(ctx.items, frame.movieId.to_numpy())
    if np.any(ii >= len(ctx.items)) or not np.array_equal(ctx.items[ii], frame.movieId.to_numpy()):
        raise ValueError('Candidate movie ID has no canonical mapping')
    ui = np.array([ctx.user_index(u) for u in frame.userId],dtype=np.int64)
    valid = ui >= 0
    pos, obs, genre = np.zeros(len(frame)), np.zeros(len(frame)), np.zeros(len(frame))
    pos[valid] = ctx.positive_count[ui[valid]]
    obs[valid] = ctx.user_count[ui[valid]]
    genre[valid] = (ctx.user_genres[ui[valid]]*ctx.genres[ii[valid]]).sum(axis=1)
    frame['item_positive_log'] = np.log1p(ctx.popularity[ii])
    frame['item_observed_log'] = np.log1p(ctx.item_count[ii])
    frame['user_positive_log'], frame['user_observed_log'] = np.log1p(pos), np.log1p(obs)
    frame['genre_similarity'] = genre
    frame[feature_columns(channels)] = frame[feature_columns(channels)].astype('float32')
    return frame
