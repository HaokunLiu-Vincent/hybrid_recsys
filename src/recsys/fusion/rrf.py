"""Reciprocal rank fusion: raw source scores are never summed."""

def rrf(frame, channels=None, constant=60):
    channels = [c[:-8] for c in frame if c.endswith('_present')] if channels is None else list(channels)
    score = sum(frame[s+'_present']/(constant+frame[s+'_rank']) for s in channels)
    keep = frame[[s+'_present' for s in channels]].sum(axis=1) > 0
    return frame.loc[keep].assign(ranking_score=score[keep]).sort_values(
        ['ranking_score','movieId'],ascending=[False,True],kind='stable').movieId.tolist()

