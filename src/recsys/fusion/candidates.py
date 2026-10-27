"""Deduplicated union preserving raw per-source scores/ranks/presence."""
import numpy as np
import pandas as pd

def fuse(outputs, channels=None, k=100, rrf_constant=60):
    channels = tuple(outputs) if channels is None else tuple(channels)
    records = {}
    for source in channels:
        for user, candidates in outputs[source].items():
            if len({r['movieId'] for r in candidates}) != len(candidates):
                raise ValueError('Duplicate within a retrieval channel')
            for candidate in candidates:
                key = (int(user),candidate['movieId'])
                if key not in records:
                    records[key] = dict(userId=key[0],movieId=key[1])
                records[key].update({source+'_score':candidate['score'],source+'_rank':candidate['rank'],
                                     source+'_present':1,source+'_fallback':int(candidate.get('fallback',False))})
    frame = pd.DataFrame(records.values()) if records else pd.DataFrame(columns=['userId','movieId'])
    for source in channels:
        for suffix, default in [('score',np.nan),('rank',k+1),('present',0),('fallback',0)]:
            col = source+'_'+suffix
            if col not in frame:
                frame[col] = default
            frame[col] = frame[col].fillna(default) if suffix != 'score' else frame[col]
    frame['source_count'] = frame[[s+'_present' for s in channels]].sum(axis=1)
    frame['rrf_score'] = sum(frame[s+'_present']/(rrf_constant+frame[s+'_rank']) for s in channels)
    return frame.sort_values(['userId','movieId'],kind='stable').reset_index(drop=True)

