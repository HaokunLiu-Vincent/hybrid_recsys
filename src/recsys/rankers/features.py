"""Future weak labels, query sampling and feature-shift diagnostics; no retriever fitting."""
import numpy as np
import pandas as pd
from ..features import feature_columns

def attach_labels(frame, future):
    """Label-side fields NEVER appear in FEATURES. Unknown is weak label zero, not dislike."""
    known = future[['userId','movieId','rating']]
    if known.duplicated(['userId','movieId']).any():
        raise ValueError('Duplicate future pairs in frozen labels')
    result = frame.merge(known,on=['userId','movieId'],how='left',validate='one_to_one')
    result['feedback_kind'] = np.where(result.rating.isna(),'unobserved',
                                      np.where(result.rating >= 4,'observed_positive','observed_low'))
    result['label'] = (result.feedback_kind == 'observed_positive').astype('int8')
    return result.drop(columns='rating')


def sample_training(frame, rng, unlabeled_per_query=100):
    rows, stats = [], dict(queries=0,effective_queries=0,queries_no_retrieved_positive=0,queries_no_contrast=0,
                           positive_candidate_labels=0,observed_low_candidates=0,sampled_unlabeled_candidates=0)
    for _, group in frame.groupby('userId',sort=True):
        stats['queries'] += 1
        positives = group[group.label == 1]
        if positives.empty:
            stats['queries_no_retrieved_positive'] += 1
            continue
        low = group[group.feedback_kind == 'observed_low']
        unknown = group[group.feedback_kind == 'unobserved']
        if len(unknown) > unlabeled_per_query:
            unknown = unknown.iloc[np.sort(rng.choice(len(unknown),unlabeled_per_query,replace=False))]
        sampled = pd.concat([positives,low,unknown]).sort_values('movieId',kind='stable')
        if sampled.label.nunique() < 2:
            stats['queries_no_contrast'] += 1
            continue
        stats['effective_queries'] += 1
        stats['positive_candidate_labels'] += len(positives)
        stats['observed_low_candidates'] += len(low)
        stats['sampled_unlabeled_candidates'] += len(unknown)
        rows.append(sampled)
    return (pd.concat(rows,ignore_index=True) if rows else frame.iloc[:0].copy()), stats


def distribution(frame, columns=None):
    columns = feature_columns() if columns is None else columns
    if frame.empty:
        return {c:dict(mean=None,std=None,missing_fraction=None,quantiles=[None,None,None]) for c in columns}
    # Use a deterministic bounded sample for quantiles; missing scores remain missing, presence reported separately.
    return {c:dict(mean=float(frame[c].mean()) if frame[c].notna().any() else None,
                   std=float(frame[c].std(ddof=0)) if frame[c].notna().any() else None,
                   missing_fraction=float(frame[c].isna().mean()),
                   quantiles=frame[c].quantile([.1,.5,.9]).replace({np.nan:None}).tolist()) for c in columns}


def mismatch(train_summary, inference_summary, columns=None):
    columns = feature_columns() if columns is None else columns
    rows = []
    for col in columns:
        a,b = train_summary[col],inference_summary[col]
        delta = None if a['mean'] is None or b['mean'] is None else (b['mean']-a['mean'])/max(a['std'],1e-6)
        rows.append(dict(feature=col,standardized_mean_shift=delta,
                         snapshot=a,full_train=b))
    return rows
