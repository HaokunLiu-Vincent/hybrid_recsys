"""User-macro metrics, deterministic user-bootstrap intervals, zero-hit queries retained."""
import numpy as np
import pandas as pd

METRICS = ('recall20','recall50','recall100','hit20','hit100','ndcg20','pool_recall')


def query_metrics(ranked, relevant):
    if not relevant:
        raise ValueError('A relevance-bearing query is required; do not invent denominator')
    ranked = list(ranked)
    if len(ranked) != len(set(ranked)):
        raise ValueError('Duplicate candidates in ranking')
    hit = np.array([i in relevant for i in ranked], dtype=np.float64)
    discount = 1/np.log2(np.arange(min(20,len(ranked)))+2)
    ideal = (1/np.log2(np.arange(min(20,len(relevant)))+2)).sum()
    return dict(recall20=float(hit[:20].sum()/len(relevant)),
                recall50=float(hit[:50].sum()/len(relevant)),
                recall100=float(hit[:100].sum()/len(relevant)),
                hit20=float(hit[:20].any()),hit100=float(hit[:100].any()),
                ndcg20=float((hit[:20]*discount).sum()/ideal),
                pool_recall=float(hit.sum()/len(relevant)), retrieved_positive=int(hit.sum()))


def summarize(rows, seed=42, bootstrap=300):
    """Paired query rows are persisted for further paired bootstrap comparisons."""
    rows = pd.DataFrame(rows)
    reports = []
    if rows.empty:
        return reports
    for name, group in [('all',rows)] + list(rows.groupby('cohort',sort=True)):
        entry = dict(cohort=name,queries=len(group))
        values = group[list(METRICS)].to_numpy(dtype=float)
        rng = np.random.default_rng(seed)
        means = np.empty((bootstrap,len(METRICS)))
        for b in range(bootstrap):
            means[b] = values[rng.integers(0,len(values),len(values))].mean(axis=0)
        for j, metric in enumerate(METRICS):
            entry[metric] = float(values[:,j].mean())
            entry[metric+'_ci95'] = np.quantile(means[:,j],[.025,.975]).tolist()
        reports.append(entry)
    return reports
