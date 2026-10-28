"""Predeclared paired evaluation of a common union using retained metric definitions."""
import numpy as np
import pandas as pd

METHODS = ('als', 'rrf', 'lambdamart')
COHORTS = ('cold_0', 'sparse_1_4', 'moderate_5_19', 'warm_20_plus')
METRICS = ('recall20', 'recall50', 'recall100', 'ndcg20', 'hit20', 'hit100', 'pool_recall')
NAMES = dict(recall20='recall_at_20', recall50='recall_at_50', recall100='recall_at_100',
             ndcg20='ndcg_at_20', hit20='hitrate_at_20', hit100='hitrate_at_100',
             pool_recall='macro_recall_ceiling', catalog_coverage20='coverage_at_20',
             catalog_coverage100='coverage_at_100')
FORBIDDEN = {'rating', 'label', 'feedback_kind', 'timestamp', 'submitted_at',
             'future_positives', 'cohort', 'target', 'is_positive'}


def reject_labels(frame):
    if FORBIDDEN.intersection(frame.columns):
        raise ValueError('Label-side fields in label-free prediction input')


def rankings(pool):
    """ALS source ranks 1..100 first; deterministic movie-ID tail, as in Phase 4."""
    from recsys.fusion.small_rrf import ranked
    reject_labels(pool)
    result = {}
    for method, score in [('als', 'als_order_score'), ('rrf', 'rrf_score'),
                          ('lambdamart', 'ranking_score')]:
        frame = pool.copy()
        if method == 'als':
            frame[score] = frame.implicit_als_present * (101 - frame.implicit_als_rank)
        if not np.isfinite(frame[score]).all():
            raise ValueError('Nonfinite ranking score')
        result[method] = ranked(frame, score)
    return result


def population(future, ctx, split_hash, phase, denominator='eligible'):
    from recsys.data import relevance
    from recsys.contracts import Population
    from recsys.runtime import fingerprint
    if phase not in ('validation', 'test') or denominator not in ('eligible', 'all_observed'):
        raise ValueError('Unsupported evaluation identity')
    truth, report = relevance(future, ctx)
    if denominator == 'all_observed':
        truth = {int(u): set(map(int, g.movieId)) for u, g in
                 future[future.rating >= 4].groupby('userId', sort=True)}
    names = dict(zip(('cold', 'sparse', 'moderate', 'warm'), COHORTS))
    cohorts = {u: names[ctx.cohort(u)] for u in truth}
    identity = fingerprint(dict(split_manifest_sha256=split_hash, phase=phase,
                                denominator=denominator, users=sorted(truth)))
    report.update(phase=phase, denominator=denominator, users_with_events=int(future.userId.nunique()),
                  users_without_positive=int(future.userId.nunique()) - int(future.loc[future.rating >= 4, 'userId'].nunique()))
    return Population(truth, cohorts, report, identity)


def evaluate(pool, pop, ctx):
    """All relevance-bearing queries, including empty predictions and zero hits."""
    from recsys.fusion.small_rrf import evaluate_rankings
    summaries, per_query = {}, []
    for method, frame in rankings(pool).items():
        score = dict(als='als_order_score', rrf='rrf_score', lambdamart='ranking_score')[method]
        pq, groups = evaluate_rankings(frame, pop, ctx, score)
        # Retained helper already computes hit100 per query; expose it in summaries.
        for group in groups:
            part = pq if group['cohort'] == 'all' else pq[pq.cohort.eq(group['cohort'])]
            group.update(hit100=float(part.hit100.mean()) if len(part) else None,
                         zero_hit20_queries=int(part.hit20.eq(0).sum()),
                         zero_hit100_queries=int(part.hit100.eq(0).sum()))
        summaries[method] = groups
        pq['method'] = method
        per_query.append(pq)
    pq = pd.concat(per_query, ignore_index=True)
    if pq.duplicated(['method', 'userId']).any():
        raise ValueError('Duplicate evaluation query')
    selected = pool[pool.userId.isin(pop.users)]
    sizes = selected.groupby('userId').size().reindex(pop.users, fill_value=0)
    base = pq[pq.method.eq('rrf')]
    total = sum(map(len, pop.truth.values()))
    union = dict(candidate_pairs=len(selected), queries=len(pop.users), positives=total,
                 candidates_mean=float(sizes.mean()) if len(sizes) else None,
                 candidates_median=float(sizes.median()) if len(sizes) else None,
                 candidates_min=int(sizes.min()) if len(sizes) else None,
                 candidates_max=int(sizes.max()) if len(sizes) else None,
                 macro_recall_ceiling=summaries['rrf'][0]['pool_recall'],
                 retrieved_positives=int(base.retrieved_positive.sum()),
                 absent_positives=total-int(base.retrieved_positive.sum()),
                 micro_positive_coverage=float(base.retrieved_positive.sum()/total) if total else None,
                 zero_union_hit_queries=int(base.retrieved_positive.eq(0).sum()),
                 catalog_coverage=selected.movieId.nunique()/int(ctx.eligible.sum()))
    return dict(evaluation_id=pop.identifier, denominator=pop.report['denominator'],
                population=pop.report, methods=summaries, union=union), pq


def paired_bootstrap(per_query, replicates=2000, seed=42):
    """Paired user resampling; the same user indices for every contrast/metric."""
    wide = per_query.pivot(index='userId', columns='method', values=['recall20', 'ndcg20']).sort_index()
    if wide.isna().any().any() or not len(wide):
        raise ValueError('Paired bootstrap requires identical nonempty query populations')
    differences = {}
    for comparator in ('rrf', 'als'):
        for metric in ('recall20', 'ndcg20'):
            differences[f'lambdamart_minus_{comparator}/{metric}'] = (
                wide[metric]['lambdamart'] - wide[metric][comparator]).to_numpy()
    data = np.column_stack(list(differences.values()))
    rng = np.random.default_rng(seed)
    samples = np.empty((replicates, data.shape[1]))
    for i in range(replicates):
        samples[i] = data[rng.integers(0, len(data), len(data))].mean(axis=0)
    return dict(seed=seed, replicates=replicates, queries=len(data),
                policy='Eligible user queries resampled with replacement, paired methods; 95% percentile intervals; descriptive after Validation selection.',
                differences={key: dict(mean=float(data[:, j].mean()),
                    ci95=np.quantile(samples[:, j], [.025, .975]).tolist())
                    for j, key in enumerate(differences)})


def generalization(validation, test):
    result = {}
    for method in METHODS:
        groups = {}
        for old, new in zip(validation['methods'][method], test['methods'][method]):
            values = {}
            for key in (*METRICS, 'catalog_coverage20', 'catalog_coverage100'):
                a, b = old.get(key), new.get(key)
                values[key] = dict(validation=a, test=b,
                    absolute=None if a is None or b is None else b-a,
                    relative=None if a is None or b is None or a == 0 else (b-a)/a)
            groups[new['cohort']] = dict(validation_queries=old['queries'], test_queries=new['queries'], metrics=values)
        result[method] = groups
    return result


def mlflow_metrics(results):
    """Public aggregate namespaces; no per-user labels or identities in metrics."""
    metrics = {}
    for denominator, report in [('policy_b', results['primary']), ('all_observed', results['all_observed'])]:
        for method, groups in report['methods'].items():
            for group in groups:
                suffix = '' if group['cohort'] == 'all' else '/cohorts/' + group['cohort']
                prefix = f'test/{denominator}/{method}' + suffix
                for key in (*NAMES, 'queries', 'relevant_positives', 'zero_hit20_queries', 'zero_hit100_queries'):
                    if group.get(key) is not None:
                        metrics[prefix+'/'+NAMES.get(key, key)] = float(group[key])
    for key, value in results['primary']['union'].items():
        if value is not None:
            metrics['test/union/'+key] = float(value)
    for name, value in results['bootstrap']['differences'].items():
        for field, number in [('mean', value['mean']), ('ci95_lower', value['ci95'][0]), ('ci95_upper', value['ci95'][1])]:
            metrics['test/paired/'+name+'/'+field] = number
    for method, groups in results['generalization'].items():
        for cohort, group in groups.items():
            for metric, values in group['metrics'].items():
                for field in ('absolute', 'relative'):
                    if values[field] is not None:
                        metrics[f'validation_to_test/{method}/{cohort}/{metric}/{field}'] = values[field]
    return metrics
