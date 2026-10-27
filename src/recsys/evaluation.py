"""Shared scoring/metric helpers; callers must supply an explicitly approved dataset/population."""
import time
import numpy as np
import pandas as pd
from .metrics import query_metrics,summarize
from .runtime import require_memory

class MetricAccumulator:
    """Streaming rankings; retains only per-query metrics and compact catalog sets."""
    def __init__(self, context, population, config):
        self.ctx, self.population, self.config = context, population, config
        self.rows, self.coverage, self.counts = [], {}, {}
        ids = np.flatnonzero(context.eligible)
        order = np.lexsort((ids, -context.popularity[ids]))
        self.popular = set(context.items[ids[order[:max(1,len(ids)//10)]]].tolist())

    def add(self, model, user, ranked, coverage_k=20):
        ranked = list(map(int, ranked))
        self.rows.append(dict(model=model, userId=int(user), cohort=self.population.cohorts[user],
                              **query_metrics(ranked, self.population.truth[user])))
        self.coverage.setdefault(model,set()).update(ranked[:coverage_k])
        hits,total = self.counts.get(model,(0,0))
        self.counts[model] = (hits+sum(i in self.popular for i in ranked[:20]), total+len(ranked[:20]))

    def finish(self, root):
        from .artifacts import dump_json
        frame = pd.DataFrame(self.rows)
        if frame.empty:
            raise ValueError('Empty eligible-positive evaluation population')
        expected = set(self.population.truth)
        for name, group in frame.groupby('model', sort=True):
            if set(group.userId) != expected or group.userId.duplicated().any():
                raise ValueError(f'{name}: metrics must include every fixed evaluation query exactly once')
        frame.to_parquet(root/'per_query_metrics.parquet',index=False)
        metrics = {name:summarize(group,self.config['seed'],self.config['bootstrap'])
                   for name,group in frame.groupby('model',sort=True)}
        conditional = {name:summarize(group[group.retrieved_positive > 0],self.config['seed'],self.config['bootstrap'])
                       for name,group in frame.groupby('model',sort=True)}
        return dict(metrics=metrics, conditional_on_pool_hit=conditional,
                    eligibility=self.population.report, evaluation_id=self.population.identifier,
                    catalog_coverage={name:len(ids)/int(self.ctx.eligible.sum()) for name,ids in self.coverage.items()},
                    top_decile_popularity_share={name:h/t if t else None for name,(h,t) in self.counts.items()},
                    note='Unconditional macro metrics retain no-hit queries; pool_recall uses every supplied candidate.')


def generate_candidates(model, data, config, root):
    """One retriever -> standardized query-sorted Parquet + shared evaluation, no other model."""
    from .artifacts import ParquetSink, candidate_schema
    from .contracts import validate_candidates
    require_memory(cpu=len(data.population.users)*2000+config['query_batch']*config['retrieval_k']*128,
                   disk=len(data.population.users)*config['retrieval_k']*64,root=root,
                   stage='standardized candidate staging and per-query metrics')
    sink = ParquetSink(root/'candidates.parquet', schema=candidate_schema())
    metrics = MetricAccumulator(data.ctx, data.population, config)
    users = data.population.users
    allowed = set(map(int,users))
    fallback, query_times = 0, []
    for start in range(0,len(users),config['query_batch']):
        chunk = users[start:start+config['query_batch']]
        before = time.perf_counter()
        output = model.retrieve_many(chunk, config['retrieval_k'], config['item_block'])
        query_times.append((time.perf_counter()-before,len(chunk)))
        for user in chunk:
            records = output[int(user)]
            frame = pd.DataFrame(records, columns=['userId','movieId','score','rank','source','fallback'])
            if len(frame):
                frame['fallback'] = frame.fallback.fillna(False).astype(bool)
                frame = frame.astype({'userId':'int64','movieId':'int64','score':'float64','rank':'int32'})
            validate_candidates(frame, data.ctx, model.name, config['retrieval_k'], allowed)
            sink.add(frame)
            fallback += int(bool(records) and bool(records[0].get('fallback',False)))
            metrics.add(model.name, int(user), frame.movieId.tolist(), coverage_k=100)
        if start % (config['query_batch']*100) == 0:
            print(f'{model.name}: {min(start+len(chunk),len(users)):,}/{len(users):,} evaluation users')
    sink.close()
    summary = metrics.finish(root)
    seconds = sum(s for s,n in query_times)
    summary.update(candidate_rows=sink.rows, fallback_queries=fallback,
                   retrieval_seconds=seconds, mean_amortized_query_ms=1000*seconds/len(users),
                   amortized_query_ms_p95=float(np.quantile([1000*s/n for s,n in query_times],.95)))
    return summary
