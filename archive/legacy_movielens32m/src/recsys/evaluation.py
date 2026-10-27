"""Bounded candidate shards, identical-budget comparisons, ablations and runtime evidence."""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from .candidate_fusion import fuse,rrf,diagnostics
from .data import relevance
from .metrics import query_metrics,summarize
from .ranking_features import features,attach_labels,distribution,FEATURES
from .retrievers.base import CHANNELS
from .runtime import require_memory,profile


def evaluate(models,ctx,future,config,store,stage,ranker=None,save_features=True):
    existing = store.load(stage)
    if existing:
        return json.loads((existing/'summary.json').read_text()),existing
    truth,eligibility = relevance(future,ctx)
    users = np.array(sorted(truth),dtype=np.int64)
    if not len(users):
        raise ValueError('No eligible future positives; evaluation population is empty')
    reports = {}
    def write(root):
        require_memory(disk=len(users)*600*len(FEATURES)*8,root=root,stage='evaluation shards')
        require_memory(cpu=config['query_batch']*600*len(FEATURES)*16+len(users)*10000,
                       stage='candidate features and metric rows')
        (root/'features').mkdir()
        metric_rows,coverage,concentration = [],{},{}
        timing = {s:[] for s in CHANNELS}
        diag_sums, unique, counts = {},{s:0 for s in CHANNELS},[]
        summary_reservoir = []
        reservoir_n = 0
        n_top = max(1,int(ctx.eligible.sum()*.1))
        pop_ids = np.flatnonzero(ctx.eligible)
        top_pop = set(ctx.items[pop_ids[np.lexsort((pop_ids,-ctx.popularity[pop_ids]))[:n_top]]].tolist())
        with profile() as total_profile:
            for shard,start in enumerate(range(0,len(users),config['query_batch'])):
                batch = users[start:start+config['query_batch']]
                outputs = {}
                for source in CHANNELS:
                    before = time.perf_counter()
                    outputs[source] = models[source].retrieve_many(batch,config['retrieval_k'],config['item_block'])
                    timing[source].append(dict(queries=len(batch),seconds=time.perf_counter()-before))
                    for user in batch:
                        candidates = outputs[source][int(user)]
                        # All channels must return the same budget, or the complete remaining catalog if fewer.
                        u = ctx.user_index(user)
                        budget = min(config['retrieval_k'],int(ctx.eligible.sum())-len(ctx.seen(u)))
                        if len(candidates) != budget:
                            raise ValueError(f'{source}: inconsistent candidate budget for {user}')
                pool = features(fuse(outputs),ctx)
                labeled = attach_labels(pool,future[future.userId.isin(batch)])
                if save_features:
                    labeled.to_parquet(root/'features'/f'part-{shard:05d}.parquet',index=False)
                # Deterministic bounded feature reservoir, spread across the complete population.
                stride = max(1,int(np.ceil(len(users)*600/config['feature_summary_rows'])))
                picked = labeled.iloc[np.arange(0,len(labeled),stride)][FEATURES]
                if reservoir_n < config['feature_summary_rows']:
                    picked = picked.head(config['feature_summary_rows']-reservoir_n)
                    summary_reservoir.append(picked)
                    reservoir_n += len(picked)
                batch_diag = diagnostics(outputs,{int(u):truth[int(u)] for u in batch})
                for key,value in batch_diag['overlap_jaccard_mean'].items():
                    diag_sums[key] = diag_sums.get(key,0.)+value*len(batch)
                for key,value in batch_diag['unique_relevant_interactions'].items():
                    unique[key] += value
                counts.extend(batch_diag['union_candidates'])
                for user,group in labeled.groupby('userId',sort=True):
                    rankings = {s:[c['movieId'] for c in outputs[s][int(user)]] for s in CHANNELS}
                    rankings['rrf'] = rrf(group)
                    for removed in CHANNELS:
                        rankings['rrf_without_'+removed] = rrf(group,[s for s in CHANNELS if s != removed])
                    if ranker is not None:
                        from .rankers.lightgbm_ranker import rank
                        rankings['lightgbm'] = rank(ranker,group)
                    for name,ranked in rankings.items():
                        row = dict(model=name,userId=int(user),cohort=ctx.cohort(user),
                                   **query_metrics(ranked,truth[int(user)]))
                        metric_rows.append(row)
                        coverage.setdefault(name,set()).update(ranked[:100] if name in CHANNELS else ranked[:20])
                        hits,total = concentration.get(name,(0,0))
                        concentration[name] = (hits+sum(i in top_pop for i in ranked[:20]),total+len(ranked[:20]))
                if shard % 20 == 0:
                    print(f'{stage}: {min(start+len(batch),len(users)):,}/{len(users):,} eligible users')
        metric_frame = pd.DataFrame(metric_rows)
        metric_frame.to_parquet(root/'per_query_metrics.parquet',index=False)
        summaries = {name:summarize(group,config['seed'],config['bootstrap'])
                     for name,group in metric_frame.groupby('model',sort=True)}
        conditional = {}
        for name in ['rrf','lightgbm']:
            rows = metric_frame[(metric_frame.model==name)&(metric_frame.retrieved_positive > 0)]
            if name in set(metric_frame.model):
                conditional[name] = summarize(rows,config['seed'],config['bootstrap'])
        fallback = {s:None for s in CHANNELS}
        # Count from persisted provenance, not infer model success from baseline scores.
        fallback_counts = {s:0 for s in CHANNELS}
        for p in sorted((root/'features').glob('*.parquet')):
            f = pd.read_parquet(p,columns=['userId']+[s+'_fallback' for s in CHANNELS])
            for s in CHANNELS:
                fallback_counts[s] += int(f.groupby('userId')[s+'_fallback'].max().sum())
        fallback = {s:dict(queries=fallback_counts[s],fraction=fallback_counts[s]/len(users)) for s in CHANNELS}
        reports.update(eligibility=eligibility,query_count=len(users),budget_per_channel=config['retrieval_k'],
            policy='B, snapshot Train-observed; all observed Train movies excluded; no history updates',
            metrics=summaries,conditional_on_pool_hit=conditional,
            catalog_coverage={n:len(ids)/int(ctx.eligible.sum()) for n,ids in coverage.items()},
            top_decile_popularity_share={n:h/t if t else None for n,(h,t) in concentration.items()},
            overlap_jaccard={k:v/len(users) for k,v in diag_sums.items()},
            unique_relevant_interactions=unique,
            union_candidate_count_quantiles=np.quantile(counts,[0,.1,.5,.9,1]).tolist(),
            fallback=fallback,
            retrieval_timing={s:dict(total_seconds=sum(x['seconds'] for x in values),
                     mean_query_ms=1000*sum(x['seconds'] for x in values)/len(users),
                     batch_amortized_query_ms_p50=float(np.median([1000*x['seconds']/x['queries'] for x in values])),
                     batch_amortized_query_ms_p95=float(np.quantile([1000*x['seconds']/x['queries'] for x in values],.95)))
                  for s,values in timing.items()},
            profile=total_profile,feature_summary=distribution(pd.concat(summary_reservoir,ignore_index=True)),
            feature_summary_method='deterministic strided sample across all eligible query shards; bounded rows',
            union_recall_ceiling=summaries['rrf'][0]['pool_recall'],
            comparison_population='identical eligible-positive user IDs, no-hit queries included',
            caveat='Latency is batch-amortized, not isolated online request latency. Coverage: channels Top100; ranked Top20.')
        (root/'summary.json').write_text(json.dumps(reports,indent=2,allow_nan=False))
    path = store.write(stage,write,dict(context_cutoff=ctx.cutoff,train_rows=ctx.train_rows,channels=list(CHANNELS)))
    return json.loads((path/'summary.json').read_text()),path


def load_feature_shards(path, columns=None):
    paths = sorted((Path(path)/'features').glob('*.parquet'))
    if not paths:
        raise FileNotFoundError('Missing candidate feature shards')
    import pyarrow.parquet as pq
    rows = sum(pq.ParquetFile(p).metadata.num_rows for p in paths)
    require_memory(cpu=rows*(len(columns) if columns else len(FEATURES)+8)*8*4,
                   stage='materialize candidate training table; lower sampled-unlabeled budget or increase RAM')
    return pd.concat([pd.read_parquet(p,columns=columns) for p in paths],ignore_index=True)


def evaluate_saved_ranker(model,feature_path,ctx,future,config,store,stage):
    old = store.load(stage)
    if old:
        return json.loads((old/'summary.json').read_text())
    from .rankers.lightgbm_ranker import rank
    truth,_ = relevance(future,ctx)
    rows,coverage = [],set()
    pop_ids = np.flatnonzero(ctx.eligible)
    top = set(ctx.items[pop_ids[np.lexsort((pop_ids,-ctx.popularity[pop_ids]))[:max(1,len(pop_ids)//10)]]])
    popular,total = 0,0
    with profile() as timing:
        for path in sorted((feature_path/'features').glob('*.parquet')):
            for user,frame in pd.read_parquet(path).groupby('userId',sort=True):
                ranked = rank(model,frame)
                coverage.update(ranked[:20])
                popular += sum(i in top for i in ranked[:20])
                total += len(ranked[:20])
                rows.append(dict(userId=int(user),cohort=ctx.cohort(user),**query_metrics(ranked,truth[int(user)])))
    frame = pd.DataFrame(rows)
    summary = dict(unconditional=summarize(rows,config['seed'],config['bootstrap']),
                   conditional_on_pool_hit=summarize(frame[frame.retrieved_positive > 0],config['seed'],config['bootstrap']),
                   catalog_coverage_top20=len(coverage)/ctx.eligible.sum(),
                   top_decile_popularity_share=popular/total if total else None,profile=timing)
    def write(p):
        frame.to_parquet(p/'per_query_metrics.parquet',index=False)
        (p/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    store.write(stage,write)
    return summary


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
