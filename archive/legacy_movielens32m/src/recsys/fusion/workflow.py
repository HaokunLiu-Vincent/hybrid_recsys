"""Fusion consumes compatible saved candidates; this module never fits retrievers."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from ..artifacts import read_artifact, artifact_reference, iter_queries, ParquetSink, dump_json, source_hashes
from ..contracts import CHANNELS, compatible, validate_candidates, mapping_id, catalog_id
from ..data import FrozenData, EXPECTED_ROWS
from ..evaluation import MetricAccumulator
from ..features import candidate_features, feature_columns
from ..rankers.features import attach_labels, distribution
from ..runtime import Store, fingerprint, digest, profile, require_memory
from .candidates import fuse
from .rrf import rrf
from .diagnostics import diagnostics


def joined_queries(paths, population):
    """Six sorted streams, one query at a time; no all-user candidate concatenation."""
    streams = {s:iter(iter_queries(Path(p)/'candidates.parquet')) for s,p in paths.items()}
    next_row = {s:next(stream,None) for s,stream in streams.items()}
    for user in population.users:
        output = {}
        for source,stream in streams.items():
            current = next_row[source]
            if current is not None and current[0] < user:
                raise ValueError('Candidate query outside the declared fixed population')
            if current is not None and current[0] == user:
                output[source] = current[1]
                next_row[source] = next(stream,None)
            else:
                output[source] = pd.DataFrame(columns=['userId','movieId','score','rank','source','fallback'])
        yield int(user),output
    if any(value is not None for value in next_row.values()):
        raise ValueError('Candidate artifact contains undeclared users')


class FusionExperiment:
    def __init__(self, project, candidates, config, expected=EXPECTED_ROWS, opened_test_data=None):
        self.project = Path(project)
        self.paths = {s:Path(p) for s,p in candidates.items()}
        self.config = config
        self.manifests = {s:read_artifact(p) for s,p in self.paths.items()}
        self.reference = compatible(self.manifests)
        first = self.reference
        if first['partition'] == 'test':
            if opened_test_data is None or opened_test_data.partition != 'test':
                raise PermissionError('Test fusion requires an explicitly opened final-evaluation context')
            self.data = opened_test_data
        else:
            self.data = FrozenData(project, expected).prepare(first['training_kind'],
                first.get('snapshot_cutoff_utc'),first.get('snapshot_label_end_utc'))
        for key,value in dict(split_manifest_sha256=self.data.split_sha,mapping_id=mapping_id(self.data.ctx),
                              catalog_id=catalog_id(self.data.ctx),evaluation_id=self.data.population.identifier).items():
            if first[key] != value:
                raise ValueError(f'Saved candidates incompatible with current {key}')
        for source,p in self.paths.items():
            if digest(p/'candidates.parquet') != self.manifests[source]['artifact_checksum']:
                raise IOError(f'Candidate checksum mismatch for {source}')
        self.run_id = fingerprint(dict(inputs={s:artifact_reference(p) for s,p in self.paths.items()},
                                       config=config,source_hashes=source_hashes('fusion'),version='fusion_v2'))[:20]
        self.store = Store(self.project/'results'/'fusion'/first['partition'],self.run_id)

    def run(self):
        old = self.store.load('fusion')
        if old:
            return old
        config = self.config
        upper_rows = sum(m['candidate_rows'] for m in self.manifests.values())
        estimated_bytes = upper_rows*len(feature_columns())*8
        require_memory(cpu=config['query_batch']*600*len(feature_columns())*16+len(self.data.population.users)*15000,
                       disk=estimated_bytes,root='/tmp',stage='bounded fusion features and metric staging')
        require_memory(disk=estimated_bytes,root=self.store.root,stage='fusion Drive persistence reserve')
        # Training fields remain cutoff-only. Future rating information is attached only as label-side columns.
        def write(root):
            sink = ParquetSink(root/'features.parquet',batch_queries=config['query_batch'])
            metrics = MetricAccumulator(self.data.ctx,self.data.population,config)
            overlaps,unique,counts = {},{s:0 for s in CHANNELS},[]
            reservoir, reservoir_rows = [],0
            stride = max(1,int(np.ceil(len(self.data.population.users)*600/100000)))
            future_by_user = self.data.future.groupby('userId',sort=False)
            with profile() as timing:
                for user,frames in joined_queries(self.paths,self.data.population):
                    outputs = {}
                    for source,frame in frames.items():
                        validate_candidates(frame,self.data.ctx,source,self.reference['top_k'],{user})
                        outputs[source] = {user:frame.to_dict('records')}
                        metrics.add(source,user,frame.movieId.tolist(),coverage_k=100)
                    pool = candidate_features(fuse(outputs,CHANNELS,k=self.reference['top_k'],rrf_constant=config.get('rrf_constant',60)),self.data.ctx)
                    future = future_by_user.get_group(user)
                    labeled = attach_labels(pool,future)
                    sink.add(labeled)
                    if reservoir_rows < 100000:
                        sample = pool.iloc[::stride][feature_columns()].head(100000-reservoir_rows)
                        reservoir.append(sample)
                        reservoir_rows += len(sample)
                    metrics.add('rrf',user,rrf(pool,CHANNELS,config.get('rrf_constant',60)))
                    for removed in CHANNELS:
                        keep = [s for s in CHANNELS if s != removed]
                        metrics.add('rrf_without_'+removed,user,rrf(pool,keep,config.get('rrf_constant',60)))
                    d = diagnostics(outputs,{user:self.data.population.truth[user]})
                    for key,value in d['overlap_jaccard_mean'].items():
                        overlaps[key] = overlaps.get(key,0)+value
                    for key,value in d['unique_relevant_interactions'].items():
                        unique[key] += value
                    counts.extend(d['union_candidates'])
                sink.close()
                summary = metrics.finish(root)
            n = len(self.data.population.users)
            summary.update(profile=timing,overlap_jaccard={k:v/n for k,v in overlaps.items()},
                unique_relevant_interactions=unique,
                union_candidate_count_quantiles=np.quantile(counts,[0,.1,.5,.9,1]).tolist(),
                union_recall_ceiling=summary['metrics']['rrf'][0]['pool_recall'],
                feature_summary=distribution(pd.concat(reservoir,ignore_index=True)),
                feature_summary_method='bounded deterministic strided sample across the full query population')
            dump_json(root/'metrics.json',summary)
            manifest = dict(self.reference,model_name='fusion',run_id=self.run_id,
                source_hashes=source_hashes('fusion'),input_candidates={s:artifact_reference(p) for s,p in self.paths.items()},
                feature_columns=feature_columns(),config=config,artifact_checksum=digest(root/'features.parquet'),
                candidate_rows=sink.rows)
            dump_json(root/'manifest.json',manifest)
        return self.store.write('fusion',write,self.reference)
