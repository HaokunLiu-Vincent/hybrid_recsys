"""Readable source provenance, verified immutable artifacts and candidate streaming."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from .runtime import Store, digest, fingerprint


def source_hashes(model=None, root=None):
    """Hash shared protocol code plus this producer, so unrelated model edits do not invalidate it."""
    root = Path(root) if root else Path(__file__).parent
    common = {'__init__.py','data.py','contracts.py','artifacts.py','runtime.py','metrics.py',
              'evaluation.py','features.py','retrievers/base.py'}
    if model in ('popularity','genre','itemcf','implicit_als','lightgcn','two_tower'):
        names = common | {'training.py',f'retrievers/{model}.py'}
        if model in ('lightgcn','two_tower'): names.add('retrievers/neural.py')
    elif model == 'fusion':
        names = common | {'fusion/candidates.py','fusion/rrf.py','fusion/diagnostics.py',
                          'rankers/features.py'}
    elif model == 'lightgbm':
        names = common | {'rankers/features.py','rankers/dataset.py','rankers/lightgbm_ranker.py',
                          'fusion/rrf.py'}
    else:
        names = {str(p.relative_to(root)) for p in root.rglob('*.py')}
    return {name:digest(root/name) for name in sorted(names)}


def read_artifact(path):
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f'Artifact unavailable: {path}; run its producer notebook first')
    Store(path.parent, path.name).load('.')
    manifest = json.loads((path/'manifest.json').read_text())
    return manifest


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=str)+'\n')


def candidate_schema():
    return pa.schema([('userId', pa.int64()), ('movieId', pa.int64()), ('score', pa.float64()),
                      ('rank', pa.int32()), ('source', pa.string()), ('fallback', pa.bool_())])


class ParquetSink:
    """Bounded list of complete-query frames. Never concatenate the full candidate table."""
    def __init__(self, path, batch_queries=64, schema=None):
        self.path, self.batch_queries, self.schema = Path(path), batch_queries, schema
        self.frames, self.writer, self.rows = [], None, 0

    def add(self, frame):
        if len(frame):
            self.frames.append(frame)
        if len(self.frames) >= self.batch_queries:
            self.flush()

    def flush(self):
        if not self.frames:
            return
        frame = pd.concat(self.frames, ignore_index=True)
        table = pa.Table.from_pandas(frame, schema=self.schema, preserve_index=False)
        if self.writer is None:
            self.schema = table.schema
            self.writer = pq.ParquetWriter(self.path, self.schema, compression='zstd')
        self.writer.write_table(table, row_group_size=16384)
        self.rows += len(frame)
        self.frames.clear()

    def close(self):
        self.flush()
        if self.writer is not None:
            self.writer.close()
        elif self.schema is not None:
            pq.write_table(pa.Table.from_batches([], schema=self.schema), self.path)
        else:
            raise ValueError('No rows were generated for the artifact')


def iter_queries(path, columns=None):
    """Read complete queries in increasing ID order, even across row-group boundaries."""
    carry = None
    last = None
    for batch in pq.ParquetFile(path).iter_batches(batch_size=16384, columns=columns):
        frame = batch.to_pandas()
        if carry is not None:
            frame = pd.concat([carry, frame], ignore_index=True)
        if frame.empty:
            continue
        if not frame.userId.is_monotonic_increasing:
            raise ValueError(f'Query rows must be ordered by userId: {path}')
        end_user = frame.userId.iloc[-1]
        carry = frame[frame.userId == end_user].copy()
        for user, group in frame[frame.userId != end_user].groupby('userId', sort=False):
            if last is not None and user <= last:
                raise ValueError('Query appears in multiple noncontiguous groups')
            last = int(user)
            yield int(user), group.reset_index(drop=True)
    if carry is not None and len(carry):
        user = int(carry.userId.iloc[0])
        if last is not None and user <= last:
            raise ValueError('Duplicate query group')
        yield user, carry.reset_index(drop=True)


def artifact_reference(path):
    path = Path(path)
    return dict(path=str(path), manifest_sha256=digest(path/'manifest.json'))
