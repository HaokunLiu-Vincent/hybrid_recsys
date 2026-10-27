"""Bounded Parquet access for LightGBM Sequence; only label and query vectors are resident."""
from pathlib import Path
import bisect
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from ..runtime import require_memory


class ParquetRows:
    """Random rows/slices over row groups; at most one feature row group is cached."""
    def __init__(self, paths, columns, batch_size=4096):
        self.columns, self.batch_size = list(columns), batch_size
        self.groups, self.ends = [], []
        total = 0
        for path in paths:
            pf = pq.ParquetFile(path)
            for group in range(pf.metadata.num_row_groups):
                n = pf.metadata.row_group(group).num_rows
                self.groups.append((Path(path),group,n))
                total += n
                self.ends.append(total)
        self.total, self.cached_index, self.cached = total, None, None
        self.max_group = max((n for p,g,n in self.groups),default=0)
        require_memory(cpu=max(self.max_group,batch_size)*len(columns)*8*4,stage='LightGBM Parquet sequence')

    def __len__(self):
        return self.total

    def _group(self, index):
        if self.cached_index != index:
            path,group,_ = self.groups[index]
            self.cached = np.ascontiguousarray(pq.ParquetFile(path).read_row_group(group,columns=self.columns)
                                               .to_pandas().to_numpy(dtype=np.float64))
            self.cached_index = index
        return self.cached

    def __getitem__(self, index):
        if isinstance(index,(int,np.integer)):
            if index < 0 or index >= self.total:
                raise IndexError(index)
            group = bisect.bisect_right(self.ends,int(index))
            start = self.ends[group-1] if group else 0
            return self._group(group)[int(index)-start].copy()
        if isinstance(index,slice):
            start,stop,step = index.indices(self.total)
            indices = np.arange(start,stop,step,dtype=np.int64)
        else:
            indices = np.asarray(index,dtype=np.int64)
        result = np.empty((len(indices),len(self.columns)),dtype=np.float64)
        if not len(indices):
            return result
        if (indices < 0).any() or (indices >= self.total).any():
            raise IndexError('Sequence index outside data')
        groups = np.searchsorted(self.ends,indices,side='right')
        for group in np.unique(groups):
            mask = groups == group
            offset = self.ends[group-1] if group else 0
            result[mask] = self._group(int(group))[indices[mask]-offset]
        return np.ascontiguousarray(result)


def labels_and_groups(paths):
    """One-dimensional labels/groups only; feature matrices are never concatenated here."""
    labels,groups = [],[]
    last_user, count = None,0
    previous = None
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16384,columns=['userId','label']):
            frame = batch.to_pandas()
            if not frame.userId.is_monotonic_increasing:
                raise ValueError('Ranking query rows are not sorted')
            labels.append(frame.label.to_numpy(dtype=np.int8))
            for user,n in frame.groupby('userId',sort=False).size().items():
                user = int(user)
                if previous is not None and user < previous:
                    raise ValueError('Ranking data must preserve contiguous, sorted query groups')
                previous = user
                if last_user is not None and user != last_user:
                    groups.append(count)
                    count = 0
                last_user = user
                count += int(n)
    if last_user is not None:
        groups.append(count)
    return np.concatenate(labels) if labels else np.empty(0,dtype=np.int8),np.asarray(groups,dtype=np.int32)
