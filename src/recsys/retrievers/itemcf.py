"""Binary positive cosine, exact bounded source-item blocks, full-history neighbor sum."""
import numpy as np
from scipy import sparse
from .base import Retriever, topk
from ..runtime import require_memory


class ItemCF(Retriever):
    def __init__(self, context, neighbors=None):
        super().__init__(context, 'itemcf')
        self.neighbors = neighbors

    def fit(self, neighbors_per_item=100, block=64):
        x = self.ctx.positive
        n = x.shape[1]
        require_memory(cpu=block*n*24 + x.nnz*16 + n*neighbors_per_item*16,
                       stage='ItemCF bounded sparse cosine blocks')
        norms = np.sqrt(np.asarray(x.sum(axis=0)).ravel())
        normed = x @ sparse.diags(1/np.maximum(norms, 1))
        normed = normed.tocsr()
        data = np.empty(n*neighbors_per_item, dtype=np.float32)
        indices = np.empty(n*neighbors_per_item, dtype=np.int32)
        indptr = np.zeros(n+1, dtype=np.int64)
        cursor = 0
        for start in range(0, n, block):
            similarity = (normed[:, start:start+block].T @ normed).tocsr()
            for row in range(similarity.shape[0]):
                a, b = similarity.indptr[row:row+2]
                ids, scores = similarity.indices[a:b], similarity.data[a:b]
                keep = (ids != start+row) & self.ctx.eligible[ids]
                ids, scores = topk(scores[keep], ids[keep], neighbors_per_item)
                end = cursor+len(ids)
                indices[cursor:end], data[cursor:end] = ids, scores
                cursor = end
                indptr[start+row+1] = cursor
            if start % (block*100) == 0:
                print(f'ItemCF cosine rows {min(start+block,n):,}/{n:,}')
        self.neighbors = sparse.csr_matrix((data[:cursor], indices[:cursor], indptr), shape=(n,n))
        self.neighbors.sort_indices()
        return self

    def retrieve(self, user_context, k=100, item_block=4096):
        u = self.ctx.user_index(user_context)
        if self.fallback(u):
            return super().retrieve(user_context, k, item_block)
        # The complete positive history participates; only a 1 x catalog row is dense.
        row = (self.ctx.positive[u] @ self.neighbors).toarray().ravel()
        ids = np.flatnonzero(self.ctx.eligible)
        ids = ids[~np.isin(ids, self.ctx.seen(u))]
        ids, scores = topk(row[ids], ids, k)
        return [dict(userId=int(user_context), movieId=int(self.ctx.items[i]), score=float(s),
                     rank=r+1, source=self.name, fallback=False) for r,(i,s) in enumerate(zip(ids,scores))]

    def scores(self, user, items):
        return (self.ctx.positive[user] @ self.neighbors[:, items]).toarray().ravel()

    def save(self, path):
        sparse.save_npz(path/'neighbors.npz', self.neighbors)
