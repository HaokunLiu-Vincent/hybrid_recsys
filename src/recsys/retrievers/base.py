"""One exact eligible-catalog retrieval contract for every channel."""
import numpy as np
from ..runtime import require_memory

CHANNELS = ('popularity', 'itemcf', 'als', 'genre', 'lightgcn', 'two_tower')


def topk(scores, ids, k):
    ids, scores = np.asarray(ids), np.asarray(scores)
    ok = np.isfinite(scores)
    ids, scores = ids[ok], scores[ok]
    order = np.lexsort((ids, -scores))[:k]
    return ids[order], scores[order]


class Retriever:
    def __init__(self, context, name):
        self.ctx, self.name = context, name
        self.logs = []

    def scores(self, user, items):
        raise NotImplementedError

    def fallback(self, user):
        return user < 0 or self.ctx.positive_count[user] == 0

    def retrieve(self, user_context, k=100, item_block=4096):
        """user_context is a canonical external userId; scores include source and fallback provenance."""
        u = self.ctx.user_index(user_context)
        fallback = self.fallback(u)
        seen = self.ctx.seen(u)
        best_ids, best_scores = np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        require_memory(cpu=item_block*64, stage='exact chunked TopK')
        for start in range(0, len(self.ctx.items), item_block):
            ids = np.arange(start, min(start+item_block, len(self.ctx.items)))
            ids = ids[self.ctx.eligible[ids] & ~np.isin(ids, seen, assume_unique=True)]
            scores = self.ctx.popularity[ids] if fallback else self.scores(u, ids)
            best_ids, best_scores = topk(np.concatenate([best_scores, scores]),
                                        np.concatenate([best_ids, ids]), k)
        return [dict(userId=int(user_context), movieId=int(self.ctx.items[i]), score=float(s),
                     rank=r+1, source=self.name, fallback=bool(fallback))
                for r, (i, s) in enumerate(zip(best_ids, best_scores))]

    def retrieve_many(self, users, k=100, item_block=4096):
        return {int(u): self.retrieve(int(u), k, item_block) for u in users}


class VectorRetriever(Retriever):
    def __init__(self, context, name, user_vectors, item_vectors, device='cpu'):
        super().__init__(context, name)
        self.U = np.asarray(user_vectors, dtype=np.float32)
        self.V = np.asarray(item_vectors, dtype=np.float32)
        self.device = device
        if self.U.shape[0] != len(context.users) or self.V.shape[0] != len(context.items):
            raise ValueError('Canonical factor mapping shape mismatch')
        if not np.isfinite(self.U).all() or not np.isfinite(self.V).all():
            raise ValueError('Nonfinite embedding checkpoint')

    def scores(self, user, items):
        return self.V[items] @ self.U[user]

    def retrieve_many(self, users, k=100, item_block=4096):
        """Batched exact full-catalog scores; never allocate all-users x all-items."""
        if self.device == 'cuda':
            import torch
        if len(users) > 128:
            result = {}
            for start in range(0, len(users), 128):
                result.update(self.retrieve_many(users[start:start+128], k, item_block))
            return result
        ui = np.array([self.ctx.user_index(u) for u in users])
        cold = np.array([self.fallback(u) for u in ui])
        vectors = self.U[np.maximum(ui, 0)].copy()
        vectors[cold] = 0
        require_memory(cpu=len(users)*item_block*32, stage='batched exact TopK')
        use_gpu = self.device == 'cuda'
        if use_gpu:
            # Small query and catalog chunks only; inference VRAM remains bounded.
            q = torch.as_tensor(vectors, device='cuda')
        best = [(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)) for _ in users]
        for start in range(0, len(self.ctx.items), item_block):
            ids = np.arange(start, min(start+item_block, len(self.ctx.items)))
            if use_gpu:
                with torch.no_grad():
                    scores = (q @ torch.as_tensor(self.V[ids].T.copy(), device='cuda')).cpu().numpy()
            else:
                scores = vectors @ self.V[ids].T
            scores[cold] = self.ctx.popularity[ids]
            scores[:, ~self.ctx.eligible[ids]] = -np.inf
            for row, u in enumerate(ui):
                seen = self.ctx.seen(u)
                local = seen[(seen >= start) & (seen < start+len(ids))]-start
                scores[row, local] = -np.inf
                old_ids, old_scores = best[row]
                best[row] = topk(np.concatenate([old_scores, scores[row]]), np.concatenate([old_ids, ids]), k)
        return {int(user): [dict(userId=int(user), movieId=int(self.ctx.items[i]), score=float(s),
                                rank=r+1, source=self.name, fallback=bool(cold[row]))
                            for r, (i, s) in enumerate(zip(*best[row]))]
                for row, user in enumerate(users)}

    def save(self, path):
        np.savez(path/'vectors.npz', U=self.U, V=self.V)
