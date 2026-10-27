"""Shared positive-pair streams, unknown sampling, and reproducible neural training."""
import random
import numpy as np
import torch
from ..runtime import require_memory


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # CSR GPU kernels may lack deterministic implementations. Fail/warn visibly rather than promise bitwise replay.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


class PairStream:
    """Uniform unknown draws, rejecting EVERY observed rating (including low ratings)."""
    def __init__(self, ctx, seed):
        self.ctx, self.rng = ctx, np.random.default_rng(seed)
        require_memory(cpu=ctx.observed.nnz*24 + ctx.positive.nnz*24, stage='positive stream / unknown membership')
        self.rows = np.repeat(np.arange(len(ctx.users), dtype=np.int32), np.diff(ctx.positive.indptr))
        self.cols = ctx.positive.indices
        self.catalog = np.flatnonzero(ctx.eligible)
        self.keys = (np.repeat(np.arange(len(ctx.users), dtype=np.int64), np.diff(ctx.observed.indptr))
                     *len(ctx.items)+ctx.observed.indices)
        self.n_items = len(ctx.items)
        self.unknown_sampling = 'Uniform over snapshot Train-observed catalog, rejecting all observed user-item pairs'

    def unknown(self, users):
        draw = self.rng.choice(self.catalog, len(users))
        for _ in range(1000):
            keys = users.astype(np.int64)*self.n_items+draw
            pos = np.searchsorted(self.keys, keys)
            bad = (pos < len(self.keys)) & (self.keys[np.minimum(pos,len(self.keys)-1)] == keys)
            if not bad.any():
                return draw
            draw[bad] = self.rng.choice(self.catalog, int(bad.sum()))
        raise RuntimeError('Unknown sampler exhausted: user may have rated almost/all catalog; inspect data/config')

    def batches(self, size):
        order = self.rng.permutation(len(self.rows))
        for start in range(0, len(order), size):
            ix = order[start:start+size]
            users, items = self.rows[ix], self.cols[ix]
            yield users, items, self.unknown(users)


def as_tensor(a, device):
    return torch.as_tensor(a, dtype=torch.long, device=device)


def bpr_loss(user, positive, unknown):
    return torch.nn.functional.softplus((user*unknown).sum(-1)-(user*positive).sum(-1)).mean()


def require_device(ctx, c, device, graph=False):
    nodes, edges = len(ctx.users)+len(ctx.items), ctx.positive.nnz
    estimate = nodes*c['dim']*4*16 + c['batch_size']*c['dim']*4*20
    if graph:
        estimate += edges*2*12 + nodes*16
    require_memory(cpu=edges*64 if graph else edges*24, gpu=estimate if device=='cuda' else 0,
                   stage='LightGCN' if graph else 'Hybrid Two-Tower')
    return estimate
