"""Independent reusable model factories. No fitting occurs at import; configuration is explicit."""
import numpy as np
from scipy import sparse
from .retrievers.base import VectorRetriever
from .retrievers.popularity import Popularity
from .retrievers.itemcf import ItemCF
from .retrievers.genre import Genre
from .metrics import query_metrics

def load_model(ctx, name, path, device='cpu'):
    if name == 'popularity':
        if not np.array_equal(np.load(path/'positive_counts.npy'),ctx.popularity):
            raise ValueError('Popularity checkpoint/history mismatch')
        return Popularity(ctx)
    if name == 'itemcf':
        return ItemCF(ctx,sparse.load_npz(path/'neighbors.npz'))
    if name == 'genre':
        model = Genre(ctx)
        with np.load(path/'vectors.npz') as f:
            if not np.array_equal(model.U,f['U']) or not np.array_equal(model.V,f['V']):
                raise ValueError('Genre checkpoint/history mismatch')
        return model
    with np.load(path/'vectors.npz') as f:
        return VectorRetriever(ctx,name,f['U'],f['V'],device=device if name in ('lightgcn','two_tower') else 'cpu')


def validation_selector(truth, config):
    # Validation-only tuning panel, fixed deterministic users shared by neural models.
    users = np.array(sorted(truth),dtype=np.int64)
    if len(users) > config['early_stop_users']:
        users = np.sort(np.random.default_rng(config['seed']).choice(users,config['early_stop_users'],replace=False))
    if not len(users):
        raise ValueError('No eligible Validation positives for model selection')
    def evaluate(model):
        values = []
        for start in range(0,len(users),config['query_batch']):
            chunk = users[start:start+config['query_batch']]
            result = model.retrieve_many(chunk,config['retrieval_k'],config['item_block'])
            values.extend(query_metrics([r['movieId'] for r in result[int(u)]],truth[int(u)])['recall100'] for u in chunk)
        return np.mean(values)
    return evaluate,users


def fit_retriever(ctx, name, config, selector=None, checkpoint=None, device='cpu'):
    """Fit exactly one model. This is the sole algorithm dispatcher for old and new workflows."""
    public_name = name
    name = 'als' if name == 'implicit_als' else name
    if name == 'popularity':
        model = Popularity(ctx)
    elif name == 'itemcf':
        model = ItemCF(ctx).fit(**config['itemcf'])
    elif name == 'genre':
        model = Genre(ctx)
    elif name == 'als':
        from .retrievers.implicit_als import fit_als
        model = fit_als(ctx,config)
    elif name == 'lightgcn':
        from .retrievers.lightgcn import fit_lightgcn
        model = fit_lightgcn(ctx,config,selector,checkpoint,device)
    elif name == 'two_tower':
        from .retrievers.two_tower import fit_two_tower
        model = fit_two_tower(ctx,config,selector,checkpoint,device)
    else:
        raise ValueError(f'Unknown channel {name}')
    model.name = public_name
    return model
