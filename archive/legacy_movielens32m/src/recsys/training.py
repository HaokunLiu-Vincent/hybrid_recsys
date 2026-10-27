"""Stage-scoped retriever persistence: resumable completed models, immutable epoch snapshots."""
from pathlib import Path
import json
import numpy as np
from scipy import sparse
from .retrievers.base import CHANNELS, VectorRetriever
from .retrievers.popularity import Popularity
from .retrievers.itemcf import ItemCF
from .retrievers.genre import Genre
from .runtime import profile,environment
from .metrics import query_metrics


def load_model(ctx, name, path, device='cuda'):
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


def train_channels(ctx, config, store, namespace, truth=None, device='cuda', names=CHANNELS):
    """Snapshots receive truth=None: fixed epoch counts, never internal future labels for stopping."""
    models, failures = {}, {}
    selector, selection_users = validation_selector(truth,config) if truth else (None,[])
    for name in names:
        stage = f'{namespace}/{name}'
        try:
            saved = store.load(stage)
            if saved:
                models[name] = load_model(ctx,name,saved,device)
                print(f'Reuse verified checkpoint: {stage}')
                continue
            def checkpoint(epoch,model,logs,state):
                # Each immutable epoch exports a reusable embedding index. Restart resumes completed channels;
                # an interrupted neural channel can be restarted from scratch with the identical seed.
                import torch
                def write(p):
                    model.save(p)
                    torch.save(state,p/'training_state.pt')
                    (p/'training_log.json').write_text(json.dumps(logs))
                store.write(f'{namespace}/epochs/{name}-{epoch:03d}',write,
                            dict(epoch=epoch,cutoff=ctx.cutoff,channel=name))
            with profile() as timing:
                model = fit_retriever(ctx, name, config, selector, checkpoint, device)
            def write_model(p):
                model.save(p)
                (p/'training.json').write_text(json.dumps(dict(channel=name,train_rows=ctx.train_rows,
                         cutoff=ctx.cutoff,config=config,profile=timing,environment=environment('/tmp'),logs=model.logs,
                         best_epoch=max(model.logs,key=lambda x:x['selection_score'])['epoch'] if model.logs else None,
                         tuning_population=list(map(int,selection_users)),
                         selection='Validation Recall@100' if selector else 'fixed epoch count; no label-side tuning'),indent=2))
            store.write(stage,write_model,dict(cutoff=ctx.cutoff,channel=name))
            models[name] = model
        except (MemoryError,OSError,RuntimeError,ImportError,ValueError) as exc:
            failures[name] = f'{type(exc).__name__}: {exc}'
            print(f'INCOMPLETE {stage}: {failures[name]}')
            # Preserve the actual failure; never replace a failed channel with a baseline.
    if failures:
        raise RuntimeError(f'Incomplete six-channel milestone; completed checkpoints preserved: {failures}')
    return models


def load_channels(ctx, store, namespace, device='cuda'):
    models = {}
    for name in CHANNELS:
        path = store.load(f'{namespace}/{name}')
        if path is None:
            raise FileNotFoundError(f'Missing selected model: {namespace}/{name}; final mode never trains')
        models[name] = load_model(ctx,name,path,device)
    return models


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
