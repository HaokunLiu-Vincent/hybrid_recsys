"""Exact full-graph LightGCN with one propagation/adjoint per epoch, microbatched BPR.

Z=P(A)E, P(A)=(I+A+...+A^L)/(L+1). The graph is fixed and symmetric.
Accumulate dLoss/dZ across every positive edge in bounded microbatches; then
compute dLoss/dE=P(A)dLoss/dZ exactly using sparse matrix multiplication.
One Adam step per epoch is an explicitly large-batch optimizer, NOT ordinary
per-minibatch SGD. This avoids full propagation per loss minibatch and avoids
edge-by-embedding tensors. FP32 only. No sampled subgraph or truncated histories.
"""
import numpy as np
import torch
from scipy import sparse
from .base import VectorRetriever
from .neural import PairStream, seed_all, as_tensor, bpr_loss, require_device


def propagation(graph, embedding, layers):
    result, current = embedding.clone(), embedding
    for _ in range(layers):
        current = torch.sparse.mm(graph, current)
        result = result+current
    return result/(layers+1)


def graph_tensor(ctx, device):
    p = ctx.positive
    graph = sparse.bmat([[None,p], [p.T,None]], format='csr', dtype=np.float32)
    degree = np.asarray(graph.sum(axis=1)).ravel()
    inv = np.zeros_like(degree)
    np.divide(1., np.sqrt(degree), out=inv, where=degree > 0)
    graph = (sparse.diags(inv) @ graph @ sparse.diags(inv)).tocsr()
    graph.sort_indices()
    return torch.sparse_csr_tensor(torch.as_tensor(graph.indptr.astype(np.int64), device=device),
           torch.as_tensor(graph.indices.astype(np.int64), device=device),
           torch.as_tensor(graph.data, device=device), size=graph.shape, device=device, check_invariants=True)


def fit_lightgcn(ctx, config, evaluate=None, checkpoint=None, device='cpu'):
    c = config['lightgcn']
    seed_all(config['seed'])
    require_device(ctx, c, device, graph=True)
    stream = PairStream(ctx, config['seed'])
    graph = graph_tensor(ctx, device)
    nu = len(ctx.users)
    base = torch.nn.Parameter(torch.randn(nu+len(ctx.items), c['dim'], device=device)*.05)
    opt = torch.optim.Adam([base], lr=c['lr'])
    best, stale, logs = -np.inf, 0, []
    best_vectors = None
    for epoch in range(c['epochs']):
        with torch.no_grad():
            z = propagation(graph, base, c['layers'])
        z.requires_grad_(True)
        total = 0.
        for users, positive, unknown in stream.batches(c['batch_size']):
            u, p, n = [as_tensor(a, device) for a in (users,positive+nu,unknown+nu)]
            loss = bpr_loss(z[u],z[p],z[n])
            if not torch.isfinite(loss):raise FloatingPointError('LightGCN nonfinite BPR loss')
            weight = len(users)/len(stream.rows)
            (loss*weight).backward()
            total += float(loss.detach())*weight
        with torch.no_grad():
            gradient = propagation(graph, z.grad, c['layers'])
            gradient.add_(base, alpha=2*c['regularization']/base.numel())
            if not torch.isfinite(gradient).all():raise FloatingPointError('LightGCN nonfinite gradient')
        opt.zero_grad(set_to_none=True)
        base.grad = gradient
        opt.step()
        with torch.no_grad():
            final = propagation(graph, base, c['layers']).cpu().numpy()
        model = VectorRetriever(ctx, 'lightgcn', final[:nu], final[nu:], device=device)
        score = float(evaluate(model)) if evaluate else -total
        logs.append(dict(epoch=epoch+1, mean_bpr=total, selection_score=score,
                         optimizer_steps=epoch+1, positive_edges=len(stream.rows)))
        print('LightGCN', logs[-1])
        if score > best:
            best, stale = score, 0
            best_vectors = (model.U.copy(), model.V.copy())
        else:
            stale += 1
        if checkpoint:
            checkpoint(epoch+1, model, logs, dict(base=base.detach().cpu(), optimizer=opt.state_dict(),
                                                 rng=stream.rng.bit_generator.state))
        del z, gradient
        if evaluate and stale >= c['patience']:
            break
    model = VectorRetriever(ctx, 'lightgcn', *best_vectors, device=device)
    model.logs = logs
    return model
