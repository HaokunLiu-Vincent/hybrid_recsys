"""Hybrid ID + cutoff-frozen genre towers, BPR proxy objective with unknown draws."""
import numpy as np
import torch
from .base import VectorRetriever
from .neural import PairStream, seed_all, as_tensor, bpr_loss, require_device


class Towers(torch.nn.Module):
    def __init__(self, ctx, c):
        super().__init__()
        d, g = c['dim'], ctx.genres.shape[1]
        self.users = torch.nn.Embedding(len(ctx.users),d)
        self.items = torch.nn.Embedding(len(ctx.items),d)
        self.user_mlp = torch.nn.Sequential(torch.nn.Linear(d+g,c['hidden']), torch.nn.ReLU(),
                                             torch.nn.Linear(c['hidden'],d))
        self.item_mlp = torch.nn.Sequential(torch.nn.Linear(d+g,c['hidden']), torch.nn.ReLU(),
                                             torch.nn.Linear(c['hidden'],d))
        torch.nn.init.normal_(self.users.weight, std=.05)
        torch.nn.init.normal_(self.items.weight, std=.05)

    def user(self, ids, genres):
        return torch.nn.functional.normalize(self.user_mlp(torch.cat([self.users(ids),genres],dim=1)),dim=1)

    def item(self, ids, genres):
        return torch.nn.functional.normalize(self.item_mlp(torch.cat([self.items(ids),genres],dim=1)),dim=1)


def export_vectors(net, ctx, device, batch=8192):
    result = []
    with torch.no_grad():
        for which, genres in [('user',ctx.user_genres), ('item',ctx.genres)]:
            chunks = []
            for start in range(0, len(genres), batch):
                ix = np.arange(start,min(start+batch,len(genres)))
                chunks.append(getattr(net,which)(as_tensor(ix,device),
                    torch.as_tensor(genres[ix],device=device)).cpu().numpy())
            result.append(np.concatenate(chunks))
    return VectorRetriever(ctx, 'two_tower', *result, device=device)


def fit_two_tower(ctx, config, evaluate=None, checkpoint=None, device='cpu'):
    c = config['two_tower']
    seed_all(config['seed'])
    require_device(ctx, c, device)
    stream = PairStream(ctx, config['seed'])
    net = Towers(ctx,c).to(device)
    optimizer = torch.optim.AdamW(net.parameters(),lr=c['lr'],weight_decay=c['regularization'])
    best, stale, logs, vectors = -np.inf, 0, [], None
    for epoch in range(c['epochs']):
        total = 0.
        for users,positive,unknown in stream.batches(c['batch_size']):
            u,p,n = [as_tensor(a,device) for a in (users,positive,unknown)]
            ug = torch.as_tensor(ctx.user_genres[users],device=device)
            pg = torch.as_tensor(ctx.genres[positive],device=device)
            ng = torch.as_tensor(ctx.genres[unknown],device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = bpr_loss(net.user(u,ug),net.item(p,pg),net.item(n,ng))
            if not torch.isfinite(loss):raise FloatingPointError('Two-Tower nonfinite BPR loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()):
                raise FloatingPointError('Two-Tower nonfinite gradient')
            optimizer.step()
            total += float(loss.detach())*len(users)/len(stream.rows)
        model = export_vectors(net,ctx,device)
        score = float(evaluate(model)) if evaluate else -total
        logs.append(dict(epoch=epoch+1,mean_bpr=total,selection_score=score,positive_edges=len(stream.rows)))
        print('Two-Tower',logs[-1])
        if score > best:
            best, stale, vectors = score, 0, (model.U.copy(), model.V.copy())
        else:
            stale += 1
        if checkpoint:
            checkpoint(epoch+1,model,logs,dict(network=net.state_dict(),optimizer=optimizer.state_dict(),
                                            rng=stream.rng.bit_generator.state))
        if evaluate and stale >= c['patience']:
            break
    model = VectorRetriever(ctx,'two_tower',*vectors,device=device)
    model.logs = logs
    return model
