"""Query-level overlap and unique-positive contributions."""
import numpy as np

def diagnostics(outputs, truth):
    channels = tuple(outputs)
    overlap = {a+'|'+b:[] for i,a in enumerate(channels) for b in channels[i+1:]}
    unique = {s:0 for s in channels}
    counts = []
    for user, relevant in truth.items():
        sets = {s:{r['movieId'] for r in outputs[s].get(user,[])} for s in channels}
        counts.append(len(set().union(*sets.values())))
        for key in overlap:
            a,b = key.split('|')
            union = sets[a] | sets[b]
            overlap[key].append(len(sets[a]&sets[b])/len(union) if union else 0.)
        for s in channels:
            other = set().union(*(sets[t] for t in channels if t != s))
            unique[s] += len((sets[s]-other)&relevant)
    return dict(overlap_jaccard_mean={k:float(np.mean(v)) if v else 0. for k,v in overlap.items()},
                unique_relevant_interactions=unique,
                union_candidates=counts)
