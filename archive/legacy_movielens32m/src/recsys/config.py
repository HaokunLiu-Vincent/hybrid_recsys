"""Explicit experiment defaults: all Train rows, no hidden caps or sampled evaluation."""
from copy import deepcopy

DEFAULT = dict(seed=42, cpu_threads=4, retrieval_k=100, query_batch=64, item_block=4096,
               bootstrap=300, internal_tail_days=30, unlabeled_per_query=100,
               feature_summary_rows=100_000, early_stop_users=2000,
               itemcf=dict(neighbors_per_item=100, block=64),
               als=dict(factors=64,iterations=15,regularization=.05,positive_confidence=20.,low_confidence=5.),
               lightgcn=dict(dim=64,layers=2,epochs=30,batch_size=8192,lr=.01,regularization=.001,patience=4),
               two_tower=dict(dim=64,hidden=128,epochs=15,batch_size=4096,lr=.001,regularization=.0001,patience=3),
               ranker=dict(trees=500,num_leaves=31,lr=.05,patience=40))


def default_config():
    return deepcopy(DEFAULT)


def load_model_config(config_dir, model):
    """Readable JSON -> existing algorithm configuration; aliases only at the legacy boundary."""
    from pathlib import Path
    import json
    from .contracts import CHANNELS
    if model not in CHANNELS:
        raise ValueError(f'Unknown retrieval model {model}')
    root = Path(config_dir)
    common = json.loads((root/'common.json').read_text())
    specific = json.loads((root/(model+'.json')).read_text())
    if specific['model_name'] != model:
        raise ValueError('Configuration model_name mismatch')
    engine = default_config()
    engine.update(common)
    key = 'als' if model == 'implicit_als' else model
    if specific['hyperparameters']:
        engine[key] = specific['hyperparameters']
    return engine
