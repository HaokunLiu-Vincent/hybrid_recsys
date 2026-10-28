"""Portable retrieval checkpoints built with the existing immutable/checksummed Store."""
from pathlib import Path
import json
import importlib.metadata
import numpy as np
from scipy import sparse
from .data import Context
from .runtime import Store, digest, fingerprint
from .artifacts import source_hashes
from .contracts import mapping_id,catalog_id
from .training import load_model
from .small_protocol import POLICY

VERSION='small_retrieval_v1'
FEATURES=('genres','user_genres','item_count','popularity','user_count','positive_count')


def critical_sources(model):
    root=Path(__file__).parent
    result=source_hashes(model)
    for filename in ['small_benchmark.py','retrieval_artifacts.py']:
        result[filename]=digest(root/filename)
    return result


def save_context(ctx,path):
    ctx.save(path)
    meta=json.loads((path/'context.json').read_text())
    meta['cutoff_semantics']='Maximum Train timestamp only; not a global boundary under per-user chronology'
    (path/'context.json').write_text(json.dumps(meta,indent=2)+'\n')
    sparse.save_npz(path/'observed.npz',ctx.observed)
    sparse.save_npz(path/'positive.npz',ctx.positive)
    np.savez(path/'context_features.npz',**{name:getattr(ctx,name) for name in FEATURES})


def load_context(path):
    path=Path(path)
    with np.load(path/'mappings.npz',allow_pickle=False) as arrays:
        users,items,eligible=[arrays[name].copy() for name in ['users','items','eligible']]
    with np.load(path/'context_features.npz',allow_pickle=False) as arrays:
        features={name:arrays[name].copy() for name in FEATURES}
    meta=json.loads((path/'context.json').read_text())
    observed,positive=[sparse.load_npz(path/(name+'.npz')) for name in ['observed','positive']]
    ctx=Context(users,items,observed,positive,features['genres'],meta['genre_names'],features['user_genres'],
                features['item_count'],features['popularity'],features['user_count'],features['positive_count'],
                eligible,meta['cutoff'],meta['train_rows'])
    if (observed.shape!=(len(users),len(items)) or positive.shape!=observed.shape or
        np.any(np.diff(users)<=0) or np.any(np.diff(items)<=0) or eligible.dtype!=np.bool_):
        raise ValueError('Invalid portable context mapping/shape')
    if observed.nnz!=ctx.train_rows or not np.isfinite(observed.data).all() or not np.all(positive.data==1):
        raise ValueError('Invalid serialized Train histories')
    if not np.array_equal(np.asarray((observed!=0).sum(axis=0)).ravel()>0,eligible):
        raise ValueError('Catalog must derive exclusively from serialized Train observations')
    if not np.array_equal(np.diff(positive.indptr),ctx.positive_count):
        raise ValueError('Positive-history counts disagree with serialized histories')
    return ctx


def checked_manifest(path):
    path=Path(path)
    if not path.is_dir():raise FileNotFoundError('Completed artifact missing: '+str(path))
    marker=json.loads((path/'COMPLETE.json').read_text())
    if any(Path(p).is_absolute() or '..' in Path(p).parts for p in marker['files']):
        raise ValueError('Unsafe artifact member path')
    Store(path.parent.parent,path.parent.name).load(path.name)
    actual={str(p.relative_to(path)) for p in path.rglob('*') if p.is_file()}
    if actual!=set(marker['files'])|{'COMPLETE.json'}:raise ValueError('Unlisted/missing artifact files')
    return json.loads((path/'manifest.json').read_text())


def load_checkpoint(path,model=None,split_manifest_sha256=None,expected_config=None):
    path=Path(path);manifest=checked_manifest(path)
    name=manifest['model_name']
    if model is not None and name!=model:raise ValueError('Checkpoint producer/model mismatch')
    if manifest['framework_version']!=VERSION or manifest['policy']!=POLICY:
        raise ValueError('Incompatible retrieval framework/policy')
    if manifest['critical_source_hashes']!=critical_sources(name):
        raise ValueError('Incompatible model/context source version; use the recorded source revision')
    for package,version in manifest['packages'].items():
        if importlib.metadata.version(package)!=version:
            raise ValueError('Incompatible checkpoint dependency version: '+package)
    if manifest['config_sha256']!=fingerprint(manifest['config']):raise ValueError('Configuration checksum mismatch')
    if expected_config is not None and manifest['config_sha256']!=fingerprint(expected_config):
        raise ValueError('Requested model configuration differs from checkpoint')
    if split_manifest_sha256 is not None and manifest['split_manifest_sha256']!=split_manifest_sha256:
        raise ValueError('Checkpoint belongs to another frozen dataset/split')
    if manifest['training_kind']!='external_train_per_user' or not manifest['training_cutoff'].startswith('per_user:'):
        raise ValueError('Primary training cutoffs must not be represented as one global timestamp')
    ctx=load_context(path)
    if mapping_id(ctx)!=manifest['mapping_id'] or catalog_id(ctx)!=manifest['catalog_id']:
        raise ValueError('Incompatible checkpoint mappings/catalog')
    model=load_model(ctx,name,path,device='cpu')
    if model.name!=name:model.name=name
    if name=='itemcf':
        if model.neighbors.shape!=(len(ctx.items),len(ctx.items)) or not np.isfinite(model.neighbors.data).all():
            raise ValueError('Invalid ItemCF neighbor checkpoint')
    return model,manifest
