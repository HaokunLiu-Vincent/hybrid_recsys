"""Portable inference-only releases; original completed artifacts remain immutable."""
from pathlib import Path
import importlib.metadata
import json
import platform
import shutil
import subprocess
import time
import numpy as np
import pandas as pd
from .contracts import CHANNELS,mapping_id,catalog_id
from .runtime import Store,digest,fingerprint
from .artifacts import dump_json
from .retrieval_artifacts import load_checkpoint,checked_manifest
from .rankers.small_lambdamart import load_ranker
from .retrieval_workflow import ROOT,identifier
from .inference import VERSION,HybridRecommender,load_hybrid_recommender

DEFAULT_CHECKPOINTS={s:ROOT/'artifacts'/s/'phase3_v1_seed42_main' for s in CHANNELS}
DEFAULT_RANKER=ROOT/'artifacts/lightgbm/phase4_v1'
CONTEXT_FILES=['context.json','context_features.npz','mappings.npz','observed.npz','positive.npz','train_cutoffs.parquet']
WEIGHTS={s:('positive_counts.npy' if s=='popularity' else 'neighbors.npz' if s=='itemcf' else 'vectors.npz') for s in CHANNELS}
PACKAGES=['numpy','pandas','scipy','pyarrow','psutil','PyYAML','pytz','torch','implicit','threadpoolctl','lightgbm','mlflow']


def reference_recommender(checkpoints=None,ranker=DEFAULT_RANKER):
    checkpoints=checkpoints or DEFAULT_CHECKPOINTS
    if set(checkpoints)!=set(CHANNELS):raise ValueError('Exactly six Full-Train checkpoints are required')
    models={};manifests={}
    for source in CHANNELS:models[source],manifests[source]=load_checkpoint(Path(checkpoints[source]),source)
    ctx=models['popularity'].ctx;first=manifests['popularity']
    for m in manifests.values():
        for key in ['split_manifest_sha256','training_kind','training_cutoff','catalog_id','mapping_id']:
            if m[key]!=first[key]:raise ValueError('Incompatible inference retriever '+key)
    booster,schema,rm=load_ranker(ranker)
    if rm['split_manifest_sha256']!=first['split_manifest_sha256']:raise ValueError('Ranker belongs to a different dataset')
    recommender=HybridRecommender(ctx,models,booster,schema,{s:manifests[s]['config'] for s in CHANNELS},rm['validation_evaluation_id'])
    return recommender,manifests,rm


def package_release(checkpoints=None,ranker=DEFAULT_RANKER,output_root=ROOT/'artifacts/releases',
                    parity_root=ROOT/'outputs/release_checks',run_id=None):
    checkpoints=checkpoints or DEFAULT_CHECKPOINTS;ranker=Path(ranker);run_id=identifier(run_id)
    if (Path(output_root)/run_id).exists() or (Path(parity_root)/run_id).exists():raise FileExistsError('Release/check run exists; choose a new identity')
    reference,models,rm=reference_recommender(checkpoints,ranker)
    # Label-independent: every serialized Train user plus one previously unknown positive ID.
    users=list(map(int,reference.ctx.users));unknown=int(reference.ctx.users.max())+1000000
    inputs=pd.DataFrame({'userId':users+[unknown],'top_k':20},dtype='int64')
    start=time.perf_counter();canonical=reference.predict(inputs);reference_seconds=time.perf_counter()-start
    first=models['popularity'];source_root=Path(__file__).parent
    code={str(p.relative_to(source_root)):digest(p) for p in source_root.rglob('*.py')}
    originals={s:dict(manifest_sha256=digest(Path(checkpoints[s])/'manifest.json'),completion_sha256=digest(Path(checkpoints[s])/'COMPLETE.json')) for s in CHANNELS}
    release_id=fingerprint(dict(retrievers=originals,ranker_sha256=digest(ranker/'model.txt'),schema_sha256=reference.schema['sha256'],
        application_source_sha256=code,lock_sha256=digest(ROOT/'uv.lock'),pyproject_sha256=digest(ROOT/'pyproject.toml')))
    for source in CHANNELS:
        for name in CONTEXT_FILES:
            if digest(Path(checkpoints[source])/name)!=digest(Path(checkpoints['popularity'])/name):
                raise ValueError('Shared inference context differs: '+source+'/'+name)
    metadata=dict(kind='complete_hybrid_recommender',version=VERSION,run_id=run_id,release_id=release_id,
        release_status='validation_candidate',test_evaluated=False,dataset=first['dataset_id'],split='small_v1',
        split_manifest_sha256=first['split_manifest_sha256'],evaluation_id=rm['validation_evaluation_id'],
        mapping_id=first['mapping_id'],catalog_id=first['catalog_id'],training_cutoff=first['training_cutoff'],policy=first['policy'],
        rrf_constant=60,retrieval_k=100,feature_schema_sha256=reference.schema['sha256'],feature_count=len(reference.schema['names']),
        retrievers=models,original_artifact_checksums=originals,ranker_manifest_sha256=digest(ranker/'manifest.json'),
        ranker_model_sha256=digest(ranker/'model.txt'),ranker_source_sha256=rm['source_sha256'],
        ranker_schema_source_sha256=rm['schema_source_sha256'],validation_metrics=rm['comparison'],
        runtime_packages={p:importlib.metadata.version(p) for p in PACKAGES},python=platform.python_version(),
        application_source_sha256=code,lock_sha256=digest(ROOT/'uv.lock'),pyproject_sha256=digest(ROOT/'pyproject.toml'),
        source_git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True).splitlines(),
        dvc_metadata_sha256={p:digest(ROOT/p) for p in ['dvc.yaml','dvc.lock','params.yaml','data/raw/ml-latest-small.dvc']},
        input_contract='Unique positive int64 userId and integer top_k in [1,100]; output follows request user order then rank.',
        output_columns=['userId','movieId','rank','score','fallback'],
        unknown_user_policy='Train-derived popularity, no invented history/embedding; bypass ranker; raw popularity score.',
        known_zero_positive_policy='Retained six-channel fallbacks and actual observed-low history features, then ranker; quality not validated for cold start.',
        privacy='Educational local package contains Train interaction histories. Not a public data distribution.',
        inference_inputs='Only serialized Train context, model weights, fixed schema and code; no held-out rows or cached candidates.')
    def writer(path):
        (path/'context').mkdir();(path/'ranker').mkdir()
        for name in CONTEXT_FILES:shutil.copyfile(Path(checkpoints['popularity'])/name,path/'context'/name)
        for source in CHANNELS:
            destination=path/'retrievers'/source;destination.mkdir(parents=True)
            shutil.copyfile(Path(checkpoints[source])/WEIGHTS[source],destination/WEIGHTS[source])
        for name in ['model.txt','feature_schema.json']:shutil.copyfile(ranker/name,path/'ranker'/name)
        for filename in code:
            destination=path/'src/recsys'/filename;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source_root/filename,destination)
        for name in ['pyproject.toml','uv.lock']:shutil.copyfile(ROOT/name,path/name)
        (path/'requirements.txt').write_text('\n'.join(p+'=='+version for p,version in metadata['runtime_packages'].items())+'\n')
        (path/'README.md').write_text(metadata['privacy']+'\n'+metadata['inference_inputs']+'\n'
            'Use Python '+metadata['python']+'; uv sync --locked --no-default-groups --group train --group rank --group mlops.\n'
            'from recsys.inference import load_hybrid_recommender; load_hybrid_recommender(RELEASE_PATH).recommend([1], 20)\n')
        dump_json(path/'manifest.json',metadata)
    release=Store(Path(output_root).parent,Path(output_root).name).write(run_id,writer)
    model=load_hybrid_recommender(release);start=time.perf_counter();actual=model.predict(inputs);inference_seconds=time.perf_counter()-start
    pd.testing.assert_frame_equal(canonical,actual,check_exact=True)
    checks=dict(release_id=release_id,release_manifest_sha256=digest(release/'manifest.json'),
        release_complete_sha256=digest(release/'COMPLETE.json'),queries=len(inputs),known_users=len(users),
        unknown_user_id=unknown,rows=len(actual),exact_prediction_parity=True,reference_seconds=reference_seconds,
        packaged_seconds=inference_seconds,selection='All Train-context users plus an unknown ID; no future labels used')
    def parity_writer(path):
        inputs.to_parquet(path/'inputs.parquet',index=False);canonical.to_parquet(path/'reference.parquet',index=False)
        actual.to_parquet(path/'packaged.parquet',index=False);dump_json(path/'manifest.json',checks)
    parity=Store(Path(parity_root).parent,Path(parity_root).name).write(run_id,parity_writer)
    return dict(gate='C',release=str(release),parity=str(parity),**checks)
