"""Read-only pinned-registry resolution and checksummed offline model delivery."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import shutil
import tempfile
import time
import mlflow
from mlflow import MlflowClient
from .settings import MODEL_NAME,MODEL_VERSION,MODEL_URI,SCHEMA_ID,RELEASE_ID


def sha256(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):result.update(block)
    return result.hexdigest()


def relative_file(root,name):
    member=Path(name)
    if member.is_absolute() or '..' in member.parts:raise ValueError('Unsafe model member')
    root=Path(root).resolve();path=root/member;cursor=path
    while cursor!=root:
        if cursor.is_symlink():raise ValueError('Symlinks are not permitted in model assets')
        cursor=cursor.parent
    return path


def check_files(root,marker,extra=()):
    records=json.loads((Path(root)/marker).read_text())['files']
    actual={str(p.relative_to(root)) for p in Path(root).rglob('*') if p.is_file()}
    if actual!=set(records)|{marker}|set(extra):raise ValueError('Unlisted or missing deployment files')
    for name,info in records.items():
        path=relative_file(root,name)
        if path.stat().st_size!=info['bytes'] or sha256(path)!=info['sha256']:raise ValueError('Model file checksum mismatch')
    return records


def inspect_model(path,expected_release_id=RELEASE_ID,registry_tags=None):
    path=Path(path);descriptor=mlflow.models.Model.load(str(path/'MLmodel'))
    flavor=descriptor.flavors.get('python_function',{})
    release=relative_file(path,flavor['artifacts']['release']['path'])
    check_files(release,'COMPLETE.json')
    manifest=json.loads((release/'manifest.json').read_text())
    schema=json.loads((release/'ranker/feature_schema.json').read_text())
    if manifest.get('kind')!='complete_hybrid_recommender' or manifest['release_id']!=expected_release_id:
        raise ValueError('Unexpected complete model release')
    if manifest['feature_schema_sha256']!=SCHEMA_ID or len(schema['names'])!=53 or schema['sha256']!=SCHEMA_ID:
        raise ValueError('Unexpected 53-feature contract')
    if manifest['rrf_constant']!=60 or len(manifest['retrievers'])!=6 or manifest['test_evaluated'] is not False:
        raise ValueError('Unexpected inference/evaluation contract')
    if manifest['release_status']!='validation_candidate':raise ValueError('Unexpected release status')
    expected_inputs=[('userId','long'),('top_k','long')]
    expected_outputs=[('userId','long'),('movieId','long'),('rank','integer'),('score','double'),('fallback','string')]
    for columns,expected in [(descriptor.signature.inputs,expected_inputs),(descriptor.signature.outputs,expected_outputs)]:
        if [(c.name,c.type.name) for c in columns]!=expected:raise ValueError('MLflow input/output signature mismatch')
    for key in ['release_id','feature_schema_sha256','lock_sha256']:
        if descriptor.metadata.get(key)!=manifest[key]:raise ValueError('MLflow metadata/release mismatch')
    if registry_tags:
        for key in ['release_id','feature_schema_sha256','lock_sha256']:
            if registry_tags.get(key)!=manifest[key]:raise ValueError('Registry identity mismatch')
        if registry_tags['release_manifest_sha256']!=sha256(release/'manifest.json') or registry_tags['release_complete_sha256']!=sha256(release/'COMPLETE.json'):
            raise ValueError('Registered payload checksum mismatch')
    return manifest,dict(name=MODEL_NAME,version=MODEL_VERSION,release_status='validation_candidate',
        release_id=manifest['release_id'],protocol='per_user_70_15_15',retriever_count=6,rrf_constant=60,
        ranker='lightgbm_lambdamart',feature_schema_version=schema['version'],feature_schema_fingerprint=SCHEMA_ID,test_evaluated=False)


def registered_model(tracking_uri,destination):
    client=MlflowClient(tracking_uri=tracking_uri,registry_uri=tracking_uri)
    version=client.get_model_version(MODEL_NAME,MODEL_VERSION)
    if version.tags.get('release_id')!=RELEASE_ID:raise ValueError('Registry Version 1 has the wrong release identity')
    mlflow.set_tracking_uri(tracking_uri);mlflow.set_registry_uri(tracking_uri)
    path=Path(mlflow.artifacts.download_artifacts(artifact_uri=MODEL_URI,dst_path=str(destination)))
    manifest,info=inspect_model(path,registry_tags=version.tags)
    return path,manifest,info,dict(name=version.name,version=version.version,source=version.source,run_id=version.run_id,tags=version.tags)


def export_model(destination,tracking_uri):
    destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('Deployment export exists; no overwrite performed')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='recsys-export-',dir=destination.parent) as temporary:
        stage=Path(temporary);source,manifest,info,registry=registered_model(tracking_uri,stage/'download')
        shutil.copytree(source,stage/'model');target=stage/'model'
        deployment=dict(format='mlflow_offline_deployment_v1',model_uri=MODEL_URI,registered_model=registry,
            release_id=manifest['release_id'],feature_schema_sha256=SCHEMA_ID,release_lock_sha256=manifest['lock_sha256'],
            note='Unmodified Registered Model Version 1; contains Train histories; no public distribution.')
        (target/'deployment_manifest.json').write_text(json.dumps(deployment,indent=2)+'\n')
        files={str(p.relative_to(target)):dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted(target.rglob('*')) if p.is_file()}
        (target/'DEPLOYMENT_COMPLETE.json').write_text(json.dumps(dict(files=files),indent=2)+'\n')
        verify_export(target)
        if destination.exists():raise FileExistsError('Concurrent export; refusing replacement')
        target.rename(destination)
    return dict(path=str(destination),model_uri=MODEL_URI,release_id=manifest['release_id'],
        deployment_sha256=sha256(destination/'DEPLOYMENT_COMPLETE.json'),files=len(files))


def verify_export(path,expected_release_id=RELEASE_ID):
    path=Path(path);check_files(path,'DEPLOYMENT_COMPLETE.json')
    deployment=json.loads((path/'deployment_manifest.json').read_text());registry=deployment['registered_model']
    if deployment['format']!='mlflow_offline_deployment_v1' or deployment['model_uri']!=MODEL_URI or registry['name']!=MODEL_NAME or registry['version']!=MODEL_VERSION:
        raise ValueError('Deployment must select immutable Registered Model Version 1')
    if deployment['release_id']!=expected_release_id:raise ValueError('Deployment release differs from expected release')
    return inspect_model(path,expected_release_id,registry['tags'])


@dataclass
class LoadedModel:
    model: object
    info: dict


def load_model(settings):
    if settings.model_source=='local':
        path=settings.model_path
        _,info=verify_export(path,settings.expected_release_id)
    else:
        # Cache survives only this load; pyfunc retains native in-memory inference state.
        with tempfile.TemporaryDirectory(prefix='recsys-registry-startup-') as temporary:
            path,_,info,_=registered_model(settings.tracking_uri,temporary)
            if info['release_id']!=settings.expected_release_id:raise ValueError('Wrong expected release ID')
            model=mlflow.pyfunc.load_model(str(path))
        info['source_mode']='registry'
        return LoadedModel(model,info)
    model=mlflow.pyfunc.load_model(str(path));info['source_mode']='local'
    return LoadedModel(model,info)
