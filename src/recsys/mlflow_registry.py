"""Register/resolve complete releases with candidate-only aliases and immutable identity."""
from pathlib import Path
import json
import re
import tempfile
import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models import ModelSignature
from mlflow.types import Schema,ColSpec
from .runtime import digest
from .retrieval_artifacts import checked_manifest
from .inference import load_hybrid_recommender
from .mlflow_tracking import client_for,experiment,RELEASE


def candidate_target(client,name):
    version=client.get_registered_model(name).aliases.get('candidate')
    return client.get_model_version(name,version) if version is not None else None


def release_tags(manifest):
    return dict(release_id=manifest['release_id'],release_status='validation_candidate',dataset='movielens_latest_small',
        split='small_v1',protocol='per_user_70_15_15',ranking_model='lightgbm_lambdamart',retriever_count='6',
        rrf_constant='60',feature_count=str(manifest['feature_count']),test_evaluated='false',
        feature_schema_sha256=manifest['feature_schema_sha256'],split_manifest_sha256=manifest['split_manifest_sha256'],
        source_git_revision=manifest['source_git_revision'],lock_sha256=manifest['lock_sha256'])


def register_release(release,parity,backfill_summary,name='MovieLensHybridRecommender',uri=None):
    if not re.fullmatch('[A-Za-z0-9_.-]+',name):raise ValueError('Invalid registered model name')
    release=Path(release);parity=Path(parity);manifest=checked_manifest(release);checks=checked_manifest(parity)
    if not checks.get('exact_prediction_parity') or checks['release_id']!=manifest['release_id']:
        raise ValueError('Registration requires a passed matching Gate C parity record')
    if checks['release_complete_sha256']!=digest(release/'COMPLETE.json'):raise ValueError('Parity record belongs to different release bytes')
    load_hybrid_recommender(release)
    lineage=json.loads(Path(backfill_summary).read_text());lineage=lineage.get('first',lineage)
    if not lineage.get('runs') or not lineage['runs'].get('phase4/lambdamart'):raise ValueError('Verified backfill lineage is required')
    client=client_for(uri);mlflow.set_tracking_uri(client.tracking_uri);mlflow.set_registry_uri(client.tracking_uri)
    models=client.search_registered_models(filter_string=f"name = '{name}'")
    if not models:client.create_registered_model(name,description='Complete six-retriever + RRF + historical LambdaMART, Validation candidate only.')
    target=candidate_target(client,name)
    if target and target.tags.get('release_id')!=manifest['release_id']:
        raise ValueError('Existing candidate alias belongs to a different release; explicit approval is required to reassign it')
    versions=[v for v in client.search_model_versions(f"name = '{name}'") if v.tags.get('release_id')==manifest['release_id']]
    if len(versions)>1:raise ValueError('Ambiguous duplicate registered release identity')
    tags=release_tags(manifest)|dict(release_complete_sha256=digest(release/'COMPLETE.json'),release_manifest_sha256=digest(release/'manifest.json'))
    if versions:
        version=versions[0]
        if any(version.tags.get(k)!=v for k,v in tags.items()):raise ValueError('Registered release metadata conflicts')
        created=False
    else:
        eid=experiment(client,RELEASE)
        existing=client.search_runs([eid],filter_string=f"tags.release_id = '{manifest['release_id']}'")
        if existing:raise ValueError('Release run already exists without a complete version; inspect it before retrying registration')
        with mlflow.start_run(experiment_id=eid,run_name='complete_hybrid_'+manifest['run_id'],tags=tags|{
                'integration_action':'new_release_registration','retrospective_import':'false',
                'source_runs':json.dumps(lineage['runs'],sort_keys=True),
                'source_dirty':str(bool(manifest['source_git_status'])).lower()}) as run:
            mlflow.log_dict(manifest,'provenance/release_manifest.json')
            mlflow.log_dict(checks,'verification/gate_c_parity.json')
            mlflow.log_dict(lineage,'provenance/historical_mlflow_lineage.json')
            from .mlflow_tracking import metric_names
            mlflow.log_metrics(metric_names(manifest['validation_metrics']['lambdamart'][0]))
            signature=ModelSignature(inputs=Schema([ColSpec('long','userId'),ColSpec('long','top_k')]),
                outputs=Schema([ColSpec('long','userId'),ColSpec('long','movieId'),ColSpec('integer','rank'),
                                ColSpec('double','score'),ColSpec('string','fallback')]))
            info=mlflow.pyfunc.log_model(name='hybrid_recommender',
                python_model=str(release/'src/recsys/mlflow_model.py'),code_paths=[str(release/'src/recsys')],
                artifacts={'release':str(release)},pip_requirements=(release/'requirements.txt').read_text().splitlines(),
                signature=signature,input_example=pd.DataFrame({'userId':[int(load_hybrid_recommender(release).ctx.users[0])],'top_k':[20]},dtype='int64'),
                metadata=tags,tags=tags)
            version=mlflow.register_model(info.model_uri,name,tags=tags)
        created=True
    client.set_registered_model_alias(name,'candidate',version.version)
    resolved=client.get_model_version_by_alias(name,'candidate')
    if resolved.version!=version.version:raise ValueError('Candidate alias round-trip differs')
    return dict(gate='D',tracking_uri=client.tracking_uri,name=name,version=version.version,alias='candidate',
        model_uri=f'models:/{name}@candidate',release_id=manifest['release_id'],run_id=version.run_id,
        created=created,tags=resolved.tags)


def verify_release(model_uri,parity,uri=None):
    """Read-only registry verification; never registers a new version or fits models."""
    client=client_for(uri);mlflow.set_tracking_uri(client.tracking_uri);mlflow.set_registry_uri(client.tracking_uri)
    parity=Path(parity);checks=checked_manifest(parity)
    model=mlflow.pyfunc.load_model(model_uri)
    inputs=pd.read_parquet(parity/'inputs.parquet');expected=pd.read_parquet(parity/'reference.parquet')
    actual=model.predict(inputs);pd.testing.assert_frame_equal(actual,expected,check_exact=True)
    recommender=model.unwrap_python_model().recommender
    return dict(model_uri=model_uri,queries=len(inputs),rows=len(actual),exact_parity=True,
        feature_schema_sha256=recommender.schema['sha256'],reference_release_id=checks['release_id'])
