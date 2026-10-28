"""Synthetic preflight/registry-boundary tests; no real model or registry mutation."""
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch,MagicMock
from mlflow.models import ModelSignature
from mlflow.types import Schema,ColSpec
from recsys_serving.settings import RELEASE_ID,SCHEMA_ID,MODEL_URI,Settings
from recsys_serving.model_loader import inspect_model,verify_export,registered_model,sha256,load_model


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.release=self.root/'artifacts/release';(self.release/'ranker').mkdir(parents=True)
        self.manifest=dict(kind='complete_hybrid_recommender',release_id=RELEASE_ID,feature_schema_sha256=SCHEMA_ID,
            rrf_constant=60,retrievers={str(n):{} for n in range(6)},test_evaluated=False,release_status='validation_candidate',lock_sha256='fixture')
        self.schema=dict(names=[f'fixture_{n}' for n in range(53)],sha256=SCHEMA_ID,version='synthetic_schema_preflight_only')
        self.descriptor=SimpleNamespace(flavors={'python_function':{'artifacts':{'release':{'path':'artifacts/release'}}}},
            signature=ModelSignature(inputs=Schema([ColSpec('long','userId'),ColSpec('long','top_k')]),
                outputs=Schema([ColSpec('long','userId'),ColSpec('long','movieId'),ColSpec('integer','rank'),ColSpec('double','score'),ColSpec('string','fallback')])),
            metadata={'release_id':RELEASE_ID,'feature_schema_sha256':SCHEMA_ID,'lock_sha256':'fixture'})
        self.write_fixture()

    def tearDown(self):self.temp.cleanup()

    def write_fixture(self):
        (self.release/'manifest.json').write_text(json.dumps(self.manifest))
        (self.release/'ranker/feature_schema.json').write_text(json.dumps(self.schema))
        files={str(p.relative_to(self.release)):dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in self.release.rglob('*') if p.is_file() and p.name!='COMPLETE.json'}
        (self.release/'COMPLETE.json').write_text(json.dumps({'files':files}))
        self.tags=dict(self.descriptor.metadata,release_manifest_sha256=sha256(self.release/'manifest.json'),release_complete_sha256=sha256(self.release/'COMPLETE.json'))

    def test_preflight_identity_signature_schema_and_registry_hashes(self):
        with patch('mlflow.models.Model.load',return_value=self.descriptor):
            m,info=inspect_model(self.root,registry_tags=self.tags)
            self.assertEqual(info['version'],'1');self.assertEqual(info['feature_schema_fingerprint'],SCHEMA_ID)
            self.assertFalse(info['test_evaluated']);self.assertNotIn('path',info)
            with self.assertRaises(ValueError):inspect_model(self.root,expected_release_id='wrong')
            with self.assertRaises(ValueError):inspect_model(self.root,registry_tags=dict(self.tags,release_manifest_sha256='wrong'))

    def test_incompatible_schema_or_signature_fails(self):
        with patch('mlflow.models.Model.load',return_value=self.descriptor):
            self.schema['names'].pop();self.write_fixture()
            with self.assertRaises(ValueError):inspect_model(self.root)
            self.schema['names'].append('fixture_52');self.write_fixture()
            self.descriptor.signature=ModelSignature(inputs=Schema([ColSpec('string','userId')]))
            with self.assertRaises((ValueError,TypeError)):inspect_model(self.root)

    def test_no_alias_resolution_and_wrong_registry_identity_rejected(self):
        client=MagicMock();version=SimpleNamespace(name='MovieLensHybridRecommender',version='1',source='fixture',run_id='fixture',tags=self.tags)
        client.get_model_version.return_value=version
        with patch('recsys_serving.model_loader.MlflowClient',return_value=client),patch('mlflow.artifacts.download_artifacts',return_value=str(self.root)) as download,patch('mlflow.models.Model.load',return_value=self.descriptor):
            registered_model('http://127.0.0.1:1',self.root/'download')
            client.get_model_version.assert_called_once_with('MovieLensHybridRecommender','1')
            self.assertEqual(download.call_args.kwargs['artifact_uri'],MODEL_URI)
            version.tags=dict(self.tags,release_id='different')
            with self.assertRaises(ValueError):registered_model('http://127.0.0.1:1',self.root/'download')

    def test_local_mode_has_no_registry_or_dataset_reads(self):
        fake=object()
        with patch('recsys_serving.model_loader.verify_export',return_value=(self.manifest,{'release_id':RELEASE_ID})),patch('mlflow.pyfunc.load_model',return_value=fake) as load,patch('recsys_serving.model_loader.registered_model',side_effect=AssertionError('Offline must not resolve registry')):
            loaded=load_model(Settings(model_path=self.root))
            self.assertIs(loaded.model,fake);self.assertEqual(loaded.info['source_mode'],'local')
            load.assert_called_once_with(str(self.root))

    def test_export_marker_catches_corruption_missing_and_extra_files(self):
        (self.root/'MLmodel').write_text('synthetic descriptor mocked in this preflight unit fixture')
        (self.root/'deployment_manifest.json').write_text(json.dumps(dict(format='mlflow_offline_deployment_v1',model_uri=MODEL_URI,
            registered_model=dict(name='MovieLensHybridRecommender',version='1',tags=self.tags),release_id=RELEASE_ID)))
        files={str(p.relative_to(self.root)):dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in self.root.rglob('*') if p.is_file()}
        (self.root/'DEPLOYMENT_COMPLETE.json').write_text(json.dumps({'files':files}))
        with patch('mlflow.models.Model.load',return_value=self.descriptor):verify_export(self.root)
        (self.root/'MLmodel').write_text('corrupt disposable file')
        with self.assertRaises(ValueError):verify_export(self.root)
