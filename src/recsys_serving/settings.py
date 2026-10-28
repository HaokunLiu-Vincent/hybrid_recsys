"""Explicit deployment settings for the fixed, Validation-only Version 1."""
from dataclasses import dataclass
from pathlib import Path
import os

MODEL_NAME='MovieLensHybridRecommender'
MODEL_VERSION='1'
MODEL_URI=f'models:/{MODEL_NAME}/{MODEL_VERSION}'
RELEASE_ID='e4b38328a35c992114eb3e6acda90803f0aa16c61a027686d9e86b06bf5e475d'
SCHEMA_ID='21b298dc0cef61b0de9514c0db66f4e38cd9a89818bb8fe94694d0251ad48ae0'


@dataclass(frozen=True)
class Settings:
    model_source: str='local'
    model_path: Path=Path('outputs/deployment_models/movielens_hybrid_v1')
    tracking_uri: str|None=None
    expected_version: str=MODEL_VERSION
    expected_release_id: str=RELEASE_ID
    max_batch_users: int=32
    max_body_bytes: int=65536
    cpu_threads: int=1

    def __post_init__(self):
        if self.model_source not in ('local','registry'):raise ValueError('RECSYS_MODEL_SOURCE must be local or registry')
        if self.expected_version!=MODEL_VERSION:raise ValueError('This deployment selects immutable Model Version 1 only')
        if not 1<=self.max_batch_users<=64:raise ValueError('Batch limit must be 1..64')
        if not 1024<=self.max_body_bytes<=1048576:raise ValueError('Body limit must be 1024..1048576 bytes')
        if self.cpu_threads!=1:raise ValueError('Initial CPU service requires one numerical thread')
        if self.model_source=='registry' and not self.tracking_uri:raise ValueError('Registry startup requires an explicit Tracking URI')

    @classmethod
    def from_env(cls):
        return cls(model_source=os.getenv('RECSYS_MODEL_SOURCE','local'),
            model_path=Path(os.getenv('RECSYS_MODEL_PATH','outputs/deployment_models/movielens_hybrid_v1')),
            tracking_uri=os.getenv('RECSYS_TRACKING_URI'),expected_version=os.getenv('RECSYS_EXPECTED_MODEL_VERSION',MODEL_VERSION),
            expected_release_id=os.getenv('RECSYS_EXPECTED_RELEASE_ID',RELEASE_ID),
            max_batch_users=int(os.getenv('RECSYS_MAX_BATCH_USERS','32')),
            max_body_bytes=int(os.getenv('RECSYS_MAX_BODY_BYTES','65536')),cpu_threads=int(os.getenv('RECSYS_CPU_THREADS','1')))
