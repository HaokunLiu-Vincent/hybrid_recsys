"""Legacy API for the preserved monolithic notebook; new code uses features/rankers.features."""
from .features import candidate_features, feature_columns
from .rankers.features import attach_labels, sample_training
from .rankers.features import distribution as _distribution, mismatch as _mismatch
from .retrievers.base import CHANNELS
FEATURES = feature_columns(CHANNELS)

def features(pool, ctx):
    return candidate_features(pool, ctx, CHANNELS)

def distribution(frame):
    return _distribution(frame, FEATURES)

def mismatch(a, b):
    return _mismatch(a, b, FEATURES)
