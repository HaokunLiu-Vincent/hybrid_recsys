"""Optimized CPU ALS; explicit low-rating confidence differs from missing feedback."""
from .base import VectorRetriever
from ..runtime import require_memory


def fit_als(context, config):
    from implicit.cpu.als import AlternatingLeastSquares
    c = config['als']
    require_memory(cpu=context.observed.nnz*24 + (len(context.users)+len(context.items))*c['factors']*16,
                   stage='CPU implicit ALS')
    confidence = context.observed.copy()
    positive = confidence.data >= 4
    # implicit uses positive values for p=1, negative values for p=0 with abs(value) confidence.
    # Missing entries: p=0,c=1; observed low: p=0,c=1+low_confidence (not missing).
    confidence.data[positive] = 1+c['positive_confidence']
    confidence.data[~positive] = -(1+c['low_confidence'])
    model = AlternatingLeastSquares(factors=c['factors'], iterations=c['iterations'],
             regularization=c['regularization'], alpha=1., random_state=config['seed'],
             num_threads=config['cpu_threads'])
    model.fit(confidence)
    return VectorRetriever(context, 'als', model.user_factors, model.item_factors)
