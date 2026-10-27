from .base import Retriever


class Popularity(Retriever):
    def __init__(self, context):
        super().__init__(context, 'popularity')

    def scores(self, user, items):
        return self.ctx.popularity[items]

    def fallback(self, user):
        return False

    def save(self, path):
        import numpy as np
        np.save(path/'positive_counts.npy', self.ctx.popularity)
