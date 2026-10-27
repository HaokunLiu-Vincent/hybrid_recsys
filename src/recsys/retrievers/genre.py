from .base import VectorRetriever


class Genre(VectorRetriever):
    def __init__(self, context):
        super().__init__(context, 'genre', context.user_genres, context.genres)

    def fallback(self, user):
        return super().fallback(user) or not self.U[user].any()
