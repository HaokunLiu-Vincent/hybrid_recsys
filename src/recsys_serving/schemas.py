"""Strict HTTP contracts; scores are uncalibrated ranking scores."""
from typing import Annotated,Literal
from pydantic import BaseModel,ConfigDict,Field,field_validator

UserId=Annotated[int,Field(strict=True,ge=1,le=9223372036854775807)]


class RecommendRequest(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    user_ids: list[UserId]=Field(min_length=1,max_length=64)
    top_k: int=Field(strict=True,ge=1,le=100)

    @field_validator('user_ids')
    @classmethod
    def distinct(cls,users):
        if len(users)!=len(set(users)):raise ValueError('Duplicate user IDs are not allowed')
        return users


class ModelIdentity(BaseModel):
    name: Literal['MovieLensHybridRecommender']
    version: Literal['1']
    release_status: Literal['validation_candidate']


class Recommendation(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True,allow_inf_nan=False)
    movie_id: UserId
    rank: int=Field(strict=True,ge=1,le=100)
    score: float
    fallback: Literal['none','known_zero_positive_history','unknown_user_popularity']


class UserRecommendations(BaseModel):
    user_id: UserId
    recommendations: list[Recommendation]


class RecommendResponse(BaseModel):
    model: ModelIdentity
    results: list[UserRecommendations]
