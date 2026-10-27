"""
Hybrid Recommendation System - Inference Script
For AWS SageMaker Real-Time Endpoint
Input:  {"userId": int, "top_n": int}
Output: Top-N recommended movies with movieId, title, genres
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------------------------
# model_fn: Load all trained models once at endpoint startup
# ------------------------------------------------------------------------------
def model_fn(model_dir):
    """
    Load all serialized models and metadata from model directory
    Args:
        model_dir (str): Path to model folder mounted from S3
    Returns:
        dict: All models and data for prediction
    """
    model_bundle = {}

    # Load Collaborative Filtering components
    model_bundle["cf_model"]      = joblib.load(os.path.join(model_dir, "cf_nmf_model.pkl"))
    model_bundle["movie_ids"]     = joblib.load(os.path.join(model_dir, "movie_ids.pkl"))
    model_bundle["user_emb"]      = joblib.load(os.path.join(model_dir, "user_embedding.pkl"))
    model_bundle["item_emb"]      = joblib.load(os.path.join(model_dir, "item_embedding.pkl"))

    # Load Content-Based components
    model_bundle["tfidf"]         = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.pkl"))
    model_bundle["tfidf_mat"]     = joblib.load(os.path.join(model_dir, "tfidf_matrix.pkl"))

    # Load NLP semantic embeddings
    model_bundle["nlp_emb"]       = joblib.load(os.path.join(model_dir, "nlp_embeddings.pkl"))

    # Load movie metadata
    model_bundle["movies"]        = pd.read_csv(os.path.join(model_dir, "movies_meta.csv"))

    return model_bundle

# ------------------------------------------------------------------------------
# input_fn: Parse incoming request body
# ------------------------------------------------------------------------------
def input_fn(body, content_type):
    """Parse input JSON from request"""
    return json.loads(body)

# ------------------------------------------------------------------------------
# predict_fn: Core hybrid recommendation logic
# ------------------------------------------------------------------------------
def predict_fn(input_data, model_bundle):
    """
    Hybrid scoring = 0.5*CF + 0.25*CB + 0.25*NLP
    Args:
        input_data (dict): {"userId": int, "top_n": int}
        model_bundle (dict): All preloaded models
    Returns:
        list: Top-N recommended movies
    """
    user_id   = input_data["userId"]
    top_n     = input_data.get("top_n", 10)

    # Unpack models
    movies      = model_bundle["movies"]
    movie_ids   = model_bundle["movie_ids"]
    user_emb    = model_bundle["user_emb"]
    item_emb    = model_bundle["item_emb"]
    tfidf_mat   = model_bundle["tfidf_mat"]
    nlp_emb     = model_bundle["nlp_emb"]

    # Map userId to index (user ID starts at 1)
    try:
        user_idx = user_id - 1
        user_cf_vec = user_emb[user_idx]
    except IndexError:
        return {"error": "User ID not found in training data"}

    # --------------------------
    # 1. CF Score Prediction
    # --------------------------
    pred_ratings = np.dot(user_cf_vec, item_emb)
    cf_score_map = dict(zip(movie_ids, pred_ratings))

    # Map movieId to matrix index
    movie_to_index = {mid: i for i, mid in enumerate(movie_ids)}

    # Collect all valid movies with CF score
    candidate_movies = []
    for mid in movies["movieId"].unique():
        if mid in movie_to_index:
            idx = movie_to_index[mid]
            cf_score = cf_score_map[mid]
            candidate_movies.append((mid, cf_score, idx))

    # Get top 200 candidates from CF for reranking
    candidate_movies = sorted(candidate_movies, key=lambda x: -x[1])[:200]
    candidate_indices = [item[2] for item in candidate_movies]

    # --------------------------
    # 2. Content-Based Similarity
    # --------------------------
    user_cb_vec = np.mean(tfidf_mat[candidate_indices], axis=0).reshape(1, -1)
    cb_sim = cosine_similarity(user_cb_vec, tfidf_mat).flatten()

    # --------------------------
    # 3. NLP Semantic Similarity
    # --------------------------
    user_nlp_vec = np.mean(nlp_emb[candidate_indices], axis=0).reshape(1, -1)
    nlp_sim = cosine_similarity(user_nlp_vec, nlp_emb).flatten()

    # --------------------------
    # 4. Hybrid Reranking
    # --------------------------
    hybrid_results = []
    for (mid, cf, idx) in candidate_movies:
        cb_score = cb_sim[idx]
        nlp_score = nlp_sim[idx]
        hybrid_score = 0.5 * cf + 0.25 * cb_score + 0.25 * nlp_score
        hybrid_results.append((mid, hybrid_score))

    # Sort and select top N
    hybrid_results = sorted(hybrid_results, key=lambda x: -x[1])[:top_n]
    recommended_ids = [item[0] for item in hybrid_results]

    # Format output
    result_df = movies[movies["movieId"].isin(recommended_ids)]
    result_df = result_df.set_index("movieId").loc[recommended_ids].reset_index()

    return result_df[["movieId", "title", "genres"]].to_dict(orient="records")

# ------------------------------------------------------------------------------
# output_fn: Format response as JSON
# ------------------------------------------------------------------------------
def output_fn(prediction, content_type):
    return json.dumps(prediction)