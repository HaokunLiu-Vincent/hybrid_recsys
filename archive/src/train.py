"""
Hybrid Recommendation System - Training Script
Model: Collaborative Filtering (NMF) + Content-Based (TF-IDF) + NLP (SBERT)
Framework: scikit-learn (no surprise)
Platform: AWS SageMaker
Purpose: Train and serialize all models for real-time inference
"""

import os
import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

def main():
    # --------------------------
    # SageMaker Environment Paths
    # --------------------------
    input_dir  = "/opt/ml/input/data/training/"   # Data mounted from S3
    model_dir  = "/opt/ml/model/"                 # Models saved to S3 automatically

    print("=== Starting Hybrid Model Training Pipeline ===")

    # -------------------------------------------------------------------------
    # Step 1: Load and merge datasets (ratings, movies, tags)
    # -------------------------------------------------------------------------
    print("Loading raw datasets...")
    ratings = pd.read_csv(os.path.join(input_dir, "ratings.csv"))
    movies  = pd.read_csv(os.path.join(input_dir, "movies.csv"))
    tags    = pd.read_csv(os.path.join(input_dir, "tags.csv"))

    # Aggregate user tags per movie for semantic embedding
    tag_agg = tags.groupby("movieId")["tag"].apply(
        lambda x: " ".join(x.astype(str))
    ).reset_index()

    # Merge tags into movie metadata
    movies = movies.merge(tag_agg, on="movieId", how="left").fillna("")

    # Construct unified text feature for SBERT embedding
    movies["text"] = (
        movies["title"] + " "
        + movies["genres"].str.replace("|", " ") + " "
        + movies["tag"]
    )

    # -------------------------------------------------------------------------
    # Step 2: Train Collaborative Filtering (NMF Matrix Factorization)
    # -------------------------------------------------------------------------
    print("Training Collaborative Filtering (NMF)...")
    user_movie_matrix = ratings.pivot_table(
        index="userId", columns="movieId", values="rating"
    ).fillna(0)

    # NMF: Decompose user-item matrix into latent factors
    nmf_model = NMF(n_components=100, random_state=42, max_iter=500)
    user_embedding = nmf_model.fit_transform(user_movie_matrix)
    item_embedding = nmf_model.components_

    # Save CF artifacts
    joblib.dump(nmf_model, os.path.join(model_dir, "cf_nmf_model.pkl"))
    joblib.dump(user_movie_matrix.columns.tolist(), os.path.join(model_dir, "movie_ids.pkl"))
    joblib.dump(user_embedding, os.path.join(model_dir, "user_embedding.pkl"))
    joblib.dump(item_embedding, os.path.join(model_dir, "item_embedding.pkl"))

    # -------------------------------------------------------------------------
    # Step 3: Train Content-Based Model (TF-IDF on Genres)
    # -------------------------------------------------------------------------
    print("Training Content-Based Model (TF-IDF)...")
    tfidf = TfidfVectorizer(stop_words="english")
    genre_text = movies["genres"].str.replace("|", " ")
    tfidf_matrix = tfidf.fit_transform(genre_text)

    # Save TF-IDF artifacts
    joblib.dump(tfidf, os.path.join(model_dir, "tfidf_vectorizer.pkl"))
    joblib.dump(tfidf_matrix.toarray(), os.path.join(model_dir, "tfidf_matrix.pkl"))

    # -------------------------------------------------------------------------
    # Step 4: Generate SBERT Semantic Embeddings (Local Model)
    # -------------------------------------------------------------------------
    print("Generating SBERT semantic embeddings...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sbert_model_path = os.path.join(os.path.dirname(__file__), "sbert_model")
    sbert_model = SentenceTransformer(sbert_model_path, device=device)
    nlp_embeddings = sbert_model.encode(movies["text"].tolist(), convert_to_numpy=True)

    # Save NLP embeddings
    joblib.dump(nlp_embeddings, os.path.join(model_dir, "nlp_embeddings.pkl"))

    # -------------------------------------------------------------------------
    # Step 5: Save movie metadata for inference
    # -------------------------------------------------------------------------
    movies.to_csv(os.path.join(model_dir, "movies_meta.csv"), index=False)

    print("=== All models trained and saved successfully to S3 ===")

if __name__ == "__main__":
    main()