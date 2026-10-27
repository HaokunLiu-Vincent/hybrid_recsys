# Repository Instructions — MovieLens Small Local MLOps

## 1. Mission and current phase

This repository (`hybrid_recsys`) is being rebuilt **in place** as a local-first, reproducible MovieLens Latest Small recommendation-system MLOps MVP. Do not create a second repository or a parallel implementation. The intended end-to-end lifecycle is:

Kaggle data acquisition → DuckDB/Parquet preparation → DVC dataset versioning → six independent retrieval models → candidate fusion (RRF) → leakage-aware LightGBM LambdaMART → offline evaluation → MLflow tracking and model registration → FastAPI/Docker serving → integration tests.

The immediate task is **repository audit, preservation, and cleanup only**. Do not preemptively implement later stages, download data, train models, install tooling, provision AWS/GCP/Colab, deploy services, commit, or push unless separately requested.

Prefer correctness and one working lifecycle over infrastructure breadth. No Feast, Redis, Kafka, Spark, Kubernetes, cloud endpoints, or GPU requirements for the MVP unless a future requirement justifies them.

## 2. Existing repository and non-destructive migration

- Inspect the actual filesystem and `git status` before any changes. The user has **uncommitted changes**. Never assume Git can restore an untracked or uncommitted file.
- Audit existing Python code, tests, configs, notebooks, documentation, data references, and generated artifacts before classifying them.
- Retain and reuse sound, readable source code in `src/recsys/`, particularly Popularity, Genre CBF, ItemCF, Implicit ALS, LightGCN, Hybrid Two-Tower, RRF, LightGBM, metrics, evaluation, feature and artifact contracts. Validate compatibility before reuse; do not blindly preserve invalid logic.
- Keep the existing repository and Git history. Move obsolete but meaningful 32M-specific work to `archive/legacy_movielens32m/` with an archive README and path mapping. Examples include Colab-centric notebooks, 32M split scripts/configs, old orchestration, and legacy documentation.
- Preserve already archived materials unless a collision requires a clearly documented new destination. Never silently overwrite archive contents.
- Only delete confirmed disposable/reproducible caches and clutter (e.g. `__pycache__`, `.pytest_cache`, `.ipynb_checkpoints`, build debris) after listing them. Respect `.gitignore`; never assume a directory is safe to delete because it is ignored.
- Never remove unique user data, raw datasets, frozen split files, trained checkpoints, experiment results, credentials, or untracked source files without independently verifiable backups and explicit approval for their deletion. When in doubt, leave them untouched and report them.
- First create a recovery inventory and a backup strategy for uncommitted tracked and untracked work. Verify recoverability before moves/deletion. Do not commit or push without explicit approval. `git clean -fdx`, `git reset --hard`, wholesale `rm -rf`, and force pushes are prohibited.
- The user has authorized a major architectural reset, but authorization does not override the preservation and review rules above.
- Do not rewrite or regenerate algorithms just to achieve a cleaner directory tree. Prefer minimal moves and small compatibility fixes.

## 3. Local-first technical target (future stages)

- Python-first packaging, preferably `pyproject.toml` and a reproducible environment; normal imports from `src/recsys`. Jupyter notebooks only for EDA/explanation, never as the authoritative implementation.
- Human-readable `.py` sources only. No Base64/ZIP-embedded packages, giant code-generating notebooks, or duplicate model implementations.
- Default execution on a MacBook CPU, including LightGCN and Two-Tower in appropriately bounded configurations. GPU optional, not a condition for a valid MVP.
- Kaggle dataset identifier: `grouplens/movielens-latest-small`. Do not fall back silently to GroupLens website or a third-party mirror. Only `ratings.csv` and `movies.csv` are modeling inputs for the initial benchmark. Before model work, record downloaded file hashes, provenance, schema, expected counts, and licence attribution as available.
- Use DuckDB to transform raw data to Parquet. Use DVC to version raw and processed datasets; initial DVC remote can be local. Do not assume S3 or Google Drive credentials. Keep DuckDB writable database files and MLflow backend databases out of DVC remotes and cloud-synced folders.
- Use MLflow for experiment tracking and model registration. Register a deployable, versioned release bundle, not just the LightGBM booster: include retriever artifacts, mappings, known-item histories, feature definitions, ranker, fusion settings and a compatibility manifest. Avoid representing model registry as equivalent to data version control.
- FastAPI + Docker should load an explicitly selected release and expose health/version and recommendations. No retraining at service startup. Test offline-versus-serving parity, model loading, top-K and exclusion behavior.

## 4. Scientific validity: data and evaluation

The old MovieLens 32M split manifest and experimental results are **not valid inputs** for MovieLens Small. Define a new, versioned, deterministic chronological split, preserving tie handling and a documented validation/test protocol. Never implicitly merge the old and new datasets.

- Rating timestamps reflect **rating submissions**, not verified viewing timestamps.
- Observed high ratings (e.g. >=4) can define positive feedback only by documented convention. Low ratings and unobserved pairs are distinct; unobserved does not mean disliked.
- Use one documented eligible item catalog, shared seen-item filtering and comparable per-user evaluation cohorts. Report sparse/cold-user and empty-relevance cases transparently.
- Fit each retriever only on allowed historical training data; do not use future validation/test ratings to construct models, profiles, popularity or features.
- Measure per-channel Recall@K, NDCG@K, HitRate@K, coverage, candidate-pool ceiling, and latency/resource characteristics; compare on the same population.
- Never describe offline observed-rating prediction as verified online CTR or causal user satisfaction. No exposure logs exist.

### Retrieval contract

Keep six independently trainable, loadable, testable channels: Popularity, Genre CBF, ItemCF, Implicit ALS, LightGCN, and Hybrid Two-Tower. Generate standardized, deterministic Top-K outputs with `userId,movieId,score,rank,source`; no duplicate `(userId,movieId,source)`, finite scores, deterministic tie-breaking, consistent catalog and seen-item exclusion. `Top-K` is an upper bound, not an obligation to output K rows. Raw scores across models are not automatically comparable. Store manifests and run provenance.

### Fusion and ranker contract

Keep retrieval and ranker fitting separate. Implement RRF over compatible saved candidates. The LightGBM LambdaMART training examples must use **out-of-time or otherwise genuinely out-of-sample retrieval predictions generated within outer Train**, rather than in-sample predictions from full-Train retrievers. This requirement is not waived for a small dataset. Ranking features must be available at their corresponding cutoff, and candidate generation must not see their labels. Group by query/user correctly, distinguish observed positives, observed low ratings and sampled unknowns, and report how many queries have useful contrast. Tune on Validation only; open Test once after a recorded selection. A ranker cannot recover a relevant item absent from the candidate pool.

## 5. Code organization, reproducibility, and testing

- Use small, inspectable modules and a shared model interface; avoid duplicate business logic between scripts and notebooks.
- Record dataset and split hashes, code revision/source hashes, configuration, seeds, model/version, candidate policies, and evaluation cohort. Keep run outputs distinct and do not overwrite them silently.
- Unit tests should cover split integrity/leakage, retrieval eligibility, deterministic sorting and Top-K edge cases, ranking group semantics, metrics, serialization, and API parity. Use synthetic fixtures for fast tests; distinguish them from actual MovieLens Small experiments.
- Guard against dense user-item or item-item matrices when unnecessary. Resource limits and reproducibility are still important on a small dataset.
- `.gitignore` should exclude raw data, derived datasets, caches, artifacts, checkpoints, MLflow stores/DBs, local secrets, and runtime files as appropriate; include DVC pointer files and pipeline definitions in Git.
- Never embed credentials in source or logs. Do not enable billable cloud resources without approval.
- Do not install dependencies, delete unique files, commit, or push without the user's explicit approval.

## 6. Required cleanup handoff (immediate task)

Produce a `docs/repository_cleanup_audit.md` (or comparable report) with:

1. Initial `git status` summary and uncommitted/untracked recovery approach; do not expose sensitive content.
2. Inventory grouped by **KEEP ACTIVE**, **ARCHIVE**, **DELETE SAFE CACHE**, **REVIEW / DO NOT TOUCH**, including reasons and old→new paths.
3. What was actually moved/deleted and which recoverability checks passed.
4. A concise proposed target tree for the local-first MVP and a list of intentionally deferred dependencies/features.
5. Exact tests/validation run, failures, and follow-up steps; do not claim training or data download occurred.

Stop after the cleanup phase. Wait for the next task before Kaggle download, DVC setup, pipeline implementation, or model training.
