# hybrid_recsys — MovieLens Small local MLOps

**Phase 3 is complete:** six independent retrievers were trained on real MovieLens Small Train, saved, reloaded and benchmarked on Validation. Each was also trained and evaluated in a fresh locked Python environment; candidate/per-query metric files matched byte-for-byte. No real-data RRF, historical ranker snapshot fitting, LightGBM, MLflow or serving stage has run.

Planned lifecycle: Kaggle → DuckDB/Parquet → local DVC → six retrievers → RRF → historical-snapshot LambdaMART → offline evaluation → MLflow release registration → FastAPI/Docker.

## Actual Validation benchmark

All channels use the same **579 queries, 6,491 eligible positives**, Train-known catalog of 7,525 movies and Top-100 budget. Every channel produced 57,900 primary candidates, zero empty queries and zero popularity fallback queries in this eligible Validation cohort. Zero-hit queries remain in metric means.

| Retriever | Recall@20 | Recall@50 | Recall@100 | NDCG@20 | HitRate@20 | Coverage@100 |
|---|---:|---:|---:|---:|---:|---:|
| Popularity | 0.112275 | 0.190662 | 0.286786 | 0.091781 | 0.431779 | 5.33% |
| Genre CBF | 0.015106 | 0.025285 | 0.041407 | 0.011679 | 0.103627 | 52.52% |
| ItemCF | 0.161575 | 0.262102 | 0.368582 | 0.123064 | 0.523316 | 37.25% |
| Implicit ALS | 0.171301 | 0.294286 | 0.399429 | 0.126085 | 0.583765 | 26.72% |
| LightGCN | 0.136687 | 0.242046 | 0.365442 | 0.107823 | 0.493955 | 18.18% |
| Hybrid Two-Tower | 0.045533 | 0.106658 | 0.203064 | 0.034236 | 0.267703 | 16.96% |

These are fixed-configuration, seed-42 CPU baselines, not a tuning study. Full timings, cohorts, all-observed diagnostics, model audits, checksums and exact commands are in the [Phase 3 report](docs/phase3_retrieval_benchmark.md) and [measured JSON](docs/phase3_retrieval_benchmark.json). There are no eligible-positive cold-history Validation queries; cold subgroup metrics are undefined. Sparse/moderate/warm cohorts have 12 / 200 / 367 queries.

## Frozen data and interpretation

The exact Kaggle source is [`grouplens/movielens-latest-small`](https://www.kaggle.com/datasets/grouplens/movielens-latest-small): 100,836 ratings, 610 users, 9,742 metadata movies (9,724 rated). Only ratings and allowed movie genres are model inputs; no tags, links, external metadata or pretrained embeddings. Original acquisition and SHA256 provenance remain under ignored, DVC-tracked `data/raw/ml-latest-small/`.

Primary `small_v1` is per-user chronological 70/15/15: **70,186 Train / 15,143 Validation / 15,507 Test**. Whole timestamps stay together; histories are frozen Train, positive = rating ≥4, Policy B catalog, all Train-rated movies excluded including low ratings. Primary denominators use eligible positives; 417 of 6,908 observed Validation positives are Train-unknown. Separate all-observed diagnostics use 581 users / 6,908 positives and a different evaluation identity.

**The primary permits cross-user future-information exposure and is not globally leakage-free.** Ratings are submissions, not verified viewing or exposure. Unknown pairs are not verified dislikes; no online CTR/causal claims are supported. The separately frozen global sensitivity protocol was not trained/evaluated here. Test targets were not opened. See the unchanged [evaluation protocol](docs/evaluation_protocol.md).

## Environment and independent CLI

Python 3.11.9 + uv + `.venv` + `pyproject.toml` + `uv.lock`. The `train` group pins PyTorch 2.11.0, implicit 0.7.2 and threadpoolctl 3.6.0; the timezone dependency exposed by clean testing is now pinned too. No Conda migration or current-environment PyTorch reinstallation occurred.

For a **fresh** environment with uv available:

```sh
uv venv --python 3.11.9
uv sync --locked --group train
```

For the already verified existing environment, `--no-sync` preserves its inherited model wheels. Commands are independent; evaluation/benchmarking never invoke retriever fitting. Use new run IDs, since completed runs cannot be overwritten:

```sh
UV_CACHE_DIR=.uv-cache .venv/bin/uv run --locked --no-sync --group train recsys train \
  --model popularity --config configs/popularity.yaml --run-id my_popularity_run

UV_CACHE_DIR=.uv-cache .venv/bin/uv run --locked --no-sync --group train recsys evaluate \
  --model popularity --split validation \
  --checkpoint artifacts/popularity/my_popularity_run --run-id validation_run

# Consume the six existing selected artifacts; never silently fit a missing channel.
UV_CACHE_DIR=.uv-cache .venv/bin/uv run --locked --no-sync --group train recsys benchmark \
  --split validation \
  --evaluations artifacts/candidates/{popularity,genre,itemcf,implicit_als,lightgcn,two_tower}/phase3_v1_seed42_main/validation_main

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B tools/run_tests.py
UV_CACHE_DIR=.uv-cache .venv/bin/uv lock --check --offline
```

The same train command works independently for `genre`, `itemcf`, `implicit_als`, `lightgcn`, `two_tower` with their corresponding YAML configurations. CLI help is `recsys train --help`, `recsys evaluate --help`, `recsys benchmark --help`. Phase 3 rejects `--split test`.

## Artifacts and preservation

- Selected checkpoints: `artifacts/<model>/phase3_v1_seed42_main/`. Portable weights/factors, mappings, sparse Train histories/features, per-user cutoff identity, configuration/seed, source/dependency provenance and checksummed completion markers. Neural original training states are also retained and verified against exported vectors.
- Selected independent candidates/metrics: `artifacts/candidates/<model>/phase3_v1_seed42_main/validation_main/`. Primary and all-observed files/identities are separate. No six-channel candidate union/fusion was computed.
- Aggregate table: `outputs/phase3/phase3_v1_validation_benchmark/benchmark.csv`. Clean replay runs use `phase3_v1_seed42_replay`. Earlier provisional baseline runs were preserved and are not selected.
- **45 tests passed in each of the working and isolated environments**, with zero failures/skips after the dependency correction. Real-data checkpoints and all candidate checksums passed; all six clean-environment training replays matched.
- Frozen data, manifests, DVC metadata and historical reports/notebook were preserved. No re-splitting, real DVC repro/push, Git commit/push or cloud/GPU jobs occurred.

**DVC note:** its frozen Phase 2 stages record the older `pyproject.toml`/`uv.lock` hashes. Training dependency additions change those source dependencies without changing data objects. Do not regenerate immutable `small_v1` under the training lock to conceal this drift. Data/recovery history remains in [Phase 2 report](docs/phase2_data_foundation_report.md) and [pipeline instructions](docs/data_pipeline.md); the [Phase 3 verification](docs/phase3_verification.json) records preservation checks without reopening Test targets.

Next, on a separate request: RRF over the six compatible candidate artifacts. LambdaMART still needs separately fitted historical snapshot retrievers and out-of-time labels inside External Train; these full-Train models must not provide in-sample ranker-training features. MLflow, API/Docker and final Test evaluation remain future work. Historical references: [Phase 1 EDA](notebooks/01_MovieLens_Small_EDA.ipynb), [cleanup audit](docs/repository_cleanup_audit.md), [32M archive](archive/legacy_movielens32m/README.md).
