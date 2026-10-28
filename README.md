# hybrid_recsys — MovieLens Small local MLOps

**Phase 4 is complete:** six-channel RRF, six independently fitted historical retrievers, and out-of-time LightGBM LambdaMART training/Validation evaluation ran locally on Mac CPU. Checkpoint reload and prediction parity passed in separate locked processes. Test remains unopened; MLflow and serving remain future work.

Lifecycle: Kaggle → DuckDB/Parquet → local DVC → six retrievers → RRF → historical-snapshot LambdaMART → offline evaluation → MLflow release registration → FastAPI/Docker.

## Actual Validation result

The same **579 users, 6,491 eligible positives and 213,507 union candidate pairs** are used for every ranking below. The ALS baseline preserves its original Top-20 within that union.

| Ranking | Recall@20 | NDCG@20 | HitRate@20 | Catalog coverage@20 |
|---|---:|---:|---:|---:|
| Best individual: Implicit ALS | 0.171301 | 0.126085 | 0.583765 | 13.18% |
| Six-channel RRF, c=60 | 0.164938 | 0.127578 | 0.552677 | 5.98% |
| LambdaMART, selected 31 trees | 0.187106 | 0.142799 | 0.614853 | 10.52% |

Full-union query-macro recall ceiling is **0.596664**; micro positive coverage is **3,270/6,491 = 50.38%**. These differ from ranked Top-K metrics. All 45 users with no relevant union candidate remain in evaluation. Validation selected the ranker iteration; these are not final Test estimates or statistically established improvements. The 12-user sparse cohort has worse Recall@20 after ranking.

The ranker uses **528 effective historical queries / 57,936 rows**, not the 571 potentially eligible queries. All six historical models were independently fitted on 55,807 earlier Train interactions before attaching labels from 14,379 later Train rows. Full-Train Phase 3 models were not retrained or used for in-sample ranker examples. A strict **53-feature schema** includes RRF, source scores/ranks/membership, historical statistics and movie genres; unknown candidates are weak zero labels, not verified dislikes.

Detailed gates, source overlap/contributions, six ablations, sampling, feature shift, timings, checksums and limitations: [Phase 4 report](docs/phase4_fusion_and_ranking.md), [measured JSON](docs/phase4_fusion_and_ranking.json), [verification](docs/phase4_verification.json), [actual commands](docs/phase4_commands.json). The unchanged [Phase 3 report](docs/phase3_retrieval_benchmark.md) retains all six individual benchmarks.

## Frozen data and interpretation

Source: Kaggle [`grouplens/movielens-latest-small`](https://www.kaggle.com/datasets/grouplens/movielens-latest-small), 100,836 ratings / 610 users / 9,742 metadata movies. Modeling uses ratings and movie genres only. Acquisition hashes/provenance remain under ignored, DVC-tracked `data/raw/ml-latest-small/`.

Primary `small_v1`: per-user chronological 70/15/15, **70,186 Train / 15,143 Validation / 15,507 Test**. Timestamp ties stay together. Positive = rating ≥4; Policy B uses Train-known items and excludes every Train-rated item, including lows. Validation histories stay frozen. Its 6,491 eligible positives exclude 417 Train-unknown positives; all-observed metrics remain separate.

**Per-user chronology permits cross-user future-information exposure.** This is not globally leakage-free or prospective evaluation. Ratings are submissions, not verified viewing/exposure; unknown pairs are not confirmed dislikes. No online CTR or causal satisfaction claim is supported. The separate global sensitivity protocol was not modeled. See the unchanged [evaluation protocol](docs/evaluation_protocol.md).

## Environment and stage commands

Python **3.11.9**, uv, `.venv`, `pyproject.toml`, `uv.lock`; CPU-first. The `train` group pins Torch 2.11.0 / implicit 0.7.2 / threadpoolctl 3.6.0; `rank` pins native LightGBM 4.6.0. For a fresh environment:

```sh
uv venv --python 3.11.9
uv sync --locked --group train --group rank
```

For the already installed environment, `--no-sync` preserves verified inherited wheels. Completed artifacts are immutable: **use new run IDs**. CLI stages are independent; fusion/evaluation never silently fit missing models.

```sh
# Existing selected Phase 3 streams → Validation fusion; no retriever fitting.
UV_CACHE_DIR=.uv-cache .venv/bin/uv run --locked --no-sync --group train --group rank recsys fuse-validation \
  --evaluations artifacts/candidates/{popularity,genre,itemcf,implicit_als,lightgcn,two_tower}/phase3_v1_seed42_main/validation_main \
  --run-id my_validation_fusion

# Independent historical fitting; repeat for each of the six model configs.
UV_CACHE_DIR=.uv-cache .venv/bin/uv run --locked --no-sync --group train --group rank recsys historical-train \
  --model popularity --config configs/popularity.yaml --run-id my_snapshot

# Reload only: verify the existing ranker against all saved Validation feature rows.
UV_CACHE_DIR=.uv-cache .venv/bin/uv run --locked --no-sync --group train --group rank recsys rank-evaluate \
  --checkpoint artifacts/lightgbm/phase4_v1 --training-data artifacts/ranking_training/phase4_v1 \
  --run-id my_reload_check

MPLCONFIGDIR=.venv/mplconfig PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B tools/run_tests.py
UV_CACHE_DIR=.uv-cache .venv/bin/uv lock --check --offline
```

The [Phase 4 command sequence](docs/phase4_fusion_and_ranking.md#commands-and-verification) covers historical-candidates, fuse-historical, rank-features and rank-train. Existing train/evaluate/benchmark commands remain available. Phase 4 provides no Test command.

## Artifacts and verification

- Full-Train checkpoints: `artifacts/<model>/phase3_v1_seed42_main/`; selected candidates: `artifacts/candidates/<model>/phase3_v1_seed42_main/validation_main/`.
- Validation fusion: `artifacts/fusion/phase4_v1_validation/`; historical checkpoints/candidates: `artifacts/{historical,historical_candidates}/<model>/phase4_v1_snapshot/`; historical fusion: `artifacts/fusion/phase4_v1_rank_train/`.
- Versioned features/training table: `artifacts/{ranking_features,ranking_training}/phase4_v1/`; native ranker, complete rankings, history and importance: `artifacts/lightgbm/phase4_v1/`.
- Clean-process parity: `artifacts/ranking_evaluations/phase4_v1_clean_reload/` and six historical candidate reloads. All 213,507 ranker predictions match exactly; all six full historical candidate Parquets match byte-for-byte.
- **61 tests passed, zero failures/skips, in both working and isolated locked environments.** All 45 original tests remain. Every new artifact completion marker/checksum passed. All 127 protected hashes and 66 frozen data/document metadata records stayed unchanged.
- Generated artifacts/logs remain ignored local files; source, configuration, lock and reports are Git-visible. A source clone alone does not contain the datasets/checkpoints.

**DVC dependency drift remains explicit:** all three frozen Phase 2 stages depend on global pyproject/uv.lock; the writer also embeds that entire lock in immutable data identity. A disposable synthetic test confirms changing an unrelated training dependency cannot safely rebuild the same version. The [Phase 4 report](docs/phase4_fusion_and_ranking.md#dvc-environment-drift-assessed-and-deliberately-deferred) proposes a dedicated data-environment contract and future versioned migration. No frozen data or DVC metadata was changed to conceal drift; do not force-repro `small_v1` under the training lock.

Next, on a separate request: MLflow tracking and registration of a complete compatible release candidate, including Full-Train retrievers, mappings/history, fusion settings, feature schema and ranker. Historical models remain training lineage. No MLflow, API/Docker, cloud resources, new data download or final Test evaluation was added in Phase 4. Changes remain uncommitted.

Earlier references: [Phase 1 EDA](notebooks/01_MovieLens_Small_EDA.ipynb), [Phase 2 report](docs/phase2_data_foundation_report.md), [pipeline](docs/data_pipeline.md), [cleanup audit](docs/repository_cleanup_audit.md), [32M archive](archive/legacy_movielens32m/README.md).
