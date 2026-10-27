# hybrid_recsys — MovieLens Small local MLOps

The existing repository now has a verified **Phase 2 data foundation**: Kaggle acquisition, executed EDA, deterministic DuckDB/Parquet preparation and local DVC recovery. Six retrieval implementations, RRF and ranking utilities remain available; **no Small models have been trained or evaluated**.

Planned lifecycle: Kaggle → DuckDB/Parquet → local DVC → six retrievers → RRF → historical-snapshot LightGBM LambdaMART → offline evaluation → MLflow tracking/release registration → FastAPI/Docker. Later training, tracking and serving stages are not implemented here.

## Frozen evaluation contract

**Primary: per-user chronological 70/15/15**, preserving whole timestamp groups. **This is not globally leakage-free:** other users' Train interactions may occur later than a user's held-out targets. That cross-user temporal limitation is explicitly accepted for this educational offline benchmark and disclosed in manifests.

| Version | Train | Validation | Test | Eligible-positive users V / T |
|---|---:|---:|---:|---:|
| `small_v1` — primary | 70,186 | 15,143 | 15,507 | 579 / 583 |
| `small_global_v1` — global 80/10/10 sensitivity | 80,668 | 10,084 | 10,084 | 62 / 72 |

Positive = rating ≥4. Candidate items must be Train-observed; exclude every Train-rated item, including low ratings. Histories stay frozen through Test. Primary Recall/NDCG use eligible positives, with separate all-observed and cold/sparse/warm diagnostics. The primary catalog has 7,525 movies; 417 Validation and 584 Test positives are outside it. No user lacks primary Train ratings, but four users have no Train positives. Secondary Test has 54 of 74 users without Train history; never pool its results with primary metrics.

The [evaluation protocol](docs/evaluation_protocol.md) specifies boundary/rounding/short-history rules, all four measured comparisons, cohorts, metric denominators, leakage disclosure and ranker contracts. No protocol was chosen by model scores. Primary inner ranker history/labels contain 55,807 / 14,379 rows and at most 571 catalog-positive queries; effective query counts/actual candidate recall remain unknown until retrieval runs.

## Local commands

The authoritative environment is `pyproject.toml` + `uv.lock`, Python 3.11.9. Preserve an existing `.venv`; do not recreate it. With uv already available inside this project's environment:

```sh
UV_CACHE_DIR=.uv-cache .venv/bin/uv sync --locked --inexact --all-extras
UV_CACHE_DIR=.uv-cache .venv/bin/uv lock --check --offline

DVC_NO_ANALYTICS=1 .venv/bin/dvc pull
DVC_NO_ANALYTICS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/dvc repro
DVC_NO_ANALYTICS=1 .venv/bin/dvc status
DVC_NO_ANALYTICS=1 .venv/bin/dvc push

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B tools/run_tests.py
DVC_NO_ANALYTICS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B tools/verify_small_repro.py
```

Configure a persistent **local filesystem** remote outside this repository before pull/push; this machine uses `/Users/vincent/Documents/projects/hybrid_recsys_dvc_remote`, stored only in ignored local DVC configuration. `dvc push` copies data locally; no Git push or cloud service is involved. See [environment, initialization, pipeline and safe recovery instructions](docs/data_pipeline.md) for fresh setup and explicit stage commands. The requirements files are compatibility shims into the same project. PyTorch/implicit were preserved from the working environment; model-training dependencies are intentionally outside this data-phase lock.

## Verified artifacts and checks

- Exact source: [`grouplens/movielens-latest-small`](https://www.kaggle.com/datasets/grouplens/movielens-latest-small), 100,836 ratings, 610 users, **9,742 metadata movies / 9,724 rated movies**. SHA256, schemas and acquisition evidence are in ignored `data/raw/ml-latest-small/provenance.json`. Original Kaggle files/ZIP are unchanged; only ratings and movies enter processing. Kaggle version remains unavailable; consult the bundled publisher README for licence/citation terms.
- [Executed EDA notebook](notebooks/01_MovieLens_Small_EDA.ipynb): 16 code cells, seven figures, zero errors. Phase 1 notebook/report remain unchanged. Download and EDA entry points remain `tools/download_movielens_small.py` and `tools/execute_small_eda.py`; DVC normally restores the frozen acquisition rather than downloading a different provenance snapshot.
- `data/splits/small_v1/` and `data/splits/small_global_v1/`: Train/Validation/Test, movies, Train-only eligibility, evaluation cohorts/targets, canonical mappings, cutoffs, historical ranker history/label windows and manifests. Data is ignored by Git and tracked through DVC.
- `src/recsys/small_benchmark.py` loads frozen artifacts into retained context/population interfaces without fitting. Its default is Validation and Policy-B relevance. Future training must use historical snapshot retrievers for ranker examples, never full-Train in-sample predictions.
- **33 tests passed, zero failures/skips** in this environment. Existing 30 Phase 1 source files, the EDA notebook/report and old archive contents remain unchanged.
- Independent real-data builds were byte-identical. No-change DVC repro rewrote no outputs. Cold-cache local-remote recovery restored all 40 files with matching SHA256; a missing disposable Train file was also restored. Original datasets/cache/remote were never deleted.

Full measurements and execution evidence: [Phase 2 report](docs/phase2_data_foundation_report.md), [measured summary](docs/movielens_small_phase2_summary.json), [verification](docs/movielens_small_phase2_verification.json). Historical context: [Phase 1 report](docs/phase1_data_acquisition_and_eda.md), [cleanup audit](docs/repository_cleanup_audit.md), [32M reference archive](archive/legacy_movielens32m/README.md). Old 32M splits/results are not Small inputs.

Rating timestamps are submissions, not verified consumption or exposure. Unobserved pairs are not verified dislikes. No model metrics, online CTR, causal effects, MLflow registration, serving, containers or cloud execution are claimed. All current changes remain uncommitted; the next task can address bounded CPU retrieval experiments under the recorded protocol.
