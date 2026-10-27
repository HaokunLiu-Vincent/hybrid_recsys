# hybrid_recsys — MovieLens Small local MLOps transition

This **existing repository** is being rebuilt in place as a local-first MovieLens Latest Small recommendation-system MVP. Its Git history and uncommitted research work are preserved. The current phase is **audit, preservation and cleanup only**; this is not a completed Small-dataset pipeline.

Planned lifecycle (not implemented here):

`Kaggle grouplens/movielens-latest-small → DuckDB / Parquet → local DVC → six independent retrievers → RRF → out-of-time LightGBM LambdaMART → temporal evaluation → MLflow tracking + deployable release registration → FastAPI / Docker → parity tests`

## Verified current state

- Readable reusable code remains under `src/recsys/`: Popularity, Genre CBF, ItemCF, Implicit ALS, LightGCN, Hybrid Two-Tower, fusion/RRF, ranker features/LightGBM, metrics, candidate contracts and artifact utilities.
- Shared in-memory history/feature and evaluation helpers are retained. The old 32M loaders, fixed counts, Colab orchestration, notebook generators, benchmark configs and historical reports are archived; they are not the active execution path.
- No MovieLens Small data was downloaded. No new split, DuckDB/DVC setup, MLflow service, serving API or Docker implementation exists yet. Existing model logic is retained for review/adaptation, not claimed validated on MovieLens Small.
- Existing archive entries are untouched. Full 32M notebooks—including outputs—and original mixed-module versions are retained in the [32M reference archive](archive/legacy_movielens32m/README.md).
- Recovery inventory, backup location, exact changes, compatibility concerns and check results are recorded in [the cleanup audit](docs/repository_cleanup_audit.md). No commit or push was made.

```text
hybrid_recsys/
├── AGENTS.md
├── README.md
├── requirements.txt             # existing dependency reference, not a new lockfile
├── pytest.ini                   # confines optional pytest discovery to active tests/
├── src/recsys/                  # reusable code; no automatic dataset loading or fitting
│   ├── retrievers/
│   ├── fusion/
│   ├── rankers/
│   └── data / features / metrics / contracts / artifacts / runtime helpers
├── tests/                       # synthetic, non-fitting cleanup checks
├── tools/run_tests.py           # existing local, thread-bounded test wrapper
├── docs/repository_cleanup_audit.md
└── archive/
    ├── ...                     # older SageMaker/SBERT materials preserved in place
    └── legacy_movielens32m/     # superseded 32M/Colab reference work
```

## Lightweight checks

Use an **already installed** Python environment with NumPy, pandas, SciPy, PyArrow and psutil. Existing PyTorch enables two additional CPU mathematical/sampling checks; no GPU is required. Nothing installs packages automatically.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -B tools/run_tests.py
```

The active suite checks synthetic metrics, candidate eligibility/seen filtering, deterministic Top-K, unknown-pair sampling, provenance, serialization, Parquet query grouping and archive isolation. It does not fit a model, run an experiment, acquire data or open archived notebooks. Training-based and old 32M workflow tests remain in the archive; the old full benchmark/smoke commands are no longer the recommended active checks.

## Scientific and transition boundaries

Rating timestamps are submissions, not verified viewing or exposure. Observed high ratings, observed low ratings and unobserved pairs are distinct. Unknown pairs are not verified dislikes. Keep one explicit catalog/seen-item policy and comparable per-user cohorts; Top-K is an upper bound, raw channel scores are not calibrated, and candidate-pool recall limits ranking.

Before Small-dataset model work, approve a new versioned chronological protocol and out-of-time ranker-feature procedure within outer Train. Old 32M split manifests and results must never be reused as Small inputs. Record a Validation-only selection before opening Test. Offline observed-rating metrics cannot establish causal satisfaction or online CTR.

Next, after a separate task: acquire only `grouplens/movielens-latest-small`, record provenance/licence/schema/file hashes, transform `ratings.csv` and `movies.csv` using DuckDB, then configure local DVC versioning. Packaging, bounded CPU experiment configs, MLflow release bundles, API/Docker and parity tests remain intentionally deferred. No cloud, GPU, credentials or external infrastructure are prerequisites for that planned MVP.
