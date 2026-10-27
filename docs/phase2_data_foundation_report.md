# Phase 2 — data foundation completion report

**Completed locally, 28 October 2026 (Australia/Sydney).** This is data preparation and protocol validation, not model performance. The existing repository/branch/history were retained; no model was fitted, evaluated or registered and no Git commit/push occurred.

## Preserved starting state

Branch `main`, HEAD `e2b160b0049802d7c1cf5bed0a26d2504ab0a47f`. Phase 1 changes were still uncommitted, despite cleanup having been committed. Before editing shared files, a verified private recovery snapshot was created at:

`/Users/vincent/Documents/projects/hybrid_recsys_recovery/phase2_20261027T164541Z`

It contains 19 modified/untracked Phase 1/raw files totaling 5,007,659 bytes, with per-file SHA256 checked after copying and again before handoff. The snapshot also records initial Git status/HEAD and the 30-source baseline. No credentials or Git internals were copied. Existing archives were left untouched. All 30 pre-existing `src/recsys` Python files and the Phase 1 notebook/report/summary/verification retain their original hashes; README, ignore and dependency entry points were intentionally updated.

## Protocol decision and actual measurements

The user-specified primary is **per-user chronological 70/15/15**, subject to integrity/coverage checks; it passed. Whole-timestamp groups select boundaries nearest floor-rounded cumulative targets, with deterministic earlier ties and explicit short-history fallbacks. No rating is duplicated, dropped or changed. The global 80/10/10 sensitivity is materialized separately, with a distinct evaluation ID and strict global chronology. Neither protocol was selected by model scores.

| Candidate protocol | Train | Validation | Test | Train-known movies | Eligible-positive users V / T |
|---|---:|---:|---:|---:|---:|
| Per-user 70/15/15 — primary | 70,186 | 15,143 | 15,507 | 7,525 | 579 / 583 |
| Per-user 80/10/10 | 80,241 | 10,153 | 10,442 | 8,225 | 562 / 566 |
| Global 70/15/15 | 70,585 | 15,125 | 15,126 | 7,056 | 106 / 95 |
| Global 80/10/10 — sensitivity | 80,668 | 10,084 | 10,084 | 7,867 | 62 / 72 |

Primary percentages are 69.6041% / 15.0175% / 15.3784%; all 610 users appear in every external partition. Among users with evaluation events, primary Train-positive cohorts are 4 with zero positives, 18 with 1–4, 217 with 5–19, and 371 with 20+. None lacks Train ratings. Primary Validation/Test have 6,908 / 7,309 positives, of which 6,491 / 6,725 are eligible under Policy B; 417 / 584 are Train-unknown. Positive-bearing users with zero eligible targets are explicitly reported (2 / 1), and none has zero available candidates. Secondary Test has 54/74 no-Train-history users.

**Accepted limitation:** the primary preserves same-user chronology but permits other users' Train events after a user's held-out targets. It is an educational offline benchmark, not globally leakage-free or prospective deployment evaluation. Both primary external and inner manifests disclose this. Frozen Train histories, rating ≥4 positivity, Train-observed catalog, all-Train-rated seen filtering and Validation-only future selection are shared contracts. Detailed cohort/coverage/cold-target shares for all four candidates, denominators and undefined cases are in [evaluation_protocol.md](evaluation_protocol.md) and [the measured JSON](movielens_small_phase2_summary.json).

## Historical ranker feasibility

Primary External Train is split into an earlier historical snapshot (55,807 rows) and later labels (14,379 rows), using per-user whole timestamps. No external Validation/Test row is in either. Snapshot users: 610; users with ≥10 historical ratings: 605; with ≥5 historical positives: 576. There are 607 users with later observations, 574 with a positive label, and **571 catalog-positive queries as an upper bound**, all with possible full-catalog contrast. Of those, 566 have ≥10 historical ratings and 551 ≥5 historical positives.

Later positives are 6,532; snapshot-catalog eligible positives are 6,160. Actual retrieved-positive queries, sampled contrast queries, candidate-pool recall and effective LambdaMART groups remain unknown/null: no retrievers or candidates were generated. Full-Train retrieval predictions are prohibited for these historical examples. The protocol documents future feature allowlists, composite query identity, separate snapshot/full-Train artifacts and required feature-shift measurements. The existing fusion utility already provides `rrf_score`, but the future ranker allowlist must add it explicitly; no ranker implementation was rewritten here.

## Implementation and generated files

New source/configuration:

- `src/recsys/small_protocol.py`: deterministic group assignment, Train-only eligibility/cohorts, label diagnostics and inner windows.
- `src/recsys/small_pipeline.py`: verified pinned raw input, explicit DuckDB CSV schemas/quality/UTC normalization, immutable versioned Parquet/manifests, exact-source and artifact validation.
- `src/recsys/small_benchmark.py`: frozen artifact adapter to retained context/population interfaces, defaulting to Validation and eligible relevance. No fitting or prediction.
- `tools/prepare_small_data.py`, `tools/verify_small_repro.py`: three stage commands plus independent-build/cold-cache recovery checks.
- `tests/test_small_pipeline.py`: 12 synthetic tests; the 21 existing checks remain intact.
- `params.yaml`, `dvc.yaml`, `dvc.lock`, `pyproject.toml`, `uv.lock`, `.python-version`, `.dvc/config`, `.dvc/.gitignore`, `.dvcignore`, raw DVC pointer and data-directory ignore metadata.
- Documentation: protocol, pipeline/environment/recovery instructions, this report, Phase 2 summary/verification JSON. Updated README and requirements compatibility shims; `.gitignore` covers local UV caches and permits data-directory DVC ignore metadata.

Generated/ignored outputs:

```text
data/
  raw/ml-latest-small/                       # unchanged Kaggle inputs/provenance/ZIP
  raw/ml-latest-small.dvc                    # track this pointer in Git
  splits/small_v1/                           # primary
  splits/small_global_v1/                    # separate sensitivity
  diagnostics/small_protocol_v1.json          # measured four-way comparison
  validation/small_raw_v1.json
  validation/small_splits_v1.json
```

Each split directory has 14 Parquet files and one manifest: train, validation, test, movies, item eligibility, user cohorts, label-side targets, user/movie mappings, user cutoffs, ranker snapshot Train, ranker label window, ranker query diagnostics and ranker cutoffs. Manifests contain source/provenance SHA256, policy/ratios/rounding/ties, counts/cohorts, artifact hashes/types, processing-source hashes, Python/DuckDB versions and lockfile hash. CSV/ZIP/Parquet/generated manifests are excluded from Git, versioned through DVC instead. The local cache/remote are approximately 7.6 MB each; the remote is outside the source tree at `/Users/vincent/Documents/projects/hybrid_recsys_dvc_remote`.

## Actual environment and commands

Python 3.11.9; uv 0.13.6; DVC 3.67.1; DuckDB 1.5.2; NumPy 2.4.4; Pandas 3.0.2; PyArrow 24.0.0; SciPy 1.17.1. The existing `.venv` was preserved. No Conda migration/command occurred. PyTorch 2.11.0/implicit 0.7.2 remain available from the pre-existing installation and unchanged, outside this data-phase lock. LightGBM/MLflow/FastAPI were not installed. The authoritative pyproject has base data dependencies, DVC dev group and notebook/acquisition extras; old requirements files now reference it without duplicate pins.

Core commands actually executed from repository root:

```sh
pwd
git status --short
git branch --show-current
git rev-parse HEAD
.venv/bin/python -m pip install uv dvc
.venv/bin/uv lock
.venv/bin/uv sync --locked --inexact --all-extras
UV_CACHE_DIR=.uv-cache .venv/bin/uv sync --locked --inexact --all-extras
UV_CACHE_DIR=.uv-cache .venv/bin/uv lock --check --offline

DVC_NO_ANALYTICS=1 .venv/bin/dvc init
DVC_NO_ANALYTICS=1 .venv/bin/dvc config core.analytics false
DVC_NO_ANALYTICS=1 .venv/bin/dvc config cache.type copy
DVC_NO_ANALYTICS=1 .venv/bin/dvc config core.autostage false
DVC_NO_ANALYTICS=1 .venv/bin/dvc remote add --default --local local /Users/vincent/Documents/projects/hybrid_recsys_dvc_remote
DVC_NO_ANALYTICS=1 .venv/bin/dvc add data/raw/ml-latest-small
git restore --staged -- .dvc/.gitignore .dvc/config .dvcignore
DVC_NO_ANALYTICS=1 .venv/bin/dvc config --local core.site_cache_dir /Users/vincent/Documents/projects/hybrid_recsys/.dvc/tmp/site-cache

DVC_NO_ANALYTICS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/dvc repro
DVC_NO_ANALYTICS=1 .venv/bin/dvc status
DVC_NO_ANALYTICS=1 .venv/bin/dvc push
DVC_NO_ANALYTICS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B tools/verify_small_repro.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B tools/run_tests.py
git diff --check
```

DVC initialization initially staged its own three files; the listed Git command unstaged only those new paths, returning to the initially empty index. No existing user staging was changed. Local DVC push copied 41 object files, **not Git changes**, to the local remote. The verifier's exact scratch `dvc pull/status` commands and stdout are recorded in `outputs/phase2/reproducibility_573d417bf2c84faf8071aa4c7d751775.json` and summarized in the checked verification JSON. Reproduction commands for future use are in [data_pipeline.md](data_pipeline.md), including the distinction between frozen DVC recovery and a newly timestamped Kaggle acquisition.

## Results and resolved issues

- **33 tests passed, zero failures/skips** in the final current environment. Schemas, source integrity, rounding/ties/short histories, deterministic shuffled-input assignment, no dropped/changed rows, Train-only eligibility/frozen histories, cold/empty queries, metric denominators, inner labels and imports are checked. Retained PyTorch sparse warnings, restricted PyArrow hardware probes and the intentional duplicate-ZIP fixture warning did not fail tests; none was suppressed.
- Both real-data split versions passed exact source-value/multiplicity coverage, within-user chronology/ties, global chronology for the secondary, typed Parquet checks, catalog/cohort/mapping recomputation and historical-window validation. Real-data retained-interface checks load Validation without fitting and match 579 / 62 queries, 6,491 / 4,810 eligible positives and 7,525 / 7,867 catalog items.
- Independent real-data regeneration into disposable output paths was **byte-identical**, including manifests. The no-change `dvc repro` skipped every stage and left all 40 data/report SHA256 values and mtimes unchanged.
- An empty-cache disposable **DVC-only, no-Git workspace** pulled from the persistent local remote: 41 objects fetched, 40 workspace files restored, every SHA256 matched. Only its restored Train copy was then removed; a second pull restored it. Originals, original cache and remote were never deleted. Scratch parameter changes invalidated only preparation/validation; scratch raw changes invalidated all downstream stages and the raw pointer. No altered scratch stage was executed. Final original DVC status is clean.
- Initial CSV parsing exposed a legitimate doubled quote in a movie title; explicit quote/escape handling fixed parsing without ignored errors or row repair. Two initial new-test failures were SQL fixture aliases lacking `AS` for reserved `partition`; corrected fixtures now pass. A first recovery-helper attempt used unsupported DVC `remote get-url`; it was replaced by the installed CLI's `config remote.local.url` and recovery then passed.
- Sandbox cache restrictions affected initial `uv lock --check` and an unprivileged DVC status. Project-local `.uv-cache` and `.dvc/tmp/site-cache` resolved them. They were not package compatibility failures. Lock resolution/sync passed; no heavyweight model package was substituted or reinstalled.

No remaining data-pipeline blocker is known. Remaining limitations are explicit: Kaggle version unavailable from Phase 1, publisher licence conditions require the retained README, accepted cross-user temporal exposure, submission/exposure/selection bias, small sample/cohorts, unknown effective ranker queries and future feature-schema integration. Byte identity is tested on this Mac/runtime, not promised across arbitrary writer versions/platforms. The complete model-training environment and effective recommendation performance remain future work.

Final Git status, raw/source/Phase 1 preservation checks, stage integrity and recovery evidence hashes are in [movielens_small_phase2_verification.json](movielens_small_phase2_verification.json). Data/cache/remote/credentials are excluded from Git; no files are staged, committed or Git-pushed. Existing uncommitted Phase 1 files remain present alongside Phase 2 additions. No MLflow, service/container, cloud provisioning or model training occurred.
