# Phase 1 — acquisition and EDA execution report

Executed locally on 28 October 2026 (Australia/Sydney; UTC acquisition date 27 October). Work continued in the existing repository on `main`, starting from clean status at `e2b160b0049802d7c1cf5bed0a26d2504ab0a47f`. Read `AGENTS.md`, README and the cleanup audit before implementation. This report concerns real Small data, not archived 32M results or synthetic benchmark results.

## Acquisition and immutable raw inputs

Source: [Kaggle `grouplens/movielens-latest-small`](https://www.kaggle.com/datasets/grouplens/movielens-latest-small). Official CLI 2.2.5 downloaded the public dataset without configured credentials. No other source was attempted. The separate optional metadata request required authentication, so a Kaggle version number is **unavailable**. The publisher README says generated September 26, 2018; it is not a Kaggle version identifier.

The ZIP contained five flat files. They were safely extracted and normalized to `data/raw/ml-latest-small/`, with the original ZIP preserved at `_source/kaggle-download.zip`. Nested layouts are also supported by the reusable utility, but multiple matching datasets fail rather than being guessed. Existing directories are never replaced. An existing complete acquisition is hash/schema-verified and reused without network activity; a partial or altered dataset fails for inspection.

`data/raw/ml-latest-small/provenance.json` records the exact CLI download command, download timestamp and its basis, all file sizes/SHA256, typed schemas, observed counts, publisher expectations and discrepancies. Download time is `2026-10-27T15:53:42.683065+00:00`, based on local archive ctime after transfer. Kaggle preserved a server last-modified time of `2019-10-08T10:01:31+00:00` as mtime; that historical date is explicitly not treated as this download time.

| File | Bytes | SHA256 |
|---|---:|---|
| ratings.csv | 2,483,723 | `aa289ca83157595d0df6aea1be6a4ded676ddc4385472e8313a8ed9805352646` |
| movies.csv | 494,431 | `5a5f32dd9bb3797b8e728a1b98958789d2b13f294a69fdfbc5727f8a9611aa07` |
| Original ZIP | 993,937 | `12b11d77fdb2c221daa631e5724104f51ecab1298c0e5f535d56e9ceff02ef8b` |

Publisher README, tags and links are preserved and hashed. Only ratings and movies are loaded into Pandas/DuckDB; tags and links are not analytical/modeling inputs. Kaggle reports licence `other`; the bundled unmodified README is the licence/citation reference. No permissive licence or commercial permission is inferred.

## Actual observations

| Check | Result |
|---|---:|
| Ratings / distinct users | 100,836 / 610 |
| Metadata movies / rated movies / unrated metadata movies | 9,742 / 9,724 / 18 |
| Observed positives (`rating >= 4`) / observed low ratings | 48,580 / 52,256 |
| Rating range | 0.5–5.0, half-star increments |
| Required nulls / duplicate user–movie pairs / duplicate movie IDs | 0 / 0 / 0 |
| Ratings without movie metadata | 0 |
| Metadata-catalog interaction sparsity | 98.3032% |
| Adjacent submission pairs with gap ≤60 seconds | 83,398 / 100,226 (83.21%) |
| Adjacent submission pairs with gap ≤30 minutes | 93,691 / 100,226 (93.48%) |
| Users with recorded timestamp span ≤1 day | 362 / 610 (59.34%) |

UTC submissions span `1996-03-29T18:36:55+00:00` to `2018-09-24T14:27:30+00:00`. Publisher expectations for ratings, movies and users match observed counts; no discrepancies were found. Ratings load as int32 IDs, float32 rating and int64 timestamp; metadata uses int32 movie ID and string title/genres. IDs are initially parsed as int64 and downcast only after range checks to prevent integer wraparound.

Rating submissions do not establish movie consumption times. Brief submission gaps and short user spans make naive personalized temporal evaluation fragile. Dataset-wide minimum activity is not evidence of history before a cutoff. Unobserved pairs are unknown, not verified dislikes; no exposure probabilities, online CTR or causal satisfaction can be estimated here.

## Exploratory split recommendation — no partitions frozen

Each alternative uses its own strictly earlier Train prefix for the catalog and seen histories. Eligible future positives are rating ≥4, in the Train-observed movie catalog, and not already rated by that user in Train. Validation and prospective Test use the same frozen Train catalog/history. Equal timestamps stay together. These are aggregate feasibility queries, not stored assignments or model evaluation.

| Alternative | Train rows | Validation / Test rows | Eligible-positive users V / T | Train-known positive coverage V / T | Users without Train ratings V / T |
|---|---:|---:|---:|---:|---:|
| 3-month V / 3-month T | 97,438 | 1,403 / 1,995 | 22 / 28 | 81.47% / 90.22% | 6 / 17 |
| 6-month V / 6-month T | 93,249 | 4,189 / 3,398 | 33 / 39 | 81.77% / 84.22% | 14 / 24 |
| 6-month V / 12-month T | 87,371 | 5,878 / 7,587 | 36 / 59 | 95.47% / 80.18% | 21 / 39 |
| Timestamp 80/10/10 | 80,668 | 10,084 / 10,084 | 62 / 72 | 92.54% / 78.32% | 41 / 54 |

Recommend discussing the timestamp 80/10/10 alternative first because it offers more eligible label queries than the calendar options. Its observed boundaries are `2016-03-22T08:26:11Z` and `2017-06-21T07:17:30Z`: Train is before the first boundary; intervals thereafter include their left boundary. It is not an exact row-count prescription for future versions. This proposal does **not** solve personalization sparsity: 54 of 74 prospective Test users have no Train ratings. Only 21 Validation and 18 Test eligible-positive queries have at least five Train positives. Report all-user results plus prespecified cold/sparse/warm cohorts, retaining empty-relevance counts rather than silently dropping them.

For this option, Validation/Test have 5,198 / 4,128 observed future positives and 4,810 / 3,233 catalog-eligible positives. Train-unknown movies exclude 388 / 895 future-positive interactions. The internal historical snapshot diagnostic has an upper bound of 124 potential ranker queries, before retrieval misses and lack of label contrast. This is not a candidate recall ceiling or effective ranking-training query count. No retrieval or ranking examples were generated.

Inspect the one Validation-boundary crossing within 30 minutes under the timestamp option; exact ties alone do not prevent splitting a submission burst. Prespecify an embargo or sensitivity protocol if warranted. Retain the 6-month/12-month calendar option as a temporal sensitivity. If warm-history estimates remain too small, seek approval for an expanding-window backtest rather than silently replacing chronology or excluding users. Freeze a documented decision before final splits and subsequent Validation-only model selection; prospective Test aggregates here were inspected only for protocol feasibility.

Popularity provides a cold-user baseline; Genre CBF needs a historical profile/fallback; ItemCF needs bounded sparse co-occurrences; ALS/LightGCN/Two-Tower need explicit missing-versus-low conventions and cold-ID treatment. Six-channel budgets/catalogs/cohorts must match before RRF. LambdaMART needs out-of-time retrieval snapshots inside outer Train, useful query contrast, pool-recall accounting and snapshot/inference feature-distribution diagnostics. The notebook details each component without fitting it.

## Files added or updated

- `notebooks/01_MovieLens_Small_EDA.ipynb`: new 28-cell notebook, all requested sections, visible DuckDB analytical SQL, 16 executed code cells.
- `src/recsys/small_data.py`: source-specific safe acquisition, schema/quality validation, provenance/hash verification and no-overwrite reuse.
- `src/recsys/small_eda.py`: read-only in-memory DuckDB diagnostics and exploratory chronological views; no persistent split writer.
- `tools/download_movielens_small.py`, `tools/execute_small_eda.py`: normal source entry points; selected-interpreter execution, no automatic installation or global kernel requirement.
- `tests/test_small_data.py`: nine synthetic tests for schemas, invalid data, ID overflow, traversal/symlink/duplicate ZIP rejection, nested ambiguity, provenance/tampering, no-overwrite and chronological ties/cold counts.
- `requirements-eda.txt`: pinned versions used for Phase 1. No model packages added.
- `README.md`: current real-data state, commands and protocol limitations. `.gitignore`: clarified existing raw/output exclusions; functional rules already covered this data layout and continue to permit DVC metadata.
- This report and `docs/movielens_small_phase1_summary.json`: observed statistics, full cutoff/cohort tables, execution and unchanged-source hashes.
- Ignored local additions: `.venv/`, raw acquisition and `outputs/eda/` visual-check/partial-execution evidence. No old archive entries or retained model code edited.

## Actual commands and validation

The repository had pip requirements and no active pyproject/uv/Poetry configuration. Python 3.11.9 in the existing SINA environment already supplied numerical/DuckDB and core-test libraries. A project environment reused those packages without changing Conda. Commands below were actually executed from the repository root (not a claimed fresh-environment reproduction):

```sh
pwd
git status --short
git branch --show-current
git rev-parse HEAD
/Users/vincent/miniconda3/envs/SINA-mv-Recsys/bin/python -m venv --system-site-packages .venv
.venv/bin/python -m pip install kaggle matplotlib nbformat nbclient
.venv/bin/kaggle --version
.venv/bin/kaggle datasets files grouplens/movielens-latest-small --csv
.venv/bin/kaggle datasets metadata grouplens/movielens-latest-small -p /private/tmp/movielens_small_phase1_kaggle_metadata
.venv/bin/kaggle datasets download grouplens/movielens-latest-small -p /private/tmp/movielens_small_phase1_kaggle_download --quiet
.venv/bin/python -m ipykernel install --prefix .venv --name hybrid-recsys-eda --display-name 'hybrid_recsys Phase 1 EDA'
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B tools/execute_small_eda.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B tools/download_movielens_small.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -B tools/run_tests.py > /private/tmp/movielens_small_phase1_tests.log 2>&1
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B /private/tmp/verify_movielens_small_phase1.py
git diff --check
```

The first downloaded ZIP was published using the new `publish_archive` Python function with the actual CLI receipt now stored in provenance, avoiding a second download. Subsequent downloader entry-point runs verified and reused that acquisition. Project-local kernel registration was performed during setup, but the final executor directly selects its Python in memory and does not depend on registration or alter global kernels.

Results:

- Acquisition/listing succeeded; **optional metadata command failed with authentication required**. No credentials or tokens written or displayed, no alternate data source used. Required data acquisition is not blocked.
- The first notebook launch hit the restricted execution sandbox's localhost socket permission. No analytical cell ran; its partial file is retained under ignored `outputs/eda/execution_failed_1fa8c697b0024d6290ad495d304c1be7.ipynb`. Repeating the same execution command with local IPC permitted succeeded. This was a local IPython kernel, not a hosted service.
- Actual notebook: **16/16 code cells, execution counts 1–16, seven inline PNG figures, 12 code cells with HTML tables, zero error outputs**. All DuckDB queries completed. Extracted figure contact sheet was visually inspected; charts and labels rendered. Tables contain actual outputs.
- **21 tests passed, zero failures, zero skips**: 12 retained non-fitting core checks plus nine new data checks. PyTorch emitted existing sparse-CSR warnings, PyArrow reported restricted hardware probes, and the deliberate duplicate-ZIP fixture emitted its expected duplicate-name warning; none caused a failure. No optimizer/training job ran.
- Raw file hashes verified before/after EDA and reuse. All **28 pre-existing `src/recsys` Python source hashes** match the pre-edit baseline; the two Small modules are additions. Existing synthetic tests check fusion, ranker contracts, metrics, sampling and serialization, not Small model accuracy.
- Final static/ignore checks are recorded in `movielens_small_phase1_verification.json`: Python AST, nbformat, sequential execution, hashes, Git whitespace, raw/output ignore coverage and trackable DVC metadata. The initial and final HEAD match. No files staged, committed or pushed.

The minimal EDA pins do not recreate all older optional model-test dependencies: the complete suite used existing SciPy/PyArrow/psutil/PyTorch from the reused environment. New users can run the nine data tests with `requirements-eda.txt`; the README distinguishes that from the full core suite.

No final Parquet splits, DVC configuration, model training, model benchmark, MLflow service/registration, API/Docker or cloud infrastructure was created. The next phase is an explicit protocol decision, deterministic DuckDB/Parquet preparation with hashes/tie rules, then local DVC versioning.
