# Existing repository cleanup audit

**Date:** 28 October 2026, Australia/Sydney. **Scope:** audit, recoverability, archival and minimal cleanup of the existing repository only.

## 1. Initial state and recovery

- Working directory / repository root: `/Users/vincent/Documents/projects/hybrid_recsys`.
- Branch: `main`; HEAD: `946fad13331cfb0191e33e47104bdecf00242aba`. The repository and Git history were retained in place; no new repository or parallel Small codebase was created.
- Initially **21 tracked paths**, **89 untracked files**, **19 tracked working-tree deletions**, and a modified root README. There were no staged changes. The 19 deletions already existed from the earlier archive operation; they were not caused by this cleanup.
- The README described a Colab/32M workflow, but the actual filesystem was inspected independently: Python definitions/imports, all tests and configs, tools, notebook cell/output inventories, the existing archive, ignored caches and potential data/runtime paths.

Initial `git status --short`:

```text
 D 01_EDA_From_S3.ipynb
 D 02_Model_Development.ipynb
 D 03_SageMaker_Pipeline.ipynb
 M README.md
 D env_set_up.txt
 D setup_env.sh
 D src/inference.py
 D src/requirements.txt
 D src/sbert_model/1_Pooling/config.json
 D src/sbert_model/README.md
 D src/sbert_model/config.json
 D src/sbert_model/config_sentence_transformers.json
 D src/sbert_model/model.safetensors
 D src/sbert_model/modules.json
 D src/sbert_model/sentence_bert_config.json
 D src/sbert_model/special_tokens_map.json
 D src/sbert_model/tokenizer.json
 D src/sbert_model/tokenizer_config.json
 D src/sbert_model/vocab.txt
 D src/train.py
?? AGENTS.md
?? archive/
?? configs/
?? notebooks/
?? requirements.txt
?? src/recsys/
?? tests/
?? tools/
```

### Recovery snapshot created before any repository edits

`/Users/vincent/Documents/projects/hybrid_recsys_recovery/cleanup_20261027T151822Z`

This private sibling directory is outside the repository and all cleanup/deletion targets. It is a backup, not another Git repository or active implementation. Its root was created with mode `0700`. It contains `workspace/`, `inventory.json`, `git_status_before.txt`, `verification.json` and recovery instructions.

- **91 current non-cache files, 95,085,276 bytes** copied, including every present tracked/untracked authored file and all pre-existing archive files/weights.
- All copied files were checked against their source SHA256 values. **Every one of the 91 files was then copied back into disposable external staging and checked again**, demonstrating a full restore round trip before moves/edits.
- All backup hashes were rechecked after cleanup. The original `.git/` was neither copied nor altered; the snapshot records the baseline HEAD and paths already absent. Those historical tracked versions remain recoverable from the unchanged existing Git history, and already-archived local versions are also captured.
- `.git/`, compiled caches and recognized secret paths are excluded from the snapshot. No recognized credential files or high-confidence secret-pattern matches were detected among candidate files. This was a name/pattern screen, not a guarantee of an exhaustive security audit. No credentials, remote URLs or secret values were printed. The Git credential/config store was not read into the backup.
- The snapshot name uses UTC (`20261027T151822Z`), corresponding to 28 October locally. Keep this backup until the migration is reviewed; do not remove it as cache. Restore selected files into a review directory first, rather than blindly overwriting subsequent work.

## 2. Four-way inventory and disposition

The complete **pre-edit per-file classification**, sizes, hashes, reasons and old/new destinations are in [repository_cleanup_inventory.json](repository_cleanup_inventory.json). The action log is [repository_cleanup_actions.json](repository_cleanup_actions.json). Counts refer to the initial inventory; newly written cleanup documentation/tests are additional files.

| Classification | Initial entries | Decision and reason |
|---|---:|---|
| KEEP ACTIVE | 33 | Reusable readable source, fixtures, local test wrapper, instructions, ignore/dependency references. Seven mixed files were first preserved in full in the new archive, then minimally edited. |
| ARCHIVE | 38 | Clearly superseded 32M/Colab notebooks, benchmark configs, old orchestration, generators, mixed old suites and historical documentation. Moved, never deleted. |
| DELETE SAFE CACHE | 60 | Individually listed CPython `.pyc` files, each with an existing corresponding source file. Their hashes were rechecked before unlinking. |
| REVIEW / DO NOT TOUCH | 20 | Every pre-existing archive file, including the 90,864,192-byte SBERT `model.safetensors`, tokenizer files, SageMaker code and notebooks. Kept at original locations and rehashed. |

### KEEP ACTIVE

- `src/recsys/retrievers/`: Popularity, Genre CBF, ItemCF, Implicit ALS, LightGCN, Two-Tower, the common interface and negative-sampling/math utilities.
- `fusion/`: candidate deduplication/provenance, RRF and diagnostics; `rankers/`: weak-label features, query-preserving Parquet access and LightGBM helpers.
- `metrics.py`, `contracts.py`, `features.py`, `artifacts.py`, `runtime.py` and generic history/evaluation/factory functions.
- `tests/helpers.py`, and the new `tests/test_core_contracts.py` carrying nine unchanged non-fitting test bodies from the original suites plus three cleanup/serialization isolation checks.
- `tools/run_tests.py` remains the same local thread-bounded runner. New `pytest.ini` confines optional pytest discovery to `tests/` and excludes `archive/`; pytest was not installed or run.
- `AGENTS.md` was read and left unchanged. `README.md`, `.gitignore` and dependency-reference prose were updated conservatively.

### ARCHIVE — exact destination

`archive/legacy_movielens32m/`

| Original group | New location | Disposition |
|---|---|---|
| All 11 `notebooks/*.ipynb` | `archive/legacy_movielens32m/notebooks/` | Byte-preserving moves; EDA/Split/legacy outputs retained. |
| All nine `configs/*.json` | `archive/legacy_movielens32m/configs/` | Old benchmark sizing/boundaries/configuration retained; no replacement Small configs invented. |
| Previous top-level README | `archive/legacy_movielens32m/README.movielens32m.md` | Original moved before writing the transition-state root README. |
| `experiment.py`, `workflows.py`, `final_evaluation.py`, `smoke.py`, `config.py`, `fusion/workflow.py`, `rankers/workflow.py` under `src/recsys/` | Same relative paths below `archive/legacy_movielens32m/` | Old 32M/Colab execution paths removed from active imports. |
| Five notebook build/smoke/validation tools | `archive/legacy_movielens32m/tools/` | Superseded Colab tooling retained as reference; `tools/run_tests.py` stays active. |
| Two old mixed test suites, two historical validation reports, preservation-hash JSON | `archive/legacy_movielens32m/tests/` | Originals preserved; training/32M workflow tests are not run during cleanup. |

The archive has its own [README and full 45-entry old→new table](../archive/legacy_movielens32m/README.md) and a machine-readable [path_mapping.json](../archive/legacy_movielens32m/path_mapping.json). The **45 original files** comprise 38 moves plus seven deliberate original-version copies. No destination existed beforehand, and exclusive destination creation prevented overwrites.

### Mixed-file compatibility work — not an algorithm rewrite

Original versions were archived before changes:

- `data.py`: retained `Context`, `build_context`, `relevance` verbatim; removed the 32M constants, frozen-v1 loader/checker and old internal split procedure from the active module.
- `evaluation.py`: retained `MetricAccumulator` and `generate_candidates` verbatim; old all-channel benchmark orchestration/full-table helpers remain archived.
- `training.py`: retained independent `load_model`, `validation_selector`, `fit_retriever`; removed six-channel milestone wrappers. Loader default changes to CPU.
- `artifacts.py`: provenance hash lists no longer reference moved configs/orchestrators. Existing artifact schemas were not silently relabeled as MovieLens Small.
- `runtime.py`: default environment path is `/tmp` rather than `/content`; CUDA diagnostics no longer mandate an L4; generic artifact-copy wording replaces Drive-specific wording.
- LightGCN and Two-Tower fitting functions default to CPU rather than CUDA. Their model/optimization bodies are unchanged; no fitting was performed.

AST comparison verified all retained computation definitions, allowing only those explicit device-default changes. Four other concrete retriever files remain byte-identical to the recovery snapshot. Generic compatibility imports (`candidate_fusion.py`, `ranking_features.py`) remain because reusable code still uses them. No duplicate active model implementations were generated.

### DELETE SAFE CACHE — actual removals

Only these inventoried cache groups were touched:

| Cache directory | `.pyc` files | Bytes |
|---|---:|---:|
| `src/recsys/__pycache__` | 26 | 257,561 |
| `src/recsys/fusion/__pycache__` | 5 | 23,759 |
| `src/recsys/rankers/__pycache__` | 7 | 42,474 |
| `src/recsys/retrievers/__pycache__` | 18 | 86,605 |
| `tests/__pycache__` | 4 | 92,349 |

Total: **60 files, 502,748 bytes**. Only exact inventoried `.pyc` paths with matching pre-cleanup hashes were unlinked. Empty cache directories were removed afterward. Empty original `notebooks/` and `configs/` organizer directories were removed only after their contents had been archived; unexpected contents would have prevented removal. No broad `rm -rf`, Git cleanup, reset or stash was used.

### REVIEW / DO NOT TOUCH and deferred decisions

- All 20 older archive files were left untouched. In particular, the existing SBERT weights were not treated as disposable merely because a new ignore rule covers model files. Explicit archive visibility exceptions preserve their existing Git visibility.
- No local root `data/`, `artifacts/`, `results/`, `mlruns/`, `mlartifacts/`, `.dvc/`, `.env` or `.aws` was found. This is a local filesystem observation, not a claim that no external/Drive datasets or checkpoints exist.
- Historical references to `/content/drive/MyDrive/Colab Notebooks/personal_projects/movielens_hybrid_recsys/data/ml-32m`, `data/splits/v1`, artifacts and results were not followed, mounted, moved or deleted. Any actual external datasets/checkpoints require a separate location/retention decision.
- No unclassified source file was deleted. Untracked status was never considered evidence of disposability; clear obsolete roles were archived only after full recovery verification.
- Keep the external backup and pre-existing archived model until a separate retention decision. Packaging/version pinning, bounded CPU configs and credential/auth choices remain for later authorized work.

## 3. Ignore rules and current/proposed tree

`.gitignore` now excludes data/output/model paths, compiled/test/build caches, MLflow stores/databases, DuckDB/SQLite runtime databases, DVC cache/tmp/local config, and local secrets. It allows ordinary source, `.env.example`, DVC pointer files (including pointers under `data/`), `dvc.yaml`, `dvc.lock` and `.dvc/config`. **23 `git check-ignore --no-index -q` cases passed**, including visibility of the existing archived SBERT weights. See [repository_cleanup_ignore_checks.json](repository_cleanup_ignore_checks.json). No DVC files or runtime directories were initialized.

Actual transition tree (archive internals omitted):

```text
hybrid_recsys/
├── .git/                         # same history/index; nothing staged
├── .gitignore
├── AGENTS.md                     # unchanged
├── README.md                     # transition state, no completion claims
├── requirements.txt              # retained reference; no install/lockfile created
├── pytest.ini                    # archive-safe optional discovery only
├── src/recsys/
│   ├── data.py, evaluation.py, training.py
│   ├── contracts.py, metrics.py, features.py, artifacts.py, runtime.py
│   ├── candidate_fusion.py, ranking_features.py
│   ├── retrievers/               # six reusable models + shared utilities
│   ├── fusion/                   # reusable functions, no legacy workflow
│   └── rankers/                  # features/Parquet/LightGBM, no old workflow
├── tests/
│   ├── helpers.py
│   └── test_core_contracts.py
├── tools/run_tests.py
├── docs/repository_cleanup_*     # inventory, actions, checks and this audit
└── archive/
    ├── ...                      # 20 prior files unchanged
    └── legacy_movielens32m/      # preserved references + original-path mapping
```

Proposed later MVP additions, **not created today**:

```text
pyproject.toml + reproducible environment
src/recsys/ acquisition/preparation/split/release/serving adapters
DVC pipeline definitions + selected local remote
ignored data/raw and data/processed Parquet
ignored model/run/release outputs
local non-synced DuckDB and MLflow stores, outside DVC remotes
FastAPI/Docker entry points + integration/parity tests
```

This is a future layout within the same repository. DuckDB/DVC/MLflow/FastAPI/Docker setup, new temporal protocol, data acquisition, serving, model training, cloud/GPU resources, commits and pushes are all intentionally deferred.

## 4. Verification commands, actual results and limitations

No dependencies were installed. Checks used the existing `/Users/vincent/miniconda3/envs/SINA-mv-Recsys/bin/python` environment. Installed base LightGBM and nbformat were unavailable; neither was installed or needed for the selected cleanup checks. Core ranker code imports lazily without fitting. The earlier task's test counts/reports are historical, not substituted for current validation.

Actual targeted test command:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /Users/vincent/miniconda3/envs/SINA-mv-Recsys/bin/python -B tools/run_tests.py
```

- **Final: 12 tests passed, zero skipped, zero failures**, with no model fitting, optimizer steps, real dataset evaluation or acquisition. The known-answer/vector/gradient/sampling checks use tiny synthetic values; they are not Small-dataset experiment results.
- First attempt: 11 passed, one failed with `NameError: iter_queries is not defined` in the migrated Parquet-query test. The missing original import was restored; the assertion/test body was not removed or weakened.
- Old tests that fit ItemCF/ALS/neural/ranker models or invoke obsolete frozen-32M workflows remain unchanged in the archive and were deliberately not run. Useful non-fitting assertions were retained. No failing 32M-contract test was hidden by weakening its expected values.
- PyTorch sparse-beta warnings and sandbox-restricted Arrow CPU-cache `sysctlbyname` diagnostics appeared; all final assertions passed. No warnings were suppressed. Future CPU training/performance still needs a separate bounded configuration review.
- The full final test output is [repository_cleanup_test_results.txt](repository_cleanup_test_results.txt).

Additional exact commands run (stdlib scripts were created under `/private/tmp` for this cleanup, not installed into a new application):

```sh
python3 /private/tmp/hybrid_cleanup_inventory.py
python3 /private/tmp/hybrid_cleanup_backup.py
python3 /private/tmp/hybrid_cleanup_plan.py
python3 /private/tmp/hybrid_cleanup_apply.py
python3 /private/tmp/hybrid_cleanup_verify.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /Users/vincent/miniconda3/envs/SINA-mv-Recsys/bin/python -B /private/tmp/hybrid_cleanup_static.py
git check-ignore --no-index -q <each of 23 recorded fixture paths>
git status --short
git diff --cached --quiet
```

The backup command alone used the authorized external sibling location. No secrets, patches containing private values or Git configuration were included in output. Evidence files make the relevant assertions inspectable without relying on temporary scripts:

- [repository_cleanup_verification.json](repository_cleanup_verification.json): **91 backup hashes**, **45 archived originals**, **20 unchanged previous archive files**, **60 deleted bytecodes**, AST identity of retained computation, unchanged HEAD/branch and empty staged diff.
- [repository_cleanup_static_checks.json](repository_cleanup_static_checks.json): **31 active Python files** syntax-checked, **28 current-source modules** imported, no fitting calls in active tests, old loaders/orchestrators absent, and **zero cache directories** remaining.
- All **11 archived notebook JSON structures** checked read-only. Byte hashes—not reserialization—confirm preservation of their complete source/output content. No archived notebook was executed, no full EDA was run, and `nbformat` validation was not claimed for this cleanup.
- All moved/copied originals equal both the pre-edit inventory and the independently verified backup. The seven intentional original/reference copies are distinguished from moves in the path map. No unique input, checkpoint or result was deleted.

### Remaining compatibility review before any Small model work

- Existing candidate conventions, legacy `als` versus canonical `implicit_als` feature names, `modular_v2` manifest labels and source hashes must be reviewed against a new Small dataset/split identity. Nothing was silently migrated into a Small experiment.
- The retained metric helper requires nonempty relevance sets; the new Small protocol must explicitly account for empty-relevance and cold/sparse cohorts rather than silently inheriting the old primary population.
- Historical ranking labels must follow snapshot fitting inside outer Train. Old fixed-32M split loaders and the 30-day internal-tail procedure are archived; no replacement split was invented today.
- LightGCN retains its one-large-batch-optimizer-step-per-epoch formulation. Bounded CPU sizing, optimization checks and reproducibility need future model-stage work. GPU is optional; core defaults are now CPU.
- Existing `Context.save` is not a complete serving release: mappings alone do not constitute histories/retrievers/features/ranker/fusion. A future registered release must bundle and validate all required components, with parity tests. No model registry, endpoint or container was built today.
- Requirements remain a reference, not a tested/reproducible Small environment or a newly resolved lockfile. No claim of end-to-end MVP readiness is made.

## 5. Final Git state and next authorized phase

Branch/HEAD unchanged; **19 pre-existing tracked deletions**, **two tracked modifications** (`README.md`, `.gitignore`), **108 untracked files**; no staged changes. The active source was already untracked at the start and remains uncommitted. No stash/reset/clean/commit/push command was run.

Final `git status --short`:

```text
 M .gitignore
 D 01_EDA_From_S3.ipynb
 D 02_Model_Development.ipynb
 D 03_SageMaker_Pipeline.ipynb
 M README.md
 D env_set_up.txt
 D setup_env.sh
 D src/inference.py
 D src/requirements.txt
 D src/sbert_model/1_Pooling/config.json
 D src/sbert_model/README.md
 D src/sbert_model/config.json
 D src/sbert_model/config_sentence_transformers.json
 D src/sbert_model/model.safetensors
 D src/sbert_model/modules.json
 D src/sbert_model/sentence_bert_config.json
 D src/sbert_model/special_tokens_map.json
 D src/sbert_model/tokenizer.json
 D src/sbert_model/tokenizer_config.json
 D src/sbert_model/vocab.txt
 D src/train.py
?? AGENTS.md
?? archive/
?? docs/
?? pytest.ini
?? requirements.txt
?? src/recsys/
?? tests/
?? tools/
```

Cleanup is complete and stops here. Minimal next phase, only when requested:

1. Acquire **only** Kaggle `grouplens/movielens-latest-small` using an explicitly chosen local/auth method; never silently substitute another source. Record provenance, archive/member/file hashes, schema/count checks and bundled licence attribution. Do not assume already-downloaded data or invent observed counts.
2. Transform `ratings.csv` and `movies.csv` using DuckDB into versioned Parquet with input/output validation. Keep any writable DuckDB database local, outside DVC remotes and cloud-synced locations. Do not fit models or reuse old 32M splits.
3. Configure DVC dataset pointers/pipeline metadata and an explicitly chosen local remote; preserve raw files and processed hashes. Keep tracking metadata in Git and raw/derived data outside Git. Review a new deterministic chronological split/ranker protocol before later model work.

Decisions for later: recovery/archive retention, the approved acquisition/auth approach, local dataset storage/DVC remote locations and the new protocol/environment versions. None is required to preserve this completed cleanup, and no later phase was started.
