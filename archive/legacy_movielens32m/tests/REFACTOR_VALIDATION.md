# Modular framework migration and validation

The eight requested notebooks, readable producer modules, JSON configurations, candidate contracts, tests and project README are implemented. No model implementation is a placeholder. No real MovieLens 32M experiment or historical snapshot was trained during this structural refactor.

## Final source tree

```text
hybrid_recsys/
├── .git/ and .gitignore                         # preserved
├── AGENTS.md                                   # read, not modified
├── README.md
├── requirements.txt
├── archive/                                    # untouched
├── configs/
│   ├── common.json
│   ├── popularity.json
│   ├── genre.json
│   ├── itemcf.json
│   ├── implicit_als.json
│   ├── lightgcn.json
│   ├── two_tower.json
│   ├── fusion.json
│   └── lightgbm.json
├── notebooks/
│   ├── 01_MovieLens32M_EDA.ipynb                 # preserved with outputs
│   ├── 02_MovieLens32M_Data_Split.ipynb          # preserved with outputs
│   ├── 03_MovieLens32M_Retrieval_and_Ranking.ipynb # legacy, preserved with outputs
│   ├── 03_Popularity_Genre.ipynb
│   ├── 04_ItemCF.ipynb
│   ├── 05_Implicit_ALS.ipynb
│   ├── 06_LightGCN.ipynb
│   ├── 07_Two_Tower.ipynb
│   ├── 08_Retrieval_Fusion.ipynb
│   ├── 09_LightGBM_Ranking.ipynb
│   └── 10_Final_Evaluation.ipynb
├── src/recsys/
│   ├── __init__.py
│   ├── data.py
│   ├── contracts.py
│   ├── artifacts.py
│   ├── features.py
│   ├── config.py
│   ├── runtime.py
│   ├── metrics.py
│   ├── evaluation.py
│   ├── workflows.py
│   ├── smoke.py
│   ├── final_evaluation.py
│   ├── retrievers/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── neural.py
│   │   ├── popularity.py
│   │   ├── genre.py
│   │   ├── itemcf.py
│   │   ├── implicit_als.py
│   │   ├── lightgcn.py
│   │   └── two_tower.py
│   ├── fusion/
│   │   ├── __init__.py
│   │   ├── candidates.py
│   │   ├── rrf.py
│   │   ├── diagnostics.py
│   │   └── workflow.py
│   ├── rankers/
│   │   ├── __init__.py
│   │   ├── features.py
│   │   ├── dataset.py
│   │   ├── lightgbm_ranker.py
│   │   └── workflow.py
│   ├── experiment.py                            # legacy orchestration API
│   ├── candidate_fusion.py                      # compatibility imports
│   └── ranking_features.py                     # compatibility imports
├── tests/
│   ├── helpers.py
│   ├── test_recsys.py
│   ├── test_modular.py
│   ├── refactor_preserved_notebooks.json
│   ├── NOTEBOOK03_VALIDATION.md                 # historical previous-task report
│   └── REFACTOR_VALIDATION.md
└── tools/
    ├── build_framework_notebooks.py
    ├── validate_framework.py
    ├── run_tests.py
    ├── smoke_notebook.py
    ├── build_notebook03.py                      # old encoded generator disabled
    └── validate_notebook03.py                   # redirects to framework validation
```

The tree omits ignored caches and existing Drive synchronization temporary directories. Runtime `data/`, `artifacts/` and `results/` paths are documented in README; this task did not create real datasets or benchmark outputs there.

## Created and modified files

Created eight new notebooks; nine JSON configurations; `README.md` and `requirements.txt`; new contracts/artifacts/features/retrieval-workflow/smoke/final-evaluation modules; the `fusion/` package; ranker feature/dataset/workflow modules; synthetic fixture/contract tests; notebook-preservation hashes; and four framework build/test/validation tools.

Modified existing shared `data.py`, `config.py`, `runtime.py`, `training.py`, `evaluation.py` and `retrievers/base.py` for development/Test separation, independent factories and CPU operation. Migrated common candidate/feature functions into their new responsibility-specific modules while preserving compatibility imports in `candidate_fusion.py` and `ranking_features.py`. Extended `rankers/lightgbm_ranker.py` with bounded Parquet/Sequence fitting. Updated `tests/test_recsys.py` to share fixtures, bound synthetic CPU threading and fix the Top-10 assumption. Disabled the encoded legacy builder and redirected its validator.

The six algorithm files—Popularity, Genre, ItemCF, Implicit ALS, LightGCN and Two-Tower—were reused. Their bytes were compared with the source members of the preserved legacy notebook's bundle and match exactly. The bundle was inspected only; it is not extracted or embedded by any new notebook. The common retrieval adapter adds the canonical new source name `implicit_als`; legacy `als` remains supported by legacy APIs.

All three existing notebooks remained byte-for-byte unchanged, including execution counts, errors and saved charts/tables. Their recorded hashes are in `refactor_preserved_notebooks.json`. The archive, source CSVs and actual Drive split artifacts were not modified. Existing Git changes from the earlier archive operation were left in place. No commit or push was made.

## Actual local checks

After explicit user approval, `nbformat` and LightGBM with their required dependencies were installed only in `/private/tmp/recsys_refactor_test_deps_20261026`. Existing Conda environments were not altered.

- **21 synthetic tests passed, zero skipped.** Real optimized implicit ALS and real LightGBM fitting were exercised. This includes the native LightGBM Sequence interface and the migrated sklearn helper; the Top-10 test preserves the returned candidate set when only seven items are eligible.
- Tests cover known-answer metrics, canonical mappings, eligibility/seen filtering, finite scores, deterministic ties/ranks, duplicate rejection, shared baseline preparation, temporal boundaries, development without Test deserialization, compatible/incompatible manifests, corrupt checkpoints, independent model save/load, and model-specific source hashes.
- A disposable 48-row Train / 25-row historical-prefix experiment exercised six independent full-Train-style synthetic runs, six synthetic historical snapshots, saved candidate fusion, sampling, actual LambdaMART fitting, Validation comparison, selection freezing, explicitly gated Test inference, result reuse and unchanged frozen fixture checksums. It is not a MovieLens benchmark.
- **All 11 notebooks passed `nbformat.validate()`**, including the three preserved notebooks and all eight new ones. All code cells and Python modules passed syntax checks; existing IPython syntax was transformed for static checking where needed. All readable package modules imported successfully.
- **All eight new notebooks passed top-to-bottom synthetic smoke execution in fresh local processes.** Only the project path was overridden to this checkout. No notebook output was written back, no real dataset was loaded, and all distributed new code cells retain null execution counts and empty outputs.

Locally used versions include Python 3.11.9, PyTorch 2.11.0, implicit 0.7.2, LightGBM 4.7.0 and nbformat 5.11.1. Local hardware was CPU-only macOS, not an L4 GPU. The local native BLAS/OpenMP environment can fail when CPU pools are resized/unbounded in a mixed-library process. The supplied runner starts a fresh thread-bounded process, and synthetic LightGBM uses one CPU thread. This is a test setting, not a measured throughput configuration. Tiny-data LightGBM warnings about uninformative features and sparse PyTorch beta warnings are not hidden or treated as benchmark findings.

Reproduce using an environment with the documented dependencies:

```sh
python tools/run_tests.py
python tools/validate_framework.py --preserved-hashes tests/refactor_preserved_notebooks.json
PYTHONPATH=src python tools/smoke_notebook.py notebooks/03_Popularity_Genre.ipynb
```

## Execution and sharing

1. Copy/clone `src/`, `configs/`, `tests/`, and `notebooks/` to the documented Google Drive project path. Ensure required packages are available; installation remains explicit.
2. Open `03_Popularity_Genre.ipynb`, keep `RUN_MODE="smoke_test"`, and run top to bottom. Choose `full_train` deliberately afterward for the real baselines. It uses all external Train rows.
3. Run notebooks 04–07 independently in fresh sessions. Their only data dependency is the frozen dataset. Explicit checkpoints support `load_evaluate` without fitting; completed matching runs are verified/reused.
4. Paste the six candidate directories into Notebook 08. It validates compatibility and streams fusion without retraining. Raw scores, ranks, source-presence and fallback provenance are retained.
5. For ranking later, explicitly configure the same historical UTC cutoff/label end in each retrieval notebook and select `rank_snapshot`. Fuse those saved historical candidates separately. Notebook 09 consumes historical fusion and full-Train Validation fusion; it contains no six-model training loop.
6. Review Validation results, freeze exact artifacts in Notebook 10, and enable Test only with the matching selection ID. No fitting, tuning or Validation-history update occurs in final mode.

## Intentional compatibility choices and remaining work

No new model module needs completion before controlled experiments. Full-scale GPU execution/resource profiling, real snapshot training and real benchmark evaluation still require Colab and were not attempted. Actual Drive manifest boundaries/counts/checksums are runtime checks, not locally attested dataset results.

The full-scale ranker intentionally uses native LightGBM `Dataset`/`Sequence` with the same `lambdarank` objective instead of concatenating all raw features for `LGBMRanker.fit`. Compressed bins, label/group vectors and histogram workspace still require sufficient RAM. Source/version, dataset and cutoff checks are strict; code/config changes create distinct runs. Full-Train versus historical feature shifts are measured for Validation and Test but remain a scientific limitation.

New `modular_v2` artifacts are not silently converted from old monolithic `recsys_v1` outputs. An explicit provenance-checked legacy converter is optional future work if those artifacts need reuse; old artifacts remain untouched. Completed-stage resume is implemented, while automatic mid-epoch neural state restoration is not. Disk/RAM/VRAM estimates are conservative and do not guarantee L4 completion or Drive quota availability.
