# MovieLens 32M / Colab reference archive

These files were preserved on 28 October 2026 (Australia/Sydney) while cleaning up the existing repository for a future MovieLens Latest Small, CPU-first local MLOps MVP. They are research/reference material, not the active execution path. No data, split, model training or cloud work was performed during this cleanup.

## Why these groups were kept

- `notebooks/`: all 11 previous 32M notebooks, including saved EDA/Split/legacy outputs and the later modular Colab notebooks. File bytes and outputs are unchanged. The old encoded source payload is retained only inside its historical notebook; active code never extracts it.
- `configs/`: the nine old benchmark/32M experiment configurations. They are not approved MovieLens Small settings.
- `src/recsys/`: superseded 32M loaders, fixed counts, orchestration and benchmark lifecycle modules, plus exact original copies of mixed modules before generic helpers were separated or CPU/path defaults adjusted.
- `tests/`: original mixed correctness/training/32M workflow suites and historical validation reports. Their claims refer to past work, not this cleanup. Useful non-fitting tests were retained in the active suite; no old test file was destroyed.
- `tools/`: old Colab notebook generators and validators. Do not invoke them as Small-dataset setup.
- `README.movielens32m.md`: the exact previous root README. Its Colab and 32M execution instructions are superseded.

Existing older files in the parent `archive/`—including SageMaker/SBERT code and weights—were not moved or edited.

## Recovery and compatibility

- Verified external recovery snapshot: `/Users/vincent/Documents/projects/hybrid_recsys_recovery/cleanup_20261027T151822Z`.
- Every archived original matches its pre-cleanup SHA256. The snapshot additionally contains the entire pre-cleanup readable source tree and original paths. This reference archive is deliberately not installed as a second package or made into a separately runnable codebase.
- Some original mixed files also have a smaller active counterpart. Those are intentional original-reference copies, not competing active implementations.
- Relative imports, old Drive paths and manifests are preserved as historical evidence, not patched to run in place. If historical reproduction is needed, restore selected originals from the recovery snapshot into an explicitly reviewed environment. Never feed old 32M manifests/results to MovieLens Small.
- No existing archive destination was overwritten. Git history and the index were left untouched.

## Exact old → new mapping

Paths below are repository-relative. `copy original` means the old version was archived before minimally editing the active counterpart.

| Old path | Archived path | Action |
|---|---|---|
| `README.md` | `archive/legacy_movielens32m/README.movielens32m.md` | move original; replace root with transition README |
| `configs/common.json` | `archive/legacy_movielens32m/configs/common.json` | move original |
| `configs/fusion.json` | `archive/legacy_movielens32m/configs/fusion.json` | move original |
| `configs/genre.json` | `archive/legacy_movielens32m/configs/genre.json` | move original |
| `configs/implicit_als.json` | `archive/legacy_movielens32m/configs/implicit_als.json` | move original |
| `configs/itemcf.json` | `archive/legacy_movielens32m/configs/itemcf.json` | move original |
| `configs/lightgbm.json` | `archive/legacy_movielens32m/configs/lightgbm.json` | move original |
| `configs/lightgcn.json` | `archive/legacy_movielens32m/configs/lightgcn.json` | move original |
| `configs/popularity.json` | `archive/legacy_movielens32m/configs/popularity.json` | move original |
| `configs/two_tower.json` | `archive/legacy_movielens32m/configs/two_tower.json` | move original |
| `notebooks/01_MovieLens32M_EDA.ipynb` | `archive/legacy_movielens32m/notebooks/01_MovieLens32M_EDA.ipynb` | move original |
| `notebooks/02_MovieLens32M_Data_Split.ipynb` | `archive/legacy_movielens32m/notebooks/02_MovieLens32M_Data_Split.ipynb` | move original |
| `notebooks/03_MovieLens32M_Retrieval_and_Ranking.ipynb` | `archive/legacy_movielens32m/notebooks/03_MovieLens32M_Retrieval_and_Ranking.ipynb` | move original |
| `notebooks/03_Popularity_Genre.ipynb` | `archive/legacy_movielens32m/notebooks/03_Popularity_Genre.ipynb` | move original |
| `notebooks/04_ItemCF.ipynb` | `archive/legacy_movielens32m/notebooks/04_ItemCF.ipynb` | move original |
| `notebooks/05_Implicit_ALS.ipynb` | `archive/legacy_movielens32m/notebooks/05_Implicit_ALS.ipynb` | move original |
| `notebooks/06_LightGCN.ipynb` | `archive/legacy_movielens32m/notebooks/06_LightGCN.ipynb` | move original |
| `notebooks/07_Two_Tower.ipynb` | `archive/legacy_movielens32m/notebooks/07_Two_Tower.ipynb` | move original |
| `notebooks/08_Retrieval_Fusion.ipynb` | `archive/legacy_movielens32m/notebooks/08_Retrieval_Fusion.ipynb` | move original |
| `notebooks/09_LightGBM_Ranking.ipynb` | `archive/legacy_movielens32m/notebooks/09_LightGBM_Ranking.ipynb` | move original |
| `notebooks/10_Final_Evaluation.ipynb` | `archive/legacy_movielens32m/notebooks/10_Final_Evaluation.ipynb` | move original |
| `src/recsys/artifacts.py` | `archive/legacy_movielens32m/src/recsys/artifacts.py` | copy original; retain generic active counterpart |
| `src/recsys/config.py` | `archive/legacy_movielens32m/src/recsys/config.py` | move original |
| `src/recsys/data.py` | `archive/legacy_movielens32m/src/recsys/data.py` | copy original; retain generic active counterpart |
| `src/recsys/evaluation.py` | `archive/legacy_movielens32m/src/recsys/evaluation.py` | copy original; retain generic active counterpart |
| `src/recsys/experiment.py` | `archive/legacy_movielens32m/src/recsys/experiment.py` | move original |
| `src/recsys/final_evaluation.py` | `archive/legacy_movielens32m/src/recsys/final_evaluation.py` | move original |
| `src/recsys/fusion/workflow.py` | `archive/legacy_movielens32m/src/recsys/fusion/workflow.py` | move original |
| `src/recsys/rankers/workflow.py` | `archive/legacy_movielens32m/src/recsys/rankers/workflow.py` | move original |
| `src/recsys/retrievers/lightgcn.py` | `archive/legacy_movielens32m/src/recsys/retrievers/lightgcn.py` | copy original; retain generic active counterpart |
| `src/recsys/retrievers/two_tower.py` | `archive/legacy_movielens32m/src/recsys/retrievers/two_tower.py` | copy original; retain generic active counterpart |
| `src/recsys/runtime.py` | `archive/legacy_movielens32m/src/recsys/runtime.py` | copy original; retain generic active counterpart |
| `src/recsys/smoke.py` | `archive/legacy_movielens32m/src/recsys/smoke.py` | move original |
| `src/recsys/training.py` | `archive/legacy_movielens32m/src/recsys/training.py` | copy original; retain generic active counterpart |
| `src/recsys/workflows.py` | `archive/legacy_movielens32m/src/recsys/workflows.py` | move original |
| `tests/NOTEBOOK03_VALIDATION.md` | `archive/legacy_movielens32m/tests/NOTEBOOK03_VALIDATION.md` | move original |
| `tests/REFACTOR_VALIDATION.md` | `archive/legacy_movielens32m/tests/REFACTOR_VALIDATION.md` | move original |
| `tests/refactor_preserved_notebooks.json` | `archive/legacy_movielens32m/tests/refactor_preserved_notebooks.json` | move original |
| `tests/test_modular.py` | `archive/legacy_movielens32m/tests/test_modular.py` | move original |
| `tests/test_recsys.py` | `archive/legacy_movielens32m/tests/test_recsys.py` | move original |
| `tools/build_framework_notebooks.py` | `archive/legacy_movielens32m/tools/build_framework_notebooks.py` | move original |
| `tools/build_notebook03.py` | `archive/legacy_movielens32m/tools/build_notebook03.py` | move original |
| `tools/smoke_notebook.py` | `archive/legacy_movielens32m/tools/smoke_notebook.py` | move original |
| `tools/validate_framework.py` | `archive/legacy_movielens32m/tools/validate_framework.py` | move original |
| `tools/validate_notebook03.py` | `archive/legacy_movielens32m/tools/validate_notebook03.py` | move original |

Full four-way classification, cache inventory and checks: [cleanup audit](../../docs/repository_cleanup_audit.md).
