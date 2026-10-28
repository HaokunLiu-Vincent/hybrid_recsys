# Phase 3 — six-retriever training and Validation benchmark

**Completed locally on the Mac CPU.** All six retained implementations were fitted on real `small_v1` External Train, serialized, independently reloaded and evaluated on Validation. Every final model was then fitted and evaluated again in a fresh locked environment; candidate and per-query metric Parquet files were byte-identical across environments. No real-data RRF, historical-snapshot training, ranker, MLflow, serving or Test evaluation occurred.

Base Git revision: `bcb7213b8a084510c26033463bbf4da6a9192cbc` (`main`, initially clean). Phase 3 code is intentionally uncommitted; actual critical-source hashes and the dirty-source path list are recorded in each run. DVC data version: `small_v1`, object MD5 `455595aeb5f24864381e436f11093797.dir`; frozen split-manifest SHA256 `905f1370e919f06da3be728f95010d706b078c05f5c5e455c28da398ee41e799`. The original DVC metadata, processed partitions, manifests, notebooks and historical reports are preserved.

**Temporal interpretation:** this is the accepted per-user chronological educational benchmark. Other users' Train events can follow a user's held-out targets. It is not globally leakage-free or a prospective deployment benchmark. Timestamps are submissions, not verified viewing/exposure. No online CTR or causal satisfaction is measured.

## Checkpoint A — integration audit

| Component | Inspected behavior / validation |
|---|---|
| Shared context | Sorted canonical Train-user / metadata-movie mappings, CSR observed/positive matrices, Train-only counts/profiles, separate 7,525-item eligibility mask; no dense user–item matrix. |
| Popularity | Counts binary Train positives (rating ≥4); deterministic Top-K and all-rated seen exclusions. |
| Genre CBF | Normalized allowed genre vectors and Train-positive user profiles; no title embeddings/tags/external metadata. Empty genre/user profiles have explicit popularity fallback. |
| ItemCF | Binary-positive column cosine; sparse bounded source-item blocks; directed Top-neighbor rows; complete positive history multiplies source→target neighbors. Verified against a tiny dense known answer and neighbor bounds. |
| Implicit ALS | CSR rows are users, columns movies; canonical factors verified. Positive preference 1/confidence 21, observed low preference 0/confidence 2, missing preference 0/confidence 1. These are implicit-feedback training proxies, not verified unknown dislikes. |
| LightGCN | Train-positive symmetric normalized bipartite graph; mean of layers 0–2; exact full-graph propagation/adjoint; one large-batch Adam step per epoch. Tiny forward/backward tests verify gradients and parameter updates. |
| Hybrid Two-Tower | Learned user/item IDs plus historical user genre profiles/item genres through two separate 64-hidden-unit MLPs and normalized 32-dimensional outputs. Minibatch BPR/AdamW; not substituted with matrix factorization. |
| Negative sampling | One uniform Train-catalog unknown per positive; rejects every Train-observed pair, high or low. Unknown means unobserved, not verified negative. |
| Metrics | Retained known-answer Recall/NDCG/HitRate definitions, user-macro averaging, deterministic ordering and zero-hit retention. Empty-relevance metrics remain undefined. |

The Phase 2 generic adapter verifies every split file, which would read Test bytes. New `load_training`/`load_validation` entry points instead use explicit input allowlists. Fitting reads Train, movies and Train-derived mapping/catalog artifacts only; it receives no Validation selector. Validation evaluation never opens Test or mixed target/cohort files. Frozen cutoff-table identity comes from manifest metadata, not a false global cutoff. Portable `context.json` explicitly labels its scalar timestamp as maximum Train timestamp only.

The six algorithm implementations were reused. Changes to the neural algorithms are only finite-loss/gradient failure checks; optimizer objectives and update schedules remain intact. New code provides stage-specific loading, readable CLI orchestration and portable/checksummed serialization. Evaluation rebuilds a Train-only context for frozen-data compatibility/relevance checks; predictions use the reloaded checkpoint context and weights, with no retriever fitting.

## Checkpoints B and C — real-data results

Both baselines passed training → save → independent load → candidates → Validation, plus clean-environment replay, before the remaining four models started. All final models use the same **579 primary queries, 6,491 eligible positives, 7,525-item Train-known catalog and Top-100 budget**. Every model emits 57,900 primary candidate rows, with zero empty-candidate and zero fallback queries in this particular eligible Validation population. Synthetic tests additionally cover legitimate cold/unknown-user fallbacks and fewer-than-K catalogs.

Primary metrics use eligible Policy-B positives only. All Train-rated movies, including low ratings, are excluded. Validation is never appended to history. Exactly 417/6,908 observed Validation positives (6.0365%) are outside the Train catalog, consistently reported for every channel; they are excluded from primary denominators and retained in the separate diagnostic below. No model-specific query cohort was selected and no zero-hit query was dropped.

| Model | Recall@20 | Recall@50 | Recall@100 | NDCG@20 | HitRate@20 | Coverage@100 | Train seconds | Inference seconds |
|---|---|---|---|---|---|---|---|---|
| Popularity | 0.112275 | 0.190662 | 0.286786 | 0.091781 | 0.431779 | 5.33% | 0.0819 | 0.3241 |
| Genre CBF | 0.015106 | 0.025285 | 0.041407 | 0.011679 | 0.103627 | 52.52% | 0.0757 | 0.2688 |
| ItemCF | 0.161575 | 0.262102 | 0.368582 | 0.123064 | 0.523316 | 37.25% | 0.3957 | 0.1808 |
| Implicit ALS | 0.171301 | 0.294286 | 0.399429 | 0.126085 | 0.583765 | 26.72% | 0.4984 | 0.3076 |
| LightGCN | 0.136687 | 0.242046 | 0.365442 | 0.107823 | 0.493955 | 18.18% | 2.7986 | 0.3060 |
| Hybrid Two-Tower | 0.045533 | 0.106658 | 0.203064 | 0.034236 | 0.267703 | 16.96% | 3.7468 | 0.3053 |

Coverage@100 is distinct eligible recommended movies / 7,525. Recall/NDCG/HitRate are user-macro means. Per-channel pool recall equals its Recall@100 because its saved pool is Top-100; no six-channel union recall was calculated. Inference seconds measure the retrieval/scoring calls across all 579 queries; candidate validation/Parquet I/O is separately recorded as candidate-pipeline time. Training seconds include shared Train-context fitting plus model-specific fitting, excluding input hashing and checkpoint serialization. These timings are observations on this Mac, not latency guarantees.

ALS is strongest on these fixed Validation metrics; this does not establish a tuned winner or production performance. Coarse genre similarity has broad catalog coverage but low recall here. No configuration was changed based on these scores, no sweeps were run and Test remains unopened.

### Train-positive-history cohorts

Cells show Recall@100 / NDCG@20. Counts are identical for every model. Cold-history primary Validation has no eligible-positive queries, so its metrics are explicitly undefined rather than fabricated zero. Sparse estimates have only 12 queries and are descriptive.

| Model | Cold 0 positives | Sparse 1–4 (n=12) | Moderate 5–19 (n=200) | Warm 20+ (n=367) | Zero-hit queries @100 |
|---|---|---|---|---|---|
| Popularity | undefined (n=0) | 0.243056 / 0.027249 | 0.339136 / 0.089930 | 0.259688 / 0.094900 | 175 |
| Genre CBF | undefined (n=0) | 0.027778 / 0.039107 | 0.041318 / 0.008111 | 0.041901 / 0.012726 | 422 |
| ItemCF | undefined (n=0) | 0.402778 / 0.088829 | 0.462243 / 0.120215 | 0.316422 / 0.125735 | 127 |
| Implicit ALS | undefined (n=0) | 0.402778 / 0.085456 | 0.495727 / 0.140245 | 0.346842 / 0.119696 | 97 |
| LightGCN | undefined (n=0) | 0.277778 / 0.027940 | 0.446112 / 0.104926 | 0.324347 / 0.112014 | 125 |
| Hybrid Two-Tower | undefined (n=0) | 0.125000 / 0.012774 | 0.234752 / 0.026646 | 0.188347 / 0.039074 | 195 |

### Separate all-observed-positive diagnostic

This namespace uses all 581 users with any observed positive and all 6,908 observed positives, including Train-unknown targets. Its evaluation ID differs from the primary ID; primary benchmark ranking/table values above never mix these denominators. Two users with no eligible targets are retained here with zero attainable hits. Each model has a separate diagnostic candidate file and per-query metrics.

| Model | Queries | Relevant positives | Recall@100 | NDCG@20 |
|---|---|---|---|---|
| Popularity | 581 | 6908 | 0.281504 | 0.091125 |
| Genre CBF | 581 | 6908 | 0.040690 | 0.011618 |
| ItemCF | 581 | 6908 | 0.362444 | 0.122086 |
| Implicit ALS | 581 | 6908 | 0.393904 | 0.125184 |
| LightGCN | 581 | 6908 | 0.360590 | 0.106994 |
| Hybrid Two-Tower | 581 | 6908 | 0.199129 | 0.034023 |

## Fixed CPU budgets and objective details

All models use seed 42, one CPU/BLAS/OpenMP/PyTorch thread, 32-query batches and 1,024-item inference blocks. All 70,186 Train interactions enter the context; appropriate positive-only models use all 34,363 Train-positive edges. There is no training subsample.

- Popularity / Genre: deterministic Train counts and genre/profile construction, no optimizer.
- ItemCF: 100 neighbors per source item; 64-item sparse similarity blocks; no full dense item–item matrix.
- ALS: 32 factors, 20 native CPU iterations, regularization 0.1, positive confidence increment 20, low-rating increment 1, alpha 1.
- LightGCN: 32 dimensions, two propagation layers, 40 epochs, batch 2,048 for loss accumulation, learning rate 0.03, mean-base-embedding L2 coefficient 0.0001. Exactly one Adam update per epoch, preserving the retained large-batch semantics.
- Hybrid Two-Tower: 32 dimensions, hidden 64, 20 epochs, batch 1,024, learning rate 0.003, AdamW weight decay 0.00001. Item/user towers consume their allowed genre features as well as learned IDs.

Neural training uses the retained minimum observed epoch Train-BPR snapshot rule, not Validation selection. LightGCN logs loss before its epoch update; Two-Tower logs losses during minibatch updates; exported snapshots are after the epoch. Both selected the final configured epoch (40 / 20) in this run. No Validation early stopping or hyperparameter tuning was used; patience settings are inactive without a Validation selector. Finite losses/gradients are required, and saved raw base/network states were reloaded and checked against exported vectors.

| Model | Context fitting s | Algorithm fitting s | Peak sampled fit RSS MiB | Selected epoch |
|---|---|---|---|---|
| Popularity | 0.0819 | 0.0001 | 309.3 | not applicable |
| Genre CBF | 0.0756 | 0.0002 | 309.2 | not applicable |
| ItemCF | 0.0707 | 0.3250 | 322.7 | not applicable |
| Implicit ALS | 0.0729 | 0.4255 | 319.2 | not applicable |
| LightGCN | 0.0732 | 2.7253 | 498.5 | 40 |
| Hybrid Two-Tower | 0.0732 | 3.6735 | 482.1 | 20 |

Peak RSS is sampled process memory (including native allocations), not an exact allocator peak. CPU resource/RAM/disk checks and full environment details are captured per run. CUDA and MPS were not used.

## Environment and reproducibility

Python 3.11.9, uv 0.13.6; training group pins PyTorch 2.11.0, implicit 0.7.2 and threadpoolctl 3.6.0. Numerical/data versions: NumPy 2.4.4, SciPy 1.17.1, Pandas 3.0.2, PyArrow 24.0.0, psutil 7.2.2, PyYAML 6.0.3, pytz 2026.5. The existing `.venv` was preserved; its PyTorch and implicit installations were not reinstalled or replaced, and no Conda migration occurred.

An isolated environment at `/private/tmp/hybrid_recsys_phase3_clean_20261027` was created from `uv.lock` with `include-system-site-packages=false`, no inherited model/data libraries and no notebook requirement. **All six models were actually trained, reloaded and evaluated there**, not merely imported. Its candidate files and per-query metric files match the main environment byte-for-byte. The working and isolated environments each pass all 45 tests, zero failures/skips. This verifies this Mac/locked CPU setup; no cross-platform bitwise promise is made.

The first clean test exposed a genuine undeclared `pytz` requirement in DuckDB's timezone conversion. It was pinned in base dependencies and both full suites rerun successfully. Initial provisional baseline artifacts under `phase3_seed42_main/replay` remain untouched and unselected. Final baseline and four-model gates use `phase3_v1_seed42_main/replay` under the corrected lock. The provisional artifact reader/source signatures predate the final compatibility checks; use their recorded source version if inspecting them. No completed run was overwritten or deleted, and these repeats were reproducibility audits, not tuning.

Official reference for ALS orientation/confidence: [implicit CPU ALS API](https://benfred.github.io/implicit/api/models/cpu/als.html). Environment groups and lock management use [uv dependency management](https://docs.astral.sh/uv/concepts/projects/dependencies/).

## Artifacts, compatibility and exact commands

Every selected checkpoint is self-contained for loading with the matching code/dependencies: model parameters or factors, canonical mappings, observed/positive sparse Train histories, Train-derived context features, derived per-user Train cutoffs, configuration/seed, training log and manifest. Neural checkpoints additionally retain the selected original graph-base / tower-network and optimizer/RNG states. `COMPLETE.json` hashes every payload file, including the manifest; loaders reject corruption, source/framework/dependency mismatches, wrong model/config, split identity, mappings or catalog. Model artifacts do not need an active fitting session to load.

| Model | Selected checkpoint directory | Primary candidate Parquet SHA256 |
|---|---|---|
| Popularity | `artifacts/popularity/phase3_v1_seed42_main` | `5da5aebbae2d153ddf739f40cb07f309b87f0c1be739f0dbf2970885535f5b2f` |
| Genre CBF | `artifacts/genre/phase3_v1_seed42_main` | `5d10f39122e293467272bc80004e16efae2a8f4973aa95b1ffc835a641216f16` |
| ItemCF | `artifacts/itemcf/phase3_v1_seed42_main` | `fafbca63145167a687865a5cdc3f92a8da8ab053c2a66976d104e530d578c1ef` |
| Implicit ALS | `artifacts/implicit_als/phase3_v1_seed42_main` | `52c2df8ec9febb1e31d7056199d79bbd562d03a48200bc8ab45d923995be8566` |
| LightGCN | `artifacts/lightgcn/phase3_v1_seed42_main` | `03a3836429911b02987e3637370971d23f7febb025f3c4117362a89debab798c` |
| Hybrid Two-Tower | `artifacts/two_tower/phase3_v1_seed42_main` | `ec801ade43dc71941f6ecb3e646013e5583cd36b363c99a4bc70c7c9840cc2a9` |

Selected candidate directory for each model is `artifacts/candidates/<model>/phase3_v1_seed42_main/validation_main/`. It contains `candidates.parquet`, separate all-observed candidates, both per-query metric tables, metrics/manifest JSON and a checksum marker. Full checksums, configurations, exact timings, per-user cutoff-table identity and source/dependency provenance are in [phase3_retrieval_benchmark.json](phase3_retrieval_benchmark.json). The six-model aggregate is `outputs/phase3/phase3_v1_validation_benchmark/benchmark.csv` and `benchmark.json`. No candidate files were merged or fused.

Exact argv/environment/log records for the 24 final train/evaluate operations plus the benchmark command are in [phase3_commands.json](phase3_commands.json). The clean replay commands additionally set `UV_PROJECT_ENVIRONMENT=/private/tmp/hybrid_recsys_phase3_clean_20261027`. Main training commands used `uv run --locked --no-sync --group train` to avoid duplicating the existing inherited model wheels; the clean environment was genuinely synced with the train group.

To reproduce into **new immutable run IDs** in the already validated environment:

```sh
for model in popularity genre itemcf implicit_als lightgcn two_tower; do
  UV_CACHE_DIR=.uv-cache .venv/bin/uv run --locked --no-sync --group train recsys train \
    --model "$model" --config "configs/$model.yaml" --run-id my_seed42_replay
  UV_CACHE_DIR=.uv-cache .venv/bin/uv run --locked --no-sync --group train recsys evaluate \
    --model "$model" --split validation \
    --checkpoint "artifacts/$model/my_seed42_replay" --run-id validation_replay
 done

UV_CACHE_DIR=.uv-cache .venv/bin/uv run --locked --no-sync --group train recsys benchmark \
  --split validation --run-id my_validation_benchmark \
  --evaluations artifacts/candidates/{popularity,genre,itemcf,implicit_als,lightgcn,two_tower}/my_seed42_replay/validation_replay
```

Do not reuse these names if already present; omit optional run IDs for unique UTC/UUID identifiers. Benchmarking requires exactly six explicit compatible evaluation artifacts and never fits missing models. `--split test` is rejected in Phase 3.

For a fresh environment with uv available, create a normal `.venv` on Python 3.11.9 and run `uv sync --locked --group train` first; then `uv run --locked --group train recsys ...` can manage that fresh environment. `--no-sync` is for an already verified environment, not a claim that dependencies need no installation. The isolated check actually executed:

```sh
UV_PROJECT_ENVIRONMENT=/private/tmp/hybrid_recsys_phase3_clean_20261027 UV_CACHE_DIR=.uv-cache \
  .venv/bin/uv sync --locked --no-default-groups --group train --python .venv/bin/python
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B tools/run_tests.py
PYTHONDONTWRITEBYTECODE=1 /private/tmp/hybrid_recsys_phase3_clean_20261027/bin/python -B tools/run_tests.py
```

## Preservation, limitations and Phase 4

No real Test target file, raw full-rating file or historical ranker-label window was read by the training/evaluation workflow. The final preservation audit checks source-data file metadata without reopening Test targets, plus hashes of the allowed Train/Validation inputs, split manifests and DVC metadata. Historical notebook/reports remain unchanged. No real split generation, DVC repro/push or data reversioning was executed.

**DVC environment dependency drift is deliberate and disclosed:** frozen `dvc.lock` records the Phase 2 pyproject/lock hashes; adding training dependencies changes those two source inputs. Do not run `dvc repro` to silently regenerate immutable `small_v1` under the new training lock. The data object, manifests and DVC metadata remain original. A future pipeline/environment separation can be reviewed independently; no false claim of a newly clean DVC repro is made here.

No retriever work remains incomplete. Limitations remain: accepted cross-user future exposure, selected observed ratings rather than exposure logs, small cohorts, one fixed configuration/seed, strict source/dependency compatibility, and no Test evaluation or model-release selection. Results are initial baselines, not an optimization study or causal comparison.

The six compatible Validation candidate artifacts are ready for a separately requested RRF phase. **LambdaMART still requires separately trained historical snapshot retrievers inside External Train.** Never use these full-Train models' in-sample predictions as leakage-free ranker examples. The existing ranker feature allowlist also still needs explicit RRF-score integration/versioning in that later phase. MLflow, API/Docker, cloud resources and final Test remain out of scope. No Git commit or push was made.
