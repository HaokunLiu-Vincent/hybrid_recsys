# Phase 5 — local MLflow and complete recommender release

Completed on 2026-10-28, locally on Mac CPU. **Gates A, B, C and D passed.** Existing real models were not fitted again. Historical experiment import, new integration actions and new label-free inference checks are distinct below. No final Test targets were opened/evaluated; no FastAPI recommendation service, Docker image or cloud infrastructure was built.

## Current release and verified result

- Tracking / Registry / UI: **http://127.0.0.1:5002**; one localhost-only MLflow worker remains running.
- Registered model: **`MovieLensHybridRecommender`, version `1`, alias `candidate`**.
- Registry URI: **`models:/MovieLensHybridRecommender@candidate`**.
- Release run: `b99a65b035ed4f6e868bd846153d14cb`, experiment `MovieLens-Small-Release` (ID 4).
- Local inference bundle: `artifacts/releases/phase5_v1/`; parity evidence: `outputs/release_checks/phase5_v1/`.
- **611 requests / 12,220 Top-20 rows matched exactly**, including all 610 known Train users and one unknown user. The independent registry process had no project installed and denied **every file read of the original repository**.
- The 579 existing Validation users' **11,580 prior Phase 4 Top-20 user/movie/rank/score rows also matched exactly**. Saved rankings were used only as a verification baseline; they are absent from the model bundle and never used by inference.
- **71 tests passed, zero failures/skips**, in both working and separately locked environments; all original 61 tests remain unchanged.

This is a **Validation-selected release candidate**, not a production approval or final Test estimate. No champion/production alias was assigned.

## Initial audit and protected provenance

Initial Git was clean on `main` at `b110fe6959ce47e7a2872acfc793210ccd6508d5`. Phase 4 had since been committed, contrary to the older report's then-current uncommitted status. Nothing was committed, staged, pushed, reset, stashed or cleaned in this phase. New Phase 5 code is currently uncommitted; the bundle records the actual preceding Git revision, dirty source status and exact packaged Python/lock hashes, not an invented clean release commit.

Inspected AGENTS/README, evaluation/data-pipeline/Phase 2–4 reports, machine-readable metrics/commands/verification, retained retrievers and interfaces, feature/fusion/ranker/artifact utilities, packaging and DVC metadata. All **29 selected Phase 3/4 completed artifact directories** passed byte/checksum and applicable source/runtime loader checks before MLflow work. Their actual selected manifests, not transcribed expected metric counts, drive import and packaging.

The before-edit audit is `outputs/phase5/initial_audit.json`: **254 protected file SHA256 values** and **43 frozen data size/mtime records**. Every one remained unchanged. Existing Python source changes are limited to CLI additions; old algorithms, contracts, features, ranker, split/data preparation, notebooks and historical reports remain byte-identical. The initial clean tree required no recovery copy of uncommitted authored work. No original data/checkpoint/output was deleted or overwritten.

Frozen source: Kaggle `grouplens/movielens-latest-small`; split `small_v1`, per-user chronological 70/15/15. Train 70,186 / Validation 15,143 / Test 15,507 (counts from the old manifest). Split manifest SHA256 **`905f1370e919f06da3be728f95010d706b078c05f5c5e455c28da398ee41e799`**. The per-user cutoff identity is **`per_user:9859a72f9f04efe290c109f073f6e845cf82a5490b488f688e5f2ef8cc2ec1d6`**, not a false global maximum timestamp. Canonical catalog/mappings and all-Train-rated exclusion remain unchanged.

MLflow, Torch, implicit, LightGBM, DuckDB, pandas and NumPy imported together. Python is **3.11.9**, uv 0.13.6, MLflow **3.17.2**, SQLAlchemy 2.1.6. MLflow 3.17.2 was the current compatible stable version resolved from PyPI and then pinned exactly. `uv.lock` SHA256 is `e405905fdb6152228784727fcafa66749ac546a9887dc6eef7254254aaa584e9`. Existing heavyweight model wheels were preserved. A fresh Python 3.11.9 environment was built at `/private/tmp/hybrid_recsys_phase5_isolated`, synced with train/rank/mlops and **`--no-install-project`**; it contains dependencies but no installed `recsys`.

## Gate A — actual local Tracking and Registry

Persistent SQLite: `outputs/mlflow/mlflow.db`; HTTP-served artifacts: `outputs/mlflow/artifacts/`; owned PID/port/URIs: `outputs/mlflow/server.json`; append-only log: `outputs/mlflow/server.log`. Absolute SQLite URI: `sqlite:////Users/vincent/Documents/projects/hybrid_recsys/outputs/mlflow/mlflow.db`. Registry and Tracking share this database. No DVC remote/cloud artifact store is used.

Port 5000 was occupied. The owned server initially selected 5001; a startup-probe TIME_WAIT issue during a development restart selected 5002. The probe was corrected to reuse a released socket address, and **the persisted 5002 URI then survived both the Gate A test and the final registered-model restart**. The server binds only 127.0.0.1; no unrelated process was killed. Graceful stop waits for the owned process to exit before restart.

Actual HTTP checks created/discovered an experiment, logged/read parameters and metrics, uploaded/downloaded exact artifact bytes, queried a run, created/read a registry probe version, and recovered those records after restart. Probe run `ef65d7f23cec4541b622ceeb1fbf6eca`, registry `MovieLensLocalHealthProbe` version `3`. Earlier labeled health probes were preserved. The final recommender version and predictions were also verified after another server restart, without creating another version.

Server/import network operations required the tool sandbox's authorized localhost/network permission; import-only tests were not substituted for real server verification. Start/stop/status and backup/recovery instructions are in [local setup](mlflow_local_setup.md).

## Gate B — retrospective experiment import

Exactly **17 logical historical records** are imported: six Full-Train retrievers, six historical retrievers, Validation RRF, historical union, label-free features, ranking dataset, and LambdaMART. Candidate manifests/checksums belong to the corresponding retriever record; hundreds of redundant runs were not created.

| Original logical record | Experiment ID | MLflow run ID |
| --- | --- | --- |
| phase3/retriever/popularity | 2 | `8f69d9f3dc7c461a99e20179cd673a64` |
| phase3/retriever/genre | 2 | `8cfa9cd3b7a542b094b79fe78e42ed21` |
| phase3/retriever/itemcf | 2 | `601e2b43c07d4142a8a2d3ce2dc7cebc` |
| phase3/retriever/implicit_als | 2 | `ec73e29974e84720ab6e65e756277583` |
| phase3/retriever/lightgcn | 2 | `9bc4801c454b47fdba52abadd06fb862` |
| phase3/retriever/two_tower | 2 | `05cbb86624284c678a5898b5c4b24e9e` |
| phase4/rrf | 3 | `e77c1f9912654d2aa6f70eaf7440e4c0` |
| phase4/historical_retriever/popularity | 2 | `94a5c3db4aca4bbdb902b00ee1952c5a` |
| phase4/historical_retriever/genre | 2 | `692c099aca754cef8ef6e2c5b087fcfc` |
| phase4/historical_retriever/itemcf | 2 | `308dcd3a4bb640588fa8917a028e6cf7` |
| phase4/historical_retriever/implicit_als | 2 | `2ba0c1ecc1f84b92bc29b8f9b1fb3a6b` |
| phase4/historical_retriever/lightgcn | 2 | `232b4bec29df4a29a5ddf2f9e3ab17df` |
| phase4/historical_retriever/two_tower | 2 | `34d65768182d46c395f38bd3c670d10f` |
| phase4/historical_union | 3 | `ba90622692e54e0aabad004bf1b83c6a` |
| phase4/features | 3 | `b945be6af80743ab9047dd53f0873a1f` |
| phase4/ranking_dataset | 3 | `5699b1d6b56442508f2d305ba7f7beec` |
| phase4/lambdamart | 3 | `8f80d7d5b6a74015801f0398fb0cd760` |

`MovieLens-Small-Retrieval` is experiment 2; `MovieLens-Small-Fusion-Ranking` is experiment 3. Explicit upstream run-ID references connect the Full-Train models to Validation fusion, historical models to their union/features, the ranking table to LambdaMART, and all of these to the release run. Historical models are **training lineage**, not deployment weights.

Each import records `retrospective_import=true`, original execution phase/run ID, original training outside MLflow, source manifest/completion identity, original Git/source and lock hashes, DVC object/metadata identity, policy/cutoffs/evaluation IDs, original configuration/seed/packages/timings, import timestamp and importer source hash/version. Original timestamps absent from manifests are explicitly **not recorded**; import time is not relabeled as training time. Some original training was dirty-source work; its actual status/hash metadata is retained.

Source manifests, checksummed file inventories, candidate identities and diagnostic artifacts are copied into distinct original-path namespaces. Full weight/history copies are concentrated in the complete release rather than duplicated into each tracking record. Original source paths are provenance references, not inference dependencies. Metadata/metrics come directly from completed manifests; none were newly invented.

Primary metrics use `validation/policy_b/*`; all-observed diagnostics use `validation/all_observed/*` with their different evaluation ID. Full-pool ceilings use `validation/union/*`; ranking feasibility uses `historical_ranker/*`. Sampled historical training trajectories remain explicitly named in the original training-history artifact and are not presented as unconditional Validation scores.

Repeated import verified original source records and exact parameter/metric values and returned the same 17 IDs, creating **zero duplicates**. Conflicting provenance/metrics and incomplete imports are rejected. During development an absent optional tag value was corrected to an explicit unavailable string; already finished records were verified/reused, not deleted or overwritten. [Imported run metadata and programmatic comparison](phase5_mlflow_runs.json) retain the actual API values.

## Programmatic comparison of completed experiments

The following was queried from MLflow by `tools/compare_mlflow.py`, not manually transcribed as a new benchmark:

| Original record | Recall@20 | NDCG@20 | HitRate@20 |
| --- | --- | --- | --- |
| phase3/retriever/popularity | 0.112275 | 0.091781 | 0.431779 |
| phase3/retriever/genre | 0.015106 | 0.011679 | 0.103627 |
| phase3/retriever/itemcf | 0.161575 | 0.123064 | 0.523316 |
| phase3/retriever/implicit_als | 0.171301 | 0.126085 | 0.583765 |
| phase3/retriever/lightgcn | 0.136687 | 0.107823 | 0.493955 |
| phase3/retriever/two_tower | 0.045533 | 0.034236 | 0.267703 |
| phase4/rrf | 0.164938 | 0.127578 | 0.552677 |
| phase4/lambdamart | 0.187106 | 0.142799 | 0.614853 |

ALS is the best individual on these Recall@20/NDCG@20/HitRate@20 metrics. Channel coverage differs: Genre's Top-100 covers 52.52% of the catalog versus ALS 26.72%, without comparable standalone recall. All six source streams use 579 queries, 6,491 eligible positives and Top-100/source. The union expands potential query-macro recall from ALS's Top-100 0.399429 to **0.596664** across **213,507** pairs; these are different candidate budgets, not a claim of universally better Top-K ranking. Micro union positive coverage is 3,270/6,491 = 0.503774.

RRF Top-20 has slightly higher NDCG but lower Recall/HitRate than ALS Top-20. The six preserved ablations show ALS contributes 417 unique relevant pairs; Genre contributes 112 despite low standalone Recall. Removing Popularity/Two-Tower can improve fixed-RRF NDCG while losing candidate recall. Full overlap, popularity concentration and ablations are linked in the imported original JSON and unchanged [Phase 4 report](phase4_fusion_and_ranking.md).

LambdaMART had **528 actual informative groups / 57,936 rows**, distinct from 571 potentially eligible queries. Its six independent snapshots used 55,807 earlier interactions; later 14,379 Train interactions supplied labels only after candidate/features were frozen. Sampling retains retrieved positives/lows within useful groups and up to 100 unknowns, with no inserted unretrieved positives. Its 53-feature schema includes RRF and source-specific score/rank/presence/fallback features. Selected iteration 31; RRF score has greatest gain importance, followed by ALS score and ItemCF rank. Feature importance, all training rounds, sampled-vs-natural feature shift, and effective counts are logged as artifacts.

The 12-user sparse cohort regresses: Recall@20 0.194444 under RRF versus 0.069444 under LambdaMART. Overall gains are descriptive and Validation-selected; no significance, production lift, exposure/CTR or causal claim follows. Cross-user future-information exposure is still accepted by the per-user protocol; it is not globally leakage-free.

## Gate C — complete portable inference bundle

`artifacts/releases/phase5_v1/` contains:

```text
manifest.json, COMPLETE.json
context/                 # ONE shared verified Train context
retrievers/<six sources>/ # native counts, sparse ItemCF or exported model vectors
ranker/                  # model.txt and exact feature_schema.json
src/recsys/              # versioned readable application code
pyproject.toml, uv.lock, requirements.txt, README.md
```

There are **65 payload files / 7,775,208 bytes**, plus the completion inventory. Shared context includes canonical mappings, Train-only observed/positive sparse histories, eligibility, counts/popularity, genre features/user profiles and the serialized per-user cutoff table. All six source contexts were verified byte-identical before storing one shared copy. Retriever states are the selected **Full-Train Phase 3** states, never the historical snapshots. Neural exported inference vectors preserve the selected models; unnecessary optimizer/training states are not packaged.

No Validation/Test target rows, label windows, candidate Parquets, saved predictions or historical retriever weights are bundled. Aggregate Validation metrics are provenance metadata. The package contains Train user interaction history and is an educational local artifact, not a public distribution of user data.

- Semantic release ID: **`e4b38328a35c992114eb3e6acda90803f0aa16c61a027686d9e86b06bf5e475d`**.
- Complete inventory SHA256: **`7776b30a4cd2f392676da1096e5a22b77b7f8d091f5259c2cd0a3b5e868a7a95`** (binds every relative filename, byte count and payload SHA256).
- Release manifest SHA256: `c859217da1492c5c4add7cdd57751da5de98c739458f001ce6ddc909977fc601`.
- Feature schema logical fingerprint: **`21b298dc0cef61b0de9514c0db66f4e38cd9a89818bb8fe94694d0251ad48ae0`**, version `small_lambdamart_features_v1`, **53 float32 inputs**.

Schema order, NaN absent-source scores, missing-rank 101/presence/fallback 0, fixed transforms and RRF constant 60 are preserved. Loaders verify complete file inventories, current packaged code hashes, Python/dependency versions, Full-Train identities, catalog/mappings/cutoffs, native ranker feature order and schema identity. All old inference algorithms are called directly; no precomputed lookup recommendations or second algorithm implementation was introduced.

Known-user inference generates Top-100 from **each** retained retriever, enforces Train-known catalog/all-rated exclusion, deduplicates/fuses the full union, builds the same 53 features from Train context, scores every union row and sorts by score then movieId. Output preserves requested user order and ranks. Known zero-positive users follow the retained per-source fallbacks using actual history statistics; unknown users bypass the personalized ranker and receive Train-derived popularity, without invented embeddings/features.

Input is `userId:int64,top_k:int64`, unique positive IDs, top_k 1..100. Output is `userId:int64,movieId:int64,rank:int32,score:float64,fallback:string`. Fallback labels are `none`, `known_zero_positive_history`, `unknown_user_popularity`. Unknown popularity scores and known LightGBM scores are not calibrated probabilities or comparable across those paths. Invalid IDs/bools/floats/strings/nulls, duplicate requests and invalid K are rejected. Fewer available candidates yields fewer rows, without padding/duplicates. No histories or training labels are returned.

Reference predictions were generated **before packaging** from original checkpoints and the retained Phase 4 fusion/features/ranking functions, using all 610 serialized Train users plus unknown ID 1000610. This covers warm/moderate/sparse and all four known zero-positive-history users without choosing users by future outcomes. Native package reload matched user/movie/rank/score/fallback exactly, tolerance zero. Reference inference took **6.655 s**; packaged inference **6.810 s**, both local batch timings, not API latency estimates.

## Gate D — registration and independent loading

MLflow **Models From Code** packages the PythonModel entry point, supported code paths, explicit input/output signature and a valid input example. Native model state is read from relative artifact paths; application objects are not cloudpickled. The release's uv project/lock and pinned requirements are retained. Supported model versions, aliases and tags are used; no deprecated stage promotion API is called.

API choices were checked against installed MLflow 3.17.2 signatures and the official [Tracking Server](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/), [pyfunc packaging](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pyfunc.html) and [registry client](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html) documentation.

Version 1 tags include `release_status=validation_candidate`, dataset/split/protocol, `ranking_model=lightgbm_lambdamart`, retriever_count=6, rrf_constant=60, feature_count=53, test_evaluated=false, and release/schema/source/lock checksums. The release run is a **new integration action**, with explicit original MLflow run references; its copied Validation metrics are the historical observations above, not a new fitting/evaluation run.

Re-registering this exact completed release returned **version 1 / created=false**. The only alias is `candidate:1`. An existing candidate for another release is rejected rather than silently reassigned. Read-only verification creates no run/version. After another owned-server restart at the same URI, the candidate resolved and all 611 requests still matched exactly.

The decisive portability test copied only verification inputs/reference and a standalone verifier to `/var/folders/tg/wlyh4rfn4cg7fnv67kjr7mq80000gn/T/phase5-registry-independent-booovdkg`. It ran the isolated dependency-only environment with **`python -I -B`**, from that disposable directory, with no installed `recsys`. An audit hook rejected **all reads under the original repository**, including original artifacts and DVC data. MLflow downloaded the registry artifacts, imported application code from `/private/var/folders/tg/wlyh4rfn4cg7fnv67kjr7mq80000gn/T/tmpz9wcku6g/code/recsys/__init__.py`, validated the bundle, and reproduced **12,220 rows exactly**. Invalid input cases also raised. Original checkpoints remained untouched; they were not renamed/deleted to simulate isolation.

This proves loading/prediction through the stored registry bundle without source-checkout/data/Validation dependencies. It does not claim binaries work across arbitrary Python/platform/version changes. Environments/caches must remain outside immutable releases; MLflow supplies application code through code paths. Exact inference runtime requirements:

| Package | Version |
| --- | --- |
| numpy | 2.4.4 |
| pandas | 3.0.2 |
| scipy | 1.17.1 |
| pyarrow | 24.0.0 |
| psutil | 7.2.2 |
| PyYAML | 6.0.3 |
| pytz | 2026.5 |
| torch | 2.11.0 |
| implicit | 0.7.2 |
| threadpoolctl | 3.6.0 |
| lightgbm | 4.6.0 |
| mlflow | 3.17.2 |

The complete transitive lock, including integration/server libraries, is also retained. MLflow emits a nonfatal recommendation to add predict type hints for inferred signatures; an explicit signature and explicit runtime validation are supplied and tested. Sandbox PyArrow hardware-probe warnings are retained. No failures were suppressed.

## Tests, failures resolved and preservation

All original **61** tests plus **10** focused Phase 5 tests passed: working runner **12.974 s**, isolated runner **17.644 s**, zero failures/skips. Added tests use disposable SQLite/artifact stores and synthetic model fixtures, distinctly separate from the actual release checks. They cover retrospective metadata, namespace separation, discovery/idempotency/conflict rejection, all-six packaging/one shared context/no target files, known/unknown prediction, invalid inputs, Top-K/seen/catalog/ordering, corruption/dependency/mapping rejection, historical-checkpoint rejection, complete pyfunc registration, candidate alias safety and native/registry prediction parity. Actual all-user checks additionally cover four cold users and all prior Validation rankings.

The first new-suite run exposed an installed-API difference for missing aliases; the implementation now reads the public registered-model alias mapping. Tests then passed without weakening assertions. Two tiny JSON provenance artifacts from the initial tracking-test attempt were inventoried under ignored `mlruns/` and preserved; final tests explicitly use temporary artifact locations. No pre-existing MLflow data, experiments or versions were deleted to make tests pass.

`ast.parse` passed for **63 active Python files**. `uv lock --check --offline`, `git diff --check`, all completed release/parity inventories, protected hashes and frozen metadata passed. [Verification JSON](phase5_verification.json) contains exact checksums, runtime, model tags, gates, tests, final status and isolated-process evidence; [command records](phase5_commands.json) contain actual argv/exit codes/timings. Complete logs are under ignored `outputs/phase5/`.

DVC tracks raw/processed data; MLflow tracks experiments/models. The unchanged Phase 2 `dvc.yaml` still depends on global pyproject/uv.lock, and its writer embeds whole-lock identity. Adding MLflow changes environment dependencies but **not** frozen data objects or split manifests. No real DVC repro/push, lock editing, new data version or hidden regeneration occurred. The dedicated future data-environment migration proposal remains deferred.

## Reproduction and next phase

See [local setup](mlflow_local_setup.md) for complete start/stop, environment, backfill, package, registry, inference, independent verification and backup/recovery instructions. Existing completed IDs refuse overwrite; use the recorded `phase5_v1` for idempotent registration, and new package IDs for new code/config releases. A different current candidate requires explicit reassignment approval. The exact active registry load is:

```python
import mlflow
import pandas as pd
mlflow.set_tracking_uri("http://127.0.0.1:5002")
mlflow.set_registry_uri("http://127.0.0.1:5002")
model = mlflow.pyfunc.load_model("models:/MovieLensHybridRecommender@candidate")
recommendations = model.predict(pd.DataFrame({"userId": [1, 42], "top_k": [20, 20]}, dtype="int64"))
```

Mutable SQLite, artifacts and logs remain ignored local files. They persist across restarts but are **not automatically backed up remotely or copied to GitHub**. The local model contains Train histories; protect it accordingly. Strict source/runtime compatibility and local store recovery are operational limitations. Final Test remains unopened; accepted temporal exposure, unknown-feedback proxies, sampling/feature-distribution mismatch, sparse-cohort regression and selection optimism remain unchanged scientific limitations.

Ready for separately requested **Phase 6: FastAPI + Docker**, using an explicitly selected complete model version, load-at-startup inference, health/version and offline-serving parity tests. No service/container implementation or production approval is implied by this phase. Current source changes are uncommitted on `main` at `b110fe6959ce47e7a2872acfc793210ccd6508d5`; full working-tree status is in verification JSON.
