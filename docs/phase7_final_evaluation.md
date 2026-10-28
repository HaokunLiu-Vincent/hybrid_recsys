# Phase 7 — frozen final Test evaluation and project handoff

Completed locally on 28 October 2026 (Australia/Sydney), using the original selected Version 1. Gates A, B, C and D passed in sequence. Test findings were preserved without model fitting, tuning, reselection, new model versions, alias changes, commits or pushes. No evaluation correction or protocol violation was required.

## Frozen identities and starting audit

Initial working tree was clean, branch `main`, HEAD **`98a92b54ee3b05fd4ee175442e3bc3991ef436f1`**. The user committed Phase 6 since its original report. There was no uncommitted authored work to recover before edits. The initial audit records **1,022 protected file hashes and 43 frozen data metadata records** in ignored `outputs/phase7/initial_audit.json`. README is the sole intentionally edited pre-existing file; new evaluator/tools/tests/docs are additive. Existing algorithms, all 85 prior tests, serving/Docker configuration, history/archive/notebooks, DVC metadata and completed checkpoints remain intact.

Inspected AGENTS/README, the protocol/data-pipeline/Phase 2–6 reports and machine records, current lock/params/DVC metadata, data adapters, six-channel evaluation/contracts, RRF, feature/ranker interfaces, immutable artifact utilities, complete release loader and serving package. The 222-package lock parsed and passed offline verification. No package installation, dataset acquisition, DVC repro/push, model fitting or cloud activity occurred.

| Identity | Frozen value |
|---|---|
| Source | Kaggle `grouplens/movielens-latest-small`; ratings and movie genres |
| Protocol | `small_v1`, per-user chronological 70/15/15 |
| Split manifest SHA256 | `905f1370e919f06da3be728f95010d706b078c05f5c5e455c28da398ee41e799` |
| DVC split directory object | `455595aeb5f24864381e436f11093797.dir` |
| DVC raw directory object | `587f1f96e195f3158547ce0c5610f030.dir` |
| Train / Validation / Test | 70,186 / 15,143 / 15,507 interactions |
| Fixed registry reference | `models:/MovieLensHybridRecommender/1` |
| Original release MLflow run | `b99a65b035ed4f6e868bd846153d14cb` |
| Semantic release ID | `e4b38328a35c992114eb3e6acda90803f0aa16c61a027686d9e86b06bf5e475d` |
| Schema logical fingerprint | `21b298dc0cef61b0de9514c0db66f4e38cd9a89818bb8fe94694d0251ad48ae0` |
| Native booster SHA256 | `a14b8ecd561056725d200cf8cce023e76086fc8d7aad96426d9a8284ffaa33a7` |
| Train catalog ID | `6b607210c9f671ac865f15a1c1ce2168573246722e66ffae32075e9405c36fd2` |
| Train mapping ID | `06951befc01e3f0ada339b3b9f8129a47639ce26d7342fb591abc1ed42f06977` |
| Per-user Train cutoff identity | `per_user:9859a72f9f04efe290c109f073f6e845cf82a5490b488f688e5f2ef8cc2ec1d6` |
| Runtime | Python 3.11.9; exact release package pins verified |
| Evaluation serving lock SHA256 | `1188c119c8ddb1f1c2bf61feec71efef494f1ac884fb6ffe3a6e5310592970bc` |
| Original release lock SHA256 | `e405905fdb6152228784727fcafa66749ac546a9887dc6eef7254254aaa584e9` |
| Primary Test evaluation ID | `f2663c0ebbb5f302829a2d4247ee378680a15fd68f339dbfcd53b2a57c61c4cf` |
| All-observed diagnostic ID | `33083bd17aa471503975c41c77b48cf6688b35b202d28562ca56673165cabcbe` |

JSON-file SHA256 and the schema's logical fingerprint are distinct. Original release provenance remains its actual earlier dirty-source revision; the current committed Phase 6 HEAD is evaluation provenance, not retroactively claimed model-training provenance. Full retriever manifest/completion identities, registry tags/source, runtime and evaluator hashes are in [public preflight](phase7_preflight.json).

Frozen evaluator source identities:

| Frozen evaluator file | SHA256 |
| --- | --- |
| src/recsys_evaluation/__init__.py | `63b25d656c8f11b30cf9ca83966daf3cb7336e4e6d793fb018b2909d863784e6` |
| src/recsys_evaluation/core.py | `9ae389f2439393b9bba4e4378aad19aa07754d12af426fea8c7d5674e4b210cc` |
| src/recsys_evaluation/workflow.py | `54e2c707a5cb5eab3c4982c2ad857ac197e2d0a35c87147fa856188e9e8494f5` |
| tests/test_final_evaluation.py | `4dddb980b657145959abd90924075f354e7b3c604de4f1a7bb78dd1ef2dc9dec` |
| tools/final_evaluation.py | `6314a6311ccff95f12a7b2be4ac855f4a428a051392131a2049d3e91769313fa` |

## Gate A — freeze and known-answer Validation replay

Completed **2026-10-28T06:26:50.878401+00:00 UTC**. The model/export inventory, live immutable Registry Version 1, six selected Full-Train checkpoints, 31 trees, RRF c=60, fixed 53-feature schema, Train catalog/mappings/history and frozen split identity passed. Registry source/tags match the Phase 6 verified export. Sparse observed-history equality was checked against frozen Train without a dense matrix.

The evaluator calls the retained inference/fusion/metric implementations through a separate `recsys_evaluation` package. A process audit hook rejects Test, mixed targets/cohorts, raw ratings and historical label windows in Gates A/B. Allowed data opens were Train, movies, canonical mapping/catalog files, split manifest and Validation only; the saved read inventory is explicit.

The predeclared contract fixed the three methods, denominator, K=20/50/100, NDCG@20, HitRate@20/100, catalog coverage, full-pool macro/micro recall, zero-hit counts, overlap, Train-positive cohorts and paired bootstrap. Original selection remains Validation NDCG@20 and Version 1, regardless of Test ordering. ALS uses original source ranks followed by a deterministic movie-ID tail; all methods rank the same full union.

Actual Validation replay reproduced **579 queries, 6,491 eligible positives and 213,507 union pairs**. All six source streams' IDs/ranks/scores, union provenance, full LambdaMART ranking IDs/ranks/scores and all original comparison/cohort metrics matched exactly. DataFrame dtype differences were permitted only for serialized type representation; numerical tolerance was zero. Original records were never rewritten. The same evaluator exercised separate all-observed aggregation, gaps, seeded bootstrap and metric serialization before Test opened. Synthetic tests additionally exercised known answers, zero hits, empty cohorts, real audit-hook denial, sealing, frozen identities and actual temporary local MLflow logging/read-only reuse.

| Method | Validation Recall@20 | Validation NDCG@20 | Test Recall@20 | Test NDCG@20 | Test HitRate@20 |
| --- | --- | --- | --- | --- | --- |
| Implicit ALS | 0.171301 | 0.126085 | 0.115124 | 0.075881 | 0.459691 |
| Six-channel RRF | 0.164938 | 0.127578 | 0.117521 | 0.085011 | 0.468268 |
| LambdaMART v1 | 0.187106 | 0.142799 | 0.128886 | 0.092294 | 0.492281 |

## Gate B — fresh label-free predictions and sealing

Sealed **2026-10-28T06:27:21.709576+00:00 UTC**. Population came solely from the packaged sorted Train mapping: **610 users**. Each source emitted **61,000 rows**, respecting Top-100, catalog and all-rated exclusions. The union has **224,127 unique pairs**. Features and full ALS/RRF/LambdaMART rankings were serialized with deterministic ordering, separate source provenance and a complete SHA256 inventory. Sealed files contain no target labels/ratings/cohorts.

The adapter calls the registered release's retained `pipeline_rankings`, reconstructs source streams from exact union provenance, revalidates them with existing contracts and rebuilds the union for exact equality. Every user's final Top-20 IDs/ranks/scores match a fresh registered pyfunc call: **610 queries / 12,200 rows**, tolerance zero. No cached Validation candidates supplied predictions. Gate B's only external data open was the split manifest; serialized Train context supplied inference inputs.

Every artifact is bound by [the sealed prediction manifest and inventory](phase7_prediction_manifest.json): model, protocol, cutoff, catalog, mappings, budgets, Train-user hash, source/runtime hashes, timestamp and payload checksums. Files remain under ignored `outputs/phase7/sealed_predictions/`; they contain user-level predictions and histories are not published.

| Artifact | SHA256 |
| --- | --- |
| preflight/COMPLETE.json | `fdb2675acfa841c72bf1c3b4a6c524bd5b03cf240d1befb75197dc9f8517da2d` |
| sealed_predictions/COMPLETE.json | `546153be660ad28ba4ccac9afe75e85cc94b6fe4dde9d88d10f5ac92aea09ae2` |
| sealed_predictions/prediction_manifest.json | `d2e94de13a8c38e5554d87a33e2d124c378b91406a0bc026eafddcb5141aaa4a` |
| sealed_predictions/union.parquet | `6721729b1b4a9a568821a56b9ad8a8d61d5a85fc7106b3c0815a2bf6a5d26ffb` |
| sealed_predictions/als.parquet | `affa644be784435f4ad38dc1f4f276891e81779ce31664bab17ddd867ad6ced1` |
| sealed_predictions/rrf.parquet | `8914e2c281d6d248b23613eee1f6ff9a0ee74315debd23e4ca82d484ba60fc67` |
| sealed_predictions/lambdamart.parquet | `6721729b1b4a9a568821a56b9ad8a8d61d5a85fc7106b3c0815a2bf6a5d26ffb` |
| final_evaluation/COMPLETE.json | `b88ddd9e49fa0b81b24e819348531709c622e5276dadfadadd993866a2499d0a` |
| final_evaluation/results.json | `01e92fb92d4d1c2b79a703c25127a975f4eb6233104c99baec439d93a0d3ace4` |

## Gate C — controlled Test unsealing

The first authorized Test payload read occurred **2026-10-28T06:27:38.950453+00:00 UTC**, after both completed markers, prediction checksums and frozen evaluator hashes were verified. `unsealing.json` was written exclusively before the read. The Test bytes were read once into a buffer, hashed and parsed from the same bytes. SHA256 is `b9c596fba7b0a7640651887b5bee9225a5ecab455e11749f6a40d042c1fad70e`. An existing final-evaluation directory blocks automatic rerun, including after interruption.

Actual Test has **15,507 rows, 610 event-bearing users, 5,565 movies and 7,309 observed positive ratings**. Submission timestamps span epoch seconds **828124762–1537799250**. Counts and exact global/per-user bounds match the authoritative manifest/cutoff table; each user's Test minimum follows their Train maximum. No split was regenerated. Validation was never appended to history.

Policy B excludes **584 catalog-unknown positives** and no additional Train-seen positives in this unique-pair acquisition. Primary evaluation has **583 users and 6,725 positives**, matching frozen expectations. There are 26 event-bearing users without positives and one further positive-bearing user without an eligible target; those undefined primary denominators are reported rather than invented zeros. They are not zero-hit queries dropped from an otherwise eligible cohort. The secondary diagnostic retains all **584 positive-bearing users / 7,309 positives** under its distinct ID.

### Final primary measured comparison

| Method | Recall@20 | NDCG@20 | HitRate@20 | Recall@50 | Recall@100 | HitRate@100 | Coverage@20 | Coverage@100 | Zero hits @20 / @100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Implicit ALS | 0.115124 | 0.075881 | 0.459691 | 0.218799 | 0.325781 | 0.773585 | 0.131163 | 0.265781 | 315 / 132 |
| Six-channel RRF | 0.117521 | 0.085011 | 0.468268 | 0.205307 | 0.306208 | 0.746141 | 0.058605 | 0.211960 | 310 / 148 |
| LambdaMART v1 | 0.128886 | 0.092294 | 0.492281 | 0.232168 | 0.339046 | 0.801029 | 0.103654 | 0.224319 | 296 / 116 |

Coverage fractions use the 7,525-item Train catalog. ALS preserves all original Top-100 ranks before its union tail, so both its Top-20 and Top-100 source baselines remain intact. RRF and LambdaMART rank the identical complete union. No method-specific population or candidate eligibility was introduced.

### Retrieval availability and union ceiling

For the primary 583 users: **213,686 candidate pairs**, mean **366.528302**, median **355**, min **100**, max **570** per query. Full-union catalog coverage is **0.713223**. The full-union macro recall ceiling is **0.520283**; micro positive coverage is **2,854/6,725 = 0.424387**. **3,871 eligible positives are absent**, and **51 users have no relevant union candidate**. All 51 remain in every method's evaluation. None has an empty candidate list. Label-free all-610 totals differ from this later label-side eligible subset.

Retrieval availability, RRF ordering and LambdaMART ordering are distinct. Pool recall is shared across methods and exceeds ranked Top-K recall. Source pair overlaps, unique relevant contributions, before-dedup counts and per-source catalog coverage are preserved in [aggregate JSON](phase7_final_metrics.json); no channels were removed in response.

### Train-positive-history cohorts

| Cohort | n | Method | Recall@20 | NDCG@20 | HitRate@20 | Recall@100 | Zero hits @20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cold_0 | 3 | Implicit ALS | 0.000000 | 0.000000 | 0.000000 | 0.233333 | 3 |
| cold_0 | 3 | Six-channel RRF | 0.000000 | 0.000000 | 0.000000 | 0.233333 | 3 |
| cold_0 | 3 | LambdaMART v1 | 0.000000 | 0.000000 | 0.000000 | 0.233333 | 3 |
| sparse_1_4 | 13 | Implicit ALS | 0.141026 | 0.056328 | 0.230769 | 0.269231 | 10 |
| sparse_1_4 | 13 | Six-channel RRF | 0.179487 | 0.075055 | 0.230769 | 0.358974 | 10 |
| sparse_1_4 | 13 | LambdaMART v1 | 0.141026 | 0.134523 | 0.230769 | 0.307692 | 10 |
| moderate_5_19 | 199 | Implicit ALS | 0.180800 | 0.089135 | 0.412060 | 0.426679 | 117 |
| moderate_5_19 | 199 | Six-channel RRF | 0.164659 | 0.089644 | 0.386935 | 0.402746 | 122 |
| moderate_5_19 | 199 | LambdaMART v1 | 0.199467 | 0.107181 | 0.437186 | 0.447713 | 112 |
| warm_20_plus | 368 | Implicit ALS | 0.079633 | 0.070024 | 0.497283 | 0.273970 | 185 |
| warm_20_plus | 368 | Six-channel RRF | 0.090800 | 0.083550 | 0.524457 | 0.252735 | 175 |
| warm_20_plus | 368 | LambdaMART v1 | 0.091340 | 0.083505 | 0.535326 | 0.282253 | 171 |

Cold n=3 and sparse n=13 are small descriptive groups; no empty Test group occurred. Empty Validation cold metrics remain undefined. Sparse Recall@20 regression persists: RRF **0.179487**, LambdaMART **0.141026**, versus Validation **0.194444 / 0.069444**. Sparse NDCG improves under ranking, so one metric does not establish universal subgroup improvement. Cold Top-20 remains zero for all methods. No cohort routing or tuning followed these findings. Complete per-cohort gaps and zero-hit counts are in the JSON.

### Validation-to-Test gaps

All deltas below are Test minus frozen Validation; relative changes divide by the Validation value. Coverage deltas are fractions.

| Method | Δ Recall@20 (relative) | Δ NDCG@20 (relative) | Δ HitRate@20 | Δ coverage@20 |
| --- | --- | --- | --- | --- |
| Implicit ALS | -0.056177 (-32.79%) | -0.050203 (-39.82%) | -0.124074 | -0.000664 |
| Six-channel RRF | -0.047416 (-28.75%) | -0.042568 (-33.37%) | -0.084409 | -0.001196 |
| LambdaMART v1 | -0.058221 (-31.12%) | -0.050505 (-35.37%) | -0.122572 | -0.001595 |

Union macro ceiling declines from **0.596664 to 0.520283** and micro positive coverage from **0.503774 to 0.424387**. All-observed catalog coverage also falls from 93.96% to 92.01%. Small data, different per-user timeline positions/catalog exposure, history/label shift, sampled ranking supervision and Validation-based early stopping are possible contributors; these diagnostics do not establish causality. Accepted cross-user temporal exposure remains. The original model selection is unchanged.

### Separate all-observed-positive diagnostic

| Method | Recall@20 | NDCG@20 | HitRate@20 | Recall@100 |
| --- | --- | --- | --- | --- |
| Implicit ALS | 0.113568 | 0.075215 | 0.458904 | 0.319531 |
| Six-channel RRF | 0.116077 | 0.084503 | 0.467466 | 0.299964 |
| LambdaMART v1 | 0.127251 | 0.091578 | 0.491438 | 0.332086 |

This denominator retains every observed positive, including impossible catalog targets: **584 users / 7,309 positives**. Catalog-only micro ceiling is **0.920099** and query-macro ceiling **0.963348**. These values cannot be pooled with primary Policy-B scores. The extra impossible-target user has zero achievable hits.

### Predeclared paired uncertainty

| Paired contrast | Mean difference | 95% percentile interval |
| --- | --- | --- |
| lambdamart_minus_als/ndcg20 | 0.016413 | [0.009691, 0.023541] |
| lambdamart_minus_als/recall20 | 0.013761 | [0.002903, 0.025530] |
| lambdamart_minus_rrf/ndcg20 | 0.007283 | [0.000476, 0.014430] |
| lambdamart_minus_rrf/recall20 | 0.011364 | [0.000804, 0.021823] |

Resampled all **583 eligible user queries** with replacement, **2,000 replicates, seed 42**; identical sampled user indices are paired across methods and metrics. Intervals are 95% percentile intervals for user-macro differences. They are descriptive uncertainty conditional on this frozen sample, after Validation model selection. They do not account for every selection/dependence/temporal bias and support no causal online or broad production significance claim. No result triggered reselection or threshold tuning.

## Gate D — new MLflow final-evaluation record

Actual server was checked healthy at **http://127.0.0.1:5002**, using the existing owned server. Experiment **`MovieLens-Small-Final-Evaluation`**, ID **5**; final run **`12f4fed76651402c97c09df96d6360dc`**. It references the fixed URI `models:/MovieLensHybridRecommender/1`. Evaluation fingerprint is `b88ddd9e49fa0b81b24e819348531709c622e5276dadfadadd993866a2499d0a`.

Logged **631 numeric metrics** with explicit primary, all-observed, union, paired-bootstrap, per-cohort and Validation-to-Test namespaces. Tags identify release/model/schema, Git/source hashes, DVC/split/policy, final held-out status, frozen-before-Test, sealed-before-labels and no Test tuning. Uploaded JSON includes contract/preflight, historical selection, model/protocol identity, prediction inventory, metrics, unsealing and checksum markers; per-query metric Parquets are in the **local private** experiment store. Test labels themselves were not uploaded. Detailed source/prediction/label-side evidence remains ignored and local.

Repeated logging returned the **same run, `created=false`**, verifying tags/metrics without writing a duplicate. All **21 pre-existing MLflow run parameter/metric/tag/status records** and Version-1 source/tags/aliases/version list remain unchanged. The `candidate` alias still points at 1. Version 1's historical `test_evaluated=false` tag and `validation_candidate` status were deliberately preserved; this new record carries the later assessment. No new model version, tag annotation, champion/production alias or automatic deployment change occurred.

## Verification, preservation and commands

All **104 tests passed, zero failures/skips**, in both environments: working **14.211 s**, isolated locked dependency environment **16.941 s**. All original 85 tests remain byte-identical; 19 new synthetic/final-assessment tests cover population, separate denominators, frozen histories, no-hit/empty cohorts, known answers, same union, ALS tails/ties, pool versus ranked K, forbidden labels, bootstrap pairing, actual process audit denial, corrupt seals/identity mismatches, overwrite refusal, absent training/mutation APIs and real disposable SQLite MLflow logging/idempotency. Tiny model fits in retained synthetic unit fixtures are not MovieLens retraining.

The final read-only verifier checks the protected baseline, all 43 frozen metadata records, **28 Parquet hashes** from the two split manifests and **7 raw acquisition/provenance hashes**, evaluator freeze, gate chronology, saved per-query aggregation, checksummed prediction/evaluation inventories, Registry/historical records and actual downloaded MLflow result bytes. **1,021 protected files** remain byte-identical; only top-level README was intentionally replaced. Existing model/serving source, raw data, frozen partitions, checkpoints, release/export, notebooks, historical reports, archive, configs and DVC metadata were not moved or rewritten. The final verifier hashes held-out files only after the authorized evaluation; this is integrity checking, not a second quality evaluation. Full checks/status are in [verification JSON](phase7_verification.json).

Exact primary commands actually executed from repository root:

```sh
git status --short
git rev-parse HEAD
git branch --show-current
scripts/start_mlflow.sh status
MPLCONFIGDIR=.venv/mplconfig PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B tools/run_tests.py
.venv/bin/python -B tools/final_evaluation.py preflight --tracking-uri http://127.0.0.1:5002
.venv/bin/python -B tools/final_evaluation.py seal
.venv/bin/python -B tools/final_evaluation.py evaluate
.venv/bin/python -B tools/final_evaluation.py log --tracking-uri http://127.0.0.1:5002
# Repeated only reporting, not evaluation; returned the same run read-only.
.venv/bin/python -B tools/final_evaluation.py log --tracking-uri http://127.0.0.1:5002
MPLCONFIGDIR=.venv/mplconfig PYTHONDONTWRITEBYTECODE=1 /private/tmp/hybrid_recsys_phase5_isolated/bin/python -B tools/run_tests.py
UV_CACHE_DIR=.uv-cache .venv/bin/uv lock --check --offline
.venv/bin/python -B tools/verify_final_evaluation.py --tracking-uri http://127.0.0.1:5002 \
  --save outputs/phase7/verification_final.json
git diff --check
git status --short
```

Logs and full private evidence are in `outputs/phase7/`. [Command/evidence index](phase7_commands.json) records argv/results/log hashes. Existing completed gate directories reject overwrites. **Do not rerun `evaluate` to seek another score.** Subsequent inspection should use the read-only verifier; an actual implementation correction would require preserving original evidence and a separately identified correction. None was needed here.

## Repository handoff and reproducibility limits

README now presents the actual lifecycle, separate Validation/Test results, architecture, data/model recovery prerequisites, native/Docker quick start, tests, licensing attribution and honest limitations. [System architecture](system_architecture.md) documents data/model/release/evaluation lineages. Public JSON contains only aggregates, source/manifest/checksum identity and cohort/overlap summaries; user IDs, raw ratings, labels, recommendation tables and histories are not added to Git.

Data byte reproduction/cold DVC recovery was verified on this Mac in Phase 2. Phase 3 actually replayed all six fits in an isolated locked Mac environment. Phase 6 proved exact native/Linux serving parity, including offline restart. Phase 7 separately measures held-out quality. None implies arbitrary-platform training bit identity or public production readiness.

Known limits remain: per-user rather than globally prospective chronology; Validation-based selection; weak unknown labels; historical/full-Train feature mismatch; small sparse/cold cohorts; no exposure logs or online CTR/causal claims; local-only DVC and MLflow stores; Train histories inside private model exports; no automatic remote backup; whole-environment DVC drift; CPU-only inference with 8.54 GB CUDA-linked Linux dependency closure; optional Docker Registry HTTP 403 under unchanged host protection. These were documented without redesigning the platform.

Git HEAD remains `98a92b54ee3b05fd4ee175442e3bc3991ef436f1`; nothing was staged, committed or pushed. Proposed review/commit scope: README; new `src/recsys_evaluation/`; `tools/final_evaluation.py` and `tools/verify_final_evaluation.py`; `tests/test_final_evaluation.py`; new Phase 7 aggregate reports and architecture. Inspect `git diff` plus new files, ensure ignored private outputs stay excluded, then the user may authorize a commit such as **`Complete frozen final Test evaluation and document local MLOps lifecycle`**. No push occurred. Future release annotation/promotion is a separate decision.

This completes a local educational lifecycle with independently evidenced data provenance, temporal contracts, multi-channel models, historical ranking supervision, complete registry packaging, API/container parity and an honest final held-out assessment. Test is now consumed for this model-selection cycle and must not become a tuning set.
