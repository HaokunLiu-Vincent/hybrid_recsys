# Phase 4 — six-channel fusion and historical LambdaMART

Completed locally on Mac CPU on 2026-10-28. Gates A, B and C passed. The six selected Phase 3 Full-Train retrievers were consumed without retraining. Six **new** historical models produced out-of-time ranking examples; native LightGBM was fitted and reloaded. Test payloads were never opened.

## Verified result and scope

| Method over identical union | Recall@20 | NDCG@20 | HitRate@20 | Recall@50 | Recall@100 | Coverage@20 |
| --- | --- | --- | --- | --- | --- | --- |
| Implicit ALS (original Top-20) | 0.171301 | 0.126085 | 0.583765 | 0.294286 | 0.399429 | 13.18% |
| RRF c=60 | 0.164938 | 0.127578 | 0.552677 | 0.279296 | 0.401886 | 5.98% |
| LambdaMART, 31 trees | 0.187106 | 0.142799 | 0.614853 | 0.311211 | 0.418571 | 10.52% |

All three rankings use the **same 213,507 candidate pairs, 579 Validation queries and 6,491 eligible positives**. The ALS baseline keeps its original 100 source ranks first and places other union members in a deterministic movie-ID tail; its Top-20 exactly matches Phase 3. The shared full-union macro recall ceiling is **0.596664**, distinct from Recall@100. Micro positive coverage is **3,270 / 6,491 = 0.503774**. No ranking changes this ceiling. There are 45 users with no relevant union candidate and zero empty candidate queries; all remain in evaluation. LambdaMART Top-20 has 223 zero-hit queries.

LambdaMART's observed NDCG@20 increase over RRF is 0.015221. This is a Validation-selected result, with no confidence interval or significance claim. ALS retains higher Top-20 catalog coverage than either fused ranking. Sparse-user performance does not improve uniformly.

## Initial audit, preservation and identities

Initial Git status was clean, branch `main`, HEAD `c68dce24bc92adf8047837ca66deedbdc038686e`. The Phase 3 work was already committed, so there was no initial uncommitted/untracked authored work requiring a separate recovery copy. The baseline audit was captured before edits at `/private/tmp/movielens_phase4_baseline.json` and retained in `outputs/phase4/baseline_audit.json`. Git HEAD remains unchanged; nothing was committed, staged or pushed.

Inspected AGENTS/README; evaluation, pipeline, Phase 2 and Phase 3 reports; selected manifests and command logs; reusable retrievers, fusion, ranker/features, metrics, contracts and immutable artifact utilities; packaging, DVC and params. All six selected candidate/checkpoint completion markers, source/runtime identities and checksums passed. Provisional Phase 3 runs were not selected.

- Dataset: Kaggle `grouplens/movielens-latest-small`; modeling uses ratings and movie genres only.
- Frozen protocol: `small_v1`, per-user chronological 70/15/15, Train **70,186**, Validation **15,143**, Test **15,507** (Test count from the existing manifest, not its payload).
- Split manifest SHA256: `905f1370e919f06da3be728f95010d706b078c05f5c5e455c28da398ee41e799`.
- Validation evaluation ID: `585db5a09f479eeadaffc9cff33bfac1d254d971ebdd67f97670000be9196793`.
- Validation per-user cutoff hash: `9859a72f9f04efe290c109f073f6e845cf82a5490b488f688e5f2ef8cc2ec1d6`.
- Full-Train catalog ID: `6b607210c9f671ac865f15a1c1ce2168573246722e66ffae32075e9405c36fd2`; 7,525 Train-observed movies.
- Shared canonical mapping ID: `06951befc01e3f0ada339b3b9f8129a47639ce26d7342fb591abc1ed42f06977`. Historical mappings were independently rebuilt from permitted metadata/history; equal sorted-ID domains yield the same mapping hash. **Historical eligibility, history, snapshot ID, model weights and cutoff identities differ**; Full-Train context/factors were never reused.
- DVC metadata hashes are recorded in [verification JSON](phase4_verification.json); `dvc.yaml`, `dvc.lock`, params and data objects were preserved. No DVC repro, push or split regeneration occurred.

The primary protocol enforces same-user chronology but **permits cross-user future-information exposure**. It is not globally leakage-free or prospective production evaluation. The frozen global sensitivity protocol remains separate and untrained. Ratings are submissions, not verified viewing/exposure. Positive means observed rating >=4; low ratings and unknown pairs remain distinct. Policy B uses historical Train-known items and excludes **all** that user's historical ratings, including lows. Validation is never appended to history.

Final preservation checks: **127 protected file hashes and 66 frozen file size/mtime records unchanged**. Existing retriever algorithms, shared feature builder, artifact I/O and data preparation sources are byte-identical; only `cli.py` changed among pre-existing Python sources. Historical documentation/notebook and selected Phase 3 artifacts are intact. Integrity verification hashed allowed data files; it did not reopen Test just to checksum it.

## Gate A — verified Validation RRF

Reused `fusion.candidates.fuse`, with `sum(1 / (60 + source_rank))` over present sources. Raw scores remain source-specific. Source ranks, scores, presence and fallback flags survive deduplication. Missing source score is NaN, rank 101 and presence/fallback 0. Ties sort by movieId. Compatibility checks cover model/checkpoint identity, split/protocol/cutoff, catalog/mappings, policy, query population, schema/budget and file checksums.

Each source contributes 57,900 candidates (100 for each of 579 queries): **347,400 input rows → 213,507 unique pairs**. Per-user union size: min 286, median 356, max 570. This is a full union with an upper budget of 600, not Top-100. Contributing-source counts: 1: 143767, 2: 30885, 3: 19317, 4: 13935, 5: 5446, 6: 157.

| Source | Recall@20 | NDCG@20 | HitRate@20 | Coverage@20 |
| --- | --- | --- | --- | --- |
| popularity | 0.112275 | 0.091781 | 0.431779 | 1.75% |
| genre | 0.015106 | 0.011679 | 0.103627 | 19.92% |
| itemcf | 0.161575 | 0.123064 | 0.523316 | 12.35% |
| implicit_als | 0.171301 | 0.126085 | 0.583765 | 13.18% |
| lightgcn | 0.136687 | 0.107823 | 0.493955 | 4.89% |
| two_tower | 0.045533 | 0.034236 | 0.267703 | 12.76% |

Unique contribution means pairs absent from the other five sources; relevance uses the same eligible Validation truth.

| Source | Input rows | Unique pairs | Unique relevant pairs | Source retrieved positives | Source movie coverage |
| --- | --- | --- | --- | --- | --- |
| popularity | 57900 | 13300 | 128 | 1534 | 401 / 7525 |
| genre | 57900 | 53442 | 112 | 233 | 3952 / 7525 |
| itemcf | 57900 | 15898 | 123 | 1771 | 2803 / 7525 |
| implicit_als | 57900 | 24060 | 417 | 1889 | 2011 / 7525 |
| lightgcn | 57900 | 7961 | 118 | 1823 | 1368 / 7525 |
| two_tower | 57900 | 29106 | 186 | 1097 | 1276 / 7525 |

Union movie coverage is **5,485 / 7,525 = 72.89%**. Source coverage and distinct relevant contributions do not by themselves establish ranking usefulness.

### Overlap and popularity concentration

| Pair | Shared pairs | Mean overlap / query | Mean query Jaccard |
| --- | --- | --- | --- |
| popularity\|genre | 1042 | 1.799655 | 0.009138 |
| popularity\|itemcf | 29626 | 51.167530 | 0.365794 |
| popularity\|implicit_als | 17739 | 30.637306 | 0.189530 |
| popularity\|lightgcn | 35839 | 61.898100 | 0.474024 |
| popularity\|two_tower | 14676 | 25.347150 | 0.157143 |
| genre\|itemcf | 1685 | 2.910190 | 0.014903 |
| genre\|implicit_als | 1978 | 3.416235 | 0.017536 |
| genre\|lightgcn | 1380 | 2.383420 | 0.012148 |
| genre\|two_tower | 1817 | 3.138169 | 0.016091 |
| itemcf\|implicit_als | 22083 | 38.139896 | 0.248065 |
| itemcf\|lightgcn | 30594 | 52.839378 | 0.381728 |
| itemcf\|two_tower | 14662 | 25.322971 | 0.152793 |
| implicit_als\|lightgcn | 22587 | 39.010363 | 0.252117 |
| implicit_als\|two_tower | 13612 | 23.509499 | 0.136119 |
| lightgcn\|two_tower | 19941 | 34.440415 | 0.216202 |

Pair Jaccard is averaged across the same 579 users; aggregate pair-set Jaccards are also saved in the JSON. Popularity/LightGCN overlap is highest (mean Jaccard 0.4740); Genre mostly expands the union. Train-positive top-decile share is **63.94%** across union rows versus **99.48%** in RRF Top-20. Mean historical positive popularity is 31.21 versus 105.74, and recommendation-frequency Gini is 0.8167 versus 0.9870. Thus RRF Top-20 remains strongly concentrated despite broad union coverage.

### Six remove-one ablations

All deltas below are reduced system minus the six-source system; coverage deltas are fractions of the 7,525-item catalog.

| Omitted source | Union macro recall | Δ union recall | Lost unique positives | RRF NDCG@20 | Δ NDCG@20 | Δ Recall@20 | Δ coverage@20 | Union catalog coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| popularity | 0.583131 | -0.013533 | 128 | 0.129121 | 0.001542 | 0.008905 | 0.036412 | 72.89% |
| genre | 0.578661 | -0.018003 | 112 | 0.127017 | -0.000561 | -0.003442 | -0.003588 | 49.65% |
| itemcf | 0.576172 | -0.020492 | 123 | 0.117404 | -0.010174 | -0.010779 | 0.004651 | 64.56% |
| implicit_als | 0.522842 | -0.073822 | 417 | 0.114543 | -0.013035 | -0.017255 | -0.010498 | 71.19% |
| lightgcn | 0.581761 | -0.014904 | 118 | 0.128043 | 0.000465 | -0.001931 | 0.032425 | 71.51% |
| two_tower | 0.570075 | -0.026589 | 186 | 0.130724 | 0.003145 | -0.003613 | -0.008372 | 72.88% |

ALS contributes the most unique positives and largest union-recall decrement when removed. Genre supplies 112 unique relevant pairs despite weak standalone Recall@20. Removing Popularity or Two-Tower increases RRF NDCG@20 in this fixed experiment while reducing candidate recall; LightGCN removal slightly increases NDCG but reduces Recall@20. These are marginal observations under fixed RRF, not a source selection/tuning exercise. All six sources remain in the ranker benchmark.

Gate A saved and reloaded the exact union before Gate B. An initial unpublished attempt failed its strict Parquet round-trip check because the sorted DataFrame retained a noncanonical index. Canonical `reset_index(drop=True)` fixed the implementation; a regression assertion was added. No completed artifact was overwritten. Both failure and successful retry logs remain under `outputs/phase4/`.

## Gate B — independent historical snapshots and actual feasibility

Frozen historical snapshot: **55,807 interactions**, 27,831 observed-positive edges; future label window **14,379 interactions**, 6,532 observed positives. Snapshot-known catalog has **6,584** items. Snapshot input SHA256 `f143d3cc9cd7c2aaa6c12d4b12516a552456097806c1ae7604002d2a9c584d87`; label-window SHA256 `e736bff782ddd0e8e7f81635dba6e60eefa4705b0eb27ed9dbd9bd33e1e67983`.

Per-user cutoff table SHA256 `379e7f17bad74836ae4b434139cfc3788e9ac2a3d08213e5108b1d9d9ab25b8f` records snapshot max and label min/max timestamps. For all 607 users with later label rows, minimum later timestamp is **strictly greater** than that user's historical maximum (minimum gap 1 second). Historical and label interaction IDs are disjoint, and together exactly reconstruct External Train. Global timestamp envelopes overlap: historical 1996-03-29T18:36:55+00:00 to 2018-09-17T04:20:46+00:00; labels 1996-04-18T11:46:45+00:00 to 2018-09-17T04:23:19+00:00. These envelopes are not a common global cutoff.

All six models use unchanged conservative Phase 3 YAML configurations and seed 42, independently recomputing histories, profiles, popularity, sparse matrices/graphs, factors/embeddings and eligibility. Neural checkpoints use minimum historical Train BPR loss; no label-window/Validation selector is passed. BPR samples unobserved pairs excluding **all observed ratings**, as an implicit-feedback training convention rather than verified dislikes. ALS retains its established positive/low confidence treatment. Historical and external checkpoint loaders reject the other's training kind.

| Source | Train rows | Catalog items | Fit incl. context (s) | Generate 610 Top-100s (s) | Peak RSS (MiB) | Selected epoch |
| --- | --- | --- | --- | --- | --- | --- |
| popularity | 55807 | 6584 | 0.069105 | 0.359896 | 306.640625 | n/a |
| genre | 55807 | 6584 | 0.069064 | 0.317475 | 310.156250 | n/a |
| itemcf | 55807 | 6584 | 0.323520 | 0.208208 | 317.390625 | n/a |
| implicit_als | 55807 | 6584 | 0.427016 | 0.338257 | 317.125000 | n/a |
| lightgcn | 55807 | 6584 | 2.301923 | 0.352020 | 484.765625 | 40 |
| two_tower | 55807 | 6584 | 3.359585 | 0.333771 | 468.359375 | 20 |

CPU timings are measured fitting/context and inference, excluding process startup, model publication and diagnostic work. Each model saved, reloaded and generated a separate **61,000-row** stream for **all 610 snapshot users**, without consulting future labels. Genre/ItemCF/ALS/LightGCN/Two-Tower each used popularity fallback for four no-positive-history users (400 rows); Popularity has no fallback flag. Validation has no fallback queries. Historical fusion has **225,658 pairs**. Candidate files and label-free feature artifacts were published before joining historical labels.

### Measured labels, sampling and groups

| Measurement | Actual count |
|---|---:|
| Historical queries generated without labels | 610 |
| Users with future observed positives | 574 |
| Users with snapshot-eligible future positives | 571 |
| Queries with retrieved positive and contrast | **528** |
| Queries with only zero labels | 82 |
| Queries with only positive labels | 0 |
| Eligible future positives | 6,160 |
| Retrieved positive candidates | 3,160 |
| Observed-low candidates, full union | 2,144 |
| Unobserved candidates, full union | 220,354 |
| Training observed positives / lows / sampled unknowns | 3,160 / 1,976 / 52,800 |
| Effective training rows / groups | **57,936 / 528** |

Historical union query-macro recall is **0.596135**, using all 571 relevance-bearing users, including 43 with no retrieved positive. Micro coverage is **3,160 / 6,160 = 0.512987**; **48.70%** of eligible future positives are absent and cannot be recovered by ranking. The 571 potential queries are not reported as the 528 effective groups.

The fixed training policy retains every retrieved positive and observed low **within informative queries**, samples up to 100 unknown candidates per composite query with seed 42 + query-ID hash, and drops no-positive/no-contrast queries from gradient training only. Of the 2,144 retrieved lows, 168 occur in dropped no-positive groups. All 3,160 positives survive. No unretrieved future-positive item is inserted. Unknowns receive binary weak label 0 alongside observed lows, while `feedback_kind` remains a diagnostic column, never a model feature. Query sizes min / median / p90 / max are **101 / 107 / 122 / 190**, with exact contiguous-group sums. The predeclared minimum feasibility threshold was 50 informative groups; 528 passed. This threshold is an execution gate, not evidence of statistical power.

## Explicit features, missingness and distribution mismatch

Feature schema `small_lambdamart_features_v1`, logical checksum `21b298dc0cef61b0de9514c0db66f4e38cd9a89818bb8fe94694d0251ad48ae0`, contains **53 float32 features in fixed order**: 24 source-specific score/rank/presence/fallback fields; source count; four log1p historical activity/popularity counts; genre similarity; **rrf_score**; best/mean present-source rank; original full-union candidate count; 19 binary allowed movie genres. Source absence uses NaN score, rank 101, presence/fallback 0; present scores and other features are finite. IDs, target label, rating, feedback category, timestamps and label-derived cohorts are excluded by projection, not merely dropped by convention. No target-derived normalization or mapping is fitted.

The active versioned schema extends the reused feature builder without changing the old `features.py` checksum required by Phase 3 checkpoints. `queryId = evaluation_or_snapshot_namespace + '|' + userId` prevents same-user groups from merging across snapshots. Feature construction is separate from label attachment; tests mutate label-side columns and reject incompatible schemas/order/dtypes. Historical features use their own context; Validation features use Full-Train context. Missing-value and feature order semantics are identical.

| Source-specific score | Natural historical mean | Sampled training mean | Validation mean | Historical missing | Validation missing |
| --- | --- | --- | --- | --- | --- |
| popularity_score | 67.971855 | 67.103989 | 76.611725 | 72.97% | 72.88% |
| genre_score | 1.323439 | 0.820855 | 0.825085 | 72.97% | 72.88% |
| itemcf_score | 3.366755 | 3.280696 | 3.597144 | 72.97% | 72.88% |
| implicit_als_score | 0.989895 | 0.522173 | 0.546246 | 72.97% | 72.88% |
| lightgcn_score | 5.958543 | 5.798548 | 5.869501 | 72.97% | 72.88% |
| two_tower_score | 1.479842 | 0.982172 | 0.983555 | 72.97% | 72.88% |

The natural historical means include four cold-history fallback users; sampled training and eligible Validation have none. Raw score shifts must not be interpreted as calibrated confidence differences. Missing fractions are over union rows (not just rows present in that source). All per-source rank, presence, fallback, p10/p50/p90 and standard-deviation diagnostics are saved in `feature_shift.json` and the report JSON.

| Feature | Natural historical mean | Sampled mean | Validation mean |
| --- | --- | --- | --- |
| rrf_score | 0.015824 | 0.016704 | 0.015875 |
| source_count | 1.621923 | 1.696993 | 1.627113 |
| user_observed_log | 3.883387 | 4.042757 | 4.161868 |
| user_positive_log | 3.257478 | 3.444361 | 3.545127 |
| item_observed_log | 3.133480 | 3.197640 | 3.302945 |
| item_positive_log | 2.590652 | 2.648683 | 2.733268 |
| candidate_count | 379.223816 | 368.909973 | 375.364929 |

Query-macro **actual row counts** before/after sampling are 369.93 historical union / 109.73 sampled training / 368.75 Validation; median 357 / 107 / 356. The `candidate_count` feature retains each query's original full-union size even in sampled training; its table means are row-weighted. Mean source overlap is 1.6219 / 1.6970 / 1.6271. Label-conditioned sampling deliberately changes the training population and score distributions.

The simple descriptive alert (absolute historical-standardized mean shift >0.5 or missingness difference >0.05) flags zero natural-history-versus-Validation features. This does **not** establish equal distributions or calibrated scores: fallback/cohort composition, larger Full-Train histories and sampling produce visible mismatch. No Validation labels were used to refit transformations. The full natural historical, sampled and Validation summaries are preserved rather than treating sampled training as online traffic.

## Gate C — native LightGBM fit and Validation selection

Added `rank = ["lightgbm==4.6.0"]` and updated `uv.lock`; other model pins remain unchanged. Python 3.11.9, uv 0.13.6, native macOS arm64 LightGBM imported and trained successfully with existing libomp. Existing working PyTorch 2.11.0 and implicit 0.7.2 were preserved. A mixed Torch/LightGBM CPU smoke also passed. Exact runtime package versions and source/lock hashes are in [verification](phase4_verification.json).

Fixed config in `configs/lambdamart.yaml`: objective `lambdarank`; binary gain `[0,1]`; learning rate 0.05; 15 leaves; max depth 5; min leaf 50; L1 0.1/L2 1; max_bin 127; truncation level 30; up to 200 rounds; early stopping 20; seed 42 for all RNG settings; one CPU thread; deterministic column-wise mode; no bagging or hyperparameter sweep.

Training gradients use only **528 historical groups / 57,936 rows**. Validation uses **579 groups / all 213,507 union rows**, including no-positive-in-pool groups, with no Validation sampling/query drops. Validation labels supervise early stopping, not gradients. Native candidate-conditional NDCG is disabled: the custom metric computes established **user-macro NDCG@20 against the full eligible-positive truth**, with deterministic movie-ID ties and zero-hit users retained. The full 6,491-positive denominator is therefore preserved even when a positive is absent from the pool. The training trajectory uses sampled informative historical groups; it is training-biased and cannot be compared directly to Validation as an unbiased generalization estimate.

- Evaluated **51 rounds**, selected **31 trees**; saved only the best-iteration model.
- Sampled training NDCG@20: 0.260576 first, 0.341422 at selected iteration, 0.353428 last evaluated.
- Full-union Validation NDCG@20: 0.111043 first, **0.142799 best**, 0.141849 last evaluated.
- LightGBM fit with custom metric: **7.330 s**, measured peak process RSS **904.1 MiB**; total rank-train process **12.175 s**.
- Native model: **59,145 bytes**; 53 schema inputs, of which 47 are used after constant-feature filtering (the six training fallback columns are zero). Prediction over all rows **0.157250 s**, score+deterministic sorting **0.213230 s**, amortized **0.368273 ms/query**. RRF score+sort **0.059538 s**, original Gate A union construction **3.301375 s**. These are local batch timings, excluding feature construction/retriever serving and concurrency; they are not API latency claims.
- Feature construction: historical 0.537 s, Validation 0.512 s.

Resource guards estimate sparse candidate/feature/bin memory against dynamic available RAM; item scoring remains blocked and CPU threads bounded. Available resources at verification are recorded separately from peak training RSS. No dense full user-item/item-item matrix, GPU or cloud execution was introduced.

### Cohorts and importance

| Cohort | Queries | RRF Recall@20 | Ranker Recall@20 | RRF NDCG@20 | Ranker NDCG@20 |
| --- | --- | --- | --- | --- | --- |
| cold_0 | 0 | undefined | undefined | undefined | undefined |
| sparse_1_4 | 12 | 0.194444 | 0.069444 | 0.066502 | 0.066224 |
| moderate_5_19 | 200 | 0.226095 | 0.265669 | 0.126121 | 0.153142 |
| warm_20_plus | 367 | 0.130645 | 0.148140 | 0.130370 | 0.139667 |

Cold subgroup metrics are undefined because it has no primary eligible-positive users. The 12-user sparse subgroup has worse Recall@20 under LambdaMART; overall gains are not universal. With 528 training groups and Validation-based stopping, overfit and selection optimism remain plausible. No nested tuning study, uncertainty interval or final Test estimate was run.

| Feature | Gain | Split count |
| --- | --- | --- |
| rrf_score | 5474.172528 | 41 |
| implicit_als_score | 1728.138141 | 41 |
| itemcf_rank | 1095.959844 | 48 |
| lightgcn_score | 718.935728 | 45 |
| item_observed_log | 674.064857 | 32 |
| itemcf_score | 578.877584 | 39 |
| genre_similarity | 563.318539 | 56 |
| implicit_als_rank | 509.148928 | 28 |
| item_positive_log | 250.294553 | 21 |
| lightgcn_rank | 171.504272 | 14 |
| best_source_rank | 134.164180 | 11 |
| popularity_score | 115.757269 | 16 |

RRF score has the highest gain. Importance is descriptive, not causal. All 53 feature importances and both 51-round trajectories are saved. No Python LightGBM warnings were captured. Sandbox PyArrow emitted `sysctlbyname` CPU-cache/NEON permission warnings; processes completed, artifacts validated, and clean-process predictions matched exactly.

## Immutable artifacts and parity

| Artifact directory | Manifest SHA256 |
| --- | --- |
| artifacts/fusion/phase4_v1_validation | `8d2d25a6d329301f2cf3341b2cda5c5213661ace1f9d008d3a24e3e6e1092c6b` |
| artifacts/fusion/phase4_v1_rank_train | `e2544abfbfd059421fbf1afbed2b8abababb05eeb76c0648afabf95fe7f3b8bb` |
| artifacts/historical/popularity/phase4_v1_snapshot | `edbc975ec6312ad9a814e54b8fd76dd18c84461d74b6c0e4e97fbb02c3964ea2` |
| artifacts/historical/genre/phase4_v1_snapshot | `8252897df2b0b132787e02f5d6e45fab55756d5e132c19f1e9110704d6257da6` |
| artifacts/historical/itemcf/phase4_v1_snapshot | `2a3490ebb4dea02390b08e1618e59ac7ccf275b7e52d5a8bcdfc2ee56b401370` |
| artifacts/historical/implicit_als/phase4_v1_snapshot | `2bff8326f2cdf0db22bc9cac96e0039c85c19eb68a7b6888b3d80e3e7210f0d7` |
| artifacts/historical/lightgcn/phase4_v1_snapshot | `efb33ff9283f14b2c5c8085fc9d1fd5451fb3ce1f92f5bba8bcc3d69fae5e1fa` |
| artifacts/historical/two_tower/phase4_v1_snapshot | `802e944be60819f7713c41c53e31d83c92f96a50def9f0f4966d55fa3ac76875` |
| artifacts/historical_candidates/popularity/phase4_v1_snapshot | `1468de1c04599ff61b11f3a89f78b0977810a3ad769f3cf0a95d03a554f9590c` |
| artifacts/historical_candidates/genre/phase4_v1_snapshot | `1585ba41c05bbfec24abb98630915ffc38611545cd5dd43f104d4eaafb4cf3ea` |
| artifacts/historical_candidates/itemcf/phase4_v1_snapshot | `7388e29637c0bb56fad1828ee4ccd7535b3427ad0e207fd16057485fa18fa068` |
| artifacts/historical_candidates/implicit_als/phase4_v1_snapshot | `59b6c46fd59210f8f8c4921ef2203a91a016e78b1d32429c98c62d092d2c182a` |
| artifacts/historical_candidates/lightgcn/phase4_v1_snapshot | `7fc11284571db7fa4762d6d7658e579dc019b812435554764dc2746646eaba30` |
| artifacts/historical_candidates/two_tower/phase4_v1_snapshot | `d055623850a27f1564805f0033cb183a44afa94326a02b79f63cac97091c6dc0` |
| artifacts/ranking_features/phase4_v1 | `4f1ffad314a1c547be312bb90a2e4e4a98f498a9db1679938647020665e1ecde` |
| artifacts/ranking_training/phase4_v1 | `de88eff4d565de6515fb706b3490c6c5252871ad0e723ee007a42d6d67e8e6e3` |
| artifacts/lightgbm/phase4_v1 | `f9d884fe1ee80dc7d7371362042d6b6e3665d0872a90006ab16720b684d0fbbc` |

Each directory has a `COMPLETE.json` with every payload's byte count/SHA256; checked loaders reject extra/missing/changed files. [Verification JSON](phase4_verification.json) records checksums for **all 24 new completed directories**, including six clean-reload candidate copies and ranker evaluation. Artifacts are intentionally ignored generated local outputs; report/source/config/lock files remain Git-visible. Copying the source repository alone does not transfer these model/data artifacts.

Full native model SHA256: `a14b8ecd561056725d200cf8cce023e76086fc8d7aad96426d9a8284ffaa33a7`. Validation union SHA256: `3fdacca0fe440460363df7a74d66b871f1dad7254ae96e8723f34d0139a47198`. Schema logical fingerprint above differs intentionally from its JSON file-byte checksum (also recorded in completion markers).

Historical loaders verify training kind, source/model/config/runtime hashes, snapshot/cutoff identity, eligibility and mapping. Native ranker loading verifies version, schema checksum/order, feature implementation and LightGBM runtime. Manifests and transitive references record split, DVC metadata, locks, source revision/hashes, model configurations, query IDs, catalog policies, sampling, metrics and timings. HEAD points to the preceding committed source; modified/new source hashes disambiguate these uncommitted Phase 4 runs.

A separate locked environment `/private/tmp/hybrid_recsys_phase3_clean_20261027` was synced offline with train+rank groups and launched fresh processes. **All six historical checkpoints regenerated byte-identical full 61,000-row candidate Parquets**. LightGBM reload produced **213,507 exactly equal predictions**, without fitting. That environment was already independently created for Phase 3; Phase 4 extended its lock-compliant dependencies, not a replay of the six historical fits. The ranker's same-process native text reload also matched. No completed original artifact was overwritten; clean-reload outputs have distinct run IDs.

## DVC environment drift: assessed and deliberately deferred

All three stages (`validate_raw`, `prepare_splits`, `validate_splits`) in unchanged `dvc.yaml` depend on **both `pyproject.toml` and the entire `uv.lock`**. Adding rank/train-only dependencies therefore invalidates those dependencies even if transformation inputs are unchanged. More fundamentally, `small_pipeline.py` includes whole `uv.lock` SHA in `environment_lock_sha256` within the split/evaluation identity and immutable validation report. Merely editing DVC dependency lists cannot make a future real repro safely reproduce the old identity.

The disposable synthetic `FrozenDVCContractTest` builds tiny frozen outputs, changes only its fake global lock for an unrelated training dependency, and confirms rebuilding the same version raises `FileExistsError` while all existing output hashes remain unchanged. This test passed in both environments. It did not read or rebuild project data.

A minimal future contract should isolate versioned Python + DuckDB + pandas/NumPy + PyYAML (+ timezone/serialization dependencies actually used) and the transformation source/params/input hashes, with its own resolved lock or projection; training Torch/implicit/LightGBM should not participate. DVC CLI is orchestration tooling, separate from transformation semantics. Test in a disposable fixture that training-only lock changes leave the data identity stable, while DuckDB/source/parameter changes invalidate it.

Applying that identity contract now would change frozen `small_v1` provenance, violating this phase. **No active DVC change was made.** Proposed migration: preserve v1 as a frozen legacy lock/manifest; implement and verify the dedicated contract in a new data version with explicit manifest lineage; only then update future DVC stages and record new outputs. Do not alter `dvc.lock` to hide drift or run force-repro against v1. Existing model artifacts remain independently versioned and bound to the current v1 data manifest.

## Commands and verification

Actual executed argument arrays, return codes, timings and log paths are recorded in [phase4_commands.json](phase4_commands.json); complete local logs are under `outputs/phase4/`. Installation used `uv lock`, then `uv sync --locked --inexact --group rank --all-extras` in the existing environment to preserve inherited train wheels. The independent environment sync was `UV_PROJECT_ENVIRONMENT=/private/tmp/hybrid_recsys_phase3_clean_20261027 UV_CACHE_DIR=.uv-cache .venv/bin/uv sync --locked --no-default-groups --group train --group rank --offline`.

For a new environment, use Python 3.11.9 and `uv sync --locked --group train --group rank`. Once installed, these are the actual stage commands (existing IDs intentionally refuse overwrite; choose new IDs consistently for a replay):

```sh
export UV_CACHE_DIR=.uv-cache
.venv/bin/uv run --locked --no-sync --group train --group rank recsys fuse-validation \
  --evaluations artifacts/candidates/{popularity,genre,itemcf,implicit_als,lightgcn,two_tower}/phase3_v1_seed42_main/validation_main \
  --run-id phase4_v1_validation

for model in popularity genre itemcf implicit_als lightgcn two_tower; do
  .venv/bin/uv run --locked --no-sync --group train --group rank recsys historical-train \
    --model "$model" --config "configs/$model.yaml" --run-id phase4_v1_snapshot
  .venv/bin/uv run --locked --no-sync --group train --group rank recsys historical-candidates \
    --checkpoint "artifacts/historical/$model/phase4_v1_snapshot" --run-id phase4_v1_snapshot
done

.venv/bin/uv run --locked --no-sync --group train --group rank recsys fuse-historical \
  --candidates artifacts/historical_candidates/{popularity,genre,itemcf,implicit_als,lightgcn,two_tower}/phase4_v1_snapshot \
  --run-id phase4_v1_rank_train
.venv/bin/uv run --locked --no-sync --group train --group rank recsys rank-features \
  --historical-fusion artifacts/fusion/phase4_v1_rank_train \
  --validation-fusion artifacts/fusion/phase4_v1_validation --run-id phase4_v1
.venv/bin/uv run --locked --no-sync --group train --group rank recsys rank-train \
  --training-data artifacts/ranking_training/phase4_v1 --run-id phase4_v1
.venv/bin/uv run --locked --no-sync --group train --group rank recsys rank-evaluate \
  --checkpoint artifacts/lightgbm/phase4_v1 --training-data artifacts/ranking_training/phase4_v1 \
  --run-id NEW_reload_id

MPLCONFIGDIR=.venv/mplconfig PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B tools/run_tests.py
UV_CACHE_DIR=.uv-cache .venv/bin/uv lock --check --offline
git diff --check
git status --short
```

Actual checks: **61 tests passed, zero failures/skips** in both working and isolated environments (8.291 s / 8.587 s test runner time); all original 45 Phase 1–3 tests retained. Added 16 tests covering exact RRF, eligibility/overlap/ceilings, snapshot-only read guards for all six fits/candidate stages, incompatible namespaces/cutoffs/mappings, no target features, schema/missingness/order, composite groups, deterministic sampling/no inserted positives, no-hit denominator, native fit/reload, and the synthetic DVC identity failure. CLI Test access remains rejected. All **52 active Python files** parsed with `ast.parse`; lock check and `git diff --check` passed. Saved ranker metrics were independently recomputed exactly against unchanged Validation truth and the exact union key set.

Final source changes: `pyproject.toml`, `uv.lock`, `src/recsys/cli.py`, `README.md`; new `configs/lambdamart.yaml`, `fusion/small_rrf.py`, `historical_retrieval.py`, `rankers/small_schema.py`, `rankers/small_lambdamart.py`, two Phase 4 test files and Phase 4 reports/JSON. No old algorithms, splits, notebooks, archived files or historical reports were edited. Full final Git status is in verification JSON; changes remain uncommitted.

## Readiness for Phase 5

The completed experiment is ready for a separately requested **MLflow tracking and model registration** phase: record these six Full-Train models, RRF config, historical training lineage, mappings/histories, feature schema/transforms, native ranker and all compatibility hashes as one deployable release candidate. Historical training checkpoints are lineage artifacts and must not replace Full-Train inference retrievers. Keep v1 data/DVC drift explicitly visible and retain clean-process parity checks.

Still unresolved by design: final Test remains unopened; selection is Validation-only; unknown labels are weak proxies; ranker training is conditioned on retrieved hits with sampled unknowns; snapshot/cohort/score distributions differ; sparse groups are small and do not uniformly benefit; cross-user temporal exposure is accepted; no prospective/online performance or statistical significance is established. No MLflow, registry, release packaging, API, Docker or final Test evaluation was implemented in Phase 4.
