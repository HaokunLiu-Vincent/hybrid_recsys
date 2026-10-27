# Evaluation protocol v1 — MovieLens Small

**Decision:** adopt per-user chronological 70/15/15 as the primary educational offline benchmark. Materialize global chronological 80/10/10 separately as a sensitivity protocol. Both were specified before any model fitting. The four-way comparison below measures data feasibility, not model performance; do not switch protocols after observing future model scores.

**Accepted temporal limitation:** per-user chronology is not global chronology. Another user's Train ratings can occur later than this user's Validation/Test targets. This cross-user future-information exposure is explicitly accepted for this educational benchmark; neither its features nor its metrics may be called globally leakage-free or prospective production evaluation. Primary split and historical-snapshot manifests repeat this disclosure. The global sensitivity uses strict global boundaries and distinct artifact/evaluation IDs; never pool metrics or model artifacts across protocols.

## Source and deterministic assignment

The unchanged Kaggle acquisition has 100,836 ratings, 610 users and **9,742 metadata movies, 9,724 of them rated**. Only `ratings.csv` and `movies.csv` enter processing. Raw identity and SHA256 are frozen in `params.yaml`; source or provenance changes require review and a new version. The Phase 1 EDA/report remain historical evidence, including its earlier exploratory global recommendation, now superseded by this explicit Phase 2 decision.

1. Assign canonical `interactionId` by `(userId, timestamp, movieId)` and normalize submission time to UTC. Movie ID is only a deterministic ordering of ties, not evidence of within-second event order.
2. Group all identical timestamps per user (primary) or across all users (global). Never split a group or randomly shuffle.
3. For n interactions, target cumulative counts are `floor(n*train_percent/100)` and `floor(n*(train_percent+validation_percent)/100)`, using integer arithmetic. Choose the legal cumulative group prefix nearest each target; equal distance chooses the earlier/smaller prefix.
4. With at least three timestamp groups, reserve at least one group for each partition: first boundary leaves two groups; second is after the first and before the last group. With two groups, the first is Train, the second Test, Validation empty. With one group, all interactions are Train. Source membership/values are preserved exactly.
5. Primary histories satisfy max Train < min Validation < min Test within a user whenever those windows exist. They overlap in calendar time across users. Secondary groups enforce this order globally. Actual boundaries and user cutoffs are artifacts, never inferred from nominal ratios alone.

## Actual four-way comparison

Each partition cell gives interactions, percentage of all source ratings, and distinct users.

| Protocol | Train | Validation | Test | Train-known movies |
|---|---|---|---|---|
| per_user_70_15_15 | 70,186 (69.604%; 610 users) | 15,143 (15.017%; 610 users) | 15,507 (15.378%; 610 users) | 7,525 |
| per_user_80_10_10 | 80,241 (79.576%; 610 users) | 10,153 (10.069%; 610 users) | 10,442 (10.355%; 610 users) | 8,225 |
| global_70_15_15 | 70,585 (70.000%; 453 users) | 15,125 (15.000%; 106 users) | 15,126 (15.001%; 96 users) | 7,056 |
| global_80_10_10 | 80,668 (79.999%; 522 users) | 10,084 (10.000%; 62 users) | 10,084 (10.000%; 74 users) | 7,867 |

All 610 users appear in every primary partition. The preferred primary retains more held-out positive queries than per-user 80/10/10 while keeping a substantial historical prefix. Global alternatives retain strict global chronology but leave most evaluation users with no history; this is the accepted coverage-versus-temporal-interpretation tradeoff, not evidence that primary model scores will be better.

### History cohorts among users with events in each evaluation window

Zero Train ratings is distinct from zero Train positives. Cohorts use only that protocol's frozen Train. `cold_0` = zero positives, `sparse_1_4`, `moderate_5_19`, `warm_20_plus`. Cold-positive users may have observed low ratings; do not pretend those are positive evidence.

| Protocol | Window | Users with events | No Train ratings | 0 Train positives | 1–4 positives | 5–19 positives | 20+ positives |
|---|---|---|---|---|---|---|---|
| per_user_70_15_15 | validation | 610 | 0 | 4 | 18 | 217 | 371 |
| per_user_70_15_15 | test | 610 | 0 | 4 | 18 | 217 | 371 |
| per_user_80_10_10 | validation | 610 | 0 | 4 | 11 | 188 | 407 |
| per_user_80_10_10 | test | 610 | 0 | 4 | 11 | 188 | 407 |
| global_70_15_15 | validation | 106 | 91 | 91 | 1 | 1 | 13 |
| global_70_15_15 | test | 96 | 87 | 87 | 1 | 1 | 7 |
| global_80_10_10 | validation | 62 | 41 | 41 | 0 | 3 | 18 |
| global_80_10_10 | test | 74 | 54 | 54 | 0 | 4 | 16 |

### Held-out relevance and catalog coverage

Positive means `rating >= 4.0`. A Train-known movie has at least one Train rating, high or low. Candidate eligibility is Policy B and excludes **every movie the user rated in Train**, including low ratings. Validation is never appended before Test. These are label-side diagnostics; held-out labels never build item eligibility, popularity, user profiles or training objectives.

| Protocol | Window | Positive users | Positives | Train-known positives | Excluded positives | Eligible users | Target coverage | Positive users with no retrievable target | No candidates |
|---|---|---|---|---|---|---|---|---|---|
| per_user_70_15_15 | validation | 581 | 6908 | 6491 | 417 | 579 | 93.96% | 2 | 0 |
| per_user_70_15_15 | test | 584 | 7309 | 6725 | 584 | 583 | 92.01% | 1 | 0 |
| per_user_80_10_10 | validation | 563 | 4651 | 4388 | 263 | 562 | 94.35% | 1 | 0 |
| per_user_80_10_10 | test | 568 | 4974 | 4607 | 367 | 566 | 92.62% | 2 | 0 |
| global_70_15_15 | validation | 106 | 7816 | 6508 | 1308 | 106 | 83.27% | 0 | 0 |
| global_70_15_15 | test | 96 | 6402 | 4592 | 1810 | 95 | 71.73% | 1 | 0 |
| global_80_10_10 | validation | 62 | 5198 | 4810 | 388 | 62 | 92.54% | 0 | 0 |
| global_80_10_10 | test | 72 | 4128 | 3233 | 895 | 72 | 78.32% | 0 | 0 |

The source has unique user–movie pairs, so no eligible positive is additionally removed by Train-seen filtering in this acquisition. The filtering rule is still required and shared across all six channels, RRF and ranking.

### Cold-start shares

User shares below use users with events in the relevant window; target shares use all held-out positives. User-cold and item-cold conditions can overlap and must not be added as disjoint fractions.

| Protocol | Window | No-Train user share | Positives from no-Train users | Zero-positive-history user share | Positives from zero-positive-history users | Train-unknown item target share |
|---|---|---|---|---|---|---|
| per_user_70_15_15 | validation | 0.00% | 0.00% | 0.66% | 0.00% | 6.04% |
| per_user_70_15_15 | test | 0.00% | 0.00% | 0.66% | 0.14% | 7.99% |
| per_user_80_10_10 | validation | 0.00% | 0.00% | 0.66% | 0.04% | 5.65% |
| per_user_80_10_10 | test | 0.00% | 0.00% | 0.66% | 0.16% | 7.38% |
| global_70_15_15 | validation | 85.85% | 89.82% | 85.85% | 89.82% | 16.73% |
| global_70_15_15 | test | 90.62% | 94.70% | 90.62% | 94.70% | 28.27% |
| global_80_10_10 | validation | 66.13% | 92.09% | 66.13% | 92.09% | 7.46% |
| global_80_10_10 | test | 72.97% | 86.56% | 72.97% | 86.56% | 21.68% |

## Common future evaluation contract

- **Primary population:** users with at least one eligible held-out positive; primary Recall/NDCG denominators contain eligible Policy-B positives only. This gives 579 Validation and 583 Test queries. All six retrievers, fusion and ranker must use the identical saved population/catalog/seen filtering and comparable candidate budgets, with at most 100 candidates per channel.
- **Unconditional within that population:** retain every zero-hit or empty candidate list. Empty predictions with nonempty relevance score zero. Catalog Coverage@K is the distinct recommended eligible movies divided by Train-known catalog size; declare K. Candidate-pool positive coverage uses the complete deduplicated candidate union before reranking, not just its first K.
- **Separate full-observed diagnostic:** use every user with any held-out positive and all those positives in the denominator, including Train-unknown movies. It retains users with impossible targets and measures the catalog ceiling. Never silently compare this denominator with primary scores. `load_benchmark(..., denominator='all_observed')` assigns a distinct evaluation identity.
- **Undefined cases:** no positives or no eligible positives makes the corresponding recall/NDCG query denominator undefined, not an invented zero. Record those users/counts explicitly; for all-observed diagnostics, catalog-impossible positive queries yield zero achievable hits. Primary Validation has 29 users with no positives and two positive users with no eligible target; Test has 26 and one. No primary user has zero available candidates.
- **Cohorts:** overall eligible population plus cold, sparse, moderate and warm positive-history groups. Empty subgroup metrics/intervals are undefined (`null` with reason), not zero. Primary Validation has no positive-bearing cold subgroup; primary Test has three eligible cold-positive-history users. Report zero-Train-history counts separately.
- **Metrics ready for later use:** retained `metrics.query_metrics` supplies Recall@20/50/100, NDCG@20, HitRate@20 and full-pool recall. Retained evaluation helpers support macro aggregation, per-query rows and catalog coverage; use shared query identities for individual-channel/RRF comparisons. Raw channel scores are not calibrated. No such model metrics are computed in Phase 2.
- **Test discipline:** these aggregate source/label feasibility tables are protocol design diagnostics. After this decision, tune hyperparameters and early stopping on Validation only; record a selection before final Test. A loader cannot enforce human experiment discipline. Do not select a protocol or hyperparameters by final Test model scores.

## Historical LambdaMART contract

Within each External Train, use one earlier/later 80/20 historical snapshot with the same nearest-group rule. For two-window snapshots, at least two groups reserves one per window; one group stays entirely in history. No External Validation/Test row enters either inner artifact. Primary cutoffs are user-specific and retain the same cross-user global future-information limitation. Secondary inner history is a strict global prefix.

| Outer protocol | Snapshot rows | Later label rows | Positive labels | Catalog-eligible positives | Potential positive queries | Potential contrast queries | Potential queries ≥10 history ratings | Potential queries ≥5 history positives |
|---|---|---|---|---|---|---|---|---|
| per_user_70_15_15 | 55807 | 14379 | 6532 | 6160 | 571 | 571 | 566 | 551 |
| per_user_80_10_10 | 63926 | 16315 | 7395 | 6943 | 583 | 583 | 580 | 564 |
| global_70_15_15 | 56468 | 14117 | 6926 | 5613 | 115 | 115 | 24 | 24 |
| global_80_10_10 | 64534 | 16134 | 8217 | 6728 | 124 | 124 | 19 | 19 |

**Primary details:** 55,807 snapshot-history rows and 14,379 later label rows; 610 snapshot users, 605 with at least ten historical ratings and 576 with at least five historical positives. 607 users have later observations, 574 have a positive label, 571 have a catalog-eligible positive and a possible contrasting item in the full eligible catalog. Among those 571, 566 have at least ten historical ratings and 551 at least five historical positives. These sufficiency thresholds are reporting conventions, not a hidden exclusion rule.

There are 6,532 later observed positives: 6,160 are in the snapshot catalog and 372 are impossible under that snapshot's Policy B. An ideal 100-item list could cover at most 6,130 of those 6,160 eligible positives in aggregate; a theoretical six-channel union budget of 600 could cover 6,160. These are oracle budget/catalog upper bounds, not observed candidate recall or macro Recall@100. **Actual recovered-positive query count, retrieved contrast count, pool recall and effective LambdaMART training queries are unknown/null until retrieval runs.** Do not call all 571 effective queries or full-catalog contrast actual sampled labels.

Future training must fit six retrievers using only `ranker_snapshot_train.parquet`, derive its own Train-only eligibility/mappings/history features, retrieve/fuse candidates, then attach labels exclusively from `ranker_label_window.parquet`. Do not substitute the external full-Train retrievers. Keep snapshot model artifacts separate from full-Train artifacts. Labels distinguish observed positive, observed low and sampled unknown; a binary relevance-zero proxy for unknown candidates does not mean verified dislike. Count sampling policy, positive candidates, observed-low candidates, sampled unknowns, missing-positive pools and no-contrast queries.

Query key is `(snapshot/evaluation_id, userId)`. For this single snapshot, user-sorted candidate shards can use existing contiguous user grouping. Multiple future snapshots need a unique composite query ID and query-sorted shards; never merge their queries by user ID alone. Every user's snapshot maximum timestamp must be strictly before their earliest label time. Snapshots/full-Train inference differ in history length, catalog and popularity; later measure feature distributions, missing/presence rates and score shifts, and document their effect on ranker generalization.

### Candidate-time feature allowlist

Per-source membership, rank, raw score and fallback flag; RRF score; source count; history length/positive length; snapshot-derived item observed/positive popularity; allowed movie genres and historical genre similarity. Fit transformations on permitted historical data. Labels, feedback kind, held-out rating, label timestamp, future counts/cohorts and future movie-frequency aggregates are label/report fields, never model inputs.

Existing `features.py` computes the historical numerical features, `fusion.candidates.fuse` retains per-source provenance and `rrf_score`, and `rankers.features` keeps labels separate. **The retained feature allowlist does not yet include `rrf_score`; the future ranker integration must include it explicitly and version the feature schema.** No algorithm or ranker feature implementation was rewritten in this phase. Raw source scores must remain source-specific; absent scores stay missing with membership flags.

## Frozen artifacts and consumption

`data/splits/small_v1/` and `data/splits/small_global_v1/` have distinct manifests/evaluation IDs, sorted Parquet, UTC timestamps and int32 IDs/int64 timestamps/float32 ratings/boolean positivity. Each directory contains train/validation/test, movies, item eligibility, user cohorts, label-side evaluation targets, canonical user/movie mappings, user cutoffs, historical snapshot/label windows and historical query/cutoff diagnostics. User mapping is sorted Train-observed IDs only; movie mapping is sorted raw metadata IDs, with a separate eligibility mask. Unknown users map to -1 in the retained context.

`small_benchmark.load_benchmark` reads these artifacts into retained context/population interfaces without fitting. Its default phase is Validation and default denominator is eligible positives. The context's scalar `cutoff` is max Train timestamp and is **not** a global cutoff in the primary protocol. Future experiment compatibility metadata must identify the protocol and per-user cutoff-table hash (for example `training_cutoff='per_user:<hash>'`), not misrepresent this scalar as a global boundary.

See [data pipeline and environment commands](data_pipeline.md), [Phase 2 measured summary](movielens_small_phase2_summary.json) and the immutable split manifests for full counts/hashes. No models were trained, scored, selected or registered. Ratings remain submission events, no exposure logs exist, and sparse historical windows/submission bursts and selection/popularity biases still limit interpretation.
