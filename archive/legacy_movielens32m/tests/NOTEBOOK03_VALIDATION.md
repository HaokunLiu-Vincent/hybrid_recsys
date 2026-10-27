# Notebook 03 implementation and validation

Created `notebooks/03_MovieLens32M_Retrieval_and_Ranking.ipynb` (29 new cells), the reusable `src/recsys/` package, synthetic tests, and notebook build/validation tools. No cells in notebooks 01 or 02 were modified; their SHA256 values remained:

- Notebook 01: `c367cb006adc4fe5f230f0ff41ba1d535b45ba019da8493117a785d11c2eb292`
- Notebook 02: `657180cb0127595655d630323209dd8e14366ba3e31f4ef33c596981ec8b73a0`

## Implemented

- Full historical Train for Popularity, sparse cosine ItemCF, optimized CPU implicit ALS, Genre CBF, exact full-graph LightGCN, and hybrid ID/genre Two-Tower.
- Primary Policy B for every retriever and evaluation label; all observed Train items filtered; common Top-100 budget; explicit cold-history fallback provenance.
- Frozen manifest/source/artifact checksum, schema, count and temporal-boundary validation. The expected 31,414,033 / 300,000 / 286,171 counts are assertions, not model dimensions or training caps.
- Exact chunked eligible-catalog retrieval, user-macro/cohort metrics with bootstrap intervals, source-preserving union, RRF, remove-one-channel ablations, coverage/popularity/latency/resource diagnostics.
- Separate strictly earlier internal-Train retriever snapshots and subsequent ranking labels, sampled-unknown weak supervision, candidate recall ceilings, no-hit query accounting, and snapshot-to-inference feature-shift reports.
- LightGBM LambdaMART, Validation-only model selection and an explicitly sealed artifact-only Test evaluation gate.
- Immutable, hash-verified per-channel and per-epoch artifacts, configuration/split/source provenance, software versions, sampled process RSS/CUDA allocator peaks and dynamic RAM/VRAM/disk guards.
- Self-contained Colab source bundle generated from repository modules; no separate source-code upload is needed.

LightGCN uses an exact polynomial graph operator and its exact adjoint with accumulated microbatched BPR gradients. It takes **one large-batch optimizer step per epoch**, not one step per loss microbatch. This choice and its tuning implications are documented in the notebook. Interrupted neural channels restart from the configured seed; completed channels are reused. Epoch weights/state remain available, but automatic mid-channel state restoration is not implemented.

## Actually checked locally

The synthetic suite contains 14 tests. In the existing `SINA-mv-Recsys` Python environment, **13 passed and 1 was skipped** because LightGBM is unavailable. Tested components include:

- Hand-computed Recall/NDCG, empty-hit queries and user-macro summaries.
- Policy B, observed-low seen filtering, unknown sampling and strict internal boundaries.
- Exact sparse ItemCF versus a tiny dense reference, vector retrieval across chunks/ties, provenance and weak labels.
- Real optimized ALS fitting and real CPU LightGCN/Two-Tower fitting on tiny synthetic data.
- Exact LightGCN adjoint gradients against autograd.
- Complete six-channel synthetic retrieval/evaluation, model save/load, completed-stage reuse and corrupt-checkpoint rejection.
- Both experiment milestones and the selected-run Test gate using a **test double only for the ranker library**, within a disposable synthetic project. The real LightGBM fit/predict integration test remains skipped; no mock output is included as a notebook or benchmark result.
- Frozen input checksums unchanged after the synthetic workflow.

Notebook JSON passed the **official nbformat v4.5 JSON schema**, and all code cells/modules passed Python syntax checks. Cell guards/order and the embedded source bundle were checked against repository files. The `nbformat` Python package is unavailable in the existing local environments, so **`nbformat.validate()` itself was not run locally**. The validation tool uses that API automatically when available; otherwise it explicitly requires the official schema. No new dependencies were installed locally.

Commands (use an environment with the required libraries):

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
python tools/validate_notebook03.py
```

Schema-only fallback when `nbformat` is unavailable:

```sh
python tools/validate_notebook03.py --schema /path/to/nbformat.v4.5.schema.json
```

Regenerate the source bundle after changing modules or tests:

```sh
python tools/build_notebook03.py
```

## Colab execution still required

The actual Drive split artifacts are unavailable locally. Their real checksums/boundaries/counts are verified at notebook startup, not asserted as locally verified results. No MovieLens 32M training, full EDA, GPU benchmark, real LightGBM experiment or final Test performance evaluation was run locally. No benchmark metrics were fabricated. All delivered notebook code cells have empty outputs and null execution counts.

1. Open Notebook 03 in Colab and run default `MODE="smoke_test"`. The runtime setup installs missing Colab packages, and smoke mode refuses to pass if a required model integration test is skipped.
2. Select an L4 runtime, set `MODE="full_train"`, and run the retrieval milestone over complete Train. Inspect the saved Validation comparisons and resource diagnostics.
3. Set `RUN_RANKER_MILESTONE=True` to generate internal snapshots and fit/evaluate LambdaMART. Review sampling counts, candidate ceilings and feature-shift limitations.
4. After Validation selection, set `FREEZE_AFTER_VALIDATION=True` and retain the exact printed run ID, code bundle, configuration and package versions.
5. Use `MODE="final_evaluation"` with `SELECTED_RUN_ID` and `CONFIRM_TEST_RUN_ID` set to that same sealed ID. No model training or tuning is called in final mode.

Resource estimates are preventative, not a guarantee of L4 completion. Expensive work is staged; resource failures report the channel and preserve completed checkpoints. Incomplete experiments must not be reported as a completed six-channel benchmark.
