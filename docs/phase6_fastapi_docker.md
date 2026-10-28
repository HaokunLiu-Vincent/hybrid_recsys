# Phase 6 — FastAPI, Linux Docker and inference parity

Completed locally on 2026-10-28. **Required Gates A, B, C and D passed**, using genuine six-retriever/RRF/LambdaMART inference from immutable MLflow Version 1. Native Mac HTTP and Linux ARM64 Docker HTTP both matched the registered model **exactly across 611 users / 12,220 Top-20 rows**. Required offline startup/restart, validation, failure and bounded latency checks passed. The optional Docker Registry-startup mode is blocked by the existing MLflow host-header policy; this is documented rather than weakened.

## Selection, audit and preservation

- Initial Git: clean `main`, HEAD **`1e7e65664723bc86eedbccef572f17dc79113b4b`**. Phase 5 had been committed since its completion report. Current Phase 6 changes remain uncommitted; nothing was staged, committed, pushed, reset, stashed or cleaned.
- Model: `MovieLensHybridRecommender`, fixed **Version 1**, URI **`models:/MovieLensHybridRecommender/1`**. The mutable candidate alias is not used for deployment selection and was not reassigned.
- Semantic release: **`e4b38328a35c992114eb3e6acda90803f0aa16c61a027686d9e86b06bf5e475d`**.
- Feature fingerprint: **`21b298dc0cef61b0de9514c0db66f4e38cd9a89818bb8fe94694d0251ad48ae0`**, `small_lambdamart_features_v1`, 53 features, selected 31-tree native LightGBM ranker.
- The complete six Full-Train inference states, RRF c=60, Train-known catalog, all-rated exclusion, mappings/profiles/history and transforms remain the selected Phase 5 model. No algorithms/features/models were changed or refitted.
- Original registry source: `models:/m-01f3866343e74d258fbc18cd178ced9e`, original release run `b99a65b035ed4f6e868bd846153d14cb`. Version metadata, tags, alias mapping and stored artifact bytes were verified unchanged.

Inspected AGENTS, README, evaluation/data-pipeline/Phase 3–5 documentation, Phase 5 machine records, actual loader/pyfunc/CLI contracts, manifests, dependency/DVC metadata and Docker installation. The initial audit `outputs/phase6/initial_audit.json` captured **744 protected file hashes** (including stored registry payloads), **43 frozen data size/mtime records**, and **46 original Python source hashes**. Every one passed the final comparison. The clean initial tree had no uncommitted authored files requiring a recovery copy. README updates are intentional; historical reports/notebooks/archive and DVC metadata remain intact.

Test targets were never read. Inputs for new checks came from serialized Train user mappings, not future outcomes. Test counts in old manifests were not recomputed. Frozen `small_v1`, raw DVC pointers, `dvc.yaml`, `dvc.lock` and params remain unchanged. Known whole-environment DVC dependency drift is still explicit: the release's stored lock is **`e405905fdb6152228784727fcafa66749ac546a9887dc6eef7254254aaa584e9`**, while this serving deployment's lock is **`1188c119c8ddb1f1c2bf61feec71efef494f1ac884fb6ffe3a6e5310592970bc`**. The release lock was not rewritten to match serving, and no real DVC repro/data reversioning occurred.

## Architecture and gate outcomes

```text
MLflow immutable Version 1 (authority)
    → verified outer MLflow export on host
    → read-only /models/movielens_hybrid_v1 mount
    → one pyfunc load in FastAPI lifespan
    → predict(DataFrame[userId,top_k])
    → existing six Top-100 retrievers → catalog/seen rules → RRF → 53 features → LambdaMART
    → typed HTTP response
```

The adapter is the small **`src/recsys_serving/`** package in the same project. Its separate namespace lets the immutable MLflow model supply its own `recsys` package without a cached-module collision. It contains no alternative retrieval, fusion, filtering or ranking implementation. Native/registry and offline loading share identity preflight; HTTP code only validates requests and transforms the returned DataFrame.

**Gate A:** actual Mac FastAPI startup against the pinned Registry URI, fresh direct registered-model inference and real HTTP validation/parity. All 611 requests matched the original Phase 5 reference and fresh Version-1 predictions exactly. Startup-to-ready **2.929 s**; measured lifespan load **1.748 s**; peak sampled native process RSS **446.1 MiB**. Subsequent requests reused load_count=1. Wrong user types, duplicates, K values, malformed/missing JSON, unsupported fields, batch/body limits, cold and unknown paths and K=1/10/100 were exercised through HTTP.

**Gate B:** read-only Registry APIs resolved Version 1, verified its tags/release hashes, and downloaded the **outer MLflow model**, including `MLmodel`, code and artifacts. Export: `outputs/deployment_models/movielens_hybrid_v1/` — **not** the raw `artifacts/releases/phase5_v1/` bundle. It includes a deployment manifest and complete relative-file inventory, **124 payload files**, inventory SHA256 **`9c9168a51fe50d4dff08d2af0342754beaba581b54c6244488435c353c7c006a`**. Symlink/traversal/extra/missing/changed files and wrong signatures/identities are rejected before readiness.

The owned MLflow server was stopped under ownership checks. The already loaded native Registry-backed API kept predicting. A new offline native process then started and predicted all 611 requests, was gracefully stopped, restarted with the same export, and matched all 611 again while MLflow remained unreachable. Offline start/restart-to-ready **2.454/2.455 s**. Four additional disposable starts rejected wrong release, missing model, corrupted manifest and wrong model version. Originals were never altered; MLflow was restored to its original localhost URI.

**Gate C:** real Linux Docker build, real non-root container, actual HTTP recommendations and Docker healthcheck. **Gate D:** the same 611-query checks, exact cross-platform comparisons, invalid inputs, offline container continuation/restart, failed wrong-release container, resource/latency and identity verification. The successful offline container remains available at **http://127.0.0.1:8001**; the offline native API is at **http://127.0.0.1:8002**. MLflow is restored at **http://127.0.0.1:5002**.

## HTTP contract and operational behavior

| Endpoint | Behavior |
|---|---|
| GET /health/live | Lightweight process liveness; no inference/Registry contact |
| GET /health/ready | 200 only after verified model load; otherwise 503 |
| GET /model-info | Sanitized name/version/source mode/release/protocol/retriever/RRF/ranker/schema/status; load count/startup/RSS counters |
| POST /recommend | Actual registered pyfunc inference; request order and backend ranks/scores/fallbacks preserved |

Request: `{"user_ids":[1,42],"top_k":20}`. Strict positive int64 IDs, unique nonempty list, integer K 1..100, unknown fields forbidden. Default configured batch limit **32**, hard schema maximum 64; 33–64 users at default limit return 413, >64 schema entries return 422. Body limit **65,536 bytes** applies before JSON parsing, including chunked requests. No silent coercion, deduplication, truncation or padding occurs.

One serving worker and a nonblocking inference lock allow **one** model call at a time; concurrent inference is rejected with **503 / Retry-After: 1**, rather than queued without bound. Numerical thread limits are one. Health routes remain independent of inference. Memory exhaustion returns 503, other prediction/output-contract failures return sanitized 500. No generic error path substitutes Popularity. Failed model startup exits clearly instead of serving another version. Lifespan owns the model, with readiness disabled on shutdown.

Response: model name/version/`validation_candidate`, ordered results by user, and ordered recommendations containing `movie_id`, `rank`, `score`, `fallback`. Scores must be finite; duplicate pairs, invalid ranks or reordered backend output fail closed. Known zero-positive users keep the original per-source fallback/history behavior; unknown IDs explicitly receive Train Popularity without invented embeddings or personalized features. Unknown popularity counts and personalized LightGBM scores have different units; neither is a click probability. No histories, checkpoints, local paths, credentials or raw manifests are returned.

Actual native offline response for `{"user_ids":[1,1000610],"top_k":3}`:

```json
{
  "model": {
    "name": "MovieLensHybridRecommender",
    "version": "1",
    "release_status": "validation_candidate"
  },
  "results": [
    {
      "user_id": 1,
      "recommendations": [
        {
          "movie_id": 2716,
          "rank": 1,
          "score": 0.7629301414592253,
          "fallback": "none"
        },
        {
          "movie_id": 1265,
          "rank": 2,
          "score": 0.7215190220266638,
          "fallback": "none"
        },
        {
          "movie_id": 1240,
          "rank": 3,
          "score": 0.6916513905789993,
          "fallback": "none"
        }
      ]
    },
    {
      "user_id": 1000610,
      "recommendations": [
        {
          "movie_id": 318,
          "rank": 1,
          "score": 224.0,
          "fallback": "unknown_user_popularity"
        },
        {
          "movie_id": 356,
          "rank": 2,
          "score": 211.0,
          "fallback": "unknown_user_popularity"
        },
        {
          "movie_id": 296,
          "rank": 3,
          "score": 199.0,
          "fallback": "unknown_user_popularity"
        }
      ]
    }
  ]
}
```

Structured request logs record timestamp, normalized endpoint, status, elapsed time, batch size, version/release and error category. They omit bodies, IDs, recommendation contents, histories and credentials. Normal framework startup messages and nonfatal library warnings remain separate; internal exceptions are not returned in HTTP responses. The service is local educational software, not an internet-hardened endpoint.

## Docker runtime, dependencies and security

Host Docker CLI/Engine **29.8.2**, Docker Desktop **4.94.0 (241994)**, host darwin/arm64, daemon linux/arm64 (aarch64). The Docker VM reported 8 CPUs / approximately 11.67 GiB RAM. The image target is **native linux/arm64**, with no AMD64 emulation.

- Python base: `python:3.11.9-slim-bookworm@sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317`.
- Build uv: `ghcr.io/astral-sh/uv:0.13.6@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0`.
- Runtime OS additions: libgomp1 and CA certificates; compiler/uv build tooling is not copied to runtime.
- Tag: **`movielens-recsys-api:phase6_v1`**.
- Engine-reported image ID: **`sha256:22c38e5a263390737b778b5bb3b4fa349a74e6fa2f819a5cf57fe1186be71a30`**.
- Linux manifest digest: `sha256:6963f3b8a5e189c0c2f4dd0bf13bb5137008034b74d8bac7c7b9f5ccd79dab04`.
- Image size: **8,535,447,513 bytes (8.54 GB decimal)**; build closure installed 122 locked packages. uv install took approximately 262 s and layer export/unpack approximately 121 s; these are BuildKit step times, not fabricated total wall timing.

The immutable model's exact Torch package guard requires **2.11.0**. Official [CPU-index wheels](https://download.pytorch.org/whl/cpu/torch/) identify the Linux ARM64 CPU build as **2.11.0+cpu**, which the existing literal version guard would reject. That incompatibility was identified from the official wheel metadata and unchanged loader, not concealed by relabeling a package or editing Version 1. The pinned PyPI Linux distribution includes CUDA libraries and reports internal Torch build **2.11.0+cu130** while distribution metadata remains **2.11.0**. This consciously incurs the large image cost.

Actual inference is **CPU-only**: retained loaders choose CPU, CUDA_VISIBLE_DEVICES=-1, no GPU mount/requests, and measured `torch.cuda.is_available() == False`. No GPU/cloud job ran. A future explicitly approved release/runtime-compatibility contract could admit official CPU builds; no new model was registered just to simplify serving. Torch, implicit and LightGBM were retained because the existing import/compatibility contract requires them. Notebook/acquisition/DVC dependency groups are absent from the image; DuckDB remains needed by retained import paths. Exact versions:

| Package | Native / Docker version |
| --- | --- |
| mlflow | 3.17.2 |
| fastapi | 0.162.0 |
| uvicorn | 0.54.0 |
| httpx | 0.28.1 |
| torch | 2.11.0 |
| implicit | 0.7.2 |
| lightgbm | 4.6.0 |
| numpy | 2.4.4 |
| pandas | 3.0.2 |
| scipy | 1.17.1 |
| pyarrow | 24.0.0 |

The deny-by-default `.dockerignore` allows only Dockerfile, ignore rules, pyproject/lock and the six HTTP-adapter Python files. The observed build context was **470.75 kB**. Git, macOS venv/cache, DVC/data, model artifacts, outputs/SQLite, MLflow stores, archive, secrets and checkpoints were excluded. `/app` inside the image contains only the adapter; core model code/weights/history come from the read-only exported-model mount, not image layers or a development source mount. No image/artifact was pushed publicly.

Actual container `05f3718cd42e37e0b28696bcbc7db5d91f15ecb53a3c4a941b2dfb75637d6bb0` (`movielens-phase6-v1`): UID **10001**, one worker, two-CPU quota, one numerical thread, **3 GiB** memory cap, PID limit 256, all Linux capabilities dropped, no-new-privileges, read-only root filesystem, writable bounded /tmp tmpfs, and one read-only bind mount for the exported model. Host publication is **127.0.0.1:8001 → container 8000**; 0.0.0.0 is used only inside the container. No repository, original checkpoints, DVC files or target/candidate Parquets are mounted.

The readiness healthcheck became **healthy** and remained healthy after restart. Container start-to-ready **5.720 s**, measured lifespan load **2.118 s**, restarted-to-ready **4.884 s**. Final checks confirmed current image adapter source hashes equal the host deployment source and that the registered payload inventory is still unchanged.

## Parity, lifecycle and measurements

| Comparison | Queries / recommendation rows | Ranking/order/fallback mismatches | Maximum absolute / relative score difference |
|---|---|---|---|
| Fresh Version 1 vs Phase 5 reference | 611 / 12,220 | 0 | 0 / 0 |
| Native Registry HTTP vs Version 1 | 611 / 12,220 | 0 | 0 / 0 |
| Native offline first/restart vs Version 1 | 611 / 12,220 each | 0 | 0 / 0 |
| Linux Docker HTTP vs Version 1 | 611 / 12,220 | 0 | 0 / 0 |
| Linux Docker offline restart vs Version 1 | 611 / 12,220 | 0 | 0 / 0 |

All 610 known Train users, including warm/moderate/sparse and **four zero-positive-history users**, plus unknown user 1000610 are covered. Scores were compared with **tolerance zero**; no tolerance was increased to conceal rank changes. All 12,220 recommendations were also checked against serialized Train eligibility/history: **zero ineligible movies, seen-item violations or duplicate pairs**. New predictions are not a quality evaluation against labels.

Both native and Docker actual HTTP checks covered 18 strict-validation cases, malformed/missing JSON, body/batch bounds, user order, unknown/cold paths and K=1/10/100. Counter evidence shows load_count=1 after many requests, and load_count=1 in each fresh restarted process. Docker was stopped/restarted while owned MLflow was unreachable and still produced the full exact result. A separate wrong-release Docker start exited nonzero; the healthy deployment did not fall back or change identity.

Warm-up requests are excluded below. These are small sequential localhost samples, not production throughput or a load-capacity estimate; no P99 is claimed. Native startup used Registry download, Docker used an offline mount, so startup-source differences are explicit.

| Mode / request | N after warm-up | P50 ms | P95 ms | Sequential requests/s |
| --- | --- | --- | --- | --- |
| Native Mac / known_single | 10 | 46.063 | 47.886 | 21.757 |
| Native Mac / unknown_single | 10 | 4.154 | 4.454 | 240.863 |
| Native Mac / batch_8 | 6 | 124.662 | 127.705 | 8.055 |
| Linux Docker / known_single | 10 | 52.670 | 55.063 | 18.812 |
| Linux Docker / unknown_single | 10 | 6.440 | 7.723 | 150.311 |
| Linux Docker / batch_8 | 6 | 160.135 | 238.926 | 5.555 |

Native peak sampled RSS **446.1 MiB**. Representative Docker process RSS **346.5 MiB**; Docker stats reported **270.5 MiB / 3 GiB** working memory. RSS and cgroup working-set accounting differ; these are not a falsely equal metric or a claimed container peak. No OOM or resource-cap bypass occurred.

### Optional Docker Registry mode: precise limitation

`host.docker.internal:5002/health` returned **200** under the unchanged localhost MLflow binding. However, an actual registry-backed container startup failed: `/api/2.0/mlflow/model-versions/get` returned **403**, `Invalid Host header - possible DNS rebinding attack detected`. A separate diagnostic reproduced the exact failure. Health reachability alone therefore did **not** establish Registry usability. No allowed-host/security workaround or 0.0.0.0 MLflow rebind was introduced. Native Registry startup is verified; **Docker should use the required, fully verified offline export mode in this configuration**. This optional attempt is reported as failed/unavailable, not as a passing Docker Registry test.

## Tests and reproducible commands

1. **Synthetic/unit/ASGI/preflight suites:** all original 71 plus 14 new tests, **85 passed**, zero failures/skips, working **13.668 s**, isolated Mac dependency environment **15.252 s**. They include strict schemas, lifespan/load-once/readiness, sanitization, busy/resource failures, finite responses, short lists/no padding, offline loader isolation, fixed version/signatures, corrupt assets and checksum rejection.
2. **Real native registered-model/HTTP integration:** passed all reference/parity/validation and four failed-start negative cases; no real models were retrained.
3. **Required real Linux Docker integration:** passed build, health, CPU/non-root/read-only identity checks, genuine predictions, parity, restart/offline and negative startup. It was actually exercised, not a syntax-only Docker test.
4. **Optional Docker Registry startup:** failed with the diagnosed host-policy 403; deliberately left unavailable. This failure is separate from the 85 passing suites and required offline Docker checks.

`ast.parse` passed **75 active Python files**. Lock check, diff whitespace check, empty Git index, complete deployment inventory, all 744 protected hashes and 43 frozen metadata records passed. No warning indicating correctness failure was suppressed; nonfatal existing MLflow signature-inference advice is retained because an explicit model signature is already supplied, and sandbox PyArrow hardware-probe warnings remain visible.

Actual command records and full Docker argv/exit codes are in [phase6_commands.json](phase6_commands.json); measured details, runtime/security metadata, hashes and final status are in [verification](phase6_verification.json) and [results](phase6_serving_results.json). Logs, actual example JSON and full request samples are under ignored `outputs/phase6/`.

From the repository root (new output paths/container names are required when replaying immutable runs):

```sh
uv sync --locked --group train --group rank --group mlops --group serve
scripts/start_mlflow.sh start

.venv/bin/python -B tools/export_serving_model.py --tracking-uri http://127.0.0.1:5002 \
  --destination outputs/deployment_models/NEW_movielens_hybrid_v1
.venv/bin/python -B tools/create_serving_reference.py --tracking-uri http://127.0.0.1:5002 \
  --output-dir outputs/phase6/NEW_reference

# Native default offline mode; one worker, no reload.
RECSYS_MODEL_SOURCE=local RECSYS_MODEL_PATH="$PWD/outputs/deployment_models/movielens_hybrid_v1" \
  RECSYS_HOST=127.0.0.1 RECSYS_PORT=8000 .venv/bin/python -B -m recsys_serving
# Native Registry-backed alternative requires explicit RECSYS_MODEL_SOURCE=registry
# and RECSYS_TRACKING_URI=http://127.0.0.1:5002; version is still fixed to 1.

curl http://127.0.0.1:8001/health/live
curl http://127.0.0.1:8001/health/ready
curl http://127.0.0.1:8001/model-info
curl -H 'Content-Type: application/json' -d '{"user_ids":[1,1000610],"top_k":3}' \
  http://127.0.0.1:8001/recommend

docker build --platform linux/arm64 --progress plain -t movielens-recsys-api:phase6_v1 .
docker run -d --name NEW_movielens_phase6 --platform linux/arm64 --cpus 2 --memory 3g --pids-limit 256 \
  --read-only --cap-drop ALL --security-opt no-new-privileges \
  --tmpfs /tmp:rw,noexec,nosuid,size=536870912 -p 127.0.0.1:8001:8000 \
  --mount "type=bind,source=$PWD/outputs/deployment_models/movielens_hybrid_v1,target=/models/movielens_hybrid_v1,readonly" \
  movielens-recsys-api:phase6_v1

.venv/bin/python -B tools/verify_serving_http.py --url http://127.0.0.1:8001 \
  --reference-dir outputs/phase6/direct_reference --output outputs/phase6/NEW_http_check.json --latency
# Complete Docker checks create a new named container, restart it with owned MLflow
# temporarily stopped, restore MLflow in finally, and retain it only with --keep-running.
.venv/bin/python -B tools/verify_docker_serving.py --image movielens-recsys-api:phase6_v1 \
  --name NEW_verification_container --port 8004 --model outputs/deployment_models/movielens_hybrid_v1 \
  --reference-dir outputs/phase6/direct_reference --output-dir outputs/phase6/NEW_docker_check

docker logs movielens-phase6-v1
docker stop --timeout 30 movielens-phase6-v1
docker start movielens-phase6-v1
MPLCONFIGDIR=.venv/mplconfig PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B tools/run_tests.py
```

The current container already occupies 8001 and native offline API 8002; stop an owned instance or choose new ports/names before replay. Manual native shutdown is Ctrl-C; verification-created native PID/argv records are in `outputs/phase6/` for targeted SIGTERM after checking ownership. Do not kill an unrelated port owner. Startup failures should be diagnosed from safe error categories/local logs; do not disable checksum/version guards or silently switch to another model. A raw Phase 5 bundle path lacks outer `MLmodel` and is not the offline deployment path.

The implementation follows supported [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/), [Docker non-root/build guidance](https://docs.docker.com/build/building/best-practices/) and [Docker healthcheck](https://docs.docker.com/reference/dockerfile/#healthcheck) interfaces, verified against installed APIs and actual execution.

## Remaining boundaries and next step

The model remains a **Validation candidate**, with accepted cross-user temporal exposure, historical weak-label/sampling limitations and sparse-user regression. Unknown Popularity is not validated personalized cold-start quality. No Test, online CTR/exposure or causal performance was measured. Loopback timings and basic local safeguards do not establish production-scale readiness. SQLite/model storage still has no automatic remote backup; the model contains Train histories and must not be publicly distributed. DVC environment drift is unchanged, and the large exact-version Linux image plus optional Registry host-policy failure remain explicit limitations.

The required local serving lifecycle is ready for a **separately controlled final Test evaluation** after explicitly recording the selected Version 1 and unchanged protocol. That future task must keep frozen Train histories/catalog, open Test only under the agreed evaluation plan, and avoid tuning from Test outcomes. Nothing in Phase 6 performs that evaluation. Final Git HEAD remains `1e7e65664723bc86eedbccef572f17dc79113b4b`; all current changes are uncommitted, with exact status in verification JSON.
