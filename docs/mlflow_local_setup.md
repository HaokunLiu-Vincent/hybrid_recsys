# Local MLflow operations

Python 3.11.9 and **MLflow 3.17.2** are pinned through `pyproject.toml` / `uv.lock`. The `mlops` group is separate from `train` and `rank`. A fresh environment uses:

```sh
uv venv --python 3.11.9
uv sync --locked --group train --group rank --group mlops
```

The existing environment was synced with `--inexact --group mlops --group rank --all-extras` to preserve its inherited Torch/implicit wheels. No Conda migration occurred. MLflow brings HTTP-server libraries and a Docker Python SDK as dependencies; this project implements no recommendation HTTP service or Docker image in Phase 5.

## Start, status and stop

Run from the repository root, or invoke the script by its absolute path:

```sh
scripts/start_mlflow.sh start
scripts/start_mlflow.sh status
scripts/start_mlflow.sh stop
```

The script resolves paths independently of the caller's working directory. It starts one worker, binds **127.0.0.1**, and persists:

```text
outputs/mlflow/
  mlflow.db          # SQLite tracking AND registry backend
  artifacts/        # HTTP-served persistent model/run artifacts
  server.json       # owned PID, stable selected port, explicit URIs
  server.log        # append-only server log
```

The verified instance uses **http://127.0.0.1:5002** for both Tracking and Registry. Port 5000 was occupied and was not killed. The selected port is remembered in `server.json`; the same URI survived subsequent restarts. A newly occupied preferred port produces a documented alternate in the returned state. Always inspect the returned URI; clients use this owned-server state, not an implicit remote MLflow environment setting. `stop` checks process ownership/backend arguments and gracefully terminates only the process group started by this script. Never kill an unrelated port owner.

The actual SQLite URI is `sqlite:////Users/vincent/Documents/projects/hybrid_recsys/outputs/mlflow/mlflow.db` (four slashes for an absolute POSIX path). The server explicitly uses this for backend and registry, `--serve-artifacts`, and an absolute `--artifacts-destination`. Clients use HTTP artifact proxy URIs, not the server's filesystem. No DVC remote, cloud storage or public bind address is configured.

Verify real HTTP operations and persistence, creating a labeled health-probe run/version without deleting older probes:

```sh
MLFLOW_ENABLE_USAGE_TRACKING=false MPLCONFIGDIR=.venv/mplconfig \
  .venv/bin/python tools/verify_mlflow_local.py --restart \
  --output outputs/phase5/NEW_health_record.json
```

This checks experiments, parameters, metrics, artifacts, run querying, registry writes and restart persistence. `status` reports local reachability; sandboxed callers may require permission to connect to localhost even when the server is running. A PID alone does not establish health.

## Experiments and historical import

- `MovieLens-Small-Retrieval`: six Full-Train and six historical retriever records.
- `MovieLens-Small-Fusion-Ranking`: Validation RRF, historical union, features, ranking dataset and LambdaMART.
- `MovieLens-Small-Release`: newly executed complete-model registration.
- `MovieLens-Small-Health`: explicit infrastructure probes.

```sh
UV_CACHE_DIR=.uv-cache .venv/bin/uv run --locked --no-sync \
  --group train --group rank --group mlops recsys mlflow-backfill
```

Save the returned JSON for release lineage; the actual verified import is in `outputs/phase5/gate_b.json`. Imports never retrain. Their `retrospective_import=true`, original source manifests/hashes, import timestamp and explicit upstream MLflow run IDs distinguish old experiment execution from today's logging. Missing original execution timestamps remain explicitly unavailable. Identical reruns verify existing source/metric records and return their IDs; conflicts and incomplete imports fail for inspection rather than overwriting or deleting records.

For a noninteractive JSON file, use the wrapper below; it keeps framework console messages out of the summary and refuses to overwrite it:

```sh
.venv/bin/python tools/backfill_mlflow.py --output outputs/phase5/NEW_backfill.json
```

## Package, register and load

Completed local releases are immutable. Use new IDs when packaging; the existing `phase5_v1` is already complete:

```sh
UV_CACHE_DIR=.uv-cache .venv/bin/uv run --locked --no-sync --group train --group rank --group mlops \
  recsys package-release --run-id NEW_release_id

UV_CACHE_DIR=.uv-cache .venv/bin/uv run --locked --no-sync --group train --group rank --group mlops \
  recsys register-release --release artifacts/releases/NEW_release_id \
  --parity outputs/release_checks/NEW_release_id --backfill-summary outputs/phase5/gate_b.json

UV_CACHE_DIR=.uv-cache .venv/bin/uv run --locked --no-sync --group train --group rank --group mlops \
  recsys verify-release --model-uri models:/MovieLensHybridRecommender@candidate \
  --parity outputs/release_checks/phase5_v1
```

Registration verifies a matching passed parity record, packages all six Full-Train retrievers and the ranker, and assigns only `candidate`. An existing candidate pointing at a different release is rejected; this command does not authorize alias promotion/reassignment. Identical registration returns the same version. Verification never fits, backfills or creates versions.

```python
import mlflow
import pandas as pd
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5002")
mlflow.set_registry_uri("http://127.0.0.1:5002")
version = MlflowClient().get_model_version_by_alias("MovieLensHybridRecommender", "candidate")
model = mlflow.pyfunc.load_model("models:/MovieLensHybridRecommender@candidate")
recommendations = model.predict(pd.DataFrame({"userId": [1, 42, 1000610], "top_k": [20, 10, 5]}, dtype="int64"))
```

Input: unique positive int64 user IDs and integer `top_k` in 1..100. Output: `userId:int64,movieId:int64,rank:int32,score:float64,fallback:string`, grouped in request-user order, then rank. No padding or duplicate recommendations. Unknown users receive Train popularity and bypass the ranker; their raw popularity scores are not calibrated with known-user LightGBM scores. Known zero-positive users retain the existing per-source fallback behavior and their actual Train-derived features; cold-start quality has not been evaluated. Invalid inputs raise informative exceptions.

Direct, version-compatible Python inference is also available:

```python
from recsys.inference import load_hybrid_recommender
recommendations = load_hybrid_recommender("artifacts/releases/phase5_v1").recommend([1, 42], top_k=20)
```

MLflow uses Models From Code and supported `code_paths`, not a pickled application object. It supplies bundled application code when the project is not installed. The release includes pinned runtime requirements plus the whole uv project/lock for reconstruction. Keep environments, caches and bytecode outside immutable release directories; do not create `.venv` inside a completed release. For an isolated dependency environment use an external `UV_PROJECT_ENVIRONMENT`, `uv sync --locked --no-default-groups --group train --group rank --group mlops --no-install-project`, and load the registry model, which supplies application code. `tools/verify_registry_model.py` is intended to be copied to a disposable directory and run with that environment's `python -I -B`; it denies reads of the original checkout. The exact executed command is in the Phase 5 verification JSON.

## Persistence, backup and recovery

SQLite and artifacts persist across server restarts. **They are not automatically uploaded to GitHub, DVC or a remote backup.** All generated stores/logs are ignored by existing `.gitignore` rules. The model contains Train interaction histories and is for local educational use, not public dataset redistribution.

For a consistent backup, stop the owned server, copy the entire `outputs/mlflow/` directory to a **new** protected local backup location, verify file hashes, then restart. Keep the database and artifact tree together, including any SQLite sidecar files present. Back up `artifacts/releases/` and source/lock separately as needed. Do not copy an actively written database as if it were guaranteed consistent. No automatic backup was created or remote backup configured in this phase.

After recovery to the same location, start and verify the alias/run IDs and artifact downloads. Relocating the *server store* may require updating SQLite experiment artifact roots and server settings using supported MLflow migration procedures; model-bundle portability does not imply arbitrary tracking-store relocation. Back up before any future MLflow/database upgrade. Never reset the database, delete experiments/versions, or edit registry rows to hide a failed import. For an interrupted run or registration, inspect the recorded source fingerprint, run status and logged-model URI before explicitly recovering it; automated code fails rather than silently duplicating an incomplete record.

MLflow tracks experiments and model releases; DVC tracks datasets. Global training/MLflow lock additions still differ from frozen Phase 2 DVC dependencies. Do not run real DVC repro or change `dvc.lock` to conceal that drift. See [Phase 5 results](phase5_mlflow_registry.md) and the unchanged [Phase 4 migration proposal](phase4_fusion_and_ranking.md#dvc-environment-drift-assessed-and-deliberately-deferred).
