# Local environment, DuckDB pipeline and DVC recovery

The authoritative dependency configuration is `pyproject.toml` + `uv.lock` (uv 0.13.6), with Python 3.11.9 pinned by `.python-version`. Base dependencies support data preparation and retained non-neural interfaces; default `dev` adds local DVC 3.67.1. `notebook` and `acquisition` are optional extras. There are no serving/tracking/cloud dependencies. Both requirements files are compatibility entry points into this same project, not independent version constraints.

## Environment

For a fresh macOS checkout with uv installed, create a normal local environment; do not rerun environment creation over an existing working `.venv`:

```sh
uv venv --python 3.11.9
uv sync --locked --all-extras
```

For this existing repository, the verified command was:

```sh
UV_CACHE_DIR=.uv-cache .venv/bin/uv sync --locked --inexact --all-extras
UV_CACHE_DIR=.uv-cache .venv/bin/uv lock --check --offline
```

`--inexact` preserves unrelated installed packages ([uv sync documentation](https://docs.astral.sh/uv/concepts/projects/sync/)). The existing `.venv` originally inherited libraries from a pre-existing Python installation; it was preserved, not migrated to Conda or recreated. The data/notebook/DVC dependencies were synced from the lock. Existing PyTorch 2.11.0 and implicit 0.7.2 were neither reinstalled nor changed. They remain **outside this phase's dependency lock**, as does uninstalled LightGBM: this is a reproducible data environment, not a claimed reproducible training environment. Fresh environments can run data tests and import the retained lazy model modules; the two existing PyTorch mathematical checks skip when PyTorch is absent. The actual current run had zero skips.

If uv is unavailable, install uv 0.13.6 using your approved installation method; in this task it was bootstrapped into the existing `.venv` via pip. No Conda command was used. The project-local UV cache avoids restricted writes to the user's global cache. Notebook extras preserve the Phase 1 workflow, but the historical executed notebook/report were not rewritten under the new environment.

## DVC layout and ownership

- Git holds source, `params.yaml`, environment lock, `dvc.yaml`, `dvc.lock`, `.dvc/config`, `.dvcignore`, `.dvc/.gitignore`, data-directory `.gitignore` files and `data/raw/ml-latest-small.dvc`.
- Raw data is owned **only** by that `.dvc` pointer. Acquisition is not a pipeline output and is never automatically rerun by `dvc repro`.
- `validate_raw` verifies pinned source/provenance SHA256, exact headers and DuckDB schema/quality checks; it owns `data/validation/small_raw_v1.json`.
- `prepare_splits` owns the two distinct versioned split directories and `data/diagnostics/small_protocol_v1.json`. No other stage owns a file inside these directories.
- `validate_splits` recomputes assignments, catalog/cohorts, mappings and inner windows against source values, and owns `data/validation/small_splits_v1.json`.
- `.dvc/cache/` is this checkout's disposable object cache, distinct from its working data. `cache.type=copy` avoids hardlink/symlink coupling to raw inputs.
- DVC's disposable state/index databases use `.dvc/tmp/site-cache/`, configured locally; they are not dataset objects or remote contents. This also avoids restricted writes to system cache locations. Recovery tests use a separate state directory per disposable workspace.
- The persistent **local remote** is `/Users/vincent/Documents/projects/hybrid_recsys_dvc_remote`, outside the repository, with a keep notice and private directory permissions. This is a data store, not a second source repository. Do not delete it or run garbage collection without reviewing recovery copies. A same-disk remote protects against workspace removal, not disk loss.
- The machine-specific remote URL/default lives in ignored `.dvc/config.local`. No secrets are present. On another machine, choose a persistent local directory outside this checkout and configure its URL locally. Do not place writable DuckDB/MLflow databases in this remote or a cloud-synced directory.

All stage outputs use DVC `persist: true` and the Python writer refuses incompatible existing versions. A no-change run skips; a direct identical rerun verifies/reuses outputs. Changed source/configuration invalidates the declared stages, but **does not authorize overwriting frozen split versions**. For a reviewed new version, update the versioned paths in configuration, CLI report locations and DVC stage outputs together. This deliberately small v1 pipeline is not a generic migration framework. Never force regeneration over user data or old 32M splits.

## Initialize on a fresh checkout only

The following initialization has already been performed here. Do not repeat `dvc init` or `dvc add` over an existing setup. DVC metadata remains uncommitted until the user reviews it.

```sh
.venv/bin/dvc init
.venv/bin/dvc config core.analytics false
.venv/bin/dvc config core.autostage false
.venv/bin/dvc config cache.type copy
.venv/bin/dvc config --local core.site_cache_dir "$PWD/.dvc/tmp/site-cache"

# Select a persistent external directory appropriate to this machine.
.venv/bin/dvc remote add --default --local local /absolute/path/outside/repository/hybrid_recsys_dvc_remote

# Only after the exact Kaggle acquisition has been verified; preserve all raw files.
.venv/bin/dvc add data/raw/ml-latest-small
```

DVC initialization may stage its own metadata; this task unstaged only those three newly created paths and left everything uncommitted. `dvc push` transfers data objects to the configured **local filesystem** remote; it is unrelated to `git push` ([DVC remote configuration](https://doc.dvc.org/command-reference/remote/add)). No cloud remote is configured.

## Reproduce and validate

Run from this repository root after syncing the locked data dependencies and configuring the local remote:

```sh
# With available DVC objects: recover raw data/artifacts if missing.
DVC_NO_ANALYTICS=1 .venv/bin/dvc pull

# Or, for a first independent acquisition only: never substitute another Kaggle source.
# .venv/bin/python tools/download_movielens_small.py
# Pinned params.yaml hashes must agree with that exact acquisition, including provenance.

DVC_NO_ANALYTICS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/dvc repro
DVC_NO_ANALYTICS=1 .venv/bin/dvc status
DVC_NO_ANALYTICS=1 .venv/bin/dvc push

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B tools/run_tests.py
DVC_NO_ANALYTICS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B tools/verify_small_repro.py
git diff --check
```

A fresh acquisition has a different provenance timestamp/hash even if its CSV hashes match; that difference is intentionally not silently adopted. Prefer pulling this frozen acquisition from DVC. A new download requires reviewing provenance and updating the source pin as a new version, rather than editing the original manifest to resemble an old acquisition.

Equivalent explicit stage commands (normally DVC manages their order/cache):

```sh
.venv/bin/python -B tools/prepare_small_data.py validate-raw
.venv/bin/python -B tools/prepare_small_data.py prepare
.venv/bin/python -B tools/prepare_small_data.py validate-splits
```

DuckDB runs in memory, in UTC, with one thread and a 1 GB memory limit. CSV parsing uses explicit schemas and RFC-style doubled quote escaping; no rows/errors are ignored. No persistent DuckDB database is created. Parquet is sorted deterministically by `(userId,timestamp,movieId)` with ZSTD compression under the locked DuckDB writer. Schemas, file SHA256, raw provenance hash, policy, actual counts, source hashes, Python/DuckDB versions and lockfile hash are in each split manifest. See [the evaluation contract](evaluation_protocol.md).

## Verified safe recovery test

`tools/verify_small_repro.py` verifies clean `dvc status`, checks output SHA256 and mtimes around a no-change `dvc repro`, and independently builds both protocols into a temporary directory for byte comparison. It then creates a **disposable DVC-only, no-Git recovery workspace**, copies metadata and stage dependencies, configures an empty separate cache and reads the external local remote. It does not reproduce a second implementation or execute model training.

In that disposable workspace, it executes `dvc init --no-scm`, local remote configuration, `dvc pull`, `dvc status --json`, removes only one restored Train file, and executes `dvc pull` again. The first pull verifies cold-cache remote recovery; the second demonstrates missing-output restoration. Original working data, original cache and persistent remote are never deleted. Finally, disposable config/raw changes verify stage invalidation **without** running those altered stages. All original data/report hashes and mtimes are rechecked.

Actual result: independent build byte-identical, 40 restored files SHA256-identical, no-change repro rewrote no outputs, final DVC status clean. `dvc push` uploaded 41 object files; cold pull fetched 41 objects and restored 40 workspace files. Object count differs because directories have metadata objects and content can be deduplicated. Exact command/output evidence is under ignored `outputs/phase2/reproducibility_*.json`, with a concise checked record in [Phase 2 verification](movielens_small_phase2_verification.json).

Byte identity was verified on this Mac with the current locked writer/runtime, not asserted across arbitrary OS/Python/DuckDB versions.

Nothing here trains a retriever/ranker, calculates model performance, configures MLflow, builds a service/container, or commits/pushes Git changes.
