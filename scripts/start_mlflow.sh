#!/bin/sh
set -eu
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export PYTHONDONTWRITEBYTECODE=1
exec "$repo_dir/.venv/bin/python" -B -m recsys.mlflow_local "${1:-start}"
