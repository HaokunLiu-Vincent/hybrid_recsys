# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.13.6@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0 AS uv
FROM python:3.11.9-slim-bookworm@sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317 AS builder
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /build
COPY pyproject.toml uv.lock ./
# The immutable Version 1 requires torch==2.11.0 exactly. Its Linux PyPI
# distribution declares CUDA runtime dependencies; official +cpu builds have a
# different version identity and are rejected by the unmodified model loader.
# Preserve the locked distribution, run on CPU, and record this image-size cost.
RUN uv sync --frozen --no-default-groups --group train --group rank --group mlops --group serve --no-install-project

FROM python:3.11.9-slim-bookworm@sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317 AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY src/recsys_serving/ ./recsys_serving/
ENV PATH=/opt/venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    MLFLOW_ENABLE_USAGE_TRACKING=false MPLCONFIGDIR=/tmp/matplotlib XDG_CACHE_HOME=/tmp/cache \
    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    RECSYS_MODEL_SOURCE=local RECSYS_MODEL_PATH=/models/movielens_hybrid_v1 \
    RECSYS_EXPECTED_MODEL_VERSION=1 RECSYS_EXPECTED_RELEASE_ID=e4b38328a35c992114eb3e6acda90803f0aa16c61a027686d9e86b06bf5e475d \
    RECSYS_HOST=0.0.0.0 RECSYS_PORT=8000 RECSYS_MAX_BATCH_USERS=32 RECSYS_CPU_THREADS=1
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=45s --retries=3 \
    CMD ["/opt/venv/bin/python", "-c", "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.getenv('RECSYS_PORT','8000')+'/health/ready',timeout=2)"]
CMD ["/opt/venv/bin/python", "-B", "-m", "recsys_serving"]
