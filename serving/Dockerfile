# syntax=docker/dockerfile:1
# API image for the MILK10k model (P1-DCK-01). Torch-free: ONNX Runtime on CPU.
# Build context: serving/ (see serving/.dockerignore), e.g. `docker build -t milk10k-api serving` or `make up`.
#
# The model package (~1.4 GB, CC BY-NC 4.0, non-commercial) is NOT in the image: mount it read-only and point
# MODEL_DIR at it (default /models/final-P1a-kg-onnx); `scripts/fetch_model.sh` downloads it from the GitHub Release.
# Research model, not a medical device.

# Tag pinned to the patch release and to its digest (tags are mutable); update both together.
FROM python:3.12.16-slim-trixie@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ORT_DISABLE_TELEMETRY=1 \
    MODEL_DIR=/models/final-P1a-kg-onnx
# ORT_DISABLE_TELEMETRY: onnxruntime 1.31 otherwise opens an HTTPS connection to mobile.events.data.microsoft.com when
# a session is created (measured 2026-11-15, docs/serving_measurements.md §9).

# Fixed UID/GID (no login, no home directory): the app owns nothing in the image, so it cannot change its own code.
RUN groupadd --system --gid 10001 app \
    && useradd --system --no-log-init --uid 10001 --gid app --home-dir /nonexistent --shell /usr/sbin/nologin app

WORKDIR /app

# 1. Dependencies first (exact pins from uv pip compile): this layer is rebuilt only when requirements.txt changes.
#    onnxruntime ships three tiny example models (datasets/*.onnx); removed so the image holds no .onnx file at all.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -f /usr/local/lib/python3.12/site-packages/onnxruntime/datasets/*.onnx

# 2. The package with its `api` extra; --no-deps so nothing unpinned is added, then pip check confirms the extra is
#    satisfied by the pinned set.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps ".[api]" \
    && pip check \
    && rm -rf /app/src /app/build

USER app:app
EXPOSE 8000

# Readiness, not liveness: 200 only once the model package is loaded (overload never flips it, API-D5).
# Python standard library only (the slim image has no curl).
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --start-interval=2s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request as u; sys.exit(0 if u.urlopen('http://127.0.0.1:8000/ready', timeout=4).status == 200 else 1)"]

# One process; concurrency comes from the in-app limiter (MAX_CONCURRENT_PREDICTIONS), scaling from replicas.
# --no-access-log: the app writes one access line per request itself, probes at DEBUG (API-D9); uvicorn's own access
# log would repeat each line and log every healthcheck at INFO.
CMD ["uvicorn", "milk10k_serving.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
