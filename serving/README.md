# serving/

Torch-free inference service for the MILK10k `final-P1a-kg` model: the `milk10k_serving` package (preprocessing, ONNX Runtime ensemble, predictor) and its FastAPI app (`milk10k_serving.api`), and the API image (`Dockerfile`, build context `serving/` with an allowlist `.dockerignore`).

Install: `make setup` (pinned `requirements.txt` + dev tools + the package), or `pip install "./serving[api]"` (the `api` extra adds FastAPI, pydantic, python-multipart, uvicorn, prometheus-client and the OpenTelemetry SDK, OTLP/HTTP exporter and FastAPI instrumentation to the predictor core). The build backend is pinned (`setuptools==84.2.1` in `pyproject.toml`), so image builds do not pick up a newer one by themselves.

Run (from the repository root, after `make setup`; settings from environment variables, see `.env.example`): `make serve`, i.e. `uvicorn milk10k_serving.api:app --host 0.0.0.0 --port 8000`. Endpoints: `GET /health` (liveness), `GET /ready` (503 while the model loads in the background, 200 when ready), `POST /predict` (multipart), `GET /metrics` (Prometheus, `metrics.py`), `GET /docs` (OpenAPI). Tracing (`tracing.py`): on when `OTEL_EXPORTER_OTLP_ENDPOINT` is set (OTLP/HTTP, e.g. `http://jaeger:4318`), off by default; a collector that is down never affects `/predict`. onnxruntime telemetry: importing `milk10k_serving.package` sets `ORT_DISABLE_TELEMETRY=1` unless the environment already has a value (onnxruntime reads it when it is imported; an earlier import without it is logged as an error). See the root README's "API" section for the status codes and `curl` examples, and `client.py` for a Python client (standard library only): `python serving/client.py ISIC_0145429`.

Container: `make up` from the repository root (Compose, see the root README) or `docker build -t milk10k-api serving` and run with the model package mounted read-only at `/models/final-P1a-kg-onnx` (`MODEL_DIR`); the image has no model files, no torch, runs as uid 10001 and its healthcheck polls `/ready`.

Research model, not a medical device.
