# tests/

PyTest suite for the serving code. Markers (see the root `pyproject.toml`): `unit` (no model files), `api` (FastAPI `TestClient` or a real uvicorn server in a thread, `live_server.py`, with the fakes in `fakes.py`; no model files), `parity` (needs the model package and the sample lesions) and `slow`.

| Command | Runs |
|---|---|
| `make test` | `unit` + `api` (no model files, a few seconds) |
| `make test-report` | the same, plus `reports/junit.xml` and `reports/coverage.xml`; fails below `COV_MIN` % coverage (default 90) |
| `make test-container` | the HTTP parity test on the sample lesions against a running API at `API_BASE_URL` (default `http://127.0.0.1:8000`, the Compose stack from `make up`); no server or model in the test process |
| `make test-all` | everything, incl. `parity`/`slow`: the in-process parity test, the HTTP parity test on the sample lesions (`test_http_parity.py`) and, when `MILK10K_DATA_DIR` points at the prepared dataset, all 479 reference test lesions through HTTP (writes `reports/parity_full.json`) |

Model files and sample lesions missing: the tests that need them are skipped, or fail with `REQUIRE_MODEL_PACKAGE=1` (CI). The 479-lesion run is skipped without `MILK10K_DATA_DIR` in either case.

`API_BASE_URL`: when set, `test_http_parity.py` talks to that server instead of starting uvicorn in the test process (P1-TST-05). The 479-lesion test also runs that way when `MILK10K_DATA_DIR` is set too; it is a one-off check (about 4 minutes), not part of `make test`.

Observability: `test_metrics.py` (`/metrics` format, counters on 200/422/503, busy counter, bounded labels with 1,000 random paths, answers while the threadpool is saturated) and `test_tracing.py` (span tree with an in-memory exporter, no request data in attributes, probes not traced, fail-soft with a collector that is down or never answers). `test_package.py` also checks the onnxruntime telemetry default in clean subprocesses.

MLflow history import: `test_mlflow_history.py` checks the mapping of `scripts/mlflow_history.py` on small fake files (never the real results; no MLflow needed, so it runs in `make test` and Jenkins). Its end-to-end test (import twice into a temporary SQLite store: no duplicates, no local path in the store) is skipped unless MLflow is installed; it runs in the MLflow venv with `artifacts/mlflow/.venv/bin/python -m pytest --noconftest tests/test_mlflow_history.py` (pytest added to that venv by hand; `--noconftest` because `conftest.py` imports the serving package).
