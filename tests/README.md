# tests/

PyTest suite for the serving code. Markers (see the root `pyproject.toml`): `unit` (no model files), `api` (FastAPI `TestClient` or a real uvicorn server in a thread, `live_server.py`, with the fakes in `fakes.py`; no model files), `parity` (needs the model package and the sample lesions) and `slow`.

| Command | Runs |
|---|---|
| `make test` | `unit` + `api` (no model files, a few seconds) |
| `make test-report` | the same, plus `reports/junit.xml` and `reports/coverage.xml`; fails below `COV_MIN` % coverage (default 90) |
| `make test-all` | everything, incl. `parity`/`slow`: the in-process parity test, the HTTP parity test on the sample lesions (`test_http_parity.py`) and, when `MILK10K_DATA_DIR` points at the prepared dataset, all 479 reference test lesions through HTTP (writes `reports/parity_full.json`) |

Model files and sample lesions missing: the tests that need them are skipped, or fail with `REQUIRE_MODEL_PACKAGE=1` (CI). The 479-lesion run is skipped without `MILK10K_DATA_DIR` in either case.
