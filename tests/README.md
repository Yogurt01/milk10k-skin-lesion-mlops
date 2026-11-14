# tests/

PyTest suite for the serving code. Markers (see the root `pyproject.toml`): `unit` (no model files), `api` (FastAPI `TestClient` with the fakes in `fakes.py`; no model files), `parity` (needs the model package and the sample lesions) and `slow`.
