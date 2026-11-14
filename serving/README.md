# serving/

Torch-free inference service for the MILK10k `final-P1a-kg` model: the `milk10k_serving` package (preprocessing, ONNX Runtime ensemble, predictor) and its FastAPI app (`milk10k_serving.api`); the Dockerfile comes later.

Install: `make setup` (pinned `requirements.txt` + dev tools + the package), or `pip install "./serving[api]"` (the `api` extra adds FastAPI, pydantic, python-multipart and uvicorn to the predictor core).

Run (from the repository root, after `make setup`; settings from environment variables, see `.env.example`): `make serve`, i.e. `uvicorn milk10k_serving.api:app --host 0.0.0.0 --port 8000`. Endpoints: `GET /health` (liveness), `GET /ready` (503 while the model loads in the background, 200 when ready), `POST /predict` (multipart), `GET /docs` (OpenAPI). See the root README's "API" section for the status codes and `curl` examples, and `client.py` for a Python client (standard library only): `python serving/client.py ISIC_0145429`.

Research model, not a medical device.
