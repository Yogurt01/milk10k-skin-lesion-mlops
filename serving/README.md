# serving/

Torch-free inference service for the MILK10k `final-P1a-kg` model: the `milk10k_serving` package (preprocessing, ONNX Runtime ensemble, predictor) and its FastAPI app (`milk10k_serving.api`); the Dockerfile comes later.

Run (from the repository root, after `make setup`; settings from environment variables, see `.env.example`): `make serve`, i.e. `uvicorn milk10k_serving.api:app --host 0.0.0.0 --port 8000`. Endpoints: `GET /health` (liveness), `GET /ready` (503 while the model loads in the background, 200 when ready), `POST /predict` (multipart), `GET /docs` (OpenAPI).

Research model, not a medical device.
