# scripts/

Repository tooling: `check_repo_hygiene.py` (no data, images, checkpoints or secrets in git; notebook output rules).

`fetch_model.sh` (`make fetch-model`): downloads the model package from the GitHub Release `model-v1.0.0` with `curl` (no token), checks `manifest.json` against a pinned sha256 and every file against the manifest; resumable and idempotent; `--check` only verifies a folder. Folder: the `DEST_DIR` argument, else `MODEL_HOST_DIR` resolved as Compose does (environment, then `.env`, relative to the repository root), else `artifacts/model_package/final-P1a-kg-onnx`. The package is CC BY-NC 4.0, non-commercial.

`smoke_test.py` (P1-BSH-04, standard library only): the post-deploy check. Waits for `/ready` (default 120 s; a failed model load stops at once), then checks `/health`, one `/predict` with a synthetic PNG generated in the script (response fields and consistency, not the prediction) and `/metrics` (the `milk10k_` metrics, model ready, the `/predict` counted). `python3 scripts/smoke_test.py [--base-url URL]` (default `API_BASE_URL` or `http://127.0.0.1:8000`); exit 0 pass, 1 fail, 2 usage error.
