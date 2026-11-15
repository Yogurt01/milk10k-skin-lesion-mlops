# scripts/

Repository tooling: `check_repo_hygiene.py` (no data, images, checkpoints or secrets in git; notebook output rules).

`fetch_model.sh` (`make fetch-model`): downloads the model package from the GitHub Release `model-v1.0.0` with `curl` (no token), checks `manifest.json` against a pinned sha256 and every file against the manifest; resumable and idempotent; `--check` only verifies a folder. Default folder: `artifacts/model_package/final-P1a-kg-onnx`. The package is CC BY-NC 4.0, non-commercial.
