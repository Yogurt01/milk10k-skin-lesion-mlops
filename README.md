# MILK10k skin-lesion classification: MLOps project

> **Research model, not a medical device.** It was developed for the ISIC MILK10k research benchmark; its output is
> not a diagnosis and must not be used for clinical decisions.

**Status:** work in progress.

## Run with Docker Compose

Needs Docker with Compose v2, `curl` and `python3`. From the repository root:

```bash
make up      # downloads the model package (~1.4 GB, once; sha256-checked), builds the image, waits until healthy
curl -s localhost:8000/ready
make down
```

- **Model download:** `make up` runs `scripts/fetch_model.sh`, which downloads the model package from the GitHub Release [`model-v1.0.0`](https://github.com/Yogurt01/milk10k-skin-lesion-mlops/releases/tag/model-v1.0.0) into `artifacts/model_package/final-P1a-kg-onnx/`. Interrupted downloads resume; files already verified are skipped.
- **Model files:** the model package (CC BY-NC 4.0, non-commercial) is not in the image. Compose mounts it read-only.
- **Network:** the API listens on `127.0.0.1:8000` only.
- **Settings:** optional; `cp .env.example .env` and edit.

## API

FastAPI service in [`serving/`](serving/) (`milk10k_serving.api`). Start it from the repository root with
`make serve` (port 8000; settings from environment variables, see [`.env.example`](.env.example)). The model package
loads in the background, so `/ready` turns 200 a few seconds after start. Interactive docs: `http://localhost:8000/docs`.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness: always 200 while the process runs (also while loading and under overload) |
| `GET /ready` | Readiness: 200 once the model is loaded and verified, 503 while loading or after a failed load |
| `POST /predict` | One lesion: dermoscopic image (required), clinical image (optional), metadata → P(malignant), 11 class probabilities, diagnosis, `disclaimer` |
| `GET /docs`, `GET /openapi.json` | OpenAPI documentation, including the allowed metadata values once the model is loaded |

```bash
curl -s localhost:8000/health
# {"status":"ok"}

curl -s localhost:8000/ready
# {"status":"ready","model":"final-P1a-kg-onnx","model_version":"1.0.0","load_seconds":3.27}

# Images from the sample lesions (built by training/sample/make_sample.py; not in git)
curl -s -X POST localhost:8000/predict \
  -H 'X-Request-ID: readme-example-1' \
  -F derm_image=@training/sample/images/test/derm/ISIC_0145429.jpg \
  -F clinical_image=@training/sample/images/test/clinical/ISIC_7021698.jpg \
  -F age_approx=60 -F sex=male -F anatomical_site_general=head/neck -F fitzpatrick_type=5
# {"p_malignant":0.8439921140670776,"probabilities":{"AKIEC":0.1065...,"SCCKA":0.5778...,...},"diagnosis":"SCCKA",
#  "clinical_image_used":true,"model":"final-P1a-kg-onnx","model_version":"1.0.0","metadata_missing":[],
#  "disclaimer":"Research model for the ISIC MILK10k benchmark, not a medical device. ..."}

curl -s localhost:8000/openapi.json | python3 -m json.tool | head
```

Python client (standard library only): `python serving/client.py ISIC_0145429` prints the diagnosis, P(malignant)
and the disclaimer for a sample lesion (`--base-url`, `--sample-dir` to change the defaults).

**Form fields of `/predict`:** `derm_image` (required), `clinical_image` (optional; a missing or undecodable clinical
image is treated as missing), `sex` and `fitzpatrick_type` (required), `age_approx` (0-120) and
`anatomical_site_general` (optional: omitted or empty is encoded as missing and listed in `metadata_missing`).
Allowed values come from the model package's `tabular_encoding.json`.

| Status | When |
|---|---|
| 200 | Prediction; `disclaimer` is always present |
| 400 | The body is not parseable multipart form data |
| 413 | The body is larger than two images at `MAX_UPLOAD_MB` plus 1 MB (from `Content-Length`), or one image is larger than `MAX_UPLOAD_MB` |
| 422 | `sex` or `fitzpatrick_type` missing; a value outside the allowed set; dermoscopic image missing, empty or not decodable; an image above `OPENCV_IO_MAX_IMAGE_PIXELS` decoded pixels |
| 503 | Model loading (`Retry-After`) or failed to load; or server busy (`Retry-After`) |
| 500 | Unexpected error; generic message only |

- **Validation order:** readiness is checked before field validation, so while the model is not ready every
  `/predict` gets 503, even one with an invalid field.
- **Overload:** at most `MAX_CONCURRENT_PREDICTIONS` predictions run at once and `MAX_QUEUED_PREDICTIONS` more wait;
  further requests get 503 with `Retry-After` at once. Overload is not "not ready": `/ready` stays 200.
- **Request ids:** every response carries `X-Request-ID` (the client's value if it is 1-64 characters of
  `[A-Za-z0-9._:-]`, otherwise a generated one); the server's log lines for that request carry it too. Logs contain
  ids, sizes, status and latency, never images or metadata values.

## Licence

- **Code:** MIT, see [LICENSE](LICENSE).
- **MILK10k data:** [ISIC Archive MILK10k](https://challenge.isic-archive.com/data/#milk10k), CC BY-NC 4.0,
  attribution "MILK study team". The images and metadata are not included in this repository.
- **Checkpoints and model package:** derived from the MILK10k data, so they may be used and shared for
  **non-commercial purposes only**, under compatible terms. They are not stored in git.
- **Pretrained initialisation:** the `timm/resnet18.a1_in1k` weights are Apache-2.0.
