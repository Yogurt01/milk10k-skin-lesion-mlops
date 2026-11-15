# Full parity through the API (P1-TST-08)

> Research model, not a medical device. MILK10k data and the derived model package: CC BY-NC 4.0, "MILK study team".

All 479 reference test lesions were sent through real HTTP `POST /predict` and compared with the stored reference
predictions `training/results/predictions/final-P1a-kg.csv` (Kaggle T4, fp16). Date: 2026-11-14.

**Result: 479 / 479 lesions within the tolerance of 0.002**. The largest difference was 1.096e-3 (column `DF`, lesion
ISIC_6071998), about half the tolerance.

## Command

```bash
MILK10K_DATA_DIR=/path/to/prepared/milk10k make test-all
# or only this test:
MILK10K_DATA_DIR=/path/to/prepared/milk10k serving/.venv/bin/python -m pytest -s \
  tests/test_http_parity.py::test_all_479_reference_lesions_through_http
```

The test (`tests/test_http_parity.py`, markers `parity` and `slow`) does the following:
- Starts uvicorn in a thread on the already-loaded model package, with default settings.
- Sends each lesion as the parity tests do: the dermoscopic image, the clinical image if its file exists, and the four metadata fields as stored, with empty ones omitted. Requests go out on 2 client threads, which is the `MAX_CONCURRENT_PREDICTIONS` default.
- Compares the 12 columns: `target_binary` = `p_malignant`, plus the 11 class probabilities.
- Writes `reports/parity_full.json`, which is gitignored.

Without `MILK10K_DATA_DIR` the test is skipped.

## Environment

| Item | Value |
|---|---|
| Machine | Intel Core i5-12500H, 16 logical CPUs, 15,198 MiB RAM; other processes used most of it |
| Python | 3.12.3 (`serving/.venv`) |
| Runtime | onnxruntime 1.31.1 (CPU, 4 intra-op threads per session), opencv-python-headless 5.0.0.93, numpy 2.5.3, fastapi 0.162.7, uvicorn 0.54.0 |
| Model package | `final-P1a-kg-onnx` 1.0.0 (15 ONNX members, sha256-verified at load) |
| Server | uvicorn, real HTTP on 127.0.0.1, `MAX_CONCURRENT_PREDICTIONS=2`, `MAX_QUEUED_PREDICTIONS=8` |

## Results

| Item | Value |
|---|---|
| Lesions sent / answered 200 | 479 / 479 |
| Within 0.002 (all 12 columns) | 479 / 479 |
| Max \|diff\| | 1.096e-3 (`DF`, ISIC_6071998) |
| Clinical image used | 479 / 479 |
| Lesions with `metadata_missing` | 6, all `["anatomical_site_general"]`: ISIC_1075872, ISIC_8231451, ISIC_5581186, ISIC_9599471, ISIC_8440783, ISIC_7277175 |
| Total time | 258.7 s and 255.2 s (two runs, identical differences) |

Max |diff| per column:

| Column | Max \|diff\| |
|---|---:|
| `target_binary` | 8.77e-4 |
| AKIEC | 6.53e-4 |
| BCC | 6.44e-4 |
| BEN_OTH | 3.13e-4 |
| BKL | 5.21e-4 |
| DF | 1.10e-3 |
| INF | 7.79e-4 |
| MAL_OTH | 3.42e-4 |
| MEL | 3.07e-4 |
| NV | 8.75e-4 |
| SCCKA | 5.00e-4 |
| VASC | 3.30e-4 |

The reference was produced on a GPU in fp16 and the API runs on a CPU in fp32. The PyTorch GPU recompute of all 479
lesions matched the reference within 3.1e-4 ([serving_measurements.md](serving_measurements.md) §3). This CPU ONNX run
through the API stays within 1.1e-3.

## Through the container (P1-TST-05, 2026-11-15)

The same tests against the API container of the Compose stack (`make up`, image `serving/Dockerfile`, model package
mounted read-only, default settings, Python 3.12.16 in the image). With `API_BASE_URL` set, `tests/test_http_parity.py`
sends its requests to that URL and starts no server and loads no model in the test process.

```bash
make up
make test-container          # sample lesions; API_BASE_URL defaults to http://127.0.0.1:8000
API_BASE_URL=http://127.0.0.1:8000 MILK10K_DATA_DIR=/path/to/prepared/milk10k serving/.venv/bin/python -m pytest -s \
  tests/test_http_parity.py::test_all_479_reference_lesions_through_http   # one-off, not in make test
```

| Item | Value |
|---|---|
| Sample lesions (3 intact) | max \|diff\| 2.55e-4; the 3 without a clinical image valid (`make test-container`: 1 passed, 3.7 s) |
| Lesions sent / answered 200 | 479 / 479 |
| Within 0.002 (all 12 columns) | 479 / 479 |
| Max \|diff\| | 1.096e-3 (`DF`, ISIC_6071998) |
| Max \|diff\| per column | the same as the table above (all 12, to 3 significant digits) |
| Clinical image used / `metadata_missing` | 479 / 479; the same 6 lesions |
| Total time | 243.1 s |

The container gives the same differences as the host run above.
