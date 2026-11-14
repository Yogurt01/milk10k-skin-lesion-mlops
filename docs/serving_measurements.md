# Serving measurements (M1)

Cost, correctness and speed of serving the released `final-P1a-kg` model (15 checkpoints: 3 seeds × 5 folds),
measured before designing the API and its deployment. Date: 2026-11-09.

> Research model, not a medical device. MILK10k data and the derived checkpoints: CC BY-NC 4.0, "MILK study team".

## Environment

| Item | Value |
|---|---|
| Machine | Laptop, Intel Core i5-12500H (12 cores / 16 threads), 16 GB RAM (15.3 GiB usable), RTX 3050 Laptop GPU 4 GB, NVIDIA driver 580.178.04 |
| Load during the runs | About 10 GiB RAM already used by other processes (5.3 GiB available, 1.9 GiB swap in use): latencies are indicative, not best-case |
| Python env | `training/.venv` from `training/requirements.txt` (Python 3.12.3, torch 2.14.0+cu130, timm 1.0.30) |
| ONNX env | Separate venv: same pins + onnx 1.23.2, onnxscript 0.7.2, onnxruntime 1.31.1 |
| Data | `MILK10K_DATA_DIR` = prepared MILK10k root (`train/`, `test/`, `*_metadata.csv`, `sample_submission.csv`) |
| Code | `training/` unchanged; measurement scripts call `milk10k.inference` read-only |

Defaults: PyTorch uses 12 intra-op threads on this CPU. CPU runs use fp32; GPU runs use fp16 autocast (as the reference run).
Latency is the median of repeated runs for **one lesion** (dermoscopic + clinical image + 4 metadata fields), models already loaded.

## 1. Size and memory (P1-DEC-01)

| Quantity | CPU (PyTorch fp32) | GPU (PyTorch fp16 autocast) |
|---|---:|---:|
| Checkpoints on disk | 15 × 90.8 MiB = 1,361.5 MiB | same |
| Parameters per member | 23.75 M | same |
| Load all 15 members | 4.4 s | 4.7 s |
| Process RSS after `import torch` + `milk10k` | 807 MiB | 907 MiB (incl. CUDA context) |
| RSS with 15 members loaded | 2,287 MiB (+1,480 MiB, ≈ 99 MiB/member) | 1,015 MiB host |
| Peak RSS during inference | 2,388 MiB (1 lesion); 2,494 MiB overall | 1,655 MiB; 1,796 MiB overall |
| VRAM with 15 members loaded / peak | — | 1,387 / 1,410 MiB |

The release path (`predict_lesions`) loads, runs and frees one member at a time. A serving process should keep all
15 members loaded; its predictions are identical to the release path (max |diff| 0.0 on the 6 sample lesions).
The 807 MiB baseline comes from the CUDA build of torch; a CPU-only torch wheel was not measured.

## 2. Latency per request (P1-DEC-01)

| Path | CPU 12 threads | CPU 4 threads | CPU 2 threads | GPU |
|---|---:|---:|---:|---:|
| Preprocessing (decode, resize, tabular encoding) | 9 ms | — | — | 15 ms |
| 1 member | 56 ms | 56 ms | 75 ms | 11 ms |
| **15 members, no TTA** | **0.78 s** | **0.82 s** | **1.20 s** | **0.12 s** |
| 15 members, TTA4 | 2.82 s | 3.31 s | 4.23 s | 0.35 s |
| 15 members, no TTA, batch of 6 lesions | 3.90 s (0.65 s/lesion) | — | — | 0.28 s |
| Release path, 6 lesions incl. loading 15 checkpoints | 11.5 s (TTA 24.4 s) | — | — | 6.9 s (TTA 7.4 s) |

Going from 12 to 4 threads costs only about 5% on CPU, so 3 replicas × 4 CPU threads is a reasonable fit for a 12-core node.

## 3. CPU parity (P1-DEC-02)

Reference: `training/results/predictions/final-P1a-kg.csv` (and `-tta.csv`), produced on a Kaggle T4 in fp16. Tolerance 0.002.

| Run | Lesions | Max abs diff (13 columns) | Within 0.002 | Output checks |
|---|---:|---:|---|---|
| CPU fp32, no TTA | 3 intact sample lesions | 2.5e-4 (`target_binary`) | yes | all pass (6 lesions) |
| CPU fp32, TTA4 | 3 intact | 1.4e-4 (`BCC`) | yes | all pass |
| GPU fp16, no TTA | 3 intact | 8.6e-5 | yes | all pass |
| GPU fp16, TTA4 | 3 intact | 4.2e-5 | yes | all pass |
| GPU fp16, no TTA | all 479 test lesions | 3.1e-4 | yes | — |

The 3 other sample lesions (`clinical_blank`, `clinical_nan`, `clinical_missing_file`) deliberately differ from the
reference inputs (their clinical image is removed), so they differ by up to 0.085: they test missing-clinical
handling, not parity. The full 479-lesion parity through the API is a later task (P1-TST-08).

## 4. Smaller variants (P1-DEC-03)

### 4a. OOF score of seed subsets

OOF of a lesion = mean over the chosen seeds of its held-out fold model (the 3-seed mean reproduces the stored
`final-P1a-kg_oof.csv` to 9e-16).

| Variant | Members | OOF score | pAUC80 | Macro-AUC₁₁ | FairGap |
|---|---:|---:|---:|---:|---:|
| **Seeds 42+43+44 (released)** | 15 | **0.6505** | 0.5716 | 0.8911 | 0.1693 |
| Seeds 42+43+44 + TTA4 | 15 (×4 views) | 0.6555 | 0.5772 | 0.8969 | 0.1760 |
| Seeds 42+44 | 10 | 0.6476 | 0.5633 | 0.8874 | 0.1295 |
| Seeds 43+44 | 10 | 0.6470 | 0.5553 | 0.8867 | 0.0774 |
| Seeds 42+43 | 10 | 0.6466 | 0.5711 | 0.8889 | 0.1983 |
| Seed 42 + TTA4 | 5 (×4 views) | 0.6499 | 0.5643 | 0.8913 | 0.1251 |
| Seed 42 | 5 | 0.6446 | 0.5565 | 0.8818 | 0.0947 |
| Seed 44 | 5 | 0.6237 | 0.5240 | 0.8770 | 0.0626 |
| Seed 43 | 5 | 0.6204 | 0.5380 | 0.8775 | 0.1951 |

The TTA gain is unconfirmed (release README: +0.0050, CI [−0.0016, +0.0109]); picking the best single seed after
seeing its OOF score is optimistic.

### 4b. Deviation from the full ensemble on the 479 test lesions

Fold subsets cannot be OOF-scored (each lesion's OOF comes from a single fold model), so every subset is compared
with the full 15-member mean (all from one GPU fp16 run).

| Variant | Members | Max abs diff | Mean abs diff | Lesions within 0.002 | Top-1 agreement | Spearman P(malignant) |
|---|---:|---:|---:|---:|---:|---:|
| 2 seeds, all folds (best / worst of 3) | 10 | 0.078 / 0.087 | 0.0053 / 0.0057 | 0–2 / 479 | 95.0–95.2% | 0.997 |
| 1 seed, all folds (best / worst of 3) | 5 | 0.157 / 0.174 | 0.0105 / 0.0113 | 0 / 479 | 90.4–92.7% | 0.990 |
| 3 folds × 3 seeds (best / worst of 10) | 9 | 0.100 / 0.169 | 0.0070 / 0.0075 | 0 / 479 | 93.5–95.0% | 0.996 |
| 2 folds × 3 seeds (best / worst of 10) | 6 | 0.150 / 0.254 | 0.0104 / 0.0112 | 0 / 479 | 91.4–92.9% | 0.990 |
| 1 fold × 3 seeds (best / worst of 5) | 3 | 0.275 / 0.360 | 0.0175 / 0.0177 | 0 / 479 | 86.2–87.7% | 0.973–0.976 |
| 1 member (best / worst of 15) | 1 | 0.432 / 0.759 | 0.030–0.031 | 0 / 479 | 79.1–79.3% | 0.930–0.934 |

**No subset stays within the 0.002 parity tolerance**: choosing one would need its own reference predictions (P1-TST-06).

## 5. Export prototype (P1-DEC-04)

One member (seed 42, fold 0) wrapped with the same softmax/sigmoid as `inference.predict_views`, exported with
`torch.onnx.export(dynamo=True)` (dynamic batch 1–64) and run with ONNX Runtime's CPU provider.

| Quantity | Value |
|---|---:|
| Export time / file size | 5.0 s / 91.0 MiB |
| Max abs diff vs PyTorch fp32, 6 sample lesions (incl. missing clinical) | 7.2e-7 (classes), 8.5e-7 (binary); top-1 identical |
| 1 member, 1 lesion, 12 / 4 / 2 threads: ONNX Runtime | 26 / 39 / 68 ms |
| Same, PyTorch in the same run | 47 / 94 / 82 ms (the earlier PyTorch run gave 56 / 56 / 75 ms) |
| Full ensemble end to end (decode + preprocessing + 15 members), 1 lesion, 4 threads, torch-free process | 0.70 s (median of 7; notebook 08: 0.61–0.70 s) |
| Same with ONNX Runtime's default thread spinning | 1.58 s: idle worker threads of the previous session compete with the next; `milk10k_serving` turns spinning off |
| RSS of a process **without torch**: after `import onnxruntime` / 15 sessions / after one run each | 44 / 1,854 / 1,859 MiB |
| RSS of the `milk10k_serving` predictor (15 sessions + OpenCV), after inference | ≈ 2.0 GiB |

The torch-free preprocessing in `milk10k_serving` (OpenCV + NumPy) is bit-identical to the release transforms,
and the full ONNX ensemble matches the reference within 2.55e-4 on the 3 intact sample lesions
(`notebooks/08_export_for_deployment.ipynb`, decision P1-DEC-05).

## 6. What it means for serving (input to P1-DEC-05)

| Option | OOF score | RAM per replica | CPU latency, 1 lesion, 4 threads | Parity vs stored reference | 3 replicas |
|---|---:|---:|---:|---|---:|
| A. Full 15, PyTorch CPU | 0.6505 | ≈ 2.4 GiB | 0.82 s | yes (2.5e-4) | ≈ 7.2 GiB |
| B. Full 15, ONNX Runtime CPU (**chosen**, P1-DEC-05) | 0.6505 | ≈ 2.0 GiB | 0.70 s (measured in M4) | yes (2.55e-4, full ONNX path) | ≈ 6.0 GiB |
| C. 2 seeds (10), PyTorch CPU | 0.6466–0.6476 | ≈ 1.9 GiB (est.) | ≈ 0.55 s (est.) | no: new reference needed | ≈ 5.7 GiB |
| D. 1 seed (5), PyTorch CPU | 0.6204–0.6446 | ≈ 1.4 GiB (est.) | ≈ 0.28 s (est.) | no: new reference needed | ≈ 4.2 GiB |
| E. Full 15 + TTA4, PyTorch CPU | 0.6555 (gain unconfirmed) | ≈ 2.4 GiB | 3.31 s | yes vs `-tta.csv` (1.4e-4) | ≈ 7.2 GiB |
| F. Full 15, PyTorch GPU (one replica, Compose only) | 0.6505 | 1.4 GiB VRAM + 1.7 GiB host | 0.12 s | yes (8.6e-5) | not with a 4 GB GPU |

Estimates (est.) scale the measured per-member numbers linearly. With the RAM already in use on this laptop during
the runs (≈ 10 GiB), three full-ensemble replicas plus the monitoring stack do not fit at the same time; either other
applications are closed, the parts run separately (hardware budget, P1-CON-05), or memory is shared across replicas
(not yet measured).

## 7. Concurrency through the API (P1-API-05, 2026-11-14)

Real model package through real HTTP (`make serve`, one uvicorn worker, `ORT_INTRA_OP_THREADS=4`), measured with
`benchmarks/serving/concurrency.py`: 24 `POST /predict` per level (after 2 warm-up requests), the 3 intact sample
lesions cycled, `c` requests in flight at a time. During each level `GET /health` is polled every 50 ms and the
server's RSS (VmRSS of the uvicorn process) is sampled.

- **Machine:** Intel Core i5-12500H, 12 cores / 16 logical CPUs, 15,198 MiB RAM.
- **Load:** other processes used ~9.5–12 GiB, with 2.9–3.1 GiB available during the runs.
- **Noise:** the same effective concurrency gave different throughput across runs (c = 2, unlimited: 2.28 req/s; limit 2 with c = 4 or 5: 1.87–1.91 req/s), so the numbers are indicative.

**A. Limiter not binding** (`MAX_CONCURRENT_PREDICTIONS=8`, `MAX_QUEUED_PREDICTIONS=32`): what `c` parallel inferences cost.

| c | Status | p50 | p95 | Throughput | `/health` p50 / max | Server RSS max |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 24 × 200 | 0.57 s | 0.68 s | 1.72 req/s | 3.0 / 5.7 ms | 2,472 MiB |
| 2 | 24 × 200 | 0.86 s | 1.00 s | 2.28 req/s | 3.1 / 6.2 ms | 2,472 MiB |
| 4 | 24 × 200 | 1.32 s | 1.73 s | 2.85 req/s | 3.1 / 6.4 ms | 2,493 MiB |

**B. Defaults** (`MAX_CONCURRENT_PREDICTIONS=2`, `MAX_QUEUED_PREDICTIONS=8`): requests beyond 2 wait in the queue.

| c | Status | p50 | p95 | Throughput | `/health` p50 / max | Server RSS max |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 24 × 200 | 0.58 s | 0.62 s | 1.71 req/s | 3.1 / 5.6 ms | 2,021 MiB |
| 2 | 24 × 200 | 1.10 s | 1.14 s | 1.81 req/s | 3.2 / 8.2 ms | 2,190 MiB |
| 4 | 24 × 200 | 2.11 s | 2.24 s | 1.87 req/s | 3.2 / 6.8 ms | 2,190 MiB |
| 5 | 24 × 200 | 2.13 s | 3.17 s | 1.91 req/s | 3.2 / 7.1 ms | 2,191 MiB |

**Overload (defaults):** 15 parallel requests gave:
- 10 × 200 (2 running + 8 queued); the slowest took 5.02 s.
- 5 × 503 with `Retry-After: 1`, each within 0.03–0.05 s.
- During the overload `/ready` stayed 200 and `/health` answered in 3.2 ms.

**Before the change** (project review, fake predictor sleeping 2 s, 60 parallel requests):
- At most 40 predictions ran at once, the size of anyio's default threadpool.
- The sync `/health` waited behind them for 1,879 ms.
- With 40 blocked predictions, the regression test measured 9,624 ms when `/health` was made sync again.

**Choice of `MAX_CONCURRENT_PREDICTIONS=2`:**
- Each running prediction uses 4 intra-op threads, so 2 running use 8 of the 16 logical CPUs. That leaves CPUs for the event loop and the rest of the stack (monitoring, Jenkins), or for other replicas on the same node.
- 4 running use all 16. In run A that gained ~25% throughput over 2 (2.85 vs 2.28 req/s), at a p95 of 1.73 s instead of 1.00 s.
- The queue of 8 keeps the worst accepted wait to a few seconds: the slowest of 10 parallel requests took 5.02 s.
- Each queued request already holds its parsed upload. At most 2 images × `MAX_UPLOAD_MB` (10 MB) each, so 160 MB for 8 waiting requests: an upper bound from the limits, not measured.
- Both values are settings, so a deployment with more CPUs or one replica per node can raise them.

## 8. Decoded image size (P1-API-09, 2026-11-14)

The upload limit (`MAX_UPLOAD_MB`) bounds the compressed size only. A small PNG can declare huge dimensions.

OpenCV 5.0.0 (`opencv-python-headless` 5.0.0.93):
- **When the cap is read:** OpenCV reads `OPENCV_IO_MAX_IMAGE_PIXELS` once, when `cv2` is imported. Set after the import, it had no effect.
- **What happens above the cap:** `cv2.imdecode` *raises* `cv2.error` (`validateInputImageSize`) instead of returning `None`, before allocating the image.
- **Exact default cap:** not measured. A 400 M-pixel image decoded and a 1.2 G-pixel image was rejected **[verify]**.

`milk10k_serving.preprocess` now sets the variable before importing `cv2` (default 25,000,000 pixels). Every MILK10k
image is at most 600 × 450 = 270,000 pixels (`training/results/tables/01_image_sizes.csv`, 11,438 images).
`ImageTooLarge` maps to a 422 for either image.

| Image (black RGB PNG) | File size | Cap | Result | Decode time | Peak RSS of the process |
|---|---:|---|---|---:|---|
| 20000 × 20000 | 1.17 MB | OpenCV default | decoded (1.2 GB array) | 1.96 s | 47 → 2,338 MiB |
| 40000 × 30000 | 3.50 MB | OpenCV default | `cv2.error` raised (an uncaught error: generic 500 in the API before this change) | — | — |
| 20000 × 20000 | 1.17 MB | env set *after* `import cv2` | decoded: the setting was ignored | 1.52 s | 47 → 2,338 MiB |
| 5000 × 5000 | 73 kB | 25,000,000 | decoded | 0.141 s | 47 → 193 MiB |
| 5001 × 5000 | 73 kB | 25,000,000 | rejected (`ImageTooLarge`) | < 0.001 s | 47 → 50 MiB |
| 20000 × 20000 / 40000 × 30000 | 1.17 / 3.50 MB | 25,000,000 | rejected | ≤ 0.002 s | +3 / +5 MiB |

## How to reproduce

Run from the repository root, with `MILK10K_DATA_DIR` pointing at the prepared dataset and the 15 checkpoints in
`training/models/final-P1a-kg/` (`make bench` runs all four):

```bash
training/.venv/bin/python benchmarks/serving/measure.py cpu      # §1–3, CPU
training/.venv/bin/python benchmarks/serving/measure.py cuda     # §1–3, GPU
training/.venv/bin/python benchmarks/serving/variants.py         # §4
notebooks/.venv/bin/python benchmarks/serving/export.py          # §5 (venv: make setup-notebooks)
```

§7 needs a running server (`make serve`, settings from the environment) and the sample lesions:
`serving/.venv/bin/python benchmarks/serving/concurrency.py --pid <uvicorn pid> --levels 1 2 4 5 --label default`
(writes `benchmarks/serving/out/concurrency_<label>.json`). §8 used temporary scripts outside the repository; the
behaviour is covered by `tests/test_preprocess.py` and `tests/test_api.py`.

Each script writes a JSON file with every number in this document to `benchmarks/serving/out/` (gitignored). The
end-to-end ONNX latency in §5 comes from notebook 08 and the `milk10k_serving` predictor. A re-run on 2026-11-09
reproduced the figures above (CPU ensemble 0.784 s vs 0.783 s; parity, OOF and RSS unchanged).
