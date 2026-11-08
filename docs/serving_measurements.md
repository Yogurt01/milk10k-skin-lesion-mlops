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
| Estimated 15 members, 4 threads (15 × 1 member) | ≈ 0.6 s (not measured end to end) |
| RSS of a process **without torch**: after `import onnxruntime` / 15 sessions / after one run each | 44 / 1,854 / 1,859 MiB |

A torch-free serving image would need the preprocessing (albumentations + OpenCV transforms and the tabular
encoding in `milk10k.dataset`) without torch tensors, and a full-ensemble parity check against the reference.

## 6. What it means for serving (input to P1-DEC-05)

| Option | OOF score | RAM per replica | CPU latency, 1 lesion, 4 threads | Parity vs stored reference | 3 replicas |
|---|---:|---:|---:|---|---:|
| A. Full 15, PyTorch CPU | 0.6505 | ≈ 2.4 GiB | 0.82 s | yes (2.5e-4) | ≈ 7.2 GiB |
| B. Full 15, ONNX Runtime CPU | 0.6505 | ≈ 1.9 GiB | ≈ 0.6 s (est.) | expected (member diff 8.5e-7); to verify on 15 | ≈ 5.6 GiB |
| C. 2 seeds (10), PyTorch CPU | 0.6466–0.6476 | ≈ 1.9 GiB (est.) | ≈ 0.55 s (est.) | no: new reference needed | ≈ 5.7 GiB |
| D. 1 seed (5), PyTorch CPU | 0.6204–0.6446 | ≈ 1.4 GiB (est.) | ≈ 0.28 s (est.) | no: new reference needed | ≈ 4.2 GiB |
| E. Full 15 + TTA4, PyTorch CPU | 0.6555 (gain unconfirmed) | ≈ 2.4 GiB | 3.31 s | yes vs `-tta.csv` (1.4e-4) | ≈ 7.2 GiB |
| F. Full 15, PyTorch GPU (one replica, Compose only) | 0.6505 | 1.4 GiB VRAM + 1.7 GiB host | 0.12 s | yes (8.6e-5) | not with a 4 GB GPU |

Estimates (est.) scale the measured per-member numbers linearly. With the RAM already in use on this laptop during
the runs (≈ 10 GiB), three full-ensemble replicas plus the monitoring stack do not fit at the same time; either other
applications are closed, the parts run separately (hardware budget, P1-CON-05), or memory is shared across replicas
(not yet measured).

## How to reproduce

Run from the repository root, with `MILK10K_DATA_DIR` pointing at the prepared dataset and the 15 checkpoints in
`training/models/final-P1a-kg/`:

```bash
training/.venv/bin/python <scripts>/measure.py cpu      # §1–3, CPU
training/.venv/bin/python <scripts>/measure.py cuda     # §1–3, GPU
training/.venv/bin/python <scripts>/variants.py         # §4
PYTHONPATH=training/src <export-venv>/bin/python <scripts>/export.py   # §5
```

The scripts are not yet in the repository (P1-SET-09 proposes `benchmarks/serving/`); each writes a JSON file with
every number in this document.
