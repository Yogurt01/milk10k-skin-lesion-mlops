# Model card: `final-P1a-kg`

**This model was developed for a research competition and is not a medical device.** It has no clinical validation
and must not be used to diagnose or manage patients.

Numbers below come from the executed notebooks (named next to them), from `results/`, or from `docs/experiment_log.md`.

## Model details

| | |
|---|---|
| Task | Per lesion: P(malignant), plus an 11-class diagnosis distribution over AKIEC, BCC, BEN_OTH, BKL, DF, INF, MAL_OTH, MEL, NV, SCCKA and VASC. Malignant = MEL, BCC, SCCKA, MAL_OTH |
| Inputs | One dermoscopic image and one clinical close-up, which may be missing. Metadata: age, sex, anatomical site, skin-tone code |
| Architecture | `MILKFusionNet`: two `resnet18.a1_in1k` encoders (dermoscopy and clinical, not shared) and a tabular MLP. Four tokens (CLS, derm, clin, tab) go through 2 Transformer encoder layers of width 256. Heads: 11-class logits and a binary logit. A missing clinical image is replaced by a learned `[NULL]` token. **23,749,008 trainable parameters** per member, of which 11,176,512 per ResNet18 and 1,054,208 in the fusion Transformer (`notebooks/04_modeling_and_training.ipynb`). 95.2 MB per fp32 checkpoint |
| Ensemble | 15 members (seeds 42/43/44 × folds 0–4); equal mean of probabilities, then class renormalisation |
| Optional | Four-view flip TTA: dermoscopy flipped horizontally and vertically, the clinical photo horizontally only. Its gain is **unconfirmed** |

## Intended use

Research and education: reproducing the experiments in `docs/experiment_log.md`, and MLOps practice. **Not** for
clinical triage, diagnosis, screening, or any decision about a real patient.

## Training data

The MILK10k training release: 5,240 labelled lesions, each with both images. Source: ISIC Archive,
<https://challenge.isic-archive.com/data/#milk10k>, licence **CC BY-NC 4.0**, attribution **"MILK study team"**
(`data/README.md`).
- **Class balance:** the malignant rate is 0.659 and BCC is 48% of lesions; MAL_OTH has 9 lesions (`notebooks/01_data_overview.ipynb`).
- **Folds:** `data/folds.csv`, a 5-fold split stratified by diagnosis and grouped by `patient_id`. The release has no patient identifier, so `patient_id := lesion_id`.
- **External data:** none.

## Preprocessing

- **Dermoscopy: square resize to 288×288**, without a crop. All images are 600×450, so every dermoscopic image is
  stretched by exactly **1.333×** (`notebooks/03_preprocessing_and_features.ipynb`).
- **Clinical: letterbox.** The longest side becomes 288 and the rest is zero-padded, which pads 25% of the canvas on every image.
- ImageNet normalisation, and a separate training augmentation per stream (`milk10k.dataset`). It includes `GaussNoise`
  inside a `OneOf`. The noise visibly corrupts images, yet removing it did not help (`docs/experiment_log.md`, §6).
- **Metadata:** a fixed encoding, identical for every fold and seed, with nothing fitted
  (`models/final-P1a-kg/tabular_encoding.json`). Fields used: site, sex, skin-tone code, and age (age/100 plus a
  missing flag). Identifiers and the training-only `diagnosis_confirm_type` are excluded.
- **Missing clinical image:** blank, NaN, missing-file and unreadable paths are all treated as missing. Training drops
  the clinical image with probability 0.15.

## Training procedure

Each fold trains for 10 epochs at batch 32, with:
- AdamW, backbone LR 3e-4 and heads 3e-3, weight decay 0.05;
- 5% linear warmup, then cosine decay;
- 11-class focal loss (γ = 2) plus binary BCE;
- sqrt-balanced sampling, and fp16 autocast with gradient scaling.

The best epoch of each fold is selected on its validation fold. The recipe is in `configs/final.yaml` and is identical to every checkpoint's stored `args`, as verified by the manifest.

| | |
|---|---|
| Hardware | Kaggle, one Tesla T4 |
| Software | Python 3.12.13, torch 2.10.0+cu128, CUDA 12.8, cuDNN 91002, timm 1.0.30, albumentations 2.0.8, albucore 0.0.24, numpy 2.0.2, pandas 2.3.3, scikit-learn 1.6.1 (`results/kaggle_runtime.json`) |
| Training time | 39.6 / 40.3 / 40.5 min per seed (5 folds × 10 epochs) |
| Reference predictions | `results/predictions/`: produced with Python 3.13.15 and torch 2.11.0+cu128 on a T4, from exactly these checkpoints |
| Checkpoints | 15 × 95.2 MB. sha256 in `models/final-P1a-kg/manifest.json`, cross-checked against the training logs and the reference inference run |

## Pretrained initialisation

| | |
|---|---|
| Weights | [`timm/resnet18.a1_in1k`](https://huggingface.co/timm/resnet18.a1_in1k), used for both image streams |
| Revision | `491b427b45c94c7fb0e78b5474cc919aff584bbf`, committed **2025-01-21**. The repository was created 2023-04-05; public and not gated |
| File | `model.safetensors`, sha256 `80c49dee3da4822c009c5a7fe591e9223c5a2cfcf95a4067ca4dfb5a7b89c612`, 46,807,446 bytes; identical to the cached copy used for training |
| Licence | Apache-2.0 |
| Training data | ImageNet-1k, using the "ResNet strikes back" A1 recipe (arXiv:2110.00476). It has no overlap with MILK10k |
| Code | [`timm`](https://github.com/huggingface/pytorch-image-models) 1.0.30 |

Inference does not need these weights: the checkpoints contain every parameter.

## Evaluation

Pooled out-of-fold (OOF) score on the 5,240 training lesions. Each lesion is scored by the mean of its three
held-out seed models. The metric is `0.7·pAUC80 + 0.3·MacroAUC₁₁ − 0.1·FairGap` (README, "The metric"). There is
no test score, because the official test labels are not public.

| Variant | Score | pAUC80 | Macro-AUC₁₁ | FairGap | pAUC80 codes 1–2 / 3–4 / 5–6 | Fold mean ± SD |
|---|---:|---:|---:|---:|---|---:|
| `final-P1a-kg` (primary) | **0.6505** | 0.5716 | 0.8911 | 0.1693 | 0.7228 / 0.5535 / 0.5899 | 0.6576 ± 0.0250 |
| `final-P1a-kg-tta` (optional) | 0.6555 | 0.5772 | 0.8969 | 0.1760 | 0.7347 / 0.5587 / 0.5971 | 0.6627 ± 0.0256 |

Source: `notebooks/05_evaluation.ipynb`. The primary score is asserted equal to 0.6505137677887615, the value of the
released OOF predictions when they were produced.

- **Seed ensembling vs the single-seed mean:** +0.020968, CI [+0.0123, +0.0380]. This is just below the decision gate for three seeds (0.0215; `docs/experiment_log.md`, §4).
- **TTA:** +0.004959, CI [−0.0016, +0.0109]. It cannot be distinguished from zero.
- **Without the clinical image** (checkpoint-level evaluation, 3-seed mean): 0.6182 vs 0.6295, and pAUC80 falls from 0.5395 to 0.5251. **Clinical-only and tabular-only models were never trained.**
- **Per-class one-vs-rest AUC:** VASC 0.980, NV 0.950, INF 0.948, DF 0.944, BCC 0.922, SCCKA 0.916, MEL 0.914, BEN_OTH 0.892, AKIEC 0.881, BKL 0.759, MAL_OTH 0.696 (9 positives).
- **Calibration** of the binary head: Brier score 0.1442. The metric is rank-based, so this is diagnostic only.
- **Operating point at ≥80% sensitivity** (cut 0.5259): sensitivity 0.800, specificity 0.779. Most benign lesions above the cut are BKL (202), AKIEC (111) and NV (54).

## Limitations

- **Optimistic CV.** There are no patient IDs, so one patient's lesions can fall in different folds. Best epochs were selected on the validation folds, and many runs were compared on the same OOF predictions.
- **Unverified skin-tone coding.** `fitzpatrick_type = skin_tone_class + 1`, and we could not verify the direction of the source scale. If it runs from dark to light, the group called "light" in the code (codes 1–2, 111 lesions) would be the darkest. Every fairness statement depends on this, so groups are reported by numeric code.
- **FairGap is dominated by noise.**
  - It is driven by the 111-lesion code 1–2 group (38 malignant, 73 benign).
  - Across three seeds its standard deviation is 0.0691, against 0.0163 for pAUC80.
  - The released ensemble's FairGap (0.169) comes from that group scoring *higher* than the others; it is not evidence for or against fairness.
- **TTA's gain is unconfirmed:** +0.005, with a CI that includes 0.
- **Weakest component:** pAUC80. The persistent misses are melanomas ranked below BKL, NV and AKIEC (`notebooks/02_eda.ipynb`).
- **Rare classes:** MAL_OTH has 9 lesions, so its estimates are unstable.
- **No external validation:** no independent cohort, no hidden-test score, and no prospective or clinical evaluation. The missing-clinical result is a checkpoint-level evaluation, not a retrained model.
- **Reproduction:** predictions reproduce within 8.6e-05 on another GPU and torch build (`notebooks/07_inference.ipynb`), not bit-for-bit.

## Ethical considerations

Dermatology datasets under-represent some skin tones. The fairness term here is noisy and its group labels are
uncertain, so no fairness claim is made. The images are de-identified ISIC data, and any attempt at re-identification is
out of bounds. Respect CC BY-NC 4.0: no commercial use, and attribute "MILK study team".
