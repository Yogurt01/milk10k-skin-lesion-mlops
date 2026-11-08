# MILK10k multi-modal skin-lesion model (`final-P1a-kg`)

**This model was developed for a research competition and is not a medical device.**

For each skin lesion in the [ISIC MILK10k](https://challenge.isic-archive.com/data/#milk10k) benchmark, the model
combines a dermoscopic image, a clinical close-up and four metadata fields. It predicts P(malignant) and an
11-class diagnosis. This folder is self-contained:
- the `milk10k` package and its configuration;
- the fixed folds;
- the 15 released checkpoints, with a hash manifest;
- the predictions and training histories of the runs;
- seven executed notebooks, a model card and an experiment log.

## Results at a glance

| | Pooled OOF score | pAUC80 | Macro-AUC₁₁ | FairGap |
|---|---:|---:|---:|---:|
| **`final-P1a-kg`** (released, primary) | **0.6505** | 0.5716 | 0.8911 | 0.1693 |
| `final-P1a-kg` with four-view TTA (optional) | 0.6555 | 0.5772 | 0.8969 | 0.1760 |
| GaussNoise-off experiment (2 seeds, vs 0.6295 for the same recipe with noise) | 0.6301 | 0.5365 | 0.8702 | 0.0655 |

These are cross-validated out-of-fold numbers (`notebooks/05_evaluation.ipynb`, `notebooks/06_experiments_summary.ipynb`).
The official MILK10k test labels are not public, so there is no test score.

## The metric

`Score = 0.7 · pAUC80 + 0.3 · MacroAUC₁₁ − 0.1 · FairGap`, implemented in `src/milk10k/metrics.py`.

- **pAUC80** is the area under the ROC curve restricted to true-positive rate ≥ 0.8, the region where at least 80% of
  malignant lesions are caught. It is divided by its maximum (0.2), so random ranking gives about 0.10 and perfect
  ranking gives 1.
- **MacroAUC₁₁** is the mean one-vs-rest ROC-AUC over the 11 diagnoses, with every class weighted equally.
- **FairGap** is the largest minus the smallest pAUC80 across skin-tone groups (`fitzpatrick_type` codes 1–2, 3–4,
  5–6). A group counts only with at least 10 malignant and 10 benign lesions.

Malignant means MEL, BCC, SCCKA or MAL_OTH; AKIEC counts as benign. All three parts are rank-based, so recalibrating
the scores never changes the result.

## How the model was built

**Phase 1: building the model.** We started from two ResNet18 encoders, a metadata MLP and a small token-fusion
Transformer, and changed one factor at a time.
- 288 px beat 224 px.
- Worse, and dropped: 384 px, melanoma loss weighting, blending the two heads, a ConvNeXt-T backbone (under our 10-epoch
  recipe) and spatial cross-attention between the image streams.
- Three seeds of the best recipe on a Kaggle T4 scored 0.6446, 0.6204 and 0.6237. That spread motivated a fixed decision
  gate: a two-seed candidate must beat the reference mean by +0.024.
- The released model averages those three seeds (0.6505). Most of the gain over the baseline comes from that averaging.

**Phase 2: diagnosis and one controlled test.**
- The training curves plateau late rather than showing undertraining.
- 427 of the 690 malignant lesions below the 80%-sensitivity cut are missed by every seed. They are mostly melanomas,
  ranked below BKL, NV and AKIEC.
- Preprocessing does not explain these misses, but `GaussNoise` visibly corrupts training images.
- Turning the noise off, with everything else identical, was a **negative result**: Δ +0.0005, CI [−0.016, +0.015].
  pAUC80 and macro-AUC both fell, and the only gain came from FairGap.

The full record, with tables, the gate and future work, is in **`docs/experiment_log.md`**.

## What is in this folder

| Path | Content |
|---|---|
| `src/milk10k/` | `metrics`, `dataset`, `models`, `train`, `inference` (ensemble, input sanitising, output checks), `evaluation`, `data_prep`, `splits`, `config` |
| `configs/final.yaml` | The released recipe, ensemble and inference settings |
| `data/` | `README.md` (download and licence) and `folds.csv` (the fixed 5-fold split) |
| `docs/experiment_log.md` | The research log: every experiment, the decision gate, the Phase 2 diagnosis, future work |
| `models/final-P1a-kg/` | 15 checkpoints (not in git), `manifest.json` (sha256 and cross-checks), `tabular_encoding.json` |
| `notebooks/` | 01–07, below |
| `results/` | OOF and test predictions, training histories and the experiment table from the runs, plus the notebooks' figures, tables and metrics |
| `sample/` | Six test-lesion IDs and `make_sample.py`, which copies their images from your local dataset |
| `scripts/` | `run_notebooks.py`, `check_static.py`, `build_manifest.py`, `export_results.py` |

## Quick start

```bash
# with uv (fast; reuses its wheel cache)
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r requirements.txt && uv pip install --python .venv/bin/python -e .
# or with plain pip (same pins; about 5.9 GB of torch/CUDA wheels on a cold cache)
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
. .venv/bin/activate

# data: follow data/README.md, or point at an existing prepared copy
export MILK10K_DATA_DIR=/path/to/prepared/milk10k
python sample/make_sample.py                              # six sample lesions for notebook 07
python scripts/run_notebooks.py 07                       # executes in place; or open the notebooks in Jupyter
```

Environment variables:
- `MILK10K_DATA_DIR`: the prepared dataset.
- `MILK10K_CHECKPOINT_DIR`: the folder holding the 15 `.pth` files, for example a DVC or LFS checkout elsewhere.
- `MILK10K_RELEASE_ROOT`: only needed if the package is installed outside this folder.

Python API:

```python
import pandas as pd
from milk10k.config import load_config
from milk10k.inference import PROBS, predict_lesions

cfg = load_config()                                       # configs/final.yaml
meta = pd.read_csv("sample/sample_metadata.csv")          # same columns as test_metadata.csv
prior = pd.read_csv(cfg["paths"]["data_dir"] / "sample_submission.csv").iloc[0][PROBS]
pred, report = predict_lesions(meta, "sample", cfg, prior, tta=False)   # 13 columns, checked
```

## Notebooks

| Notebook | Question |
|---|---|
| `01_data_overview` | What is in the data, and how are the fixed folds built? |
| `02_eda` | Which metadata carry signal, what do the images look like, and which malignant lesions does the model still rank too low? |
| `03_preprocessing_and_features` | How does a lesion become a model input: transforms, the effect of GaussNoise, missing clinical images, malformed rows, the metadata feature set? |
| `04_modeling_and_training` | The architecture and recipe, a demo run of the real training code, and the curves of the released checkpoints |
| `05_evaluation` | How good is the ensemble, where is it weak, and what do ensembling, TTA and the clinical image contribute? |
| `06_experiments_summary` | Every run of both phases, the decision gate, the negative Phase 2 result, future work |
| `07_inference` | Running the released model on new lesions, including missing clinical images, with parity checks against the reference predictions |

Two cells stay off by default:
- `PREPARE_DATA` in notebook 01 rebuilds `data/raw/`.
- `RUN_FULL_TRAINING` in notebook 04 takes about 40 minutes per seed on a T4 and may not fit a 4 GB GPU.

## Environments and parity

| Role | Environment |
|---|---|
| Training the checkpoints | Kaggle T4, Python 3.12.13, **torch 2.10.0+cu128**, fp16 |
| Producing the reference predictions (`results/predictions/`) | Kaggle T4, Python 3.13.15, **torch 2.11.0+cu128**, fp16 |
| `requirements.txt` | **torch 2.14.0** (CUDA 13.0 wheels), Python 3.12: the environment in which the notebooks were executed |

`requirements.txt` is pinned to the environment the notebooks were actually run and checked in, and the training
environment is recorded in `MODEL_CARD.md`. A different GPU and torch build change fp16 arithmetic slightly. Notebook 07 therefore
compares its predictions with the reference predictions using `inference.parity_tolerance = 0.002` rather than demanding
identical bits. The scores in this README come from the reference predictions, not from a local re-run.

## Limitations

See `MODEL_CARD.md` for details.

- **Optimistic CV:** there are no patient IDs (`patient_id := lesion_id`), best epochs were chosen on the validation folds, and many runs were compared on the same OOF predictions.
- **Unverified skin-tone coding:** we could not verify whether `skin_tone_class` runs from light to dark or the reverse. Every fairness reading depends on it, so groups are reported by numeric code.
- **FairGap is dominated by noise:** it is driven by the 111-lesion code 1–2 group (seed SD 0.0691 vs 0.0163 for pAUC80).
- **The TTA gain is unconfirmed:** +0.0050, CI [−0.0016, +0.0109].
- **Missing evaluations:** there is no independent cohort, and clinical-only and tabular-only models were never trained.

## Future work (Phase 3)

Under the decision gate, a two-seed candidate must beat the reference by about +0.024. The next experiments should
therefore target large effects. A realistic expectation is 0.66–0.68; 0.7 would be a stretch.

1. **Representation:** an EfficientNet-B3 dermoscopy encoder, and a lower backbone learning rate. Evidence: the hard
   positives are the same for every seed (427/690) and are mostly melanomas ranked below BKL, NV and AKIEC
   (`notebooks/02_eda.ipynb`).
2. **A ranking loss for the pAUC tail:** pAUC80 is the weakest and heaviest component (0.5716 vs macro-AUC 0.8911).
3. **Groundwork:** verify the skin-tone coding, train clinical-only and tabular-only models, and bootstrap per-group pAUC80.
4. **Deprioritised:** longer training (the curves plateau), different crops (no framing problem found), and
   self-supervised initialisation (we could not establish that suitable public weights were eligible).

Details: `docs/experiment_log.md`, §7.

## Run record

The notebooks were executed on an RTX 3050 Laptop GPU, in this folder's own `.venv` built from `requirements.txt` with uv (472 s).

| Step | Wall time | Result |
|---|---:|---|
| `scripts/build_manifest.py` | 6 s | 15 checkpoints; every check passed: identity, recipe = config, training logs, reference inference run |
| `scripts/export_results.py` | 7 s | 15 files collected, 150 history rows, 37 experiment rows |
| `sample/make_sample.py` | 1 s | 6 lesions, 12 images |
| Notebooks 01 / 02 / 03 / 04 | 5 / 5 / 6 / 13 s | Notebook 04 uses the pretrained weights from the local cache, with no network access |
| Notebooks 05 / 06 / 07 | 172 / 146 / 38 s | 05 and 06 are dominated by 1,000-resample bootstraps |

Checks:
- **Notebook 05:** the pooled OOF score equals **0.6505137677887615** (assertion `< 1e-12`).
- **Notebook 02:** all eight hard-positive counts match the experiment log: 690 hard, 427 hard for all 3 seeds, 621 for ≥2, a union of 1,000, and 357 / 216 / 427 hard for exactly 1 / 2 / 3 seeds. Benign inversions match too: BKL 222,424, NV 140,570, AKIEC 125,323.
- **Notebook 07:** the primary model differs from the reference predictions by at most **8.6e-05**, and the TTA variant by at most **4.2e-05**. Both are within 0.002, and every output check passes.
- **Hygiene:** no absolute local path appears in any output; library warnings that print site-packages paths are rewritten to `<release>`. Lesion images appear only in notebooks 02 and 03 (see below). Notebook sizes: 01 106 KiB, 02 163 KiB, 03 95 KiB, 04 442 KiB, 05 164 KiB, 06 143 KiB, 07 35 KiB.
- **Independence:** this folder was copied elsewhere without `.venv`, checkpoints, data or sample images, i.e. what a fresh clone contains. A new venv was built from `requirements.txt`. With `MILK10K_DATA_DIR` and `MILK10K_CHECKPOINT_DIR` pointing outside the copy, `import milk10k` and notebook 07 ran and passed every check.

## Deliberate exceptions

The repository otherwise never commits prediction files or lesion images. Two exceptions are intentional:

1. **Reference predictions.** `results/predictions/final-P1a-kg.csv` and `final-P1a-kg-tta.csv` (479 test lesions, no
   labels) are committed. They are the reference for the parity check in notebook 07 and for regression tests downstream.
2. **Lesion images in notebooks 02 and 03.** `SHOW_IMAGES = True` displays a few MILK10k lesions as small embedded JPEGs,
   each with the credit "Images: MILK10k (ISIC Archive), CC BY-NC 4.0, MILK study team". They are not saved to `results/`,
   and `scripts/check_static.py` rejects lesion images in any other notebook.

## Licence and attribution

Data: ISIC Archive MILK10k, **CC BY-NC 4.0**, attribution **"MILK study team"**; no commercial use. The pretrained
initialisation `timm/resnet18.a1_in1k` is Apache-2.0 (`MODEL_CARD.md`). The checkpoints are derived from the CC BY-NC data,
so share them only under compatible, non-commercial terms.
