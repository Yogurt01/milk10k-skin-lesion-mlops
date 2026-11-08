# models/final-P1a-kg

The released model: **15 checkpoints**, i.e. P1a-kg seeds 42/43/44 × folds 0–4, each the best epoch of its fold. They
were trained on a Kaggle T4 in fp16, and the reference predictions in `results/predictions/` were produced from exactly
these files. Each checkpoint is 95.2 MB, **1.43 GB** in total. Each file holds `model` (fp32 state dict), `args` (the
training recipe), `fold`, `epoch` (best epoch), `metrics` (validation-fold components) and `classes`; there is no
optimizer state.

| File | Purpose | In git |
|---|---|---|
| `P1a-kg-s{seed}_fold{k}.pth` | Checkpoints | **No** (`*.pth` is ignored); use DVC or Git LFS downstream |
| `manifest.json` | sha256, size, seed, fold, best epoch, validation metrics and recipe of every checkpoint, cross-checked against the training logs and the reference inference run (`all_checks_passed: true`) | Yes |
| `tabular_encoding.json` | The fixed metadata encoding shared by every fold and seed. Nothing is fitted, so there is no per-fold preprocessor | Yes |

**Getting the checkpoints.** They travel with this folder when it is copied directly, but not through git. Put the 15
files here, or point `MILK10K_CHECKPOINT_DIR` at wherever they live (for example a DVC or LFS checkout). Notebook 07
refuses to run if a file is missing or its sha256 differs from `manifest.json`. `scripts/build_manifest.py` rebuilds
the manifest after the files are placed, and moves files out of a `_download/` sub-folder first. Do not convert the
checkpoints to fp16 or safetensors: that would change the predictions.

| File | Seed | Fold | Best epoch | Val score (fold) | sha256 (first 16) |
|---|---:|---:|---:|---:|---|
| `P1a-kg-s42_fold0.pth` | 42 | 0 | 10 | 0.6400 | `fbc112c048243519` |
| `P1a-kg-s42_fold1.pth` | 42 | 1 | 9 | 0.6529 | `f7b8ee6931a8f2ed` |
| `P1a-kg-s42_fold2.pth` | 42 | 2 | 8 | 0.6769 | `b096c44e009d9421` |
| `P1a-kg-s42_fold3.pth` | 42 | 3 | 6 | 0.6032 | `657dba7ccdab597c` |
| `P1a-kg-s42_fold4.pth` | 42 | 4 | 10 | 0.6554 | `62cc7b9ebd8247ce` |
| `P1a-kg-s43_fold0.pth` | 43 | 0 | 8 | 0.6239 | `37e67d986018712b` |
| `P1a-kg-s43_fold1.pth` | 43 | 1 | 9 | 0.6359 | `0e6c51a22993206f` |
| `P1a-kg-s43_fold2.pth` | 43 | 2 | 8 | 0.6524 | `65573be1a8e1ac94` |
| `P1a-kg-s43_fold3.pth` | 43 | 3 | 9 | 0.5624 | `51c692c9cb2c5b29` |
| `P1a-kg-s43_fold4.pth` | 43 | 4 | 8 | 0.6341 | `65d8561517515e03` |
| `P1a-kg-s44_fold0.pth` | 44 | 0 | 8 | 0.5938 | `250820d11929f0b0` |
| `P1a-kg-s44_fold1.pth` | 44 | 1 | 8 | 0.6373 | `93a2600d48064094` |
| `P1a-kg-s44_fold2.pth` | 44 | 2 | 8 | 0.6298 | `447eda53f7b760f1` |
| `P1a-kg-s44_fold3.pth` | 44 | 3 | 9 | 0.6060 | `0d6b922f6501c12c` |
| `P1a-kg-s44_fold4.pth` | 44 | 4 | 10 | 0.6527 | `09bbafb579c0bc35` |
