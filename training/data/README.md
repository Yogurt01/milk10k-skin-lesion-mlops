# Data

**MILK10k** (Multi-modal Imaging of Lesions and Keratinocyte cancers), distributed by the ISIC Archive:
<https://challenge.isic-archive.com/data/#milk10k>. Licence: **CC BY-NC 4.0**, with attribution to the **MILK study team**.
Images and metadata are **not** included in this folder or in git. Download them yourself and respect the
non-commercial and attribution terms.

## 1. Download

Put the six release files in `data/raw/_source/`:

```
MILK10k_Training_Input.zip      MILK10k_Training_Metadata.csv     MILK10k_Training_GroundTruth.csv
MILK10k_Test_Input.zip          MILK10k_Test_Metadata.csv         MILK10k_Training_Supplement.csv
```

## 2. Prepare the layout

```bash
python -m milk10k.data_prep --source data/raw/_source --out data/raw   # or notebook 01 with PREPARE_DATA = True
```

This writes `data/raw/{train,test}/{derm,clinical}/*.jpg`, `train_metadata.csv` (5,240 labelled lesions), `test_metadata.csv`
(479 lesions, **no public labels**) and `sample_submission.csv` (training-set class priors). Release-specific
decisions, all made in `milk10k.data_prep`:

- The release has no `patient_id`, so **`patient_id := lesion_id`**. Grouped CV cannot keep one patient's lesions together, and CV is optimistic.
- `fitzpatrick_type = skin_tone_class + 1` (0–5 → 1–6). **The direction of the source coding is unverified**; report groups by numeric code.
- MILK10k site names are mapped to a shared site vocabulary where equivalent; `trunk`, `hand` and `foot` are kept as-is.

To use a dataset prepared elsewhere, set `MILK10K_DATA_DIR=/path/to/prepared` instead of copying it.

## 3. `folds.csv` (tracked, fixed)

Columns `isic_id, patient_id, fold`: 5,240 rows, 5 folds, `StratifiedGroupKFold` stratified by diagnosis and grouped by
`patient_id` (= `lesion_id`). This file was generated once and used by **every** reported run, ensemble and
analysis. Never regenerate it: a new split invalidates every OOF comparison. `milk10k.splits` is kept only as
documentation, and its CLI writes to a different file.
