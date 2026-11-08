# sample/

Six official **test** lesions for a quick end-to-end run of `notebooks/07_inference.ipynb`. They were chosen as the first six `isic_id` values in sorted order, without looking at any image or prediction.

**Images and metadata are not committed** (the repository's data rule), so this folder is not usable until you rebuild it:

1. Download and prepare MILK10k as described in `data/README.md`, or point `MILK10K_DATA_DIR` at an existing prepared copy.
2. Run `make_sample.py`. Notebook 07 also runs it automatically when `sample_metadata.csv` is missing.

```bash
python sample/make_sample.py                      # uses configs/final.yaml data_dir or MILK10K_DATA_DIR
python sample/make_sample.py --data-dir /path/to/prepared/milk10k
```

| File | Tracked | Content |
|---|---|---|
| `sample_ids.csv` | yes | `isic_id`, `scenario` |
| `make_sample.py` | yes | Copies the images to `sample/images/` and writes `sample_metadata.csv` |
| `images/`, `sample_metadata.csv` | no (ignored) | Rebuilt by `make_sample.py` |

| Scenario | Rows | What it tests |
|---|---:|---|
| `intact` | 3 | Normal input; also compared with the stored Kaggle predictions (parity) |
| `clinical_blank` | 1 | `clinical_image_path = ""`: the official missing-modality encoding |
| `clinical_nan` | 1 | `clinical_image_path` is NaN |
| `clinical_missing_file` | 1 | The path points to a file that does not exist |

Notebook 07 also builds unreadable-derm, out-of-range Fitzpatrick, unseen-site and duplicate-ID rows in memory. Data licence: CC BY-NC 4.0, attribution "MILK study team" (`data/README.md`).
