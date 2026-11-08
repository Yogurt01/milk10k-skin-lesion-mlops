# results/

Artifacts of the training runs, plus everything the notebooks compute from them. `provenance.json` records the source path and
sha256 of each collected file, for integrity checks.

**From the training runs** (gathered by `scripts/export_results.py`; there is no need to run it again):

| Path | Content |
|---|---|
| `oof/final-P1a-kg_oof.csv`, `oof/final-P1a-kg-tta_oof.csv` | Pooled out-of-fold predictions of the released ensemble, without and with TTA (5,240 rows) |
| `oof/P1a-kg-s{42,43,44}{,-noclin,-tta}_oof.csv` | Single-seed OOF predictions: intact, clinical image removed, and four-view TTA |
| `oof/P1a-kg-noiseoff-s{42,43}_oof.csv` | The GaussNoise-off experiment (a negative result; `docs/experiment_log.md`) |
| `predictions/final-P1a-kg.csv`, `predictions/final-P1a-kg-tta.csv` | Reference test-set predictions (479 rows, no labels). **Committed on purpose**, as an exception to the rule against committing prediction files: they are the reference for the parity check in notebook 07 and for regression tests (README, "Deliberate exceptions") |
| `training_history.csv` | Per-epoch loss, score and learning rate for 3 seeds × 5 folds × 10 epochs, parsed from the training logs |
| `experiments_scores.csv` | Every run of both phases rescored from its OOF file, with the file's sha256 |
| `kaggle_runtime.json` | Software environments and timings of the training runs and of the reference inference run |
| `provenance.json` | Source path and sha256 of every file above |

**Computed by the notebooks:** `figures/`, `tables/`, `metrics/final_metrics.json`, `metrics/notebook_runtime.json`,
`sample_predictions*.csv` and `metrics/sample_inference_report.json`. The notebook 04 smoke test writes to
`demo_training/`, which is git-ignored.
