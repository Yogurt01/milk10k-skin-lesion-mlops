"""milk10k: self-contained release of the MILK10k multi-modal skin-lesion model (final-P1a-kg).

Kept import-light: `milk10k.metrics`, `milk10k.splits`, `milk10k.config` and `milk10k.evaluation` need only
numpy/pandas/scikit-learn/PyYAML; import `milk10k.dataset`, `milk10k.models`, `milk10k.train` or
`milk10k.inference` explicitly when torch is needed.

This model was developed for a research competition and is not a medical device.
"""
__version__ = "1.0.0"
MODEL_NAME = "final-P1a-kg"
