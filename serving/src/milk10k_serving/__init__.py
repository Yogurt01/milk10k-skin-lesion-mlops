"""milk10k_serving: torch-free inference for the MILK10k final-P1a-kg model (ONNX Runtime, CPU).

Modules: `settings` (environment variables), `preprocess` (image decoding, transforms, tabular encoding equal to
the release's evaluation path), `package` (model package loading, sha256 checks, ensemble), `predictor`
(bytes + metadata -> prediction).
"""

__version__ = "0.1.0"

DISCLAIMER = (
    "Research model for the ISIC MILK10k benchmark, not a medical device. "
    "Its output is not a diagnosis and must not be used for clinical decisions."
)
