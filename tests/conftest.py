"""Shared fixtures.

Tests that need local, non-git files (the model package, the sample lesions) are skipped when the files are
absent, unless REQUIRE_MODEL_PACKAGE=1 (set in CI), in which case they fail instead.
"""

import csv
import os
from pathlib import Path

import pytest

# Import milk10k_serving's preprocessing before any test module imports cv2, so that the decoded-image cap
# (OPENCV_IO_MAX_IMAGE_PIXELS, read by OpenCV at import) is the project default in the test process too.
import milk10k_serving.preprocess  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training"
SAMPLE_DIR = TRAINING / "sample"
REFERENCE = TRAINING / "results" / "predictions" / "final-P1a-kg.csv"
TABULAR_ENCODING = TRAINING / "models" / "final-P1a-kg" / "tabular_encoding.json"
PARITY_TOLERANCE = 0.002  # training/configs/final.yaml inference.parity_tolerance


def _missing(what: str):
    if os.environ.get("REQUIRE_MODEL_PACKAGE") == "1":
        pytest.fail(f"{what} is required (REQUIRE_MODEL_PACKAGE=1) but missing")
    pytest.skip(f"{what} not available locally")


@pytest.fixture(scope="session")
def model_dir() -> Path:
    path = Path(os.environ.get("MODEL_DIR") or ROOT / "artifacts" / "model_package" / "final-P1a-kg-onnx")
    path = path if path.is_absolute() else ROOT / path
    if not (path / "manifest.json").is_file():
        _missing(f"model package ({path.relative_to(ROOT) if path.is_relative_to(ROOT) else path})")
    return path


@pytest.fixture(scope="session")
def loaded_package(model_dir):
    from milk10k_serving.package import load_package

    return load_package(model_dir, verify_sha256=True, threads=4)


@pytest.fixture(scope="session")
def sample_lesions() -> list[dict]:
    meta = SAMPLE_DIR / "sample_metadata.csv"
    if not meta.is_file():
        _missing("sample lesions (run training/sample/make_sample.py)")
    with open(meta, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        for col in ("derm_image_path", "clinical_image_path"):
            rel = row[col].strip()
            path = SAMPLE_DIR / rel if rel else None
            row[col.replace("_path", "_bytes")] = path.read_bytes() if path and path.is_file() else None
    return rows


@pytest.fixture(scope="session")
def reference() -> dict[str, dict[str, float]]:
    with open(REFERENCE, newline="") as fh:
        return {r["isic_id"]: {k: float(v) for k, v in r.items() if k != "isic_id"} for r in csv.DictReader(fh)}
