"""How far a naive `(x / 255 - mean) / std` normalisation is from the one in `milk10k_serving` (qa M3).

`milk10k_serving.preprocess` copies albumentations' uint8 path: a per-channel float32 lookup table
`(x - 255 * mean) / (255 * std)`. This script feeds the same resized uint8 pixels (the package's own resize and
padding) to both and reports max |diff|, for the naive formula in float32 and in float64. Torch-free; reads the
6 sample lesions (dermoscopic images) and the model package's `preprocessing.json`, and uses the 6 synthetic image
sizes of notebook 08 §5 (each through the dermoscopic and the clinical geometry). Writes `out/normalize_diff.json`.

Usage (from the repository root, sample images built by `training/sample/make_sample.py`):

    serving/.venv/bin/python benchmarks/serving/normalize_diff.py
"""

import csv
import json
from pathlib import Path

import numpy as np

from milk10k_serving.preprocess import PreprocessSpec, clinical, decode_rgb, dermoscopic
from milk10k_serving.settings import load_settings

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "training" / "sample"
OUT = Path(__file__).resolve().parent / "out"
SYNTHETIC_SIZES = [(450, 600), (600, 450), (287, 289), (333, 1000), (120, 90), (288, 288)]  # notebook 08 §5
IDENTITY = np.tile(np.arange(256, dtype=np.float32), (3, 1))  # LUT that returns the pixel values unchanged


def naive(pixels: np.ndarray, spec: PreprocessSpec, dtype) -> np.ndarray:
    """(x / 255 - mean) / std on CHW pixel values, computed in `dtype`."""
    mean = np.array(spec.mean, dtype=dtype).reshape(3, 1, 1)
    std = np.array(spec.std, dtype=dtype).reshape(3, 1, 1)
    return (pixels.astype(dtype) / dtype(255) - mean) / std


def compare(img: np.ndarray, geometry, spec: PreprocessSpec) -> dict:
    served = geometry(img, spec).astype(np.float64)  # the LUT normalisation used by the API
    pixels = geometry(img, spec, IDENTITY)  # same resize / padding, values 0..255
    assert np.array_equal(pixels, np.round(pixels)) and pixels.min() >= 0 and pixels.max() <= 255
    return {
        f"naive_{name}_max_abs_diff": float(np.abs(naive(pixels, spec, dtype).astype(np.float64) - served).max())
        for name, dtype in (("float32", np.float32), ("float64", np.float64))
    }


def main() -> None:
    model_dir = load_settings().model_dir
    spec = PreprocessSpec.from_package(model_dir)
    rows = []
    with open(SAMPLE / "sample_metadata.csv", newline="") as f:
        for lesion in csv.DictReader(f):
            img = decode_rgb((SAMPLE / lesion["derm_image_path"]).read_bytes())
            rows.append({"image": lesion["isic_id"], "geometry": "dermoscopic", **compare(img, dermoscopic, spec)})
    rng = np.random.default_rng(0)
    for h, w in SYNTHETIC_SIZES:
        img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        for name, geometry in (("dermoscopic", dermoscopic), ("clinical", clinical)):
            rows.append({"image": f"synthetic {h}x{w}", "geometry": name, **compare(img, geometry, spec)})
    keys = ("naive_float32_max_abs_diff", "naive_float64_max_abs_diff")
    result = {
        "model_package": model_dir.name,
        "img_size": spec.img_size,
        "images": {"sample_lesions": sum(r["image"].startswith("ISIC_") for r in rows), "synthetic_sizes": 6},
        **{f"worst_{k}": max(r[k] for r in rows) for k in keys},
        "rows": rows,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "normalize_diff.json").write_text(json.dumps(result, indent=2) + "\n")
    for k in keys:
        print(f"worst {k}: {result['worst_' + k]:.3e}")
    print(f"{len(rows)} comparisons -> {(OUT / 'normalize_diff.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
