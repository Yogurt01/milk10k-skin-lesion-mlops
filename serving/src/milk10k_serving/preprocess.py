"""Torch-free preprocessing, numerically equal to the release's evaluation path.

Source of truth: `training/src/milk10k/dataset.py` (`derm_transforms` / `clinical_transforms` with `train=False`,
`encode_tabular`, `MILKDataset.__getitem__`) and `inference.sanitize_inputs`. Notebook 08 checks the equality.

- Images are decoded with OpenCV and converted BGR -> RGB (as `read_rgb`).
- Dermoscopic: resize to size x size, bilinear (albumentations `Resize`).
- Clinical: longest side to `size` with Python `round()`, bilinear, then centre padding with 0 to size x size
  (albumentations `LongestMaxSize` + `PadIfNeeded`). A missing clinical image becomes zeros and `clin_mask = False`.
- Normalisation: albumentations `Normalize` for uint8, i.e. a per-channel float32 lookup table, output CHW float32.
- Tabular: integer codes with 0 = missing or unseen; age / 100 plus an age-missing flag.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

INPUT_NAMES = ("derm", "clin", "clin_mask", "site", "sex", "fst", "age", "age_missing")


@dataclass(frozen=True)
class PreprocessSpec:
    img_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    sites: tuple[str, ...]
    sex_codes: dict[str, int]
    fitzpatrick_values: tuple[int, ...]

    @classmethod
    def from_package(cls, model_dir: Path) -> PreprocessSpec:
        pre = json.loads((Path(model_dir) / "preprocessing.json").read_text())
        tab = json.loads((Path(model_dir) / "tabular_encoding.json").read_text())
        sex = {k: int(v) for k, v in tab["sex"].items() if k != "other_or_missing"}
        return cls(
            img_size=int(pre["img_size"]),
            mean=tuple(pre["mean"]),
            std=tuple(pre["std"]),
            sites=tuple(tab["site"]["vocabulary"]),
            sex_codes=sex,
            fitzpatrick_values=tuple(int(v) for v in tab["fitzpatrick_type"]["values"]),
        )

    def luts(self) -> np.ndarray:
        """(3, 256) float32 tables, computed exactly like albucore's `normalize_lut` for uint8 images."""
        mean = np.array(self.mean, dtype=np.float32) * 255.0
        denominator = np.reciprocal(np.array(self.std, dtype=np.float32) * 255.0)
        return (np.arange(0, 256, dtype=np.float32) - mean.reshape(-1, 1)) * denominator.reshape(-1, 1)


def decode_rgb(data: bytes | None) -> np.ndarray | None:
    """RGB uint8 image, or None if `data` is empty or not a decodable image."""
    if not data:
        return None
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _normalize(img: np.ndarray, luts: np.ndarray) -> np.ndarray:
    channels = [cv2.LUT(img[:, :, c], luts[c]) for c in range(3)]
    return np.stack(channels, axis=0).astype(np.float32, copy=False)


def dermoscopic(img: np.ndarray, spec: PreprocessSpec, luts: np.ndarray | None = None) -> np.ndarray:
    s = spec.img_size
    if img.shape[:2] != (s, s):
        img = cv2.resize(img, (s, s), interpolation=cv2.INTER_LINEAR)
    return _normalize(img, spec.luts() if luts is None else luts)


def clinical(img: np.ndarray, spec: PreprocessSpec, luts: np.ndarray | None = None) -> np.ndarray:
    s = spec.img_size
    h, w = img.shape[:2]
    scale = s / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    if (nh, nw) != (h, w):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = max(0, s - nh) // 2, max(0, s - nw) // 2
    bottom, right = max(0, s - nh) - top, max(0, s - nw) - left
    if top or bottom or left or right:
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return _normalize(img, spec.luts() if luts is None else luts)


def _number(value) -> float:
    """float(value), or NaN for None, empty, non-numeric or non-finite values."""
    if value is None or isinstance(value, bool):
        return math.nan
    try:
        number = float(str(value).strip()) if isinstance(value, str) else float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def encode_tabular(
    spec: PreprocessSpec, age_approx=None, sex=None, anatomical_site_general=None, fitzpatrick_type=None
) -> dict[str, np.ndarray]:
    """Batch-of-one tabular inputs. Unknown or invalid values are encoded as missing (0), as in the release."""
    age = _number(age_approx)
    fst = _number(fitzpatrick_type)
    site = anatomical_site_general
    return {
        "site": np.array(
            [spec.sites.index(site) + 1 if isinstance(site, str) and site in spec.sites else 0], dtype=np.int64
        ),
        "sex": np.array([spec.sex_codes.get(sex, 0) if isinstance(sex, str) else 0], dtype=np.int64),
        "fst": np.array([int(fst) if fst in spec.fitzpatrick_values else 0], dtype=np.int64),
        "age": np.array([0.0 if math.isnan(age) else age / 100], dtype=np.float32),
        "age_missing": np.array([1.0 if math.isnan(age) else 0.0], dtype=np.float32),
    }


def build_inputs(spec: PreprocessSpec, derm: np.ndarray, clin: np.ndarray | None, **metadata) -> dict[str, np.ndarray]:
    """ONNX feeds for one lesion. `derm` / `clin` are decoded RGB images; `clin=None` means missing."""
    luts = spec.luts()
    s = spec.img_size
    feeds = {
        "derm": dermoscopic(derm, spec, luts)[None],
        "clin": (clinical(clin, spec, luts) if clin is not None else np.zeros((3, s, s), dtype=np.float32))[None],
        "clin_mask": np.array([clin is not None], dtype=bool),
        **encode_tabular(spec, **metadata),
    }
    return {name: feeds[name] for name in INPUT_NAMES}
