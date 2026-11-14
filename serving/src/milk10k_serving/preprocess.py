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
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from .settings import MAX_IMAGE_PIXELS_DEFAULT

# Decoded-image cap (P1-API-09). OpenCV reads OPENCV_IO_MAX_IMAGE_PIXELS once, when cv2 is first imported (measured:
# setting it after the import has no effect), and rejects a larger image before allocating it. Without a cap a
# 1.2 MB PNG decoded to 20000 x 20000 x 3 and the process grew by ~2.3 GiB. Set the default before the import.
# If cv2 was imported earlier without the variable, OpenCV's own (much higher) default applies: say so loudly.
_CAP_ALREADY_SET = "OPENCV_IO_MAX_IMAGE_PIXELS" in os.environ
IMAGE_PIXEL_CAP_APPLIED = "cv2" not in sys.modules or _CAP_ALREADY_SET
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(MAX_IMAGE_PIXELS_DEFAULT))
if not IMAGE_PIXEL_CAP_APPLIED:
    logging.getLogger(__name__).error(
        "cv2 was imported before milk10k_serving.preprocess and OPENCV_IO_MAX_IMAGE_PIXELS was not set: the "
        "decoded-image cap of %d pixels is NOT applied (OpenCV's default applies). Import milk10k_serving first or "
        "set OPENCV_IO_MAX_IMAGE_PIXELS in the environment.",
        MAX_IMAGE_PIXELS_DEFAULT,
    )

import cv2  # noqa: E402
import numpy as np  # noqa: E402

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


class ImageTooLarge(ValueError):
    """The image's header declares more pixels (or a larger side) than OpenCV's cap allows; it was not decoded."""


def decode_rgb(data: bytes | None) -> np.ndarray | None:
    """RGB uint8 image, or None if `data` is empty or not a decodable image. Raises ImageTooLarge above the cap
    (OPENCV_IO_MAX_IMAGE_PIXELS; OpenCV 5 raises instead of returning None, which would otherwise be a 500)."""
    if not data:
        return None
    try:
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as exc:
        if "validateInputImageSize" in str(exc):
            limit = os.environ.get("OPENCV_IO_MAX_IMAGE_PIXELS", "OpenCV default")
            raise ImageTooLarge(f"image is larger than the decoding limit ({limit} pixels)") from None
        return None
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
