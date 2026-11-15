"""Test doubles that need no model files: a real `Predictor` over a fake ensemble, plus synthetic images.

The vocabulary comes from the tracked `training/models/final-P1a-kg/tabular_encoding.json` (the model package carries a
byte-identical copy) and the class names from the header of the tracked reference predictions.
"""

import csv
import json
import struct
import threading
import zlib
from pathlib import Path

import cv2
import numpy as np

from milk10k_serving.package import ModelPackage
from milk10k_serving.predictor import Predictor
from milk10k_serving.preprocess import PreprocessSpec

from .conftest import REFERENCE, TABULAR_ENCODING

with open(REFERENCE, newline="") as _fh:
    CLASSES = next(csv.reader(_fh))[2:]  # isic_id, target_binary, then the 11 classes
VALID_FORM = {"age_approx": "60", "sex": "male", "anatomical_site_general": "head/neck", "fitzpatrick_type": "4"}


def make_spec() -> PreprocessSpec:
    tab = json.loads(TABULAR_ENCODING.read_text())
    return PreprocessSpec(
        img_size=288,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        sites=tuple(tab["site"]["vocabulary"]),
        sex_codes={k: int(v) for k, v in tab["sex"].items() if k != "other_or_missing"},
        fitzpatrick_values=tuple(int(v) for v in tab["fitzpatrick_type"]["values"]),
    )


class FakeEnsemble:
    """Returns fixed probabilities and records the feeds it was called with."""

    def __init__(self, p_malignant: float = 0.8, top: str = "BCC"):
        cls = np.full(len(CLASSES), 0.02)
        cls[CLASSES.index(top)] = 1.0 - 0.02 * (len(CLASSES) - 1)
        self.row = np.concatenate([[p_malignant], cls])
        self.calls: list[dict] = []

    def predict(self, feeds: dict, stage=None) -> np.ndarray:
        self.calls.append(feeds)
        return self.row[None, :].copy()


def make_predictor(ensemble: FakeEnsemble | None = None) -> Predictor:
    manifest = {"name": "fake-package", "version": "0.0-test", "classes": CLASSES}
    return Predictor(ModelPackage(Path("fake"), manifest, make_spec(), ensemble or FakeEnsemble()))


class SlowLoader:
    """Loader that blocks until `release()` is called; raises `error` instead of returning if given."""

    def __init__(self, predictor: Predictor | None = None, error: Exception | None = None):
        self.predictor = predictor or make_predictor()
        self.error = error
        self.calls = 0
        self._go = threading.Event()

    def release(self) -> None:
        self._go.set()

    def __call__(self) -> Predictor:
        self.calls += 1
        self._go.wait(timeout=10)
        if self.error is not None:
            raise self.error
        return self.predictor


def image_bytes(h: int = 450, w: int = 600, ext: str = ".jpg", seed: int = 0) -> bytes:
    """A synthetic noise image (never a lesion image)."""
    img = np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(ext, img)
    assert ok
    return buf.tobytes()


def blank_png(w: int, h: int) -> bytes:
    """A black RGB PNG of w x h written row by row (never held decoded in memory): a tiny file with huge dimensions."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    compressor, row = zlib.compressobj(9), b"\x00" + bytes(3 * w)
    idat = b"".join(compressor.compress(row) for _ in range(h)) + compressor.flush()
    header = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")
