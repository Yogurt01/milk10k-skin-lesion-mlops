import json
import math
import subprocess
import sys

import cv2
import numpy as np
import pytest

from milk10k_serving import preprocess
from milk10k_serving.preprocess import (
    INPUT_NAMES,
    ImageTooLarge,
    PreprocessSpec,
    build_inputs,
    clinical,
    decode_rgb,
    encode_tabular,
)
from milk10k_serving.settings import MAX_IMAGE_PIXELS_DEFAULT

from .conftest import ROOT, TABULAR_ENCODING
from .fakes import blank_png

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def spec() -> PreprocessSpec:
    tab = json.loads(TABULAR_ENCODING.read_text())
    return PreprocessSpec(
        img_size=288,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        sites=tuple(tab["site"]["vocabulary"]),
        sex_codes={"male": 1, "female": 2},
        fitzpatrick_values=(1, 2, 3, 4, 5, 6),
    )


@pytest.mark.parametrize(
    ("site", "code"), [("head/neck", 1), ("foot", 11), ("Head/Neck", 0), ("nowhere", 0), (None, 0), (3, 0)]
)
def test_site_codes(spec, site, code):
    assert encode_tabular(spec, anatomical_site_general=site)["site"][0] == code


@pytest.mark.parametrize(
    ("value", "code"), [(4, 4), ("4", 4), (4.0, 4), (6, 6), (7, 0), (0, 0), (4.5, 0), (None, 0), ("x", 0)]
)
def test_fitzpatrick_codes(spec, value, code):
    assert encode_tabular(spec, fitzpatrick_type=value)["fst"][0] == code


@pytest.mark.parametrize(
    ("value", "age", "missing"),
    [(60, 0.6, 0.0), ("45", 0.45, 0.0), (None, 0.0, 1.0), ("abc", 0.0, 1.0), (math.inf, 0.0, 1.0)],
)
def test_age_and_missing_flag(spec, value, age, missing):
    out = encode_tabular(spec, age_approx=value)
    assert out["age"][0] == pytest.approx(age)
    assert out["age_missing"][0] == missing


def test_sex_codes(spec):
    assert [encode_tabular(spec, sex=s)["sex"][0] for s in ("male", "female", "other", None)] == [1, 2, 0, 0]


@pytest.mark.parametrize("data", [None, b"", b"not an image"])
def test_undecodable_bytes_give_none(data):
    assert decode_rgb(data) is None


def test_decode_converts_bgr_to_rgb():
    bgr = np.zeros((4, 5, 3), dtype=np.uint8)
    bgr[..., 0] = 255  # blue in OpenCV's BGR order
    ok, png = cv2.imencode(".png", bgr)
    rgb = decode_rgb(png.tobytes())
    assert rgb.shape == (4, 5, 3) and rgb[0, 0].tolist() == [0, 0, 255]


@pytest.mark.parametrize(("h", "w"), [(450, 600), (600, 450), (100, 333), (288, 288)])
def test_clinical_letterbox_shape_and_padding(spec, h, w):
    img = np.full((h, w, 3), 128, dtype=np.uint8)
    out = clinical(img, spec)
    assert out.shape == (3, 288, 288) and out.dtype == np.float32
    if h != w:  # a corner lies in the zero padding, which is normalised like a black pixel
        expected = (0 - np.array(spec.mean) * 255) / (np.array(spec.std) * 255)
        assert np.allclose(out[:, 0, 0], expected, atol=1e-6)


def test_build_inputs_without_clinical_image(spec):
    derm = np.full((450, 600, 3), 90, dtype=np.uint8)
    feeds = build_inputs(
        spec, derm, None, age_approx=50, sex="female", anatomical_site_general="trunk", fitzpatrick_type=3
    )
    assert tuple(feeds) == INPUT_NAMES
    assert feeds["derm"].shape == (1, 3, 288, 288) and feeds["clin"].shape == (1, 3, 288, 288)
    assert not feeds["clin"].any() and feeds["clin_mask"].tolist() == [False]
    assert {k: v.dtype.name for k, v in feeds.items()} == {
        "derm": "float32",
        "clin": "float32",
        "clin_mask": "bool",
        "site": "int64",
        "sex": "int64",
        "fst": "int64",
        "age": "float32",
        "age_missing": "float32",
    }


def test_image_above_the_default_pixel_cap_is_rejected_before_decoding():
    assert preprocess.IMAGE_PIXEL_CAP_APPLIED  # conftest imports preprocess before anything imports cv2
    w, h = 5001, 5000  # 25,005,000 pixels, a 73 kB file
    assert w * h > MAX_IMAGE_PIXELS_DEFAULT
    with pytest.raises(ImageTooLarge, match="decoding limit"):
        decode_rgb(blank_png(w, h))
    assert decode_rgb(blank_png(600, 450)).shape == (450, 600, 3)


def test_pixel_cap_comes_from_the_environment_at_import():
    code = (
        "from milk10k_serving.preprocess import ImageTooLarge, decode_rgb\n"
        "from tests.fakes import blank_png\n"
        "assert decode_rgb(blank_png(30, 30)).shape == (30, 30, 3)\n"
        "try:\n    decode_rgb(blank_png(40, 40))\nexcept ImageTooLarge:\n    print('rejected')\n"
    )
    env = {"OPENCV_IO_MAX_IMAGE_PIXELS": "1000", "PATH": ""}
    result = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "rejected"
