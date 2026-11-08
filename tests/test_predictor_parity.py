"""The torch-free ONNX path reproduces the stored reference predictions (P1-DEC-05 hard condition)."""

import math

import pytest

from milk10k_serving.predictor import InvalidInput, Predictor

from .conftest import PARITY_TOLERANCE

pytestmark = [pytest.mark.parity, pytest.mark.slow]


@pytest.fixture(scope="module")
def predictor(loaded_package):
    return Predictor(loaded_package)


def _predict(predictor, row):
    return predictor.predict(
        row["derm_image_bytes"],
        row["clinical_image_bytes"],
        age_approx=row["age_approx"],
        sex=row["sex"] or None,
        anatomical_site_general=row["anatomical_site_general"] or None,
        fitzpatrick_type=row["fitzpatrick_type"],
    )


def test_intact_sample_lesions_match_reference(predictor, sample_lesions, reference):
    intact = [r for r in sample_lesions if r["scenario"] == "intact"]
    assert len(intact) == 3
    worst = 0.0
    for row in intact:
        got = _predict(predictor, row).row()
        ref = reference[row["isic_id"]]
        worst = max(worst, max(abs(got[k] - ref[k]) for k in ref))
    print(f"max |diff| vs reference over {len(intact)} lesions x 12 columns: {worst:.2e}")
    assert worst <= PARITY_TOLERANCE


def test_missing_clinical_lesions_give_valid_outputs(predictor, sample_lesions):
    others = [r for r in sample_lesions if r["scenario"] != "intact"]
    assert len(others) == 3
    for row in others:
        pred = _predict(predictor, row)
        values = list(pred.row().values())
        assert not pred.clinical_image_used
        assert all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in values)
        assert sum(pred.probabilities.values()) == pytest.approx(1.0, abs=1e-9)
        assert pred.diagnosis == max(pred.probabilities, key=pred.probabilities.get)


@pytest.mark.parametrize("derm", [None, b"", b"\xff\xd8 truncated"])
def test_missing_or_unreadable_dermoscopic_image_is_rejected(predictor, derm):
    with pytest.raises(InvalidInput, match="dermoscopic"):
        predictor.predict(derm, None)
