"""Unit tests for the API's pure parts and the predictor adapter (no model files, no HTTP)."""

import io
import json

import pytest
from fastapi import HTTPException, UploadFile

from milk10k_serving import DISCLAIMER
from milk10k_serving.api import ModelState, allowed_values, metadata_errors, read_limited, to_response
from milk10k_serving.predictor import InvalidInput

from .conftest import TABULAR_ENCODING
from .fakes import CLASSES, FakeEnsemble, image_bytes, make_predictor, make_spec

pytestmark = pytest.mark.unit
VALID = {"sex": "female", "anatomical_site_general": "trunk", "fitzpatrick_type": 3}


def test_allowed_values_come_from_tabular_encoding():
    tab = json.loads(TABULAR_ENCODING.read_text())
    allowed = allowed_values(make_spec())
    assert allowed["sex"] == ["male", "female"]
    assert allowed["anatomical_site_general"] == tab["site"]["vocabulary"]
    assert allowed["fitzpatrick_type"] == tab["fitzpatrick_type"]["values"]


def test_valid_metadata_has_no_errors():
    assert metadata_errors(make_spec(), **VALID) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sex", "unknown"),
        ("sex", "Male"),
        ("anatomical_site_general", "Head/Neck"),
        ("fitzpatrick_type", 0),
        ("fitzpatrick_type", 7),
    ],
)
def test_value_outside_vocabulary_is_reported(field, value):
    errors = metadata_errors(make_spec(), **{**VALID, field: value})
    assert len(errors) == 1
    assert errors[0]["loc"] == ("body", field) and errors[0]["input"] == value
    assert errors[0]["msg"].startswith("Input should be one of: ")


def test_all_invalid_fields_are_reported_together():
    errors = metadata_errors(make_spec(), sex="x", anatomical_site_general="y", fitzpatrick_type=9)
    assert [e["loc"][1] for e in errors] == ["sex", "anatomical_site_general", "fitzpatrick_type"]


def test_response_assembly_carries_every_field_and_the_disclaimer():
    prediction = make_predictor().predict(image_bytes(), image_bytes(seed=1), age_approx=60, **VALID)
    response = to_response(prediction)
    assert response.p_malignant == pytest.approx(0.8)
    assert list(response.probabilities) == CLASSES
    assert response.diagnosis == "BCC"
    assert response.model == "fake-package" and response.model_version == "0.0-test"
    assert response.disclaimer == DISCLAIMER and DISCLAIMER.strip()


def test_read_limited_accepts_up_to_the_limit_and_rejects_more():
    assert read_limited(UploadFile(io.BytesIO(b"x" * 10)), "derm_image", 10) == b"x" * 10
    with pytest.raises(HTTPException) as exc:
        read_limited(UploadFile(io.BytesIO(b"x" * 11)), "derm_image", 10)
    assert exc.value.status_code == 413 and "derm_image" in exc.value.detail


def test_adapter_maps_outputs_and_flags_the_clinical_image():
    ensemble = FakeEnsemble(p_malignant=0.3, top="NV")
    predictor = make_predictor(ensemble)
    with_clin = predictor.predict(image_bytes(), image_bytes(300, 400, ".png", seed=2), age_approx=50, **VALID)
    bad_clin = predictor.predict(image_bytes(), b"not an image", age_approx=50, **VALID)
    no_clin = predictor.predict(image_bytes(), None, age_approx=50, **VALID)
    assert with_clin.diagnosis == "NV" and with_clin.p_malignant == pytest.approx(0.3)
    assert sum(with_clin.probabilities.values()) == pytest.approx(1.0)
    assert [p.clinical_image_used for p in (with_clin, bad_clin, no_clin)] == [True, False, False]
    assert [bool(c["clin_mask"][0]) for c in ensemble.calls] == [True, False, False]
    assert ensemble.calls[0]["derm"].shape == (1, 3, 288, 288)


@pytest.mark.parametrize(("derm", "reason"), [(None, "missing"), (b"", "missing"), (b"\xff\xd8 cut", "decodable")])
def test_adapter_rejects_a_missing_or_undecodable_dermoscopic_image(derm, reason):
    with pytest.raises(InvalidInput, match=reason) as exc:
        make_predictor().predict(derm, None, **VALID)
    assert exc.value.field == "dermoscopic_image"


def test_model_state_loads_once_in_the_background():
    calls = []
    state = ModelState(lambda: calls.append(1) or make_predictor())
    assert state.status == "loading"
    state.start()
    state.start()  # idempotent
    assert state.wait(5) and state.status == "ready" and state.predictor is not None
    assert calls == [1] and state.load_seconds is not None


def test_model_state_records_a_failed_load():
    def broken():
        raise RuntimeError("sha256 0000 != manifest ffff")

    state = ModelState(broken)
    state.start()
    assert state.wait(5) and state.status == "failed" and state.predictor is None
    assert state.error == "RuntimeError: sha256 0000 != manifest ffff"
