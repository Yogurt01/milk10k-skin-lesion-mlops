"""API tests: FastAPI TestClient with a fake predictor (no model files)."""

import logging
import re

import pytest
from fastapi.testclient import TestClient

from milk10k_serving import DISCLAIMER
from milk10k_serving.api import create_app, get_predictor
from milk10k_serving.package import ModelPackageError
from milk10k_serving.settings import Settings

from .fakes import CLASSES, VALID_FORM, FakeEnsemble, SlowLoader, image_bytes, make_predictor

pytestmark = pytest.mark.api
MB = 1024 * 1024


def _never_load():
    raise AssertionError("the real loader must not run in API tests")


@pytest.fixture
def ensemble():
    return FakeEnsemble()


@pytest.fixture
def client(ensemble):
    """App with a 1 MB upload limit and the predictor injected through the dependency (lifespan not started)."""
    app = create_app(Settings(max_upload_mb=1), loader=_never_load)
    app.dependency_overrides[get_predictor] = lambda: make_predictor(ensemble)
    return TestClient(app)


def _post(client, form=None, files=None, **kwargs):
    form = VALID_FORM if form is None else form
    files = {"derm_image": ("derm.jpg", image_bytes(), "image/jpeg")} if files is None else files
    return client.post("/predict", data=form, files=files, **kwargs)


def _error_locs(response):
    return [tuple(e["loc"]) for e in response.json()["detail"]]


# ---- liveness, readiness and loading ----------------------------------------------------------------------------
def test_health_is_always_200(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ready_is_503_while_loading_then_200_and_predict_follows():
    loader = SlowLoader()
    app = create_app(Settings(), loader=loader)
    with TestClient(app) as c:  # runs the lifespan: the load starts in a background thread
        assert c.get("/health").status_code == 200
        not_ready = c.get("/ready")
        assert not_ready.status_code == 503 and not_ready.json() == {"status": "loading"}
        blocked = _post(c)
        assert blocked.status_code == 503 and blocked.headers["retry-after"] == "5"
        loader.release()
        assert app.state.model.wait(5)
        ready = c.get("/ready")
        assert ready.status_code == 200
        assert ready.json()["status"] == "ready" and ready.json()["model"] == "fake-package"
        assert ready.json()["model_version"] == "0.0-test" and ready.json()["load_seconds"] >= 0
        assert _post(c).status_code == 200
    assert loader.calls == 1


def test_failed_load_keeps_ready_503_with_the_error():
    loader = SlowLoader(error=ModelPackageError("P1a-kg-s44_fold4.onnx: sha256 0000 != manifest ffff"))
    loader.release()
    app = create_app(Settings(), loader=loader)
    with TestClient(app) as c:
        assert app.state.model.wait(5)
        ready = c.get("/ready")
        assert ready.status_code == 503 and ready.json()["status"] == "failed"
        assert "sha256 0000 != manifest ffff" in ready.json()["error"]
        predict = _post(c)
        assert predict.status_code == 503 and "failed to load" in predict.json()["detail"]
        assert c.get("/health").status_code == 200


def test_failed_load_hides_absolute_paths_from_clients_and_logs_the_detail(tmp_path, caplog):
    model_dir = tmp_path / "model_package"
    message = f"{model_dir}/P1a-kg-s44_fold4.onnx: sha256 0000 != manifest ffff; loader at /opt/runtime/lib/ort.so"
    loader = SlowLoader(error=ModelPackageError(message))
    loader.release()
    app = create_app(Settings(model_dir=model_dir), loader=loader)
    caplog.set_level(logging.ERROR, logger="milk10k_serving")
    with TestClient(app) as c:
        assert app.state.model.wait(5)
        ready = c.get("/ready")
        predict = _post(c)
    assert ready.status_code == 503 and predict.status_code == 503
    assert ready.json()["error"] == (
        "ModelPackageError: <MODEL_DIR>/P1a-kg-s44_fold4.onnx: sha256 0000 != manifest ffff; loader at <path>/ort.so"
    )
    for body in (ready.text, predict.text):  # /predict's 503 repeats the error
        assert str(tmp_path) not in body and "/opt/runtime" not in body
        assert not re.search(r"(?<![\w<>])/(?:home|tmp|opt|usr|root)\b", body)
    assert message in caplog.text and "Traceback" in caplog.text  # the full detail stays in the server log


def test_openapi_lists_the_package_vocabulary_once_loaded():
    loader = SlowLoader()
    app = create_app(Settings(), loader=loader)
    with TestClient(app) as c:

        def body_schema():
            schemas = c.get("/openapi.json").json()["components"]["schemas"]
            return next(s for s in schemas.values() if "derm_image" in s.get("properties", {}))

        assert "enum" not in body_schema()["properties"]["sex"]
        loader.release()
        assert app.state.model.wait(5)
        props = body_schema()["properties"]
        assert props["sex"]["enum"] == ["male", "female"]
        assert props["fitzpatrick_type"]["enum"] == [1, 2, 3, 4, 5, 6]
        assert "head/neck" in props["anatomical_site_general"]["enum"]


def test_openapi_documents_the_status_contract(client):
    responses = client.get("/openapi.json").json()["paths"]["/predict"]["post"]["responses"]
    assert {"200", "400", "413", "422", "500", "503"} <= set(responses)
    assert "treated as missing" in responses["422"]["description"]
    assert "multipart" in responses["400"]["description"]


def test_openapi_states_the_age_range(client):
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    age = next(s for s in schemas.values() if "derm_image" in s.get("properties", {}))["properties"]["age_approx"]
    number = next(branch for branch in age["anyOf"] if branch.get("type") == "number")  # optional: number or null
    assert number["minimum"] == 0 and number["maximum"] == 120 and "0-120" in age["description"]


def test_openapi_states_which_metadata_is_required_and_the_missing_rule(client):
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    body = next(s for s in schemas.values() if "derm_image" in s.get("properties", {}))
    assert set(body["required"]) == {"derm_image", "sex", "fitzpatrick_type"}
    for field in ("age_approx", "anatomical_site_general"):
        assert "metadata_missing" in body["properties"][field]["description"]
        assert body["properties"][field]["examples"]
    assert "metadata_missing" in schemas["PredictionResponse"]["required"]


# ---- 200 paths ---------------------------------------------------------------------------------------------------
def test_valid_request_returns_every_field_and_the_disclaimer(client, ensemble):
    files = {
        "derm_image": ("derm.jpg", image_bytes(), "image/jpeg"),
        "clinical_image": ("clin.png", image_bytes(300, 400, ".png", seed=1), "image/png"),
    }
    r = _post(client, files=files)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {
        "p_malignant",
        "probabilities",
        "diagnosis",
        "clinical_image_used",
        "model",
        "model_version",
        "metadata_missing",
        "disclaimer",
    }
    assert body["disclaimer"] == DISCLAIMER and DISCLAIMER.strip()
    assert body["metadata_missing"] == []
    assert list(body["probabilities"]) == CLASSES and sum(body["probabilities"].values()) == pytest.approx(1.0)
    assert body["diagnosis"] == "BCC" and body["p_malignant"] == pytest.approx(0.8)
    assert body["clinical_image_used"] is True and body["model"] == "fake-package"
    feeds = ensemble.calls[-1]  # the form values reached the encoder
    assert (feeds["sex"][0], feeds["site"][0], feeds["fst"][0]) == (1, 1, 4)
    assert feeds["age"][0] == pytest.approx(0.6)


def test_clinical_image_is_optional(client, ensemble):
    r = _post(client)
    assert r.status_code == 200 and r.json()["clinical_image_used"] is False
    assert r.json()["disclaimer"] == DISCLAIMER
    assert bool(ensemble.calls[-1]["clin_mask"][0]) is False


@pytest.mark.parametrize("clinical", [b"", b"not an image"], ids=["empty", "undecodable"])
def test_unusable_clinical_image_is_treated_as_missing(client, clinical):
    files = {
        "derm_image": ("derm.jpg", image_bytes(), "image/jpeg"),
        "clinical_image": ("clin.jpg", clinical, "image/jpeg"),
    }
    r = _post(client, files=files)
    assert r.status_code == 200 and r.json()["clinical_image_used"] is False


@pytest.mark.parametrize("age", ["0", "120"])
def test_age_range_boundaries_are_accepted(client, ensemble, age):
    r = _post(client, form={**VALID_FORM, "age_approx": age})
    assert r.status_code == 200
    assert ensemble.calls[-1]["age"][0] == pytest.approx(int(age) / 100)


def test_empty_clinical_form_value_is_treated_as_missing(client):
    r = client.post(
        "/predict",
        data={**VALID_FORM, "clinical_image": ""},
        files={"derm_image": ("derm.jpg", image_bytes(), "image/jpeg")},
    )
    assert r.status_code == 200 and r.json()["clinical_image_used"] is False


# ---- 422 paths ---------------------------------------------------------------------------------------------------
def test_missing_dermoscopic_image_is_422(client):
    r = _post(client, files={"clinical_image": ("clin.jpg", image_bytes(), "image/jpeg")})
    assert r.status_code == 422 and ("body", "derm_image") in _error_locs(r)


@pytest.mark.parametrize(("data", "reason"), [(b"", "missing"), (b"\xff\xd8\xff\xe0 truncated", "not a decodable")])
def test_empty_or_undecodable_dermoscopic_image_is_422(client, data, reason):
    r = _post(client, files={"derm_image": ("derm.jpg", data, "image/jpeg")})
    assert r.status_code == 422 and _error_locs(r) == [("body", "derm_image")]
    assert reason in r.json()["detail"][0]["msg"]


@pytest.mark.parametrize("how", ["omitted", "empty"])
@pytest.mark.parametrize("field", ["sex", "fitzpatrick_type"])
def test_missing_required_metadata_field_is_422(client, field, how):
    form = {k: v for k, v in VALID_FORM.items() if k != field} if how == "omitted" else {**VALID_FORM, field: ""}
    r = _post(client, form=form)
    assert r.status_code == 422 and ("body", field) in _error_locs(r)


@pytest.mark.parametrize("how", ["omitted", "empty"])
@pytest.mark.parametrize(
    ("field", "feeds_when_missing"),
    [("age_approx", {"age": 0.0, "age_missing": 1.0}), ("anatomical_site_general", {"site": 0})],
)
def test_missing_optional_metadata_field_is_encoded_as_missing(
    client, ensemble, caplog, field, feeds_when_missing, how
):
    form = {k: v for k, v in VALID_FORM.items() if k != field} if how == "omitted" else {**VALID_FORM, field: ""}
    caplog.set_level(logging.INFO, logger="milk10k_serving")
    r = _post(client, form=form)
    assert r.status_code == 200 and r.json()["metadata_missing"] == [field]
    assert r.json()["disclaimer"] == DISCLAIMER
    feeds = ensemble.calls[-1]  # encoded as the release does
    assert {k: feeds[k][0] for k in feeds_when_missing} == pytest.approx(feeds_when_missing)
    assert f"metadata treated as missing: {field}" in caplog.text


def test_both_optional_fields_missing_are_listed_and_present_ones_are_not(client, ensemble):
    form = {"sex": "female", "fitzpatrick_type": "2", "age_approx": ""}
    r = _post(client, form=form)
    assert r.status_code == 200 and r.json()["metadata_missing"] == ["age_approx", "anatomical_site_general"]
    feeds = ensemble.calls[-1]
    assert (feeds["site"][0], feeds["age_missing"][0], feeds["sex"][0], feeds["fst"][0]) == (0, 1.0, 2, 2)
    assert _post(client).json()["metadata_missing"] == []  # all four sent: nothing missing


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sex", "unknown"),
        ("anatomical_site_general", "Head/Neck"),
        ("anatomical_site_general", "elbow"),
        ("fitzpatrick_type", "7"),
        ("fitzpatrick_type", "0"),
        ("fitzpatrick_type", "four"),
        ("age_approx", "-1"),
        ("age_approx", "121"),
        ("age_approx", "nan"),
        ("age_approx", "inf"),
        ("age_approx", "sixty"),
    ],
)
def test_invalid_form_value_is_422(client, field, value):
    r = _post(client, form={**VALID_FORM, field: value})
    assert r.status_code == 422 and _error_locs(r) == [("body", field)]


# ---- 413, 400, 500 -----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("field", ["derm_image", "clinical_image"])
def test_image_over_the_upload_limit_is_413(client, field):
    files = {"derm_image": ("derm.jpg", image_bytes(), "image/jpeg"), field: ("big.jpg", b"x" * (MB + 1), "image/jpeg")}
    r = _post(client, files=files)
    assert r.status_code == 413 and field in r.json()["detail"]


def test_oversized_request_body_is_413_before_it_is_parsed(client):
    r = _post(client, files={"derm_image": ("big.jpg", b"x" * (3 * MB + 1), "image/jpeg")})
    assert r.status_code == 413 and "request body" in r.json()["detail"]


def test_unparseable_multipart_body_is_400_not_500(client):
    r = client.post(
        "/predict", content=b"--xyz\r\nnot a valid part", headers={"Content-Type": "multipart/form-data; boundary=xyz"}
    )
    assert r.status_code == 400 and isinstance(r.json()["detail"], str)


def test_unexpected_error_is_a_generic_500_without_details():
    class Boom:
        package = make_predictor().package

        def predict(self, *args, **kwargs):
            raise RuntimeError("secret internal detail")

    app = create_app(Settings(), loader=_never_load)
    app.dependency_overrides[get_predictor] = lambda: Boom()
    r = _post(TestClient(app, raise_server_exceptions=False))
    assert r.status_code == 500 and r.json() == {"detail": "internal server error"}
    assert "secret" not in r.text and "Traceback" not in r.text


def test_predict_is_503_when_the_model_was_never_loaded():
    r = _post(TestClient(create_app(Settings(), loader=_never_load)))
    assert r.status_code == 503 and r.json()["detail"] == "model is loading"
