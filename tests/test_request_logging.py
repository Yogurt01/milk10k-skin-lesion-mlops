"""P1-PY-07: request id per request (accepted or generated, echoed in X-Request-ID, on every log line) and an
access log line with ids, sizes, status and latency only (never image bytes or metadata values)."""

import logging
import re

import pytest
from fastapi.testclient import TestClient

from milk10k_serving.api import RequestIdFilter, create_app, get_predictor, request_id_from
from milk10k_serving.settings import Settings

from .fakes import VALID_FORM, image_bytes, make_predictor

pytestmark = pytest.mark.api


class _Records(logging.Handler):
    """Collects records after RequestIdFilter ran in the emitting thread (as the real handler does)."""

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.addFilter(RequestIdFilter())
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def records():
    handler, package_logger = _Records(), logging.getLogger("milk10k_serving")
    package_logger.addHandler(handler)
    yield handler.records
    package_logger.removeHandler(handler)


def _client(predictor=None) -> TestClient:
    app = create_app(Settings(log_level="DEBUG"), loader=lambda: None)
    app.dependency_overrides[get_predictor] = lambda: predictor or make_predictor()
    return TestClient(app, raise_server_exceptions=False)


def _post(client, form=VALID_FORM, headers=None):
    files = {"derm_image": ("derm.jpg", image_bytes(), "image/jpeg")}
    return client.post("/predict", data=form, files=files, headers=headers or {})


def test_generated_request_id_is_returned_and_on_every_log_line_of_the_request(records):
    form = {k: v for k, v in VALID_FORM.items() if k != "anatomical_site_general"}  # logs "treated as missing"
    r = _post(_client(), form=form)
    request_id = r.headers["x-request-id"]
    assert r.status_code == 200 and re.fullmatch(r"[0-9a-f]{32}", request_id)
    mine = [rec for rec in records if rec.request_id == request_id]
    messages = [rec.getMessage() for rec in mine]
    assert any("metadata treated as missing" in m for m in messages)  # logged from the worker thread
    access = [m for m in messages if m.startswith(f"request_id={request_id} method=POST path=/predict status=200")]
    assert len(access) == 1 and re.search(r"duration_ms=\d+\.\d request_bytes=\d+", access[0])


def test_logs_never_contain_metadata_values_or_image_bytes(records):
    _post(_client(), form={**VALID_FORM, "age_approx": "37.5", "anatomical_site_general": "palms/soles"})
    text = "\n".join(rec.getMessage() for rec in records)
    for value in ("37.5", "palms/soles", "female", "male", "fitzpatrick_type=", "JFIF"):
        assert value not in text


@pytest.mark.parametrize(
    ("sent", "kept"), [("abc-123_X.y:z", True), ("a" * 64, True), ("a" * 65, False), ("has space", False), ("", False)]
)
def test_incoming_request_id_is_kept_only_when_valid(sent, kept):
    returned = _client().get("/health", headers={"X-Request-ID": sent}).headers["x-request-id"]
    assert (returned == sent) is kept
    assert request_id_from(returned) == returned


def test_generic_500_carries_the_request_id_in_header_and_log(records):
    class Boom:
        package = make_predictor().package

        def predict(self, *args, **kwargs):
            raise RuntimeError("secret internal detail")

    r = _post(_client(Boom()), headers={"X-Request-ID": "req-500"})
    assert r.status_code == 500 and r.json() == {"detail": "internal server error"}
    assert r.headers["x-request-id"] == "req-500"
    errors = [rec for rec in records if rec.levelno == logging.ERROR]
    assert errors and "request_id=req-500" in errors[0].getMessage() and errors[0].exc_info is not None
    assert any("status=500" in rec.getMessage() and rec.request_id == "req-500" for rec in records)


def test_probe_requests_are_logged_at_debug_only(records):
    _client().get("/health")
    probe = [rec for rec in records if "path=/health" in rec.getMessage()]
    assert probe and all(rec.levelno == logging.DEBUG for rec in probe)
