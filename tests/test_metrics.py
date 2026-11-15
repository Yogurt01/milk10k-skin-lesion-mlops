"""P1-OBS-01 / P1-OBS-02: `GET /metrics` (Prometheus text format) with a fake predictor; bounded labels."""

import random
import string
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from milk10k_serving.api import create_app
from milk10k_serving.metrics import UNMATCHED
from milk10k_serving.settings import Settings

from .fakes import VALID_FORM, SlowLoader, image_bytes, make_predictor
from .live_server import live_server
from .test_concurrency import TIMEOUT, BlockingEnsemble, _post, _timed_get, _wait_until

pytestmark = pytest.mark.api


def samples(text: str) -> dict[tuple, float]:
    """{(sample name, sorted label items): value} of a /metrics body."""
    out = {}
    for family in text_string_to_metric_families(text):
        for s in family.samples:
            out[(s.name, tuple(sorted(s.labels.items())))] = s.value
    return out


def value(text: str, name: str, **labels) -> float:
    return samples(text).get((name, tuple(sorted(labels.items()))), 0.0)


def requests_total(text: str, route: str, status: int, method: str = "POST") -> float:
    return value(text, "milk10k_http_requests_total", route=route, method=method, status=str(status))


@pytest.fixture
def ready_client():
    """App whose (fake) model loads immediately; the lifespan runs, so readiness and model metrics are real."""
    app = create_app(Settings(), loader=lambda: make_predictor())
    with TestClient(app) as client:
        app.state.model.wait(5)
        yield client


def _predict(client, form=None):
    files = {"derm_image": ("derm.jpg", image_bytes(), "image/jpeg")}
    return client.post("/predict", data=VALID_FORM if form is None else form, files=files)


def test_metrics_content_type_and_names(ready_client):
    ready_client.get("/health")  # a request is counted once its response is ready, so scrape after one
    response = ready_client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    for name in (
        "milk10k_http_requests_total",
        "milk10k_http_request_duration_seconds_bucket",
        "milk10k_predictions_running",
        "milk10k_predictions_waiting",
        "milk10k_predictions_rejected_busy_total",
        "milk10k_model_ready",
        "milk10k_model_load_seconds",
        "milk10k_model_info",
        "process_resident_memory_bytes",
    ):
        assert name in response.text, name


def test_counters_increase_on_200_and_422_and_model_metrics_appear(ready_client):
    before = ready_client.get("/metrics").text
    assert _predict(ready_client).status_code == 200
    assert _predict(ready_client, VALID_FORM | {"sex": "unknown-value"}).status_code == 422
    after = ready_client.get("/metrics").text
    assert requests_total(after, "/predict", 200) == requests_total(before, "/predict", 200) + 1
    assert requests_total(after, "/predict", 422) == requests_total(before, "/predict", 422) + 1
    count = "milk10k_http_request_duration_seconds_count"
    assert (
        value(after, count, route="/predict", method="POST")
        == value(before, count, route="/predict", method="POST") + 2
    )
    # model metrics after one /predict (P1-OBS-02)
    assert value(after, "milk10k_model_ready") == 1
    assert value(after, "milk10k_model_load_seconds") >= 0
    assert value(after, "milk10k_model_info", model="fake-package", model_version="0.0-test") == 1
    for stage in ("decode", "preprocess", "inference"):
        assert value(after, "milk10k_predict_stage_duration_seconds_count", stage=stage) == 1, stage
    assert value(after, "milk10k_prediction_p_malignant_count") == 1  # the 422 never reached the model
    assert value(after, "milk10k_prediction_p_malignant_bucket", le="0.8") == 1  # FakeEnsemble: 0.8
    # nothing of the request in the labels: no metadata value, no diagnosis, no file name
    for leaked in ("head/neck", "male", "unknown-value", "BCC", "derm.jpg"):
        assert leaked not in after, leaked


def test_counters_increase_on_503_while_loading():
    loader = SlowLoader()
    app = create_app(Settings(), loader=loader)
    with TestClient(app) as client:
        assert client.get("/metrics").text.count("milk10k_model_ready 0.0") == 1
        assert _predict(client).status_code == 503
        text = client.get("/metrics").text
        assert requests_total(text, "/predict", 503) == 1
        assert value(text, "milk10k_predictions_rejected_busy_total") == 0  # loading is not busy
        loader.release()
        app.state.model.wait(5)
        assert value(client.get("/metrics").text, "milk10k_model_ready") == 1


def test_busy_rejections_are_counted_and_running_waiting_gauges_follow_the_limiter():
    ensemble = BlockingEnsemble()
    app = create_app(
        Settings(max_concurrent_predictions=1, max_queued_predictions=1), loader=lambda: make_predictor(ensemble)
    )
    with live_server(app) as base, ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(_post, base) for _ in range(2)]
        _wait_until(lambda: ensemble.active == 1 and app.state.limiter.waiting == 1, "1 running and 1 queued")
        text = _timed_get(base + "/metrics")[0].text
        assert value(text, "milk10k_predictions_running") == 1
        assert value(text, "milk10k_predictions_waiting") == 1
        rejected = _post(base)
        assert rejected.status_code == 503
        text = _timed_get(base + "/metrics")[0].text
        ensemble.release.set()
        assert [f.result().status_code for f in futures] == [200, 200]
    assert value(text, "milk10k_predictions_rejected_busy_total") == 1
    assert requests_total(text, "/predict", 503) == 1


def test_random_paths_do_not_create_series(ready_client):
    """1,000 distinct random paths (and junk methods) collapse to one route label: `unmatched`."""
    rng = random.Random(0)
    paths = {"/" + "".join(rng.choices(string.ascii_lowercase + string.digits, k=12)) for _ in range(1000)}
    assert len(paths) == 1000
    for path in paths:
        assert ready_client.get(path).status_code == 404
    ready_client.request("FOO", "/predict")
    ready_client.get("/predict")  # known path, wrong method: labelled with its template
    text = ready_client.get("/metrics").text
    series = [k for k in samples(text) if k[0] == "milk10k_http_requests_total"]
    assert len(series) <= 6, series
    routes = {dict(labels)["route"] for _, labels in series}
    assert routes <= {UNMATCHED, "/predict", "/metrics"}
    assert requests_total(text, UNMATCHED, 404, method="GET") == 1000
    assert requests_total(text, "/predict", 405, method="GET") == 1
    assert requests_total(text, "/predict", 405, method="other") == 1
    for path in list(paths)[:50]:
        assert path not in text


def test_metrics_answers_while_the_threadpool_is_saturated():
    # 40 = anyio's default threadpool size: every worker thread is busy with a blocked prediction
    ensemble = BlockingEnsemble()
    app = create_app(
        Settings(max_concurrent_predictions=40, max_queued_predictions=0), loader=lambda: make_predictor(ensemble)
    )
    with live_server(app) as base, ThreadPoolExecutor(40) as pool:
        futures = [pool.submit(_post, base) for _ in range(40)]
        _wait_until(lambda: ensemble.active == 40, "40 blocked predictions")
        response, ms = _timed_get(base + "/metrics")
        ensemble.release.set()
        statuses = [f.result().status_code for f in futures]
    assert response.status_code == 200 and ms < 100, f"/metrics took {ms:.0f} ms"
    assert value(response.text, "milk10k_predictions_running") == 40
    assert statuses == [200] * 40


def test_metrics_scrapes_are_not_logged_at_info(ready_client, caplog):
    with caplog.at_level("INFO", logger="milk10k_serving"):
        ready_client.get("/metrics")
    assert not [r for r in caplog.records if r.levelname == "INFO" and "/metrics" in r.getMessage()]


def test_metrics_is_not_limited_by_the_prediction_limiter():
    """/metrics never takes a prediction slot: with every slot and queue place taken, it still answers 200."""
    ensemble = BlockingEnsemble()
    app = create_app(
        Settings(max_concurrent_predictions=1, max_queued_predictions=0), loader=lambda: make_predictor(ensemble)
    )
    with live_server(app) as base:
        worker = threading.Thread(target=_post, args=(base,))
        worker.start()
        _wait_until(lambda: ensemble.active == 1, "1 running")
        assert _post(base).status_code == 503
        assert _timed_get(base + "/metrics")[0].status_code == 200
        ensemble.release.set()
        worker.join(TIMEOUT)
