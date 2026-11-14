"""P1-API-05: /predict concurrency limit, bounded queue, overload 503, and liveness/readiness that never wait for the
threadpool. Real uvicorn server in a thread (see live_server.py) with a fake ensemble that blocks until released."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from milk10k_serving.api import RETRY_AFTER_BUSY, create_app
from milk10k_serving.settings import Settings

from .fakes import VALID_FORM, FakeEnsemble, image_bytes, make_predictor
from .live_server import live_server

pytestmark = pytest.mark.api
TIMEOUT = 10.0


class BlockingEnsemble(FakeEnsemble):
    """Each call blocks until `release` is set; tracks how many calls run at once."""

    def __init__(self):
        super().__init__()
        self.release = threading.Event()
        self._lock = threading.Lock()
        self.active = 0
        self.peak = 0

    def predict(self, feeds):
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            assert self.release.wait(TIMEOUT), "test never released the ensemble"
            return super().predict(feeds)
        finally:
            with self._lock:
                self.active -= 1


def _wait_until(condition, what: str) -> None:
    deadline = time.monotonic() + TIMEOUT
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.01)


def _post(base: str) -> httpx.Response:
    files = {"derm_image": ("derm.jpg", image_bytes(), "image/jpeg")}
    return httpx.post(base + "/predict", data=VALID_FORM, files=files, timeout=TIMEOUT * 2)


def _timed_get(url: str) -> tuple[httpx.Response, float]:
    t0 = time.perf_counter()
    response = httpx.get(url, timeout=TIMEOUT)
    return response, (time.perf_counter() - t0) * 1000


def _app(ensemble, running: int, queued: int):
    settings = Settings(max_concurrent_predictions=running, max_queued_predictions=queued)
    return create_app(settings, loader=lambda: make_predictor(ensemble))


def test_at_most_n_run_at_once_and_five_parallel_requests_all_succeed():
    ensemble = BlockingEnsemble()
    app = _app(ensemble, running=2, queued=3)
    limiter = app.state.limiter
    with live_server(app) as base, ThreadPoolExecutor(5) as pool:
        futures = [pool.submit(_post, base) for _ in range(5)]
        _wait_until(lambda: ensemble.active == 2 and limiter.waiting == 3, "2 running and 3 queued")
        health, health_ms = _timed_get(base + "/health")
        ready, _ = _timed_get(base + "/ready")
        assert health.status_code == 200 and health_ms < 100
        assert ready.status_code == 200
        ensemble.release.set()
        statuses = [f.result().status_code for f in futures]
    assert statuses == [200] * 5
    assert ensemble.peak == 2


def test_health_and_ready_answer_fast_while_the_threadpool_is_saturated():
    # 40 = anyio's default threadpool size: every worker thread is busy with a blocked prediction
    ensemble = BlockingEnsemble()
    app = _app(ensemble, running=40, queued=0)
    with live_server(app) as base, ThreadPoolExecutor(40) as pool:
        futures = [pool.submit(_post, base) for _ in range(40)]
        _wait_until(lambda: ensemble.active == 40, "40 blocked predictions")
        health, health_ms = _timed_get(base + "/health")
        ready, ready_ms = _timed_get(base + "/ready")
        ensemble.release.set()
        statuses = [f.result().status_code for f in futures]
    assert health.status_code == 200 and health_ms < 100, f"/health took {health_ms:.0f} ms"
    assert ready.status_code == 200 and ready_ms < 100, f"/ready took {ready_ms:.0f} ms"
    assert statuses == [200] * 40


def test_queue_overflow_is_503_with_retry_after_and_ready_stays_200():
    ensemble = BlockingEnsemble()
    app = _app(ensemble, running=1, queued=1)
    limiter = app.state.limiter
    with live_server(app) as base, ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(_post, base) for _ in range(2)]
        _wait_until(lambda: ensemble.active == 1 and limiter.waiting == 1, "1 running and 1 queued")
        rejected = _post(base)
        ready, _ = _timed_get(base + "/ready")
        ensemble.release.set()
        statuses = [f.result().status_code for f in futures]
    assert rejected.status_code == 503
    assert rejected.headers["retry-after"] == RETRY_AFTER_BUSY
    assert "busy" in rejected.json()["detail"]
    assert ready.status_code == 200 and ready.json()["status"] == "ready"
    assert statuses == [200, 200]
    assert ensemble.peak == 1


def test_a_rejected_or_failed_request_frees_its_slot():
    ensemble = FakeEnsemble()
    app = _app(ensemble, running=1, queued=0)
    with live_server(app) as base:
        bad = httpx.post(
            base + "/predict",
            data={**VALID_FORM, "sex": "unknown"},
            files={"derm_image": ("d.jpg", image_bytes(), "image/jpeg")},
        )
        assert bad.status_code == 422
        assert [_post(base).status_code for _ in range(3)] == [200, 200, 200]
    assert (app.state.limiter.running, app.state.limiter.waiting) == (0, 0)
