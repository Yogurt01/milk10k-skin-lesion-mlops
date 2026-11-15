"""P1-OBS-06 / P1-OBS-07: tracing. Span structure with an in-memory exporter; fail-soft with a collector that is down
or never answers; tracing off by default."""

import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import NoOpTracer

from milk10k_serving.api import create_app
from milk10k_serving.package import Ensemble, ModelPackage
from milk10k_serving.predictor import Predictor
from milk10k_serving.settings import Settings

from .fakes import CLASSES, VALID_FORM, image_bytes, make_predictor, make_spec
from .live_server import live_server
from .test_concurrency import BlockingEnsemble, _post, _wait_until

pytestmark = pytest.mark.api
MEMBERS = 3
STAGE_SPANS = {"queue_wait", "validate_input", "decode", "preprocess", "inference", "assemble_response"}


class FakeSession:
    """Stands in for an ONNX Runtime session: fixed outputs (probs_cls, prob_bin)."""

    def run(self, names, feeds):
        cls = np.full((1, len(CLASSES)), 1.0 / len(CLASSES), dtype=np.float32)
        return cls, np.array([[0.3]], dtype=np.float32)


def ensemble_predictor() -> Predictor:
    """A predictor with a real `Ensemble` of fake sessions, so the per-member spans are produced."""
    manifest = {
        "name": "fake-package",
        "version": "0.0-test",
        "classes": CLASSES,
        "ensemble": {"members": [{"file": f"m{i}.onnx"} for i in range(MEMBERS)], "tta": False},
    }
    ensemble = Ensemble([FakeSession() for _ in range(MEMBERS)], [f"m{i}.onnx" for i in range(MEMBERS)])
    return Predictor(ModelPackage(Path("fake"), manifest, make_spec(), ensemble))


@pytest.fixture
def traced():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    app = create_app(Settings(), loader=ensemble_predictor, tracer_provider=provider)
    with TestClient(app) as client:
        app.state.model.wait(5)
        yield client, exporter


def _predict(client):
    files = {
        "derm_image": ("derm.jpg", image_bytes(), "image/jpeg"),
        "clinical_image": ("clinical.jpg", image_bytes(seed=1), "image/jpeg"),
    }
    return client.post("/predict", data=VALID_FORM, files=files)


def test_one_predict_trace_has_the_stage_spans_under_the_server_span(traced):
    client, exporter = traced
    assert _predict(client).status_code == 200
    spans = exporter.get_finished_spans()
    by_name = {}
    for span in spans:
        by_name.setdefault(span.name, []).append(span)
    servers = [s for s in spans if s.parent is None]
    assert len(servers) == 1 and servers[0].name == "POST /predict"
    server = servers[0]
    assert server.attributes.get("http.route") == "/predict"
    assert {s.context.trace_id for s in spans} == {server.context.trace_id}  # one trace
    # every stage once, directly under the server span (also the ones that run in the threadpool)
    for name in STAGE_SPANS:
        assert len(by_name.get(name, [])) == 1, name
        assert by_name[name][0].parent.span_id == server.context.span_id, name
    inference = by_name["inference"][0]
    assert inference.attributes["milk10k.members"] == MEMBERS
    assert inference.attributes["milk10k.tta"] is False
    members = by_name["ensemble_member"]
    assert len(members) == MEMBERS
    assert all(m.parent.span_id == inference.context.span_id for m in members)
    assert sorted(m.attributes["milk10k.member_index"] for m in members) == list(range(MEMBERS))
    decode = by_name["decode"][0]
    assert decode.attributes["milk10k.clinical_image_provided"] is True
    assert decode.attributes["milk10k.clinical_image_used"] is True
    # stages in order of time: queue_wait, validate_input, decode, preprocess, inference, assemble_response
    order = sorted(STAGE_SPANS, key=lambda n: by_name[n][0].start_time)
    assert order == ["queue_wait", "validate_input", "decode", "preprocess", "inference", "assemble_response"]


def test_span_attributes_carry_no_request_data(traced):
    client, exporter = traced
    assert _predict(client).status_code == 200
    values = [str(v) for span in exporter.get_finished_spans() for v in span.attributes.values()]
    values += [str(e.attributes) for span in exporter.get_finished_spans() for e in span.events]
    text = " ".join(values)
    for leaked in (*VALID_FORM.values(), "derm.jpg", "clinical.jpg", "BCC", "SCCKA"):
        assert leaked not in text, leaked


def test_probes_and_scrapes_create_no_spans(traced):
    client, exporter = traced
    exporter.clear()
    for path in ("/health", "/ready", "/metrics"):
        assert client.get(path).status_code == 200
    assert exporter.get_finished_spans() == ()


def test_a_failed_validation_records_the_error_on_its_span(traced):
    client, exporter = traced
    exporter.clear()
    files = {"derm_image": ("derm.jpg", b"not an image", "image/jpeg")}
    assert client.post("/predict", data=VALID_FORM, files=files).status_code == 422
    decode = [s for s in exporter.get_finished_spans() if s.name == "decode"]
    assert len(decode) == 1 and not decode[0].status.is_ok
    assert not [s for s in exporter.get_finished_spans() if s.name == "inference"]


def test_rejected_request_ends_its_queue_wait_span():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    ensemble = BlockingEnsemble()
    settings = Settings(max_concurrent_predictions=1, max_queued_predictions=0)
    app = create_app(settings, loader=lambda: make_predictor(ensemble), tracer_provider=provider)
    with live_server(app) as base:
        worker = threading.Thread(target=_post, args=(base,))
        worker.start()
        _wait_until(lambda: ensemble.active == 1, "1 running")
        assert _post(base).status_code == 503
        ensemble.release.set()
        worker.join(10)
    waits = [s for s in exporter.get_finished_spans() if s.name == "queue_wait"]
    assert len(waits) == 2  # both ended (the in-memory exporter only receives ended spans)


def test_tracing_is_off_by_default():
    app = create_app(Settings(), loader=make_predictor)
    assert isinstance(app.state.tracer, NoOpTracer)
    assert not getattr(app, "_is_instrumented_by_opentelemetry", False)


# ---- fail-soft ----------------------------------------------------------------------------------------------------
def _closed_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]  # nothing listens on it once the socket is closed


@pytest.fixture
def silent_collector():
    """A TCP port that accepts connections (kernel backlog) but never answers: an export waits until its timeout."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(64)
    yield sock.getsockname()[1]
    sock.close()


@pytest.mark.parametrize("collector", ["down", "silent"])
def test_predict_is_unaffected_when_the_collector_is_down_or_silent(collector, silent_collector, monkeypatch):
    """Real OTLP exporter (BatchSpanProcessor, exports every 100 ms here) to a port that refuses connections or never
    answers: every /predict still returns 200 quickly, and the app still stops within the export timeout."""
    monkeypatch.setenv("OTEL_BSP_SCHEDULE_DELAY", "100")
    port = _closed_port() if collector == "down" else silent_collector
    app = create_app(Settings(otel_exporter_otlp_endpoint=f"http://127.0.0.1:{port}"), loader=make_predictor)
    assert not isinstance(app.state.tracer, NoOpTracer)
    with live_server(app) as base:
        latencies = []
        with ThreadPoolExecutor(2) as pool:
            for _ in range(5):
                t0 = time.perf_counter()
                statuses = list(pool.map(lambda _: _post(base).status_code, range(2)))
                latencies.append(time.perf_counter() - t0)
                assert statuses == [200, 200]
                time.sleep(0.2)  # let the processor try to export in between
        t_stop = time.perf_counter()
    stop_seconds = time.perf_counter() - t_stop
    assert max(latencies) < 2.0, latencies  # fake predictor: milliseconds; an export timeout would be 5 s
    assert stop_seconds < 15, stop_seconds  # shutdown flush is bounded by the export timeout (5 s)
