"""Prometheus metrics of the API (P1-OBS-01, P1-OBS-02), served at `GET /metrics`.

One registry per app (`ApiMetrics`), so tests can build many apps in one process. Label values are bounded:

- `route` is the matched route template (`/predict`, `/health`, ...), never the raw path; a path that matches no route
  is counted as `unmatched`, so random paths cannot create new series.
- `method` is one of the standard HTTP methods, anything else is `other`; `status` is the numeric status code.
- `stage` is one of the fixed `/predict` stages below.

No request ids, metadata values, diagnoses or file names appear in labels (medical data, unbounded values). The
prediction distribution is a histogram of P(malignant) without labels.
"""

from __future__ import annotations

from collections.abc import Callable

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, ProcessCollector

UNMATCHED = "unmatched"
METHODS = frozenset({"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"})
# Request latency: probes take milliseconds, one inference ~0.6 s, queued requests wait seconds (5.3 s measured for
# the slowest of 15 parallel requests, docs/serving_measurements.md §7/§9).
REQUEST_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 30.0)
# Stages of one prediction: decoding ~10 ms, preprocessing a few ms, the 15-member ensemble ~0.6 s.
STAGE_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.5, 5.0)
STAGES = ("decode", "preprocess", "inference")
P_MALIGNANT_BUCKETS = tuple(round(0.1 * i, 1) for i in range(1, 11))


def route_template(app, scope: dict) -> str:
    """The template of the route that matches the request (also for a 405 on a known path), else `unmatched`."""
    route = scope.get("route")
    if route is not None and getattr(route, "path", None):
        return route.path
    for candidate in app.router.routes:
        match, _ = candidate.matches(scope)
        if match.name != "NONE" and getattr(candidate, "path", None):  # Starlette's Match: NONE, PARTIAL, FULL
            return candidate.path
    return UNMATCHED


def method_label(method: str) -> str:
    return method if method in METHODS else "other"


class ApiMetrics:
    def __init__(
        self,
        running: Callable[[], float],
        waiting: Callable[[], float],
        ready: Callable[[], float],
        load_seconds: Callable[[], float],
    ):
        self.registry = CollectorRegistry()
        ProcessCollector(registry=self.registry)  # process_resident_memory_bytes, process_cpu_seconds_total, ...
        r = self.registry
        self.requests = Counter(
            "milk10k_http_requests",
            "HTTP requests by route template, method and status code.",
            ("route", "method", "status"),
            registry=r,
        )
        self.request_duration = Histogram(
            "milk10k_http_request_duration_seconds",
            "HTTP request latency (until the response headers are ready) by route template and method.",
            ("route", "method"),
            buckets=REQUEST_BUCKETS,
            registry=r,
        )
        self.rejected_busy = Counter(
            "milk10k_predictions_rejected_busy",
            "POST /predict requests rejected with 503 because the running slots and the queue were full.",
            registry=r,
        )
        self.stage_duration = Histogram(
            "milk10k_predict_stage_duration_seconds",
            "Duration of the stages of one prediction: decode (images), preprocess (tensors), inference (ensemble).",
            ("stage",),
            buckets=STAGE_BUCKETS,
            registry=r,
        )
        self.p_malignant = Histogram(
            "milk10k_prediction_p_malignant",
            "Distribution of the predicted P(malignant) of successful predictions.",
            buckets=P_MALIGNANT_BUCKETS,
            registry=r,
        )
        Gauge("milk10k_predictions_running", "Predictions running inference now.", registry=r).set_function(running)
        Gauge("milk10k_predictions_waiting", "Predictions waiting for a free slot now.", registry=r).set_function(
            waiting
        )
        Gauge(
            "milk10k_model_ready", "1 once the model package is loaded and verified, else 0.", registry=r
        ).set_function(ready)
        Gauge(
            "milk10k_model_load_seconds", "Seconds the model package took to load (0 until it is loaded).", registry=r
        ).set_function(load_seconds)
        self.model_info = Gauge(
            "milk10k_model_info",
            "The loaded model package (value 1): name and version from its manifest.",
            ("model", "model_version"),
            registry=r,
        )

    def observe_request(self, route: str, method: str, status: int, seconds: float) -> None:
        method = method_label(method)
        self.requests.labels(route, method, str(status)).inc()
        self.request_duration.labels(route, method).observe(seconds)
