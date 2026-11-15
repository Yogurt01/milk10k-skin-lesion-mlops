"""Tracing with OpenTelemetry (P1-OBS-06, P1-OBS-07): OTLP over HTTP to Jaeger, FastAPI auto-instrumentation and
manual spans for the stages of `/predict`.

- **On/off:** on when `OTEL_EXPORTER_OTLP_ENDPOINT` is set (e.g. `http://jaeger:4318`; `/v1/traces` is appended),
  off when it is empty (the default): then no span is recorded or exported.
- **Fail-soft:** spans are exported in the background by a `BatchSpanProcessor` (bounded queue; spans are dropped
  when it is full). A slow, down or unreachable collector costs a log line, never a failed or slower `/predict`. A
  failure while setting tracing up is logged and leaves tracing off.
- **Spans:** one server span per request (FastAPI instrumentation; `/health`, `/ready` and `/metrics` are excluded
  so probes and scrapes create no traces) with children `queue_wait`, `validate_input`, `decode`, `preprocess`,
  `inference` (one `ensemble_member` per member inside) and `assemble_response`.
- **Attributes:** counts and flags only (`milk10k.members`, `milk10k.tta`, `milk10k.member_index`, whether a clinical
  image was sent and used). Never image bytes, metadata values, diagnoses or file names.
"""

from __future__ import annotations

import logging

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from . import __version__
from .settings import Settings

logger = logging.getLogger(__name__)
ATTRIBUTE_PREFIX = "milk10k."
# Probes and scrapes are not traced (regular expressions searched in the full URL).
EXCLUDED_URLS = "/health$,/ready$,/metrics$"
# Seconds one export request may take before it is abandoned (background thread; also bounds the flush at shutdown).
EXPORT_TIMEOUT_S = 5


def build_tracer_provider(settings: Settings) -> TracerProvider | None:
    """A provider that exports to `settings.otel_exporter_otlp_endpoint`, or None when tracing is off or cannot be set
    up (logged)."""
    base = settings.otel_exporter_otlp_endpoint.strip()
    if not base:
        return None
    try:
        resource = Resource.create({SERVICE_NAME: settings.otel_service_name, SERVICE_VERSION: __version__})
        provider = TracerProvider(resource=resource, shutdown_on_exit=False)  # the app's lifespan shuts it down
        exporter = OTLPSpanExporter(endpoint=base.rstrip("/") + "/v1/traces", timeout=EXPORT_TIMEOUT_S)
        provider.add_span_processor(BatchSpanProcessor(exporter))
    except Exception:
        logger.exception("tracing could not be set up; tracing is off")
        return None
    logger.info("tracing on: OTLP/HTTP to %s, service %s", base, settings.otel_service_name)
    return provider


def tracer_for(provider: TracerProvider | None) -> trace.Tracer:
    """The tracer for manual spans: from `provider`, or a no-op tracer when tracing is off."""
    if provider is None:
        return trace.NoOpTracer()
    return provider.get_tracer("milk10k_serving", __version__)


def instrument(app, provider: TracerProvider) -> None:
    """Server span per request (no ASGI send/receive sub-spans; probes and scrapes excluded)."""
    FastAPIInstrumentor.instrument_app(
        app, tracer_provider=provider, excluded_urls=EXCLUDED_URLS, exclude_spans=["receive", "send"]
    )


def span_attributes(attributes: dict) -> dict:
    return {ATTRIBUTE_PREFIX + key: value for key, value in attributes.items()}
