"""Runtime settings, read from environment variables. Every variable is optional; empty or unset keeps the
default. The same list, with defaults, is in the repository's `.env.example`.

- `MODEL_DIR` (`artifacts/model_package/final-P1a-kg-onnx`): model package folder, contains `manifest.json`
- `MODEL_VERIFY_SHA256` (`true`): check every package file against the manifest at startup
- `ORT_INTRA_OP_THREADS` (`4`): ONNX Runtime threads per session
- `MAX_UPLOAD_MB` (`10`): largest accepted image upload
- `LOG_LEVEL` (`INFO`): DEBUG, INFO, WARNING or ERROR
- `OTEL_SERVICE_NAME` (`milk10k-api`): service name in traces
- `OTEL_EXPORTER_OTLP_ENDPOINT` (empty: tracing off): OTLP endpoint, e.g. `http://jaeger:4318`
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from pathlib import Path

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


def _bool(raw: str) -> bool:
    value = raw.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"not a boolean: {raw!r}")


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise ValueError(f"must be >= 1, got {value}")
    return value


def _log_level(raw: str) -> str:
    value = raw.strip().upper()
    if value not in LOG_LEVELS:
        raise ValueError(f"must be one of {', '.join(LOG_LEVELS)}, got {raw!r}")
    return value


@dataclass(frozen=True)
class Settings:
    model_dir: Path = Path("artifacts/model_package/final-P1a-kg-onnx")
    verify_sha256: bool = True
    ort_threads: int = 4
    max_upload_mb: int = 10
    log_level: str = "INFO"
    otel_service_name: str = "milk10k-api"
    otel_exporter_otlp_endpoint: str = ""


ENV_VARS: dict[str, tuple[str, Callable[[str], object]]] = {
    "model_dir": ("MODEL_DIR", Path),
    "verify_sha256": ("MODEL_VERIFY_SHA256", _bool),
    "ort_threads": ("ORT_INTRA_OP_THREADS", _positive_int),
    "max_upload_mb": ("MAX_UPLOAD_MB", _positive_int),
    "log_level": ("LOG_LEVEL", _log_level),
    "otel_service_name": ("OTEL_SERVICE_NAME", str),
    "otel_exporter_otlp_endpoint": ("OTEL_EXPORTER_OTLP_ENDPOINT", str),
}


def load_settings(environ: Mapping[str, str] | None = None) -> Settings:
    """Settings from `environ` (default: `os.environ`); empty or unset variables keep their defaults."""
    env = os.environ if environ is None else environ
    values: dict[str, object] = {}
    for field in fields(Settings):
        name, convert = ENV_VARS[field.name]
        raw = env.get(name, "")
        if raw.strip() == "":
            continue
        try:
            values[field.name] = convert(raw.strip())
        except ValueError as exc:
            raise ValueError(f"invalid {name}: {exc}") from exc
    return Settings(**values)
