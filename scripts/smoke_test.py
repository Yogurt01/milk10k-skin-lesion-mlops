#!/usr/bin/env python3
"""Smoke test of a running MILK10k API (P1-BSH-04): the check after a deploy. Python standard library only.

Usage (from anywhere; exit 0 = every check passed, 1 = a check failed, 2 = usage error):

    python3 scripts/smoke_test.py                                   # API_BASE_URL or http://127.0.0.1:8000
    python3 scripts/smoke_test.py --base-url http://127.0.0.1:8000 --ready-timeout 120

Checks, in order (it stops at the first failure and says which check failed and why):
  1. GET /ready: polled until 200 with status "ready" (connection errors and 503 "loading" are retried until
     --ready-timeout); a 503 with status "failed" stops at once (the model load failed; it will not recover).
  2. GET /health: 200 with {"status": "ok"}.
  3. POST /predict with a synthetic 64 x 64 PNG generated here (a colour gradient, never a lesion image) and fixed
     metadata: 200 and every response field present and consistent (11 probabilities summing to 1, diagnosis =
     the most likely class, p_malignant in [0, 1], clinical_image_used false, model and version as in /ready,
     non-empty disclaimer).
  4. GET /metrics: the milk10k_ metrics are exposed, the model is ready, and the /predict 200 above is counted.

The prediction for a synthetic image is meaningless; only the response shape is checked. Research model, not a
medical device.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time
import urllib.error
import urllib.request
import uuid
import zlib

FORM = {"age_approx": "60", "sex": "male", "anatomical_site_general": "head/neck", "fitzpatrick_type": "4"}
RESPONSE_FIELDS = (
    "p_malignant",
    "probabilities",
    "diagnosis",
    "clinical_image_used",
    "model",
    "model_version",
    "metadata_missing",
    "disclaimer",
)
N_CLASSES = 11
REQUIRED_METRICS = (
    "milk10k_model_ready",
    "milk10k_model_load_seconds",
    "milk10k_model_info",
    "milk10k_http_requests_total",
    "milk10k_http_request_duration_seconds_bucket",
    "milk10k_predict_stage_duration_seconds_bucket",
    "milk10k_prediction_p_malignant_bucket",
    "milk10k_predictions_rejected_busy_total",
    "milk10k_predictions_running",
    "milk10k_predictions_waiting",
)


class SmokeFailure(Exception):
    """A check failed; the message says which one and why."""


def synthetic_png(width: int = 64, height: int = 64) -> bytes:
    """An RGB colour-gradient PNG built with zlib (no image library, no dataset)."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    rows = b"".join(
        b"\x00" + b"".join(bytes((4 * x % 256, 4 * y % 256, 128)) for x in range(width)) for y in range(height)
    )
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


def multipart(form: dict[str, str], files: dict[str, tuple[str, bytes, str]]) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = []
    for key, value in form.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
    for name, (filename, data, content_type) in files.items():
        head = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        )
        parts.append(head.encode() + data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def request(url: str, timeout: float, data: bytes | None = None, content_type: str | None = None) -> tuple[int, bytes]:
    """(status, body). HTTP error statuses are returned; network errors raise OSError."""
    headers = {"Content-Type": content_type} if content_type else {}
    req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def as_json(status: int, body: bytes, what: str) -> dict:
    try:
        value = json.loads(body)
    except ValueError:
        raise SmokeFailure(f"{what}: HTTP {status}, body is not JSON: {body[:200]!r}") from None
    if not isinstance(value, dict):
        raise SmokeFailure(f"{what}: HTTP {status}, JSON body is not an object: {value!r}")
    return value


def check_ready(base: str, timeout: float, interval: float) -> dict:
    deadline = time.monotonic() + timeout
    last = "no answer yet"
    while True:
        try:
            status, body = request(base + "/ready", timeout=5)
        except OSError as error:  # not listening yet, refused, reset, timeout: retry
            last = f"{type(error).__name__}: {error}"
        else:
            try:
                payload = as_json(status, body, "GET /ready")
            except SmokeFailure as error:  # something else answers on that port: retry until the timeout
                last = str(error)
            else:
                if status == 200 and payload.get("status") == "ready":
                    return payload
                if payload.get("status") == "failed":
                    raise SmokeFailure(f"GET /ready: model load failed (HTTP {status}): {payload.get('error')}")
                last = f"HTTP {status} status={payload.get('status')!r}"
        if time.monotonic() >= deadline:
            raise SmokeFailure(f"GET /ready: not ready after {timeout:.0f} s (last: {last})")
        time.sleep(interval)


def check_health(base: str) -> None:
    status, body = request(base + "/health", timeout=5)
    payload = as_json(status, body, "GET /health")
    if status != 200 or payload.get("status") != "ok":
        raise SmokeFailure(f"GET /health: expected 200 {{'status': 'ok'}}, got HTTP {status} {payload}")


def check_predict(base: str, ready: dict, timeout: float) -> dict:
    data, content_type = multipart(FORM, {"derm_image": ("smoke.png", synthetic_png(), "image/png")})
    status, body = request(base + "/predict", timeout=timeout, data=data, content_type=content_type)
    payload = as_json(status, body, "POST /predict")
    if status != 200:
        raise SmokeFailure(f"POST /predict: expected 200, got HTTP {status}: {payload}")
    missing = [field for field in RESPONSE_FIELDS if field not in payload]
    if missing:
        raise SmokeFailure(f"POST /predict: response fields missing: {missing}")
    probs = payload["probabilities"]
    problems = []
    if not isinstance(probs, dict) or len(probs) != N_CLASSES:
        problems.append(f"probabilities: expected {N_CLASSES} classes, got {probs!r}")
    elif not all(isinstance(p, (int, float)) and 0 <= p <= 1 for p in probs.values()):
        problems.append(f"probabilities outside [0, 1]: {probs}")
    else:
        if not math.isclose(sum(probs.values()), 1.0, abs_tol=1e-3):
            problems.append(f"probabilities sum to {sum(probs.values())}, not 1")
        if payload["diagnosis"] != max(probs, key=probs.get):
            problems.append(f"diagnosis {payload['diagnosis']!r} is not the most likely class")
    p_mal = payload["p_malignant"]
    if not isinstance(p_mal, (int, float)) or not 0 <= p_mal <= 1:
        problems.append(f"p_malignant {p_mal!r} not in [0, 1]")
    if payload["clinical_image_used"] is not False:
        problems.append("clinical_image_used should be false (no clinical image sent)")
    if (payload["model"], payload["model_version"]) != (ready.get("model"), ready.get("model_version")):
        problems.append(f"model {payload['model']} {payload['model_version']} differs from /ready {ready}")
    if payload["metadata_missing"] != []:
        problems.append(f"metadata_missing should be [] (all metadata sent), got {payload['metadata_missing']}")
    if not isinstance(payload["disclaimer"], str) or not payload["disclaimer"].strip():
        problems.append("disclaimer is empty")
    if problems:
        raise SmokeFailure("POST /predict: " + "; ".join(problems))
    return payload


def check_metrics(base: str) -> int:
    status, body = request(base + "/metrics", timeout=10)
    if status != 200:
        raise SmokeFailure(f"GET /metrics: expected 200, got HTTP {status}")
    lines = [line for line in body.decode("utf-8", "replace").splitlines() if line and not line.startswith("#")]
    names = {line.split("{", 1)[0].split(" ", 1)[0] for line in lines}
    missing = [name for name in REQUIRED_METRICS if name not in names]
    if missing:
        raise SmokeFailure(f"GET /metrics: milk10k_ metrics missing: {missing}")
    ready = [line for line in lines if line.startswith("milk10k_model_ready ")]
    if not ready or float(ready[0].split()[1]) != 1.0:
        raise SmokeFailure(f"GET /metrics: milk10k_model_ready is not 1: {ready}")
    counted = [
        line
        for line in lines
        if line.startswith("milk10k_http_requests_total{") and 'route="/predict"' in line and 'status="200"' in line
    ]
    if not counted or float(counted[0].rsplit(" ", 1)[1]) < 1:
        raise SmokeFailure("GET /metrics: the /predict 200 is not counted in milk10k_http_requests_total")
    return sum(1 for name in names if name.startswith("milk10k_"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke test of a running MILK10k API (exit 0 = pass, 1 = fail).")
    parser.add_argument("--base-url", default=os.environ.get("API_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--ready-timeout", type=float, default=120, help="seconds to wait for /ready (default 120)")
    parser.add_argument("--interval", type=float, default=2, help="seconds between /ready polls (default 2)")
    parser.add_argument("--predict-timeout", type=float, default=60, help="seconds for /predict (default 60)")
    args = parser.parse_args(argv)
    base = args.base_url.rstrip("/")
    start = time.monotonic()
    step = "GET /ready"
    try:
        ready = check_ready(base, args.ready_timeout, args.interval)
        waited = time.monotonic() - start
        print(f"smoke: PASS /ready  {ready.get('model')} {ready.get('model_version')} (waited {waited:.1f} s)")
        step = "GET /health"
        check_health(base)
        print("smoke: PASS /health")
        step = "POST /predict"
        result = check_predict(base, ready, args.predict_timeout)
        print(f"smoke: PASS /predict  200, {len(result['probabilities'])} classes, fields complete (synthetic image)")
        step = "GET /metrics"
        count = check_metrics(base)
        print(f"smoke: PASS /metrics  {count} milk10k_ series names, model ready, /predict counted")
    except SmokeFailure as error:
        print(f"smoke: FAIL {error}", file=sys.stderr)
        return 1
    except OSError as error:  # network errors after /ready (connection refused or reset, timeout)
        print(f"smoke: FAIL {step}: {type(error).__name__}: {error} ({base})", file=sys.stderr)
        return 1
    print(f"smoke: OK  {base}  all checks passed in {time.monotonic() - start:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
