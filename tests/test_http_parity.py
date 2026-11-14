"""Parity through real HTTP: the API, served by uvicorn with the real model package, reproduces the stored reference
predictions (`final-P1a-kg.csv`) within 0.002.

- P1-TST-04: the 6 sample lesions (3 intact compared with the reference; 3 without a clinical image must be valid).
- P1-TST-08: all 479 reference test lesions with their stored metadata (needs the prepared MILK10k dataset via
  MILK10K_DATA_DIR; skipped without it, also under REQUIRE_MODEL_PACKAGE=1, because CI never needs the full set).
  Writes a summary to reports/parity_full.json (gitignored).

Request mapping (same as the in-process parity test): dermoscopic image bytes; the clinical image if its path is set
and the file exists; the four metadata fields as stored, empty ones omitted (the API encodes them as missing).
"""

import csv
import json
import math
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx2
import pytest

from milk10k_serving.api import create_app
from milk10k_serving.predictor import Predictor
from milk10k_serving.settings import Settings

from .conftest import PARITY_TOLERANCE, ROOT
from .live_server import live_server

pytestmark = [pytest.mark.parity, pytest.mark.slow]
METADATA = ("age_approx", "sex", "anatomical_site_general", "fitzpatrick_type")
REPORT = ROOT / "reports" / "parity_full.json"


@pytest.fixture(scope="module")
def base_url(loaded_package, model_dir):
    """Real uvicorn server on the already-loaded package (one model in memory), default concurrency settings."""
    app = create_app(Settings(model_dir=model_dir), loader=lambda: Predictor(loaded_package))
    with live_server(app) as base:
        yield base


def lesion_request(row: dict, derm: bytes | None, clinical: bytes | None) -> tuple[dict, dict]:
    form = {key: str(row[key]).strip() for key in METADATA if str(row[key] or "").strip()}
    files = {"derm_image": ("derm.jpg", derm or b"", "image/jpeg")}
    if clinical:
        files["clinical_image"] = ("clinical.jpg", clinical, "image/jpeg")
    return form, files


def post(client: httpx2.Client, form: dict, files: dict) -> dict:
    response = client.post("/predict", data=form, files=files)
    assert response.status_code == 200, response.text
    return response.json()


def as_row(body: dict) -> dict[str, float]:
    """The response in the reference CSV's columns: target_binary + the 11 classes."""
    return {"target_binary": body["p_malignant"], **body["probabilities"]}


def test_sample_lesions_through_http_match_the_reference(base_url, sample_lesions, reference):
    worst, checked = 0.0, 0
    with httpx2.Client(base_url=base_url, timeout=120) as client:
        for row in sample_lesions:
            body = post(client, *lesion_request(row, row["derm_image_bytes"], row["clinical_image_bytes"]))
            values = list(as_row(body).values())
            assert all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in values)
            assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-9)
            if row["scenario"] == "intact":
                ref = reference[row["isic_id"]]
                got = as_row(body)
                assert set(got) == set(ref)
                worst = max(worst, max(abs(got[k] - ref[k]) for k in ref))
                checked += 1
            else:
                assert body["clinical_image_used"] is False
    print(f"HTTP parity: max |diff| over {checked} intact lesions x 12 columns: {worst:.2e}")
    assert checked == 3
    assert worst <= PARITY_TOLERANCE


@pytest.fixture(scope="module")
def dataset_dir() -> Path:
    path = Path(os.environ.get("MILK10K_DATA_DIR", ""))
    if not os.environ.get("MILK10K_DATA_DIR") or not (path / "test_metadata.csv").is_file():
        pytest.skip("prepared MILK10k dataset not available (set MILK10K_DATA_DIR)")
    return path


def test_all_479_reference_lesions_through_http(base_url, dataset_dir, reference, loaded_package):
    with open(dataset_dir / "test_metadata.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(reference) == 479 and {r["isic_id"] for r in rows} == set(reference)

    def one(row):
        clinical = dataset_dir / row["clinical_image_path"] if row["clinical_image_path"].strip() else None
        derm = (dataset_dir / row["derm_image_path"]).read_bytes()
        form, files = lesion_request(row, derm, clinical.read_bytes() if clinical and clinical.is_file() else None)
        with httpx2.Client(base_url=base_url, timeout=300) as client:
            return row["isic_id"], post(client, form, files)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(2) as pool:  # = MAX_CONCURRENT_PREDICTIONS default: nothing waits in the queue
        bodies = dict(pool.map(one, rows))
    seconds = time.perf_counter() - t0

    columns = list(next(iter(reference.values())))
    col_max = dict.fromkeys(columns, 0.0)
    lesion_max = {}
    for isic_id, body in bodies.items():
        got, ref = as_row(body), reference[isic_id]
        assert set(got) == set(ref)
        diffs = {k: abs(got[k] - ref[k]) for k in ref}
        lesion_max[isic_id] = max(diffs.values())
        for k, d in diffs.items():
            col_max[k] = max(col_max[k], d)
    missing_site = sorted(i for i, b in bodies.items() if "anatomical_site_general" in b["metadata_missing"])
    within = sum(d <= PARITY_TOLERANCE for d in lesion_max.values())
    summary = {
        "lesions": len(bodies),
        "within_tolerance": within,
        "tolerance": PARITY_TOLERANCE,
        "max_abs_diff": max(lesion_max.values()),
        "max_abs_diff_per_column": col_max,
        "worst_lesion": max(lesion_max, key=lesion_max.get),
        "lesions_with_metadata_missing": {
            i: bodies[i]["metadata_missing"] for i in bodies if bodies[i]["metadata_missing"]
        },
        "clinical_image_used": sum(b["clinical_image_used"] for b in bodies.values()),
        "seconds_total": round(seconds, 1),
        "client_threads": 2,
        "server": "uvicorn in-process thread, real HTTP on 127.0.0.1, default settings",
        "model": f"{loaded_package.name} {loaded_package.version}",
        "python": platform.python_version(),
        "machine": {"cpu": platform.processor() or platform.machine(), "logical_cpus": os.cpu_count()},
    }
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"479-lesion HTTP parity: {within}/{len(bodies)} within {PARITY_TOLERANCE}, max |diff| "
        f"{summary['max_abs_diff']:.3e} ({summary['worst_lesion']}), {seconds:.0f} s; empty site: {len(missing_site)}"
    )
    assert len(missing_site) == 6
    assert within == len(bodies) == 479
