"""Latency, throughput, server RSS and /health latency of a running API under parallel /predict load (P1-API-05).

Sends the 3 intact sample lesions (cycled) through real HTTP at each concurrency level: `--requests` requests per
level, `c` in flight at a time, after 2 warm-up requests. Meanwhile it polls `GET /health` every 50 ms and samples the
server's RSS from /proc/<pid>/status. Torch-free; runs in `serving/.venv`. Writes `out/concurrency_<label>.json`.

Usage (from the repository root, sample images built by `training/sample/make_sample.py`; the server started with
`make serve`, settings from the environment, e.g. MAX_CONCURRENT_PREDICTIONS / MAX_QUEUED_PREDICTIONS):

    serving/.venv/bin/python benchmarks/serving/concurrency.py --pid <uvicorn pid> --levels 1 2 4 --label raw
"""

import argparse
import asyncio
import csv
import json
import os
import platform
import statistics
import time
from pathlib import Path

import httpx2

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "training" / "sample"
OUT = Path(__file__).resolve().parent / "out"


def sample_requests() -> list[tuple[dict, dict]]:
    """(form, files) for the intact sample lesions, mapped as in the parity tests (empty optional fields omitted)."""
    out = []
    with open(SAMPLE / "sample_metadata.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["scenario"] != "intact":
                continue
            form = {k: row[k].strip() for k in ("age_approx", "sex", "anatomical_site_general", "fitzpatrick_type")}
            files = {
                "derm_image": ("derm.jpg", (SAMPLE / row["derm_image_path"]).read_bytes(), "image/jpeg"),
                "clinical_image": ("clinical.jpg", (SAMPLE / row["clinical_image_path"]).read_bytes(), "image/jpeg"),
            }
            out.append(({k: v for k, v in form.items() if v}, files))
    return out


def rss_mib(pid: int) -> tuple[float, float]:
    """(VmRSS, VmHWM) of the server process in MiB."""
    fields = {}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("VmRSS", "VmHWM"):
            fields[key] = int(value.split()[0]) / 1024
    return fields["VmRSS"], fields["VmHWM"]


def pct(values: list[float], q: float) -> float:
    """Nearest-rank percentile."""
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, round(q / 100 * len(ordered) + 0.5) - 1))]


async def run_level(base: str, pid: int, payloads, concurrency: int, n_requests: int) -> dict:
    async with httpx2.AsyncClient(base_url=base, timeout=300) as client:
        for form, files in payloads[:2]:  # warm-up
            (await client.post("/predict", data=form, files=files)).raise_for_status()
        latencies, statuses, health_ms, rss = [], {}, [], []
        queue = list(range(n_requests))
        done = asyncio.Event()

        async def worker():
            while queue:
                i = queue.pop()
                form, files = payloads[i % len(payloads)]
                t0 = time.perf_counter()
                r = await client.post("/predict", data=form, files=files)
                latencies.append(time.perf_counter() - t0)
                statuses[r.status_code] = statuses.get(r.status_code, 0) + 1

        async def probe():
            while not done.is_set():
                t0 = time.perf_counter()
                await client.get("/health")
                health_ms.append((time.perf_counter() - t0) * 1000)
                rss.append(rss_mib(pid))
                await asyncio.sleep(0.05)

        prober = asyncio.create_task(probe())
        t0 = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        wall = time.perf_counter() - t0
        done.set()
        await prober
    return {
        "concurrency": concurrency,
        "requests": n_requests,
        "statuses": statuses,
        "latency_p50_s": round(statistics.median(latencies), 3),
        "latency_p95_s": round(pct(latencies, 95), 3),
        "wall_s": round(wall, 2),
        "throughput_rps": round(n_requests / wall, 2),
        "health_samples": len(health_ms),
        "health_p50_ms": round(statistics.median(health_ms), 1),
        "health_max_ms": round(max(health_ms), 1),
        "rss_max_mib": round(max(r[0] for r in rss)),
        "rss_peak_hwm_mib": round(max(r[1] for r in rss)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--pid", type=int, required=True, help="PID of the uvicorn server process (for RSS)")
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--requests", type=int, default=24, help="requests per level")
    ap.add_argument("--label", default="run")
    args = ap.parse_args()
    ready = httpx2.get(args.base_url + "/ready").json()
    payloads = sample_requests()
    env = {
        k: os.environ.get(k, "")
        for k in ("ORT_INTRA_OP_THREADS", "MAX_CONCURRENT_PREDICTIONS", "MAX_QUEUED_PREDICTIONS")
    }
    result = {
        "label": args.label,
        "model": ready.get("model"),
        "machine": {"cpu": platform.processor() or platform.machine(), "logical_cpus": os.cpu_count()},
        "env_of_this_shell": env,
        "levels": [asyncio.run(run_level(args.base_url, args.pid, payloads, c, args.requests)) for c in args.levels],
    }
    OUT.mkdir(exist_ok=True)
    path = OUT / f"concurrency_{args.label}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    for lv in result["levels"]:
        print(json.dumps(lv))
    print(f"-> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
