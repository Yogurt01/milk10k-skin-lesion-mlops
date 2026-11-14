# benchmarks/

`serving/`: the scripts behind `docs/serving_measurements.md` (size, RAM, latency, parity, smaller variants, ONNX export), plus `normalize_diff.py` (naive vs lookup-table normalisation; torch-free, runs in `serving/.venv`). Their `out/` folders are gitignored.
