# benchmarks/

`serving/`: the scripts behind `docs/serving_measurements.md` (size, RAM, latency, parity, smaller variants, ONNX export), plus `normalize_diff.py` (naive vs lookup-table normalisation; torch-free, runs in `serving/.venv`). Their `out/` folders are gitignored.

`serving/concurrency.py`: latency, throughput, server RSS and `/health` latency of a running API under parallel `/predict` load (`docs/serving_measurements.md` §7; real HTTP, needs `make serve` and the sample lesions).
