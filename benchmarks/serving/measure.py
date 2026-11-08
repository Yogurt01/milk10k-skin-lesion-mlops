"""P1-DEC-01 (size, RAM, latency) and P1-DEC-02 (parity) for one device.

Usage (from the repository root):
  MILK10K_DATA_DIR=... training/.venv/bin/python benchmarks/serving/measure.py cpu
  MILK10K_DATA_DIR=... training/.venv/bin/python benchmarks/serving/measure.py cuda
"""

import platform
import sys
import time

sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import (  # noqa: E402
    batches,
    config,
    inf,
    load_all,
    members,
    peak_rss_mb,
    predict_preloaded,
    prior,
    reference,
    rss_mb,
    sample_meta,
    save,
    timed,
    torch,
)

device_name = sys.argv[1]
device = torch.device(device_name)
gpu = device.type == "cuda"
R = {
    "device": device_name,
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_threads_default": torch.get_num_threads(),
}
if gpu:
    R["gpu"] = torch.cuda.get_device_name(0)

cfg = config(device_name)
_, amp = inf.resolve_device(cfg)
R["amp_fp16"] = amp
meta = sample_meta(cfg)
root = cfg["paths"]["sample_dir"]
intact = meta.scenario.eq("intact")

# ---- 1. Checkpoint size and parameters ---------------------------------------------------------------
mem = members(cfg)
sizes = [p.stat().st_size for _, _, p in mem]
R["checkpoints"] = {"files": len(sizes), "total_mb": sum(sizes) / 2**20, "per_file_mb": sizes[0] / 2**20}

# ---- 2. Preload all 15 members, RAM / VRAM ------------------------------------------------------------
R["rss_mb_after_imports"] = rss_mb()
if gpu:
    torch.cuda.init()
    torch.zeros(1, device=device)
    R["rss_mb_after_cuda_context"] = rss_mb()
t0 = time.perf_counter()
models = load_all(cfg, device)
R["load_15_s"] = time.perf_counter() - t0
R["rss_mb_after_load_15"] = rss_mb()
n_params = sum(p.numel() for p in models[0][2].parameters())
R["params_per_member_M"] = n_params / 1e6
if gpu:
    R["vram_mb_allocated_15"] = torch.cuda.memory_allocated() / 2**20

# ---- 3. Per-request latency: one intact lesion --------------------------------------------------------
one = meta[intact].head(1).reset_index(drop=True)
a = models[0][3]


def preprocess():
    clean, ok, _ = inf.sanitize_inputs(one, root)
    return clean, batches(clean[ok].reset_index(drop=True), root, a)


R["preprocess_1_lesion"] = timed(preprocess, 10)
clean, bs = preprocess()
df1 = clean.reset_index(drop=True)
lat = {}
if gpu:
    torch.cuda.reset_peak_memory_stats()
thread_opts = [None] if gpu else [None, 4, 2]
for th in thread_opts:
    if th:
        torch.set_num_threads(th)
    key = f"threads_{th or R['torch_threads_default']}"
    predict_preloaded(models[:1], bs, df1, device, amp, False)  # warm-up
    lat[key] = {
        "member_1_no_tta": timed(lambda: predict_preloaded(models[:1], bs, df1, device, amp, False), 10, gpu),
        "ensemble_15_no_tta": timed(lambda: predict_preloaded(models, bs, df1, device, amp, False), 5, gpu),
        "ensemble_15_tta4": timed(lambda: predict_preloaded(models, bs, df1, device, amp, True), 3, gpu),
    }
torch.set_num_threads(R["torch_threads_default"])
R["latency_1_lesion"] = lat
if gpu:
    R["vram_mb_peak_inference"] = torch.cuda.max_memory_allocated() / 2**20
R["rss_mb_after_inference"] = rss_mb()
R["peak_rss_mb"] = peak_rss_mb()

# Batch throughput: the 6 sample lesions in one batch
clean6, ok6, _ = inf.sanitize_inputs(meta, root)
df6 = clean6[ok6].reset_index(drop=True)
bs6 = batches(df6, root, a)
R["ensemble_15_no_tta_6_lesions"] = timed(lambda: predict_preloaded(models, bs6, df6, device, amp, False), 3, gpu)

# Preloaded == release path (same members, same transforms)?
pre6 = predict_preloaded(models, bs6, df6, device, amp, False)
models.clear()  # free the preloaded members before the release-path runs
if gpu:
    torch.cuda.empty_cache()

# ---- 4. Release path (loads all 15 checkpoints per call) and parity (P1-DEC-02) -------------------------
pri = prior(cfg)
par = {}
for tta in (False, True):
    t0 = time.perf_counter()
    sub, rep = inf.predict_lesions(meta, root, cfg, pri, tta=tta)
    wall = time.perf_counter() - t0
    ref = reference(cfg, tta)
    tol = cfg["inference"]["parity_tolerance"]
    p_int = inf.parity(sub[intact.to_numpy()], ref, tol)
    p_all = inf.parity(sub, ref, tol)
    s = sub.set_index("isic_id")
    diff_per_col = (
        (
            s.loc[sub.isic_id[intact.to_numpy()], inf.PROBS]
            - ref.set_index("isic_id").loc[sub.isic_id[intact.to_numpy()], inf.PROBS]
        )
        .abs()
        .max()
    )
    par["tta4" if tta else "no_tta"] = {
        "release_path_wall_s_6_lesions": wall,
        "intact_3": p_int,
        "all_6_incl_missing_clinical": p_all,
        "worst_column_intact": diff_per_col.idxmax(),
        "checks": rep["checks"],
        "sanitize_report": {k: v for k, v in rep.items() if k not in ("checks",)},
    }
    if not tta:
        rel = sub.set_index("isic_id").loc[df6.isic_id, inf.PROBS].to_numpy()
        R["preloaded_vs_release_path_max_abs_diff"] = float(
            abs(pre6.set_index("isic_id").loc[df6.isic_id, inf.PROBS].to_numpy() - rel).max()
        )
R["parity"] = par
R["peak_rss_mb_final"] = peak_rss_mb()
print(save(f"measure_{device_name}.json", R))
