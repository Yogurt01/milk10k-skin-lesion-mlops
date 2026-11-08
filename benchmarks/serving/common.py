"""Shared helpers for the serving measurements (docs/serving_measurements.md).

Read-only use of training/; outputs go to ./out/ (gitignored).
"""

import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from milk10k import inference as inf  # noqa: E402
from milk10k.config import load_config  # noqa: E402
from milk10k.dataset import MILKDataset, clinical_transforms, derm_transforms  # noqa: E402
from milk10k.train import AMP_DTYPES, make_loader  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)


def status_mb(key: str) -> float:
    for line in open("/proc/self/status"):
        if line.startswith(key + ":"):
            return int(line.split()[1]) / 1024
    raise KeyError(key)


def rss_mb() -> float:
    return status_mb("VmRSS")


def peak_rss_mb() -> float:
    return status_mb("VmHWM")


def config(device: str) -> dict:
    cfg = load_config()
    cfg["inference"]["device"] = device
    return cfg


def members(cfg: dict) -> list[tuple[int, int, Path]]:
    e, d = cfg["ensemble"], Path(cfg["paths"]["checkpoint_dir"])
    return [(s, k, d / f"{e['run_prefix']}-s{s}_fold{k}.pth") for s in e["seeds"] for k in e["folds"]]


def sample_meta(cfg: dict) -> pd.DataFrame:
    return pd.read_csv(cfg["paths"]["sample_dir"] / "sample_metadata.csv")


def prior(cfg: dict) -> pd.Series:
    return pd.read_csv(cfg["paths"]["data_dir"] / "sample_submission.csv").iloc[0][inf.PROBS]


def reference(cfg: dict, tta: bool = False) -> pd.DataFrame:
    name = "final-P1a-kg-tta.csv" if tta else "final-P1a-kg.csv"
    return pd.read_csv(cfg["paths"]["results_dir"] / "predictions" / name)


def load_all(cfg: dict, device, subset=None) -> list:
    out = []
    for s, k, p in members(cfg):
        if subset is None or (s, k) in subset:
            m, a = inf.load_member(p, s, k, cfg, device)
            m.eval()
            out.append((s, k, m, a))
    return out


def batches(df: pd.DataFrame, root, a: dict, workers: int = 0) -> list:
    """Evaluation-time preprocessing (same transforms as inference.eval_loader), materialised once."""
    s = a["img_size"]
    ds = MILKDataset(df, str(root), derm_transforms(s, train=False), clinical_transforms(s, train=False), clin_size=s)
    return list(make_loader(ds, a["batch_size"], workers, a["seed"]))


def predict_preloaded(models: list, bs: list, df: pd.DataFrame, device, amp: bool, tta: bool) -> pd.DataFrame:
    views = inf.TTA4 if tta else inf.IDENTITY
    frames = [inf.predict_views(m, bs, df, device, amp, views, AMP_DTYPES[a["amp_dtype"]]) for _, _, m, a in models]
    return inf.mean_predictions(frames, df.isic_id)


def timed(fn, n: int, sync: bool = False) -> dict:
    ts = []
    for _ in range(n):
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return {"n": n, "median_s": statistics.median(ts), "min_s": min(ts), "max_s": max(ts)}


def save(name: str, obj) -> Path:
    p = OUT / name
    p.write_text(json.dumps(obj, indent=2, default=str))
    return p
