"""Load configs/final.yaml and resolve every path relative to the release root.

Environment overrides: MILK10K_DATA_DIR (prepared dataset root: train/, test/, *_metadata.csv),
MILK10K_CHECKPOINT_DIR (folder holding the 15 .pth files, e.g. a DVC/LFS checkout) and
MILK10K_RELEASE_ROOT (if the package is installed somewhere other than <release>/src/milk10k).
"""
import os
from pathlib import Path

import yaml

PATH_KEYS = ("data_dir", "source_dir", "folds", "checkpoint_dir", "results_dir", "sample_dir")


def release_root() -> Path:
    env = os.environ.get("MILK10K_RELEASE_ROOT")
    return Path(env).resolve() if env else Path(__file__).resolve().parents[2]


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Return the config dict with cfg['paths'] values as absolute Paths."""
    root = release_root()
    cfg_path = Path(path) if path else root / "configs" / "final.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    paths = dict(cfg["paths"])
    for key, env in (("data_dir", "MILK10K_DATA_DIR"), ("checkpoint_dir", "MILK10K_CHECKPOINT_DIR")):
        if os.environ.get(env):
            paths[key] = os.environ[env]
    for key in PATH_KEYS:
        p = Path(paths[key]).expanduser()
        paths[key] = p if p.is_absolute() else (root / p)
    cfg["paths"] = paths
    cfg["release_root"] = root
    cfg["config_file"] = cfg_path.resolve()
    return cfg


def checkpoint_names(cfg: dict) -> list[str]:
    """['P1a-kg-s42_fold0.pth', ...] in seed-major order (15 files for the final ensemble)."""
    e = cfg["ensemble"]
    return [f"{e['run_prefix']}-s{s}_fold{k}.pth" for s in e["seeds"] for k in e["folds"]]
