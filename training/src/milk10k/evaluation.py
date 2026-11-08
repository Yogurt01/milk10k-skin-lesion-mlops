"""OOF scoring, paired bootstrap and hard-positive analysis (numpy/pandas/sklearn only, no torch).

`components` scores arbitrary (bootstrap) row subsets and is checked against `metrics.score_components`.
`tail_threshold` defines the hard positives used in notebooks 02 and 05 and in docs/experiment_log.md.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import (
    CLASSES,
    LAMBDA_FAIR,
    W_BINARY,
    W_MULTICLASS,
    _validate,
    fairness_gap,
    macro_ovr_auc,
    partial_auc_above_tpr,
    score_components,
)

SOLUTION_COLS = ["isic_id", "target_binary", "diagnostic_class_11", "fitzpatrick_type"]
COMPONENTS = ["score", "pauc80_norm", "macro_auc_11", "fair_gap"]
CODE_GROUPS = {"light": "codes 1-2", "medium": "codes 3-4", "dark": "codes 5-6"}  # scorer names -> numeric codes


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_solution(data_dir, folds_path) -> pd.DataFrame:
    sol = pd.read_csv(Path(data_dir) / "train_metadata.csv")[SOLUTION_COLS]
    folds = pd.read_csv(folds_path)[["isic_id", "fold"]]
    return sol.merge(folds, on="isic_id", validate="1:1")


def truth_arrays(sol: pd.DataFrame) -> dict:
    return {"y_bin": sol["target_binary"].to_numpy().astype(int),
            "y_cls": sol["diagnostic_class_11"].astype(str).to_numpy(),
            "fst": sol["fitzpatrick_type"].to_numpy(),
            "fold": sol["fold"].to_numpy()}


def load_oof(path, sol: pd.DataFrame) -> pd.DataFrame:
    """Read a pooled OOF CSV, apply the scorer's checks and align rows to `sol` order."""
    oof = pd.read_csv(path)
    return _validate(sol.set_index("isic_id"), oof, "isic_id").reset_index()


def pred_arrays(oof: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return oof["target_binary"].to_numpy(dtype=float), oof[CLASSES].to_numpy(dtype=float)


def components(t: dict, p_bin: np.ndarray, p_cls: np.ndarray, idx: np.ndarray) -> dict:
    """metrics.score_components on arrays, for rows `idx` (repeats allowed, as in a bootstrap)."""
    y_bin = t["y_bin"][idx]
    pauc = partial_auc_above_tpr(y_bin, p_bin[idx])
    macro = macro_ovr_auc(t["y_cls"][idx], p_cls[idx])
    gap, per_group = fairness_gap(y_bin, p_bin[idx], pd.Series(t["fst"][idx]))
    return {"score": W_BINARY * pauc + W_MULTICLASS * macro - LAMBDA_FAIR * gap, "pauc80_norm": pauc,
            "macro_auc_11": macro, "fair_gap": gap, "pauc80_by_group": per_group}


def evaluate(t: dict, sol: pd.DataFrame, oof: pd.DataFrame) -> dict:
    p = pred_arrays(oof)
    full = components(t, *p, np.arange(len(sol)))
    official = score_components(sol.drop(columns="fold"), oof)
    assert abs(full["score"] - official["score"]) < 1e-12, "array scoring diverged from metrics.py"
    folds = np.unique(t["fold"])
    fold_scores = [components(t, *p, np.flatnonzero(t["fold"] == f))["score"] for f in folds]
    return {**full, "fold_scores": fold_scores, "preds": p}


def paired_bootstrap(t: dict, a, b, n_boot: int = 1000, seed: int = 0) -> np.ndarray:
    """(n_boot, 4) deltas a - b of COMPONENTS, both scored on the same resampled lesions."""
    rng = np.random.default_rng(seed)
    n = len(t["y_bin"])
    out = np.empty((n_boot, len(COMPONENTS)))
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        ca, cb = components(t, *a, idx), components(t, *b, idx)
        out[i] = [ca[k] - cb[k] for k in COMPONENTS]
    return out


def bootstrap_mean_difference(t: dict, cand: list, ref: list, n_boot: int = 1000, seed: int = 0) -> np.ndarray:
    """(n_boot, 4) mean(components of cand runs) - mean(components of ref runs) on shared resamples."""
    rng = np.random.default_rng(seed)
    n = len(t["y_bin"])
    out = np.empty((n_boot, len(COMPONENTS)))
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        ca = [components(t, *p, idx) for p in cand]
        cr = [components(t, *p, idx) for p in ref]
        out[i] = [np.mean([c[k] for c in ca]) - np.mean([c[k] for c in cr]) for k in COMPONENTS]
    return out


def ci95(deltas: np.ndarray) -> dict:
    return {k: (float(np.percentile(deltas[:, i], 2.5)), float(np.percentile(deltas[:, i], 97.5)))
            for i, k in enumerate(COMPONENTS)}


def group_counts(t: dict) -> pd.DataFrame:
    """Malignant/benign counts per numeric skin-tone code group (eligibility needs >=10 of each)."""
    groups = pd.Series(t["fst"]).map({1: "codes 1-2", 2: "codes 1-2", 3: "codes 3-4", 4: "codes 3-4",
                                      5: "codes 5-6", 6: "codes 5-6"})
    df = pd.DataFrame({"group": groups, "y": t["y_bin"]}).dropna()
    out = df.groupby("group").y.agg(lesions="size", malignant="sum")
    out["benign"] = out.lesions - out.malignant
    out["eligible"] = (out.malignant >= 10) & (out.benign >= 10)
    return out


# --------------------------------------------------------------------------- #
# Hard positives: malignant lesions below the >=80%-sensitivity cut
# --------------------------------------------------------------------------- #
def tail_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Observed-positive cutoff that keeps at least 80% of malignant lesions at or above it (ties included)."""
    positives = np.sort(probabilities[labels == 1])
    if not len(positives):
        raise ValueError("No malignant examples for tail diagnostic")
    return float(positives[len(positives) // 5])


def hard_positive_flags(labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Malignant lesions scored below the >=80%-TPR cutoff."""
    return (labels == 1) & (probabilities < tail_threshold(labels, probabilities))


def benign_inversions(y_cls: np.ndarray, labels: np.ndarray, probabilities: np.ndarray,
                      hard: np.ndarray) -> pd.DataFrame:
    """Per benign class: strict pairs (benign scored above a hard malignant), and share of possible pairs."""
    hard_scores = np.sort(probabilities[hard])
    rows = []
    for c in sorted(set(y_cls[labels == 0])):
        b = probabilities[(labels == 0) & (y_cls == c)]
        pairs = int(np.searchsorted(hard_scores, b, side="left").sum())   # hard malignant strictly below each benign
        possible = len(b) * len(hard_scores)
        rows.append({"benign_class": c, "benign_lesions": len(b), "inversions": pairs,
                     "share_of_possible_pairs": pairs / possible if possible else np.nan})
    return pd.DataFrame(rows).sort_values("inversions", ascending=False).reset_index(drop=True)


__all__ = [
    "CLASSES",
    "CODE_GROUPS",
    "COMPONENTS",
    "SOLUTION_COLS",
    "benign_inversions",
    "bootstrap_mean_difference",
    "ci95",
    "components",
    "evaluate",
    "group_counts",
    "hard_positive_flags",
    "load_oof",
    "load_solution",
    "paired_bootstrap",
    "pred_arrays",
    "sha256",
    "tail_threshold",
    "truth_arrays",
]
