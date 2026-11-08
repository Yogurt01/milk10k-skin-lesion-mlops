"""Evaluation metric for MILK10k: one number that rewards three things.

    Score = 0.7 * pAUC80 + 0.3 * MacroAUC_11 - 0.1 * FairGap

- **Catching cancers at high sensitivity.** pAUC80 (`partial_auc_above_tpr`) is the area under the ROC curve restricted
  to true-positive rate >= 0.80, the region where at least 80% of malignant lesions are caught. It is divided by
  its maximum (0.2) so that it lies in [0, 1]: random ranking gives about 0.10, perfect ranking 1. It is computed by
  reflecting the ROC curve (labels 1 - y, scores -p), taking scikit-learn's low-FPR partial AUC with max_fpr = 0.2,
  and undoing scikit-learn's McClish standardisation.
- **The full differential diagnosis.** MacroAUC_11 (`macro_ovr_auc`) is the mean one-vs-rest ROC-AUC over the 11
  diagnostic classes, every class weighted equally. A class without both positives and negatives in the scored
  set is skipped.
- **Similar performance across skin tones.** FairGap (`fairness_gap`) is max - min pAUC80 over skin-tone groups
  built from `fitzpatrick_type` codes 1-2, 3-4 and 5-6. The code calls them light / medium / dark, but the direction of
  the source coding is unverified, so read them as numeric codes. A group counts only with >= 10 malignant and
  >= 10 benign lesions. Lesions without a code are left out of this term only. Fewer than two eligible groups give 0.

Malignant = MEL, BCC, SCCKA, MAL_OTH; AKIEC counts as benign. All three parts are rank-based, so a monotone
recalibration of the scores never changes the result. `score_components` returns every part, `score` the single
number. Uses only numpy, pandas and scikit-learn.
"""
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CLASSES = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF",
           "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]
MALIGNANT = {"MEL", "BCC", "SCCKA", "MAL_OTH"}

MIN_TPR = 0.80
W_BINARY, W_MULTICLASS, LAMBDA_FAIR = 0.7, 0.3, 0.1
FST_TO_GROUP = {1: "light", 2: "light", 3: "medium", 4: "medium", 5: "dark", 6: "dark"}
MIN_POS_PER_GROUP = 10
MIN_NEG_PER_GROUP = 10
PROB_SUM_TOL = 1e-3


class ParticipantVisibleError(Exception):
    """Invalid prediction file (missing/extra columns or IDs, NaN, out-of-range values, class sums != 1)."""


def partial_auc_above_tpr(y_true, y_score, min_tpr: float = MIN_TPR,
                          normalize: bool = True) -> float:
    """Partial ROC-AUC restricted to TPR >= min_tpr (ISIC 2024 definition).

    Returns a value in [0, 1 - min_tpr], or in [0, 1] if normalize=True.
    Returns NaN when y_true contains a single class.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if np.unique(y_true).size < 2:
        return float("nan")

    max_fpr = 1.0 - min_tpr
    # Reflect the ROC curve: TPR >= min_tpr  <->  FPR' <= max_fpr.
    v_gt = 1 - y_true
    v_pred = -y_score
    auc_mcclish = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    # Undo the McClish standardisation applied by scikit-learn.
    pauc = 0.5 * max_fpr ** 2 + (max_fpr - 0.5 * max_fpr ** 2) / 0.5 * (auc_mcclish - 0.5)
    return float(pauc / max_fpr) if normalize else float(pauc)


def macro_ovr_auc(y_class: np.ndarray, probs: np.ndarray) -> float:
    """Mean one-vs-rest AUC over the classes that have both positives and negatives."""
    aucs = []
    for k, c in enumerate(CLASSES):
        y_c = (y_class == c).astype(int)
        if 0 < y_c.sum() < len(y_c):
            aucs.append(roc_auc_score(y_c, probs[:, k]))
    if not aucs:
        raise ValueError("No class has both positive and negative examples.")
    return float(np.mean(aucs))


def fairness_gap(y_true: np.ndarray, y_score: np.ndarray, fst: pd.Series):
    """Max-min normalised pAUC80 across eligible Fitzpatrick groups."""
    groups = fst.map(FST_TO_GROUP).to_numpy()
    per_group: Dict[str, float] = {}
    for g in ("light", "medium", "dark"):
        m = groups == g
        n_pos = int(y_true[m].sum())
        n_neg = int(m.sum() - n_pos)
        if n_pos >= MIN_POS_PER_GROUP and n_neg >= MIN_NEG_PER_GROUP:
            per_group[g] = partial_auc_above_tpr(y_true[m], y_score[m])
    gap = max(per_group.values()) - min(per_group.values()) if len(per_group) >= 2 else 0.0
    return float(gap), per_group


def _validate(solution: pd.DataFrame, submission: pd.DataFrame, row_id: str) -> pd.DataFrame:
    required = [row_id, "target_binary"] + CLASSES
    missing = [c for c in required if c not in submission.columns]
    if missing:
        raise ParticipantVisibleError(f"Submission is missing columns: {missing}")
    extra = [c for c in submission.columns if c not in required]
    if extra:
        raise ParticipantVisibleError(f"Submission has unexpected columns: {extra}")
    if submission[row_id].duplicated().any():
        raise ParticipantVisibleError("Submission contains duplicate row ids.")

    sub = submission.set_index(row_id)
    missing_ids = solution.index.difference(sub.index)
    if len(missing_ids):
        raise ParticipantVisibleError(f"Submission is missing {len(missing_ids)} row ids.")
    sub = sub.loc[solution.index]  # align; extra ids are ignored

    values = sub[["target_binary"] + CLASSES]
    try:
        values = values.astype(float)
    except (TypeError, ValueError):
        raise ParticipantVisibleError("All probability columns must be numeric.")
    arr = values.to_numpy()
    if not np.isfinite(arr).all():
        raise ParticipantVisibleError("Submission contains NaN or infinite values.")
    if (arr < 0).any() or (arr > 1).any():
        raise ParticipantVisibleError("All probabilities must lie in [0, 1].")
    row_sums = values[CLASSES].sum(axis=1).to_numpy()
    if np.abs(row_sums - 1.0).max() > PROB_SUM_TOL:
        raise ParticipantVisibleError(
            f"The 11 class probabilities must sum to 1 (±{PROB_SUM_TOL}) in every row.")
    return values


def score_components(solution: pd.DataFrame, submission: pd.DataFrame,
                     row_id_column_name: str = "isic_id") -> Dict[str, object]:
    """Compute every metric component. Useful for local diagnostics."""
    sol = solution.drop(columns=["Usage"], errors="ignore").set_index(row_id_column_name)
    sub = _validate(sol, submission, row_id_column_name)

    y_bin = sol["target_binary"].to_numpy().astype(int)
    y_cls = sol["diagnostic_class_11"].astype(str).to_numpy()
    p_bin = sub["target_binary"].to_numpy()
    p_cls = sub[CLASSES].to_numpy()

    pauc = partial_auc_above_tpr(y_bin, p_bin)
    macro = macro_ovr_auc(y_cls, p_cls)
    gap, per_group = fairness_gap(y_bin, p_bin, sol["fitzpatrick_type"])
    total = W_BINARY * pauc + W_MULTICLASS * macro - LAMBDA_FAIR * gap
    return {"score": total, "pauc80_norm": pauc, "macro_auc_11": macro,
            "fair_gap": gap, "pauc80_by_group": per_group}


def score(solution: pd.DataFrame, submission: pd.DataFrame,
          row_id_column_name: str = "isic_id") -> float:
    """The single score (signature: solution, submission, row-id column).

    solution columns:   isic_id, target_binary, diagnostic_class_11, fitzpatrick_type[, Usage]
    submission columns: isic_id, target_binary, AKIEC, ..., VASC
    """
    return float(score_components(solution, submission, row_id_column_name)["score"])


if __name__ == "__main__":
    # Sanity checks on synthetic data.
    rng = np.random.default_rng(0)
    n = 1000
    cls = rng.choice(CLASSES, size=n, p=[.06, .45, .01, .11, .01, .01,
                                          .005, .09, .145, .1, .01])
    sol = pd.DataFrame({
        "isic_id": [f"ISIC_{i:07d}" for i in range(n)],
        "diagnostic_class_11": cls,
        "target_binary": np.isin(cls, list(MALIGNANT)).astype(int),
        "fitzpatrick_type": rng.choice([1, 2, 3, 4, 5, 6, np.nan], size=n),
    })

    onehot = (sol["diagnostic_class_11"].to_numpy()[:, None] == np.array(CLASSES)).astype(float)
    perfect = pd.DataFrame(onehot, columns=CLASSES)
    perfect.insert(0, "target_binary", sol["target_binary"].astype(float))
    perfect.insert(0, "isic_id", sol["isic_id"])
    s_perfect = score(sol, perfect)
    print(f"perfect : {s_perfect:.4f}")
    assert abs(s_perfect - 1.0) < 1e-9

    rand = rng.random((n, len(CLASSES)))
    rand /= rand.sum(1, keepdims=True)
    random_sub = pd.DataFrame(rand, columns=CLASSES)
    random_sub.insert(0, "target_binary", rng.random(n))
    random_sub.insert(0, "isic_id", sol["isic_id"])
    comps = score_components(sol, random_sub)
    print("random  :", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in comps.items()})
    assert 0.1 < comps["score"] < 0.35

    bad = perfect.copy()
    bad.loc[0, "MEL"] = 0.5  # row no longer sums to 1
    try:
        score(sol, bad)
        raise AssertionError("expected ParticipantVisibleError")
    except ParticipantVisibleError as e:
        print("invalid :", e)
    print("All metric sanity checks passed.")
