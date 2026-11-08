"""Patient-grouped, class-stratified K-fold split (leakage-safe).

Uses only numpy/pandas/scikit-learn.

Reference only: data/folds.csv is the fixed assignment used by every reported run. Do not regenerate it;
a new split invalidates every OOF comparison in results/.
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

N_SPLITS, SEED = 5, 42


def make_folds(df: pd.DataFrame, n_splits: int = N_SPLITS, seed: int = SEED,
               verbose: bool = True) -> pd.DataFrame:
    """5-fold StratifiedGroupKFold: grouped by patient, stratified by 11-class label."""
    df = df.reset_index(drop=True).copy()
    df["fold"] = -1
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (_, va_idx) in enumerate(sgkf.split(df, y=df["diagnostic_class_11"],
                                               groups=df["patient_id"])):
        df.loc[va_idx, "fold"] = k

    # ---- Leakage guardrails -------------------------------------------------
    assert (df["fold"] >= 0).all(), "unassigned rows"
    assert df.groupby("patient_id")["fold"].nunique().max() == 1, "patient spans folds!"
    for k in range(n_splits):
        tr = set(df.loc[df.fold != k, "patient_id"])
        va = set(df.loc[df.fold == k, "patient_id"])
        assert tr.isdisjoint(va), f"patient leakage in fold {k}"

    # ---- Diagnostics ----------------------------------------------------------
    if verbose:
        print(pd.crosstab(df["diagnostic_class_11"], df["fold"], margins=True))
        print("Malignant rate per fold:",
              df.groupby("fold")["target_binary"].mean().round(3).to_dict())
        rare = df.groupby("diagnostic_class_11")["patient_id"].nunique()
        rare = rare[rare < n_splits]
        if len(rare):
            print(f"⚠ Classes with < {n_splits} patients (some folds will lack them): {rare.to_dict()}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", default="data/raw/train_metadata.csv")
    ap.add_argument("--out", default="data/folds_regenerated.csv")  # never overwrite data/folds.csv
    ap.add_argument("--n-splits", type=int, default=N_SPLITS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--force", action="store_true", help="overwrite an existing folds file")
    args = ap.parse_args()

    if not Path(args.metadata).is_file():
        raise SystemExit(f"Metadata file not found: {args.metadata}\n"
                         "Prepare MILK10k into data/raw/ (data/README.md) or pass --metadata.")
    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(f"{out} already exists; folds must stay fixed across experiments. "
                         "Pass --force to regenerate deliberately.")
    df = make_folds(pd.read_csv(args.metadata), args.n_splits, args.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    df[["isic_id", "patient_id", "fold"]].to_csv(out, index=False)
    print(f"Wrote {len(df)} rows → {out}")


if __name__ == "__main__":
    main()
