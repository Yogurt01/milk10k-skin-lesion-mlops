"""Collect the training-run artifacts into results/ and record their hashes.

    python scripts/export_results.py --workspace /path/to/training/workspace

The training workspace is where the runs wrote their outputs: pooled out-of-fold (OOF) predictions in `models/`,
the reference test predictions in `submissions/`, and the per-epoch logs and environment records of the Kaggle runs
in `models/kaggle/`. The script copies the files listed below, parses the training histories of the 15 released
checkpoints, rescores every experiment from its OOF file, and writes `results/provenance.json`: the source path and
sha256 of every artifact, for integrity checks. Once `results/` exists it never needs to be run again. It does not
train, predict or contact any service.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from milk10k.config import load_config
from milk10k.evaluation import (
    evaluate,
    load_oof,
    load_solution,
    sha256,
    truth_arrays,
)

# dest (relative to this folder) <- source (relative to the training workspace)
COPIES = {
    "results/oof/final-P1a-kg_oof.csv": "models/final-P1a-kg_oof.csv",
    "results/oof/final-P1a-kg-tta_oof.csv": "models/final-P1a-kg-tta_oof.csv",
    **{f"results/oof/P1a-kg-s{s}{v}_oof.csv": f"models/P1a-kg-s{s}{v}_oof.csv"
       for s in (42, 43, 44) for v in ("", "-noclin", "-tta")},
    **{f"results/oof/P1a-kg-noiseoff-s{s}_oof.csv": f"models/P1a-kg-noiseoff-s{s}_oof.csv" for s in (42, 43)},
    "results/predictions/final-P1a-kg.csv": "submissions/final-P1a-kg.csv",
    "results/predictions/final-P1a-kg-tta.csv": "submissions/final-P1a-kg-tta.csv",
}
GENERATED = ["results/training_history.csv", "results/experiments_scores.csv", "results/kaggle_runtime.json",
             "results/provenance.json"]
EXPORTS = sorted([*COPIES, *GENERATED])          # every file this script writes (used by the static check)

# Run descriptions and the decision taken for each run (docs/experiment_log.md); scores are recomputed below.
RUNS = {  # run: (phase, environment, kind, change, recorded decision)
    "baseline": ("1", "local RTX 3050 bf16", "train", "ResNet18 x2, 224 px (before deterministic seeding)", "reference only"),
    "baseline-s42": ("1", "local RTX 3050 bf16", "train", "deterministic seeding, seed 42", "local parent"),
    "baseline-s43": ("1", "local RTX 3050 bf16", "train", "seed 43", "noise floor"),
    "P1a-s42": ("1", "local RTX 3050 bf16", "train", "288 px, seed 42", "promising, not confirmed"),
    "P1a-s43": ("1", "local RTX 3050 bf16", "train", "288 px, seed 43", "promising, not confirmed"),
    "P1a-s42-evalfix": ("1", "local RTX 3050 bf16", "eval", "clinical crop-to-fill eval", "not adopted"),
    "P1a-s43-evalfix": ("1", "local RTX 3050 bf16", "eval", "clinical crop-to-fill eval", "not adopted"),
    "P1a-s42-noclin": ("1", "local RTX 3050 bf16", "eval", "P4b: clinical image removed", "ablation"),
    "P1a-s43-noclin": ("1", "local RTX 3050 bf16", "eval", "P4b: clinical image removed", "ablation"),
    "P1a-s42-tta": ("1", "local RTX 3050 bf16", "eval", "P5a: 4-view flip TTA", "unconfirmed"),
    "P1a-s43-tta": ("1", "local RTX 3050 bf16", "eval", "P5a: 4-view flip TTA", "unconfirmed"),
    "P1b-s42": ("1", "local RTX 3050 bf16", "train", "384 px", "drop"),
    "P1b-s43": ("1", "local RTX 3050 bf16", "train", "384 px", "drop"),
    "P3b-mel-s42": ("1", "local RTX 3050 bf16", "train", "MEL weight 3 in binary loss", "drop"),
    "P3b-mel-s43": ("1", "local RTX 3050 bf16", "train", "MEL weight 3 in binary loss", "drop"),
    "P3c": ("1", "local RTX 3050 bf16", "blend", "nested binary/class-head blend (OOF only)", "drop"),
    "P1a-kg-s42": ("1", "Kaggle T4 fp16", "train", "P1a recipe on Kaggle, seed 42", "reference (parent)"),
    "P1a-kg-s43": ("1", "Kaggle T4 fp16", "train", "P1a recipe on Kaggle, seed 43", "reference (parent)"),
    "P1a-kg-s44": ("1", "Kaggle T4 fp16", "train", "P1a recipe on Kaggle, seed 44", "reference (parent)"),
    **{f"P1a-kg-s{s}-noclin": ("1", "Kaggle T4 fp16", "eval", "P4b: clinical image removed", "ablation") for s in (42, 43, 44)},
    **{f"P1a-kg-s{s}-tta": ("1", "Kaggle T4 fp16", "eval", "P5a: 4-view flip TTA", "unconfirmed") for s in (42, 43, 44)},
    "P2a-s42": ("1", "Kaggle T4 fp16", "train", "ConvNeXt-T backbone", "drop (recipe-specific)"),
    "P2a-s43": ("1", "Kaggle T4 fp16", "train", "ConvNeXt-T backbone", "drop (recipe-specific)"),
    "P4a-s42": ("1", "Kaggle T4 fp16", "train", "2 spatial cross-attention blocks", "drop"),
    "P4a-s43": ("1", "Kaggle T4 fp16", "train", "2 spatial cross-attention blocks", "drop"),
    **{f"P4a-s{s}-{v}": ("1", "Kaggle T4 fp16", "eval", d, "eval only")
       for s in (42, 43) for v, d in (("noclin", "P4b on P4a"), ("tta", "P5a on P4a"))},
    "final-P1a-kg": ("1", "Kaggle T4 fp16", "ensemble", "3 seeds x 5 folds, held-out seed mean", "FINAL (primary)"),
    "final-P1a-kg-tta": ("1", "Kaggle T4 fp16", "ensemble", "final ensemble + 4-view TTA", "optional (TTA unconfirmed)"),
    "P1a-kg-noiseoff-s42": ("2", "Kaggle T4 fp16 (same runtime as P1a-kg)", "train", "GaussNoise off, seed 42", "negative result"),
    "P1a-kg-noiseoff-s43": ("2", "Kaggle T4 fp16 (same runtime as P1a-kg)", "train", "GaussNoise off, seed 43", "negative result"),
}
LOG_LINE = re.compile(r"^fold (\d) epoch (\d+) \| loss (\S+) \| score (\S+) \| pAUC80 (\S+) \| macroAUC11 (\S+) \| "
                      r"FairGap (\S+) \[(.*?)\] \| lr (\S+) \| (\d+)s", re.MULTILINE)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", default=str(ROOT.parent), help="training workspace holding the run outputs")
    ap.add_argument("--data-dir", default=None, help="prepared dataset with train_metadata.csv (default: config)")
    args = ap.parse_args()
    ws, cfg = Path(args.workspace), load_config()
    data_dir = Path(args.data_dir) if args.data_dir else Path(cfg["paths"]["data_dir"])
    if not (data_dir / "train_metadata.csv").is_file():
        data_dir = ws / "dataset/raw"            # the workspace's prepared dataset
    prov = {"purpose": "artifact integrity: source path and sha256 of every file in results/", "files": {}}
    try:
        prov["workspace_commit"] = subprocess.run(["git", "-C", str(ws), "rev-parse", "HEAD"], capture_output=True,
                                                  text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        prov["workspace_commit"] = None

    for dest, src in COPIES.items():
        d = ROOT / dest
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ws / src, d)
        prov["files"][dest] = {"source": src, "sha256": sha256(d)}

    # Per-epoch training history of the 15 released checkpoints (Kaggle training logs)
    rows, sources = [], []
    for s in cfg["ensemble"]["seeds"]:
        log = ws / f"models/kaggle/ka/logs/{cfg['ensemble']['run_prefix']}-s{s}.log"
        for m in LOG_LINE.finditer(log.read_text()):
            groups = dict(g.split("=") for g in m[8].split())
            rows.append({"run": f"P1a-kg-s{s}", "seed": s, "fold": int(m[1]), "epoch": int(m[2]), "loss": float(m[3]),
                         "score": float(m[4]), "pauc80": float(m[5]), "macro_auc": float(m[6]), "fair_gap": float(m[7]),
                         **{f"pauc80_{k}": float(v) for k, v in groups.items()}, "lr_end": float(m[9]),
                         "seconds": int(m[10])})
        sources.append({"source": str(log.relative_to(ws)), "sha256": sha256(log)})
    hist = pd.DataFrame(rows)
    assert len(hist) == 150, f"expected 3 seeds x 5 folds x 10 epochs, got {len(hist)}"
    hist.to_csv(ROOT / "results/training_history.csv", index=False)
    prov["files"]["results/training_history.csv"] = {"parsed_from": sources,
                                                     "sha256": sha256(ROOT / "results/training_history.csv")}

    # Every experiment rescored from its pooled OOF file with the unchanged metric
    sol = load_solution(data_dir, cfg["paths"]["folds"])
    t = truth_arrays(sol)
    table = []
    for run, (phase, env, kind, change, decision) in RUNS.items():
        path = ws / f"models/{run}_oof.csv"
        r = evaluate(t, sol, load_oof(path, sol))
        table.append({"run": run, "phase": phase, "environment": env, "kind": kind, "change": change,
                      "recorded_decision": decision, "score": r["score"], "pauc80_norm": r["pauc80_norm"],
                      "macro_auc_11": r["macro_auc_11"], "fair_gap": r["fair_gap"],
                      **{f"pauc80_{k}": v for k, v in r["pauc80_by_group"].items()},
                      "fold_mean": float(np.mean(r["fold_scores"])), "fold_std": float(np.std(r["fold_scores"])),
                      "source": f"models/{run}_oof.csv", "source_sha256": sha256(path)})
    pd.DataFrame(table).to_csv(ROOT / "results/experiments_scores.csv", index=False)
    prov["files"]["results/experiments_scores.csv"] = {"sha256": sha256(ROOT / "results/experiments_scores.csv")}

    # Run environments and timings: training of the released checkpoints, the reference inference, the noise-off runs
    runtime = {"training_runs": json.loads((ws / "models/kaggle/ka/runs.json").read_text()),
               "training_env": json.loads((ws / "models/kaggle/ka/env.json").read_text())}
    d = json.loads((ws / "models/kaggle/kd/final_report.json").read_text())
    runtime["reference_inference"] = {k: d[k] for k in ("minutes", "peak_cuda_gb", "test_models",
                                                        "oof_models_per_lesion", "strict_reproduction")}
    runtime["reference_inference_env"] = json.loads((ws / "models/kaggle/kd/env.json").read_text())
    kj = ws / "models/kaggle/kj-noiseoff/output/runs.json"
    runtime["noiseoff_runs"] = json.loads(kj.read_text()) if kj.is_file() else None
    (ROOT / "results/kaggle_runtime.json").write_text(json.dumps(runtime, indent=1) + "\n")
    prov["files"]["results/kaggle_runtime.json"] = {"sha256": sha256(ROOT / "results/kaggle_runtime.json")}
    (ROOT / "results/provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    print(f"collected {len(COPIES)} files, {len(hist)} history rows, {len(table)} experiment rows")


if __name__ == "__main__":
    main()
