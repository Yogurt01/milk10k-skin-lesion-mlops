"""Build models/final-P1a-kg/manifest.json and tabular_encoding.json after placing the checkpoints.

    python scripts/build_manifest.py [--workspace /path/to/training/workspace]

Moves checkpoints left in models/final-P1a-kg/_download/ (the layout of a Kaggle output download) into
models/final-P1a-kg/. For each of the 15 checkpoints it records sha256, size and the checkpoint's own identity
(seed, fold, best epoch, validation metrics, recipe), and checks the recipe against configs/final.yaml. When the
training workspace is available, it also cross-checks each checkpoint against two records there:
- the reference inference run that produced results/predictions/ (run, seed, fold, best epoch, recipe);
- the training logs (best epoch per fold).

Without the workspace those cross-checks are skipped (recorded as null). Do not rerun it elsewhere if the
recorded cross-check results should be kept.
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from milk10k.config import checkpoint_names, load_config
from milk10k.dataset import SITES
from milk10k.evaluation import sha256
from milk10k.inference import RECIPE_KEYS


def best_epochs_from_log(log: Path) -> dict:
    """{fold: best epoch} = epoch with the highest validation score (first on ties), as train.py keeps it."""
    best = {}
    for m in re.finditer(r"^fold (\d) epoch (\d+) \| loss \S+ \| score (\S+)", log.read_text(), re.MULTILINE):
        k, e, s = int(m[1]), int(m[2]), float(m[3])
        if k not in best or s > best[k][1]:
            best[k] = (e, s)
    return {k: v[0] for k, v in best.items()}


def main() -> None:
    import torch
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace", default=str(ROOT.parent), help="training workspace (optional cross-checks)")
    args = ap.parse_args()
    cfg = load_config()
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    for p in sorted((ckpt_dir / "_download").rglob("*.pth")):
        shutil.move(str(p), ckpt_dir / p.name)
    names = checkpoint_names(cfg)
    missing = [n for n in names if not (ckpt_dir / n).is_file()]
    if missing:
        raise SystemExit(f"missing checkpoints: {missing}")

    repo = Path(args.workspace)
    d_report = repo / "models/kaggle/kd/final_report.json"
    d_records = {(r["run"], r["fold"]): r for r in json.loads(d_report.read_text())["checkpoints"]} if d_report.is_file() else {}
    logs = {f"{cfg['ensemble']['run_prefix']}-s{s}": repo / f"models/kaggle/ka/logs/{cfg['ensemble']['run_prefix']}-s{s}.log"
            for s in cfg["ensemble"]["seeds"]}
    log_best = {run: best_epochs_from_log(p) for run, p in logs.items() if p.is_file()}

    entries = []
    for name in names:
        path = ckpt_dir / name
        run, fold = name[:-len("_fold0.pth")], int(name[-5])
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        a = {**ckpt["args"], "xattn_layers": ckpt["args"].get("xattn_layers", 0),
             "mel_weight": ckpt["args"].get("mel_weight", 1.0)}
        recipe = {k: a[k] for k in RECIPE_KEYS}
        e = {"file": name, "sha256": sha256(path), "bytes": path.stat().st_size, "run": run, "seed": a["seed"],
             "fold": ckpt["fold"], "best_epoch": ckpt["epoch"],
             "val_metrics": {k: v for k, v in ckpt["metrics"].items()}, "recipe": recipe,
             "recipe_matches_config": all(recipe[k] == cfg["recipe"][k] for k in RECIPE_KEYS),
             "identity_ok": ckpt["fold"] == fold and f"-s{a['seed']}" in run}
        d = d_records.get((run, fold))
        e["reference_inference_record_match"] = None if d is None else (d["seed"] == a["seed"] and d["best_epoch"] == ckpt["epoch"]
                                                             and d["recipe"] == recipe)
        e["training_log_best_epoch_match"] = (log_best.get(run, {}).get(fold) == ckpt["epoch"]) if run in log_best else None
        entries.append(e)
    ok = all(e["identity_ok"] and e["recipe_matches_config"] and e["reference_inference_record_match"] is not False
             and e["training_log_best_epoch_match"] is not False for e in entries)
    manifest = {"model": cfg["name"], "source": "best-epoch checkpoints of the three Kaggle training runs (T4, fp16)",
                "members": len(entries), "all_checks_passed": ok,
                "cross_checks": {"reference_inference_record": str(d_report.relative_to(repo)) if d_records else None,
                                 "training_logs": sorted(str(p.relative_to(repo)) for p in logs.values() if p.is_file()),
                                 "paths_relative_to": "the training workspace"},
                "checkpoints": entries}
    (ckpt_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    encoding = {"description": "Fixed tabular encoding shared by every fold and seed (milk10k.dataset.encode_tabular). "
                               "Nothing is fitted, so there is no per-fold preprocessor to store.",
                "site": {"vocabulary": SITES, "index": "1 + position in vocabulary; 0 = missing or unseen"},
                "sex": {"male": 1, "female": 2, "other_or_missing": 0},
                "fitzpatrick_type": {"values": [1, 2, 3, 4, 5, 6], "missing": 0,
                                     "note": "skin_tone_class + 1; direction of the source coding unverified"},
                "age": {"transform": "age_approx / 100", "missing": "0 plus age_missing = 1"},
                "embedding_sizes": {"site": len(SITES) + 1, "sex": 3, "fitzpatrick_type": 7}}
    (ckpt_dir / "tabular_encoding.json").write_text(json.dumps(encoding, indent=1) + "\n")
    print(f"manifest: {len(entries)} checkpoints, all_checks_passed={ok}")
    if not ok:
        raise SystemExit("manifest checks failed; inspect models/final-P1a-kg/manifest.json")


if __name__ == "__main__":
    main()
