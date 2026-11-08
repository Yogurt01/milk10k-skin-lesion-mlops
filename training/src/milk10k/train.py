"""Training loop of the MILK10k fusion model.

    python -m milk10k.train --fold all --epochs 10 --img-size 288 --batch-size 32 --amp-dtype fp16 --seed 42
    # or from Python / notebooks:  milk10k.train.train_from_config(cfg, seed=42)   (configs/final.yaml recipe)

The CLI defaults are those of the first baseline (224 px, 5 epochs, bf16); the released recipe is in configs/final.yaml.

Outputs:
    models/baseline_fold{k}.pth        best checkpoint per fold (by validation composite score)
    models/baseline_fold{k}_oof.csv    validation predictions of that fold's best epoch
    models/baseline_oof.csv            (--fold all) all folds concatenated: one row per train lesion
    submissions/baseline_sub.csv       test predictions: the single fold's model, or the mean of all folds
"""
import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import MILKDataset, clinical_transforms, derm_transforms, make_sqrt_balanced_sampler
from .metrics import CLASSES, _validate, score_components
from .models import FocalLoss, MILKFusionNet

SOLUTION_COLS = ["isic_id", "target_binary", "diagnostic_class_11", "fitzpatrick_type"]
MEL_IDX = CLASSES.index("MEL")
AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", default="0", help="fold index, or 'all' for every fold in --folds")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--backbone", default="resnet18.a1_in1k", help="timm name, used for both streams")
    ap.add_argument("--d", type=int, default=256, help="fusion width (divisible by 8)")
    ap.add_argument("--lr", type=float, default=3e-4, help="backbone LR; heads/fusion use 10x")
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--focal-gamma", type=float, default=2.0, help="0 = plain cross-entropy")
    ap.add_argument("--mel-weight", type=float, default=1.0, help="weight of MEL lesions in the binary loss; 1 = off")
    ap.add_argument("--xattn-layers", type=int, default=0,
                    help="P4a: derm↔clinical cross-attention blocks on feature-map tokens; 0 = pooled fusion")
    ap.add_argument("--amp-dtype", choices=list(AMP_DTYPES), default="bf16",
                    help="CUDA autocast dtype; fp16 adds a GradScaler (GPUs without bf16, e.g. T4)")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--folds", default="data/folds.csv")
    ap.add_argument("--out", default="models/baseline_fold{fold}.pth", help="checkpoint path template")
    ap.add_argument("--sub", default="submissions/baseline_sub.csv")
    return ap.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)  # warn, don't crash, on non-det ops


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker from its torch seed (= loader generator's base seed + worker_id).

    Albumentations 2.x pipelines own their RNG and ignore global seeds, so reseed them too.
    """
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    info = torch.utils.data.get_worker_info()
    ds = info.dataset if info is not None else None
    for tf in (getattr(ds, "derm_tf", None), getattr(ds, "clin_tf", None)):
        if tf is not None:
            tf.set_random_seed(seed)


def make_loader(ds, batch_size: int, num_workers: int, seed: int, sampler=None,
                train: bool = False) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, shuffle=False, drop_last=train,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=num_workers > 0, worker_init_fn=seed_worker, generator=g)


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def warmup_cosine(total_steps: int, warmup_frac: float = 0.05):
    warmup = max(1, int(total_steps * warmup_frac))

    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))
    return f


def fmt_metrics(m: dict) -> str:
    groups = " ".join(f"{g}={v:.3f}" for g, v in m["pauc80_by_group"].items())
    return (f"score {m['score']:.4f} | pAUC80 {m['pauc80_norm']:.4f} | macroAUC11 {m['macro_auc_11']:.4f} | "
            f"FairGap {m['fair_gap']:.4f} [{groups}]")


@torch.inference_mode()
def predict(model, loader, df: pd.DataFrame, device, amp: bool, dtype=torch.bfloat16) -> pd.DataFrame:
    """Submission-format predictions (isic_id, target_binary, *CLASSES) for the rows of `df`."""
    model.eval()
    p_cls, p_bin = [], []
    for batch in tqdm(loader, desc="predict", leave=False):
        batch = to_device(batch, device)
        with torch.autocast(device.type, dtype=dtype, enabled=amp):
            logits_cls, logit_bin = model.forward_batch(batch)
        p_cls.append(torch.softmax(logits_cls.float(), dim=1).cpu())
        p_bin.append(torch.sigmoid(logit_bin.float()).cpu())
    probs = torch.cat(p_cls).double().numpy()
    probs /= probs.sum(axis=1, keepdims=True)
    sub = pd.DataFrame(probs, columns=CLASSES)
    sub.insert(0, "target_binary", torch.cat(p_bin).double().numpy())
    sub.insert(0, "isic_id", df["isic_id"].to_numpy())
    return sub


def train_fold(args, k: int, train: pd.DataFrame, test: pd.DataFrame, device, amp: bool):
    """Train on folds != k, keep the best epoch by validation score.

    Returns (best-epoch OOF predictions, test predictions of the best checkpoint, best metrics).
    """
    seed_everything(args.seed)  # same seed per fold: `--fold k` reproduces fold k of `--fold all`
    root = Path(args.data_root)
    out = Path(args.out.format(fold=k))
    oof_path = out.with_name(f"{out.stem}_oof.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    tr = train[train.fold != k].reset_index(drop=True)
    va = train[train.fold == k].reset_index(drop=True)
    print(f"\n=== fold {k}: train {len(tr)} / val {len(va)} | test {len(test)} | device={device} ===")

    s = args.img_size
    ds_tr = MILKDataset(tr, str(root), derm_transforms(s, train=True), clinical_transforms(s, train=True),
                        clin_size=s, train=True)
    eval_ds = lambda df: MILKDataset(df, str(root), derm_transforms(s, train=False),  # noqa: E731
                                     clinical_transforms(s, train=False), clin_size=s)
    dl_tr = make_loader(ds_tr, args.batch_size, args.num_workers, args.seed,
                        sampler=make_sqrt_balanced_sampler(tr.diagnostic_class_11), train=True)
    dl_va = make_loader(eval_ds(va), args.batch_size, args.num_workers, args.seed)

    # ---- Model / optimiser ---------------------------------------------------------
    model = MILKFusionNet(derm_name=args.backbone, clin_name=args.backbone, d=args.d,
                          pretrained=not args.no_pretrained, xattn_layers=args.xattn_layers).to(device)
    backbone = [p for n, p in model.named_parameters() if n.startswith(("derm.", "clin."))]
    rest = [p for n, p in model.named_parameters() if not n.startswith(("derm.", "clin."))]
    opt = torch.optim.AdamW([{"params": backbone, "lr": args.lr},
                             {"params": rest, "lr": args.lr * 10}], weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine(args.epochs * len(dl_tr)))
    focal = FocalLoss(args.focal_gamma)
    dtype = AMP_DTYPES[args.amp_dtype]
    scaler = torch.amp.GradScaler(device.type, enabled=amp and dtype == torch.float16)  # pass-through if off

    # ---- Train / validate ------------------------------------------------------------
    best = -math.inf
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run_loss = time.time(), 0.0
        bar = tqdm(dl_tr, desc=f"fold {k} epoch {epoch}/{args.epochs}", leave=False)
        for step, batch in enumerate(bar, 1):
            batch = to_device(batch, device)
            with torch.autocast(device.type, dtype=dtype, enabled=amp):
                logits_cls, logit_bin = model.forward_batch(batch)
            w_bin = 1 + (args.mel_weight - 1) * (batch["y_cls"] == MEL_IDX).float()  # binary loss only
            loss = (focal(logits_cls.float(), batch["y_cls"])
                    + F.binary_cross_entropy_with_logits(logit_bin.float(), batch["y_bin"], weight=w_bin))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item()
            bar.set_postfix(loss=f"{run_loss / step:.4f}")

        oof = predict(model, dl_va, va, device, amp, dtype)
        m = score_components(va[SOLUTION_COLS], oof)
        print(f"fold {k} epoch {epoch} | loss {run_loss / len(dl_tr):.4f} | {fmt_metrics(m)} | "
              f"lr {sched.get_last_lr()[0]:.2e} | {time.time() - t0:.0f}s")
        if m["score"] > best:
            best = m["score"]
            torch.save({"model": model.state_dict(), "args": vars(args), "fold": k, "epoch": epoch,
                        "metrics": m, "classes": CLASSES}, out)
            oof.to_csv(oof_path, index=False)
            print(f"  ↳ new best, saved {out}")

    # ---- Test inference with the best checkpoint ----------------------------------------
    ckpt = torch.load(out, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_pred = predict(model, make_loader(eval_ds(test), args.batch_size, args.num_workers, args.seed),
                        test, device, amp, dtype)
    print(f"fold {k} best epoch {ckpt['epoch']} | {fmt_metrics(ckpt['metrics'])}")
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pd.read_csv(oof_path), test_pred, ckpt["metrics"]


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    root = Path(args.data_root)

    folds = pd.read_csv(args.folds)[["isic_id", "fold"]]
    train = pd.read_csv(root / "train_metadata.csv").merge(folds, on="isic_id", validate="1:1")
    test = pd.read_csv(root / "test_metadata.csv")
    all_folds = sorted(int(f) for f in train.fold.unique())
    if args.fold == "all":
        run_folds = all_folds
        if "{fold}" not in args.out:
            raise SystemExit("--fold all needs a per-fold --out template containing '{fold}'.")
    elif args.fold.isdigit() and int(args.fold) in all_folds:
        run_folds = [int(args.fold)]
    else:
        raise SystemExit(f"--fold must be one of {all_folds} or 'all', got {args.fold!r}")

    oofs, test_preds, fold_scores = [], [], []
    for k in run_folds:
        oof, test_pred, m = train_fold(args, k, train, test, device, amp)
        oofs.append(oof)
        test_preds.append(test_pred.set_index("isic_id"))
        fold_scores.append(m["score"])

    # ---- Ensemble test predictions (arithmetic mean over fold models) ---------------------
    sub = sum(test_preds) / len(test_preds)
    sub[CLASSES] = sub[CLASSES].div(sub[CLASSES].sum(axis=1), axis=0)
    sub = sub.loc[test["isic_id"]].reset_index()
    _validate(test[["isic_id"]].set_index("isic_id"), sub, "isic_id")  # same checks as the scorer
    Path(args.sub).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.sub, index=False)

    print("\n" + "=" * 100)
    if len(run_folds) > 1:
        # ---- Full out-of-fold CV: every train lesion scored exactly once ----------------------
        oof = pd.concat(oofs, ignore_index=True)
        assert len(oof) == len(train) and set(oof.isic_id) == set(train.isic_id), "OOF does not cover train"
        tmpl = Path(args.out)  # models/baseline_fold{fold}.pth → models/baseline_oof.csv
        oof_path = tmpl.with_name(tmpl.stem.replace("_fold{fold}", "").replace("{fold}", "") + "_oof.csv")
        oof.to_csv(oof_path, index=False)
        m = score_components(train[SOLUTION_COLS], oof)
        print(f"Per-fold scores: {[round(x, 4) for x in fold_scores]} "
              f"(mean {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f})")
        print(f"Full OOF CV ({len(oof)} lesions) | {fmt_metrics(m)}")
        print(f"OOF → {oof_path}")
    print(f"Submission ({len(run_folds)}-model mean) → {args.sub} ({len(sub)} rows)")


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------- #
# Config-driven entry point (release addition; calls the unchanged train_fold)
# --------------------------------------------------------------------------- #
FIXED_IN_CODE = {"head_lr_multiplier": 10, "warmup_frac": 0.05, "clinical_dropout": 0.15, "sampler": "sqrt_balanced"}


def args_from_config(cfg: dict, seed: int, out_template: str, data_dir, **overrides) -> argparse.Namespace:
    """argparse.Namespace for train_fold() from configs/final.yaml; overrides only for demos/smoke runs."""
    r = cfg["recipe"]
    for key, value in FIXED_IN_CODE.items():
        if r.get(key) != value:
            raise ValueError(f"recipe.{key}={r.get(key)!r} is fixed in code at {value!r}; it cannot be configured")
    a = {"backbone": r["backbone"], "d": r["d"], "img_size": r["img_size"], "batch_size": r["batch_size"],
         "num_workers": r["num_workers"], "epochs": r["epochs"], "lr": r["lr"], "weight_decay": r["weight_decay"],
         "focal_gamma": r["focal_gamma"], "mel_weight": r["mel_weight"], "xattn_layers": r["xattn_layers"],
         "amp_dtype": r["amp_dtype"], "no_pretrained": not r["pretrained"], "seed": seed, "fold": "all",
         "data_root": str(data_dir), "folds": str(cfg["paths"]["folds"]), "out": out_template, "sub": ""}
    unknown = set(overrides) - set(a)
    if unknown:
        raise ValueError(f"unknown overrides: {sorted(unknown)}")
    a.update(overrides)
    return argparse.Namespace(**a)


def train_from_config(cfg: dict, seed: int, folds=None, out_dir="models/retrain", data_dir=None,
                      train_rows: pd.DataFrame | None = None, test_rows: pd.DataFrame | None = None,
                      **overrides) -> dict:
    """Train the configured recipe for `folds` (default all) of one seed; returns {fold: (OOF, test, metrics)}.

    `train_rows` / `test_rows` may restrict the tables (smoke demos only); real runs use the full metadata.
    """
    data_dir = Path(data_dir or cfg["paths"]["data_dir"])
    out = str(Path(out_dir) / f"{cfg['ensemble']['run_prefix']}-s{seed}_fold{{fold}}.pth")
    args = args_from_config(cfg, seed, out, data_dir, **overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    fold_table = pd.read_csv(cfg["paths"]["folds"])[["isic_id", "fold"]]
    train = train_rows if train_rows is not None else pd.read_csv(data_dir / "train_metadata.csv")
    train = train.merge(fold_table, on="isic_id", validate="1:1") if "fold" not in train else train
    test = test_rows if test_rows is not None else pd.read_csv(data_dir / "test_metadata.csv")
    results = {}
    for k in (folds if folds is not None else sorted(train.fold.unique())):
        results[int(k)] = train_fold(args, int(k), train, test, device, amp)
    return results
