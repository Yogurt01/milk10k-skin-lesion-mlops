"""Ensemble inference for final-P1a-kg with defensive input handling and output checks.

Per checkpoint: the evaluation loader (288 px, batch 32), the IDENTITY or TTA4 flip views, softmax/sigmoid averaged
over the views. Then the equal mean over the 15 checkpoints, with class renormalisation. This is the same path that
produced the reference predictions in results/predictions/. Inputs are sanitised so that malformed rows never crash
inference, and every output is checked before it is returned.
"""
from io import StringIO
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .config import checkpoint_names
from .dataset import MILKDataset, clinical_transforms, derm_transforms
from .metrics import CLASSES, _validate
from .models import MILKFusionNet
from .train import AMP_DTYPES, make_loader, seed_everything, to_device

PROBS = ["target_binary", *CLASSES]
SUBMISSION_COLUMNS = ["isic_id", *PROBS]
# (derm h-flip, derm v-flip, clinical h-flip) per view; clinical photos keep their vertical orientation
IDENTITY = [(False, False, False)]
TTA4 = [(False, False, False), (True, False, True), (False, True, False), (True, True, True)]
RECIPE_KEYS = ["backbone", "d", "img_size", "batch_size", "num_workers", "amp_dtype", "xattn_layers",
               "epochs", "lr", "weight_decay", "focal_gamma", "mel_weight"]


# --------------------------------------------------------------------------- #
# Input sanitising: never crash on missing or malformed rows
# --------------------------------------------------------------------------- #
def _resolve(root: Path, value) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return Path(root) / value.strip()        # absolute values stay absolute (as os.path.join)


def _readable(path: Path | None) -> bool:
    return path is not None and path.is_file() and cv2.imread(str(path), cv2.IMREAD_COLOR) is not None


def sanitize_inputs(metadata: pd.DataFrame, root) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Return (clean metadata, derm_ok mask, report).

    * clinical path blank / NaN / non-string / missing file / undecodable -> "" (missing clinical: [NULL] token)
    * derm path missing or undecodable -> derm_ok False (row later receives the prior row, never a crash)
    * fitzpatrick_type outside 1..6 -> NaN; age non-numeric -> NaN; non-string sex/site -> NaN
      (unknown strings already encode as 0 = missing in encode_tabular)
    * duplicate isic_id -> first occurrence kept
    """
    root = Path(root)
    df = metadata.copy()
    rep = {"rows_in": len(df)}
    dup = df.isic_id.duplicated()
    rep["duplicate_ids_dropped"] = int(dup.sum())
    df = df[~dup].reset_index(drop=True)
    for col in ("derm_image_path", "clinical_image_path", "age_approx", "sex",
                "anatomical_site_general", "fitzpatrick_type"):
        if col not in df:
            df[col] = np.nan
    raw = df.clinical_image_path
    blank = raw.isna() | raw.astype(str).str.strip().isin(["", "nan", "None"])
    clin_ok = np.array([(not b) and _readable(_resolve(root, v)) for v, b in zip(raw, blank)], dtype=bool)
    rep["clinical_blank_or_nan"] = int(blank.sum())
    rep["clinical_missing_or_unreadable_file"] = int((~blank.to_numpy() & ~clin_ok).sum())
    df["clinical_image_path"] = [str(v).strip() if ok else "" for v, ok in zip(raw, clin_ok)]
    df["clinical_available"] = clin_ok.astype("int8")
    derm_ok = np.array([_readable(_resolve(root, v)) for v in df.derm_image_path], dtype=bool)
    rep["derm_missing_or_unreadable_file"] = int((~derm_ok).sum())
    df["age_approx"] = pd.to_numeric(df.age_approx, errors="coerce")
    df["fitzpatrick_type"] = pd.to_numeric(df.fitzpatrick_type, errors="coerce")
    bad_fst = df.fitzpatrick_type.notna() & ~df.fitzpatrick_type.isin([1, 2, 3, 4, 5, 6])
    rep["fitzpatrick_out_of_range_set_missing"] = int(bad_fst.sum())
    df.loc[bad_fst, "fitzpatrick_type"] = np.nan
    for col in ("sex", "anatomical_site_general"):
        df[col] = df[col].where(df[col].apply(lambda x: isinstance(x, str)), np.nan)
    rep["rows_out"] = len(df)
    return df, derm_ok, rep


# --------------------------------------------------------------------------- #
# Ensemble prediction (src/final_inference.py test path)
# --------------------------------------------------------------------------- #
def aligned(frame: pd.DataFrame, ids: pd.Series) -> pd.DataFrame:
    if len(frame) != len(ids) or set(frame.isic_id) != set(ids):
        raise ValueError("prediction IDs do not exactly cover the requested rows")
    return _validate(pd.DataFrame(index=pd.Index(ids, name="isic_id")), frame, "isic_id")


def mean_predictions(frames: list[pd.DataFrame], ids: pd.Series) -> pd.DataFrame:
    if not frames:
        raise ValueError("cannot ensemble an empty prediction list")
    mean = aligned(frames[0], ids).copy()
    for frame in frames[1:]:
        mean = mean + aligned(frame, ids)
    mean = mean / len(frames)
    mean[CLASSES] = mean[CLASSES].div(mean[CLASSES].sum(axis=1), axis=0)
    out = mean.reset_index()
    aligned(out, ids)
    return out


def eval_loader(df: pd.DataFrame, root, a: dict):
    s = a["img_size"]
    ds = MILKDataset(df, str(root), derm_transforms(s, train=False), clinical_transforms(s, train=False), clin_size=s)
    return make_loader(ds, a["batch_size"], a["num_workers"], a["seed"])


@torch.inference_mode()
def predict_views(model, loader, df: pd.DataFrame, device, amp: bool, views: list,
                  dtype=torch.bfloat16) -> pd.DataFrame:
    """Mean softmax/sigmoid over `views`; with IDENTITY it equals the training-time predict()."""
    model.eval()
    p_cls, p_bin = [], []
    for batch in tqdm(loader, desc="predict", leave=False):
        batch = to_device(batch, device)
        cls_v, bin_v = [], []
        for dh, dv, ch in views:
            b = dict(batch)
            if dh:
                b["derm"] = b["derm"].flip(-1)
            if dv:
                b["derm"] = b["derm"].flip(-2)
            if ch:
                b["clin"] = b["clin"].flip(-1)
            with torch.autocast(device.type, dtype=dtype, enabled=amp):
                logits_cls, logit_bin = model.forward_batch(b)
            cls_v.append(torch.softmax(logits_cls.float(), dim=1).cpu())
            bin_v.append(torch.sigmoid(logit_bin.float()).cpu())
        p_cls.append(torch.stack(cls_v).mean(0))
        p_bin.append(torch.stack(bin_v).mean(0))
    probs = torch.cat(p_cls).double().numpy()
    probs /= probs.sum(axis=1, keepdims=True)
    sub = pd.DataFrame(probs, columns=CLASSES)
    sub.insert(0, "target_binary", torch.cat(p_bin).double().numpy())
    sub.insert(0, "isic_id", df["isic_id"].to_numpy())
    return sub


def resolve_device(cfg: dict) -> tuple[torch.device, bool]:
    want = cfg["inference"].get("device", "auto")
    use_cuda = torch.cuda.is_available() if want == "auto" else want == "cuda"
    device = torch.device("cuda" if use_cuda else "cpu")
    return device, bool(cfg["inference"].get("amp", True) and device.type == "cuda")


def load_member(path: Path, seed: int, fold: int, cfg: dict, device) -> tuple[MILKFusionNet, dict]:
    """Load one checkpoint after checking its identity and recipe against configs/final.yaml."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    a = {**ckpt["args"], "xattn_layers": ckpt["args"].get("xattn_layers", 0),
         "mel_weight": ckpt["args"].get("mel_weight", 1.0)}
    if ckpt["fold"] != fold or ckpt["classes"] != CLASSES or a["seed"] != seed:
        raise ValueError(f"{path.name}: checkpoint identity (fold/seed/class order) mismatch")
    r = cfg["recipe"]
    diff = {k: (a[k], r[k]) for k in RECIPE_KEYS if a[k] != r[k]}
    if diff:
        raise ValueError(f"{path.name}: recipe differs from configs/final.yaml: {diff}")
    seed_everything(a["seed"])
    model = MILKFusionNet(derm_name=a["backbone"], clin_name=a["backbone"], d=a["d"], pretrained=False,
                          xattn_layers=a["xattn_layers"])
    model.load_state_dict(ckpt["model"])
    return model.to(device), a


def predict_ensemble(df: pd.DataFrame, root, cfg: dict, tta: bool | None = None) -> pd.DataFrame:
    """Mean of all ensemble members for rows whose derm image is readable (use after sanitize_inputs)."""
    tta = cfg["inference"]["tta"] if tta is None else tta
    device, amp = resolve_device(cfg)
    views = TTA4 if tta else IDENTITY
    e, ckpt_dir = cfg["ensemble"], Path(cfg["paths"]["checkpoint_dir"])
    frames = []
    for seed in e["seeds"]:
        for k in e["folds"]:
            path = ckpt_dir / f"{e['run_prefix']}-s{seed}_fold{k}.pth"
            model, a = load_member(path, seed, k, cfg, device)
            frames.append(predict_views(model, eval_loader(df, root, a), df, device, amp, views,
                                        AMP_DTYPES[a["amp_dtype"]]))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if len(frames) != len(checkpoint_names(cfg)):
        raise RuntimeError(f"expected {len(checkpoint_names(cfg))} members, used {len(frames)}")
    return mean_predictions(frames, df.isic_id)


# --------------------------------------------------------------------------- #
# Assembly and output checks
# --------------------------------------------------------------------------- #
def assemble(pred: pd.DataFrame, metadata: pd.DataFrame, derm_ok: np.ndarray, prior: pd.Series) -> tuple[pd.DataFrame, int]:
    """Model rows + prior rows (unreadable derm) in metadata order; repair only rows that would fail the scorer."""
    bad_ids = metadata.isic_id[~derm_ok]
    if len(bad_ids):
        fill = pd.DataFrame([prior[PROBS].to_numpy(dtype=float)] * len(bad_ids), columns=PROBS)
        fill.insert(0, "isic_id", bad_ids.to_numpy())
        pred = pd.concat([pred, fill], ignore_index=True) if len(pred) else fill
    vals = pred.set_index("isic_id").reindex(metadata.isic_id)[PROBS].to_numpy(dtype=float)
    bad = (~np.isfinite(vals).all(1) | (vals < 0).any(1) | (vals > 1).any(1)
           | (np.abs(np.nan_to_num(vals[:, 1:]).sum(1) - 1) > 1e-6))
    if bad.any():
        fix = np.clip(np.nan_to_num(vals[bad], nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        cls = fix[:, 1:]
        cls[cls.sum(1) <= 0] = 1.0 / len(CLASSES)
        fix[:, 1:] = cls / cls.sum(1, keepdims=True)
        vals[bad] = fix
    sub = pd.DataFrame(vals, columns=PROBS)
    sub.insert(0, "isic_id", metadata.isic_id.to_numpy())
    return sub[SUBMISSION_COLUMNS], int(bad.sum())


def output_checks(sub: pd.DataFrame, metadata: pd.DataFrame) -> dict:
    """Row count, ID order, columns, NaN/inf, [0,1], class sums, scorer validation. Raises if any fails."""
    vals = sub[PROBS].to_numpy(dtype=float)
    checks = {"rows_equal_input_ids": len(sub) == metadata.isic_id.nunique(),
              "ids_in_input_order": list(sub.isic_id) == list(metadata.isic_id.drop_duplicates()),
              "no_duplicate_ids": not sub.isic_id.duplicated().any(),
              "columns": list(sub.columns) == SUBMISSION_COLUMNS,
              "finite_no_nan": bool(np.isfinite(vals).all()),
              "in_unit_interval": bool((vals >= 0).all() and (vals <= 1).all()),
              "class_sums_within_1e-3": bool(np.abs(vals[:, 1:].sum(1) - 1).max() <= 1e-3)}
    _validate(pd.DataFrame(index=pd.Index(sub.isic_id, name="isic_id")), sub, "isic_id")
    checks["scorer_validate"] = True
    if not all(checks.values()):
        raise ValueError(f"output checks failed: {checks}")
    return checks


def parity(sub: pd.DataFrame, reference: pd.DataFrame, tol: float) -> dict:
    """Max abs difference vs stored predictions on shared IDs, both through the same CSV writer/reader."""
    sub = pd.read_csv(StringIO(sub.to_csv(index=False)))
    shared = sorted(set(sub.isic_id) & set(reference.isic_id))
    if not shared:
        return {"checked": False, "reason": "no shared IDs"}
    a = sub.set_index("isic_id").loc[shared, PROBS].to_numpy(dtype=float)
    b = reference.set_index("isic_id").loc[shared, PROBS].to_numpy(dtype=float)
    diff = float(np.abs(a - b).max())
    return {"checked": True, "rows": len(shared), "max_abs_diff": diff, "exact": bool(np.array_equal(a, b)),
            "within_tolerance": diff <= tol, "tolerance": tol}


def predict_lesions(metadata: pd.DataFrame, root, cfg: dict, prior: pd.Series, tta: bool | None = None) -> tuple[pd.DataFrame, dict]:
    """Full pipeline: sanitize -> ensemble on readable rows -> assemble with prior rows -> output checks."""
    clean, derm_ok, rep = sanitize_inputs(metadata, root)
    ok = clean[derm_ok].reset_index(drop=True)
    pred = predict_ensemble(ok, root, cfg, tta) if len(ok) else pd.DataFrame(columns=SUBMISSION_COLUMNS)
    sub, repaired = assemble(pred, clean, derm_ok, prior)
    rep["rows_repaired_before_output"] = repaired
    rep["checks"] = output_checks(sub, clean)
    rep["tta"] = cfg["inference"]["tta"] if tta is None else tta
    return sub, rep
