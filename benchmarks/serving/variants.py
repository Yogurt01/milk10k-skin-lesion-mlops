"""Smaller ensemble variants (docs/serving_measurements.md §4).

Usage (from the repository root): MILK10K_DATA_DIR=... training/.venv/bin/python benchmarks/serving/variants.py


(a) OOF score of seed subsets (OOF of a lesion = mean of its held-out fold model over the chosen seeds).
(b) Fold subsets cannot be OOF-scored, so: per-member test predictions (479 lesions, GPU fp16 as the reference
    run), then each subset's deviation from the full 15-member mean. Also full mean vs the stored reference.
"""

import itertools
import sys
import time

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import OUT, batches, config, inf, load_all, members, pd, reference, save, torch  # noqa: E402
from milk10k.evaluation import evaluate, load_oof, load_solution, truth_arrays  # noqa: E402

cfg = config("cuda")
P = cfg["paths"]
R = {}

# ---- (a) OOF scores of seed subsets ------------------------------------------------------------------
sol = load_solution(P["data_dir"], P["folds"])
t = truth_arrays(sol)
seeds = cfg["ensemble"]["seeds"]
oof = {s: load_oof(P["results_dir"] / "oof" / f"P1a-kg-s{s}_oof.csv", sol) for s in seeds}
oof_tta = {s: load_oof(P["results_dir"] / "oof" / f"P1a-kg-s{s}-tta_oof.csv", sol) for s in seeds}


def score(frames):
    m = inf.mean_predictions(frames, sol.isic_id) if len(frames) > 1 else frames[0]
    e = evaluate(t, sol, m)
    return {k: e[k] for k in ("score", "pauc80_norm", "macro_auc_11", "fair_gap")}


rows = []
for r in (1, 2, 3):
    for combo in itertools.combinations(seeds, r):
        rows.append(
            {
                "variant": f"seeds {'+'.join(map(str, combo))}",
                "members": 5 * r,
                "tta": False,
                **score([oof[s] for s in combo]),
            }
        )
        if r == 1:
            rows.append({"variant": f"seed {combo[0]} + TTA4", "members": 5, "tta": True, **score([oof_tta[combo[0]]])})
rows.append({"variant": "seeds 42+43+44 + TTA4", "members": 15, "tta": True, **score([oof_tta[s] for s in seeds])})
stored = load_oof(P["results_dir"] / "oof" / "final-P1a-kg_oof.csv", sol)
full_mean = inf.mean_predictions([oof[s] for s in seeds], sol.isic_id)
R["oof_full_reproduces_stored"] = float(
    np.abs(
        full_mean[inf.PROBS].to_numpy() - stored.set_index("isic_id").loc[full_mean.isic_id, inf.PROBS].to_numpy()
    ).max()
)
R["oof_variants"] = rows

# ---- (b) Per-member test predictions and fold-subset deviation ------------------------------------------
device = torch.device("cuda")
_, amp = inf.resolve_device(cfg)
test = pd.read_csv(P["data_dir"] / "test_metadata.csv")
clean, ok, rep = inf.sanitize_inputs(test, P["data_dir"])
df = clean[ok].reset_index(drop=True)
R["test_sanitize"] = rep
t0 = time.perf_counter()
first = load_all(cfg, device, subset={(seeds[0], 0)})[0]
bs = batches(df, P["data_dir"], first[3], workers=4)
R["test_preprocess_s"] = time.perf_counter() - t0
per = {}
t0 = time.perf_counter()
for s, k, path in members(cfg):
    m, a = inf.load_member(path, s, k, cfg, device)
    per[(s, k)] = inf.predict_views(m.eval(), bs, df, device, amp, inf.IDENTITY, inf.AMP_DTYPES[a["amp_dtype"]])
    per[(s, k)].to_csv(OUT / f"test_member_s{s}_f{k}.csv", index=False)
    del m
    torch.cuda.empty_cache()
R["test_15_members_s"] = time.perf_counter() - t0

full = inf.mean_predictions(list(per.values()), df.isic_id).set_index("isic_id")
ref = reference(cfg).set_index("isic_id").loc[full.index]
R["full_gpu_vs_reference_479"] = {
    "max_abs_diff": float((full[inf.PROBS] - ref[inf.PROBS]).abs().to_numpy().max()),
    "within_0.002": bool((full[inf.PROBS] - ref[inf.PROBS]).abs().to_numpy().max() <= 0.002),
}


def deviation(keys, label):
    sub = inf.mean_predictions([per[k] for k in keys], df.isic_id).set_index("isic_id").loc[full.index]
    d = (sub[inf.PROBS] - full[inf.PROBS]).abs().to_numpy()
    top_agree = (sub[inf.CLASSES].to_numpy().argmax(1) == full[inf.CLASSES].to_numpy().argmax(1)).mean()
    rank = pd.Series(sub.target_binary).rank().corr(pd.Series(full.target_binary).rank())
    return {
        "variant": label,
        "members": len(keys),
        "max_abs_diff": float(d.max()),
        "mean_abs_diff": float(d.mean()),
        "lesions_within_0.002": int((d.max(1) <= 0.002).sum()),
        "top1_agreement": float(top_agree),
        "spearman_p_malignant": float(rank),
    }


folds = cfg["ensemble"]["folds"]
dev = []
for r in (1, 2):
    for combo in itertools.combinations(seeds, r):
        dev.append(deviation([(s, k) for s in combo for k in folds], f"seeds {'+'.join(map(str, combo))} (all folds)"))
for nf in (1, 2, 3):
    group = [
        deviation([(s, k) for s in seeds for k in fc], f"folds {''.join(map(str, fc))} x 3 seeds")
        for fc in itertools.combinations(folds, nf)
    ]
    worst = max(group, key=lambda x: x["max_abs_diff"])
    best = min(group, key=lambda x: x["max_abs_diff"])
    dev += [
        {**worst, "variant": f"{nf} fold(s) x 3 seeds, worst of {len(group)}: {worst['variant']}"},
        {**best, "variant": f"{nf} fold(s) x 3 seeds, best of {len(group)}: {best['variant']}"},
    ]
singles = [deviation([key], f"single member s{key[0]} f{key[1]}") for key in per]
dev.append({**max(singles, key=lambda x: x["max_abs_diff"]), "variant": "1 member, worst of 15"})
dev.append({**min(singles, key=lambda x: x["max_abs_diff"]), "variant": "1 member, best of 15"})
R["test_subset_deviation"] = dev
print(save("variants.json", R))
