"""Map the finished training run history in training/results/ to plain MLflow run specs (standard library only).

This module does not import MLflow, so its tests run in the serving and Jenkins test environments. It only reads the
files (training/ stays unchanged); scripts/mlflow_import.py writes the specs into a local MLflow store. What it maps:

- results/experiments_scores.csv: one top-level run per row (37 rows in the release): phase, environment and kind as
  params; change, decision and the OOF file's sha256 as tags; the scores as metrics.
- results/training_history.csv: one run per (run, seed, fold), nested under its seed run, with the epochs as step
  metrics (150 rows over 15 fold runs in the release).
- configs/final.yaml (as a dict; the importer parses the YAML), results/metrics/final_metrics.json,
  results/kaggle_runtime.json and results/provenance.json: params, metrics and tags of the final ensemble run.
- The registered model version: release model-v1.0.0 on GitHub (source URL, no files copied) with the manifest sha256
  pinned in scripts/fetch_model.sh.

Paths are never copied: only allowlisted fields are mapped, and any value that looks like an absolute path is refused.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

EXPERIMENT_NAME = "milk10k-run-history"
REGISTERED_MODEL = "milk10k"
RELEASE_VERSION = "1.0.0"
IMPORT_KEY_TAG = "milk10k.import_key"
INPUT_SHA_TAG = "milk10k.input_sha256"
FINAL_RUN = "final-P1a-kg"
DISCLAIMER = "Research model, not a medical device."
LICENCE = "Derived from ISIC MILK10k data (CC BY-NC 4.0, attribution: MILK study team); non-commercial use only."
NOTE = (
    "Imported after training from the files in training/results/ (scripts/mlflow_import.py); MLflow did not track "
    "these runs live. Run start and end times are import times, not training times."
)

SCORE_PARAMS = ("phase", "environment", "kind")
SCORE_TAGS = ("change", "recorded_decision", "source", "source_sha256")
SCORE_METRICS = (
    "score",
    "pauc80_norm",
    "macro_auc_11",
    "fair_gap",
    "pauc80_light",
    "pauc80_medium",
    "pauc80_dark",
    "fold_mean",
    "fold_std",
)
EPOCH_METRICS = (
    "loss",
    "score",
    "pauc80",
    "macro_auc",
    "fair_gap",
    "pauc80_light",
    "pauc80_medium",
    "pauc80_dark",
    "lr_end",
    "seconds",
)
# kaggle_runtime.json: environment fields copied as tags (never "paths": absolute Kaggle dataset paths)
ENV_FIELDS = ("python", "torch", "cuda", "cudnn", "gpus", "cpu_count", "ram_gb", "internet", "git_commit")
ENV_SOURCES = {"training_env": "train_env", "reference_inference_env": "ref_env"}
CONFIG_EXCLUDED = ("paths",)  # final.yaml: release-relative data and checkpoint paths stay out of the store

# an absolute path at the start of the value or after a space, quote, = ( or comma; a Windows drive; a file: URI
_ABS_PATH = re.compile(r"(^|[\s\"'=(,])(/|~/|[A-Za-z]:[\\/])|file:")


class MappingError(ValueError):
    """The input files do not have the expected shape."""


@dataclass
class RunSpec:
    key: str  # stable import key, unique in the experiment (idempotency)
    name: str
    parent_key: str | None = None
    params: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)  # single values (step 0)
    steps: list[tuple[str, float, int]] = field(default_factory=list)  # (key, value, step)
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class ModelVersionSpec:
    name: str
    source: str
    run_key: str
    description: str
    tags: dict[str, str]


@dataclass
class History:
    experiment_tags: dict[str, str]
    runs: list[RunSpec]
    model_version: ModelVersionSpec


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path, required: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise MappingError(f"{path.name}: missing columns {missing}")
        return list(reader)


def to_float(value: str, where: str) -> float | None:
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise MappingError(f"{where}: not a number: {value!r}") from exc


def metric_key(text: str) -> str:
    """MLflow keys allow letters, digits, _ - . / and spaces; spaces become underscores here."""
    return re.sub(r"[^\w\-./]", "_", text.strip())


def check_no_paths(values: dict[str, str], where: str) -> None:
    for key, value in values.items():
        if _ABS_PATH.search(str(value)):
            raise MappingError(f"{where}: {key} looks like an absolute path; refusing to store it")


def flatten(data: dict, prefix: str = "") -> dict[str, str]:
    """Nested dict → {"a.b": "value"}; lists are JSON-encoded, booleans lower-case."""
    flat: dict[str, str] = {}
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{name}."))
        elif isinstance(value, bool):
            flat[name] = str(value).lower()
        elif isinstance(value, list | tuple):
            flat[name] = json.dumps(value)
        else:
            flat[name] = str(value)
    return flat


def score_runs(rows: list[dict[str, str]], input_sha: str, oof_refs: dict[str, tuple[str, str]]) -> list[RunSpec]:
    """experiments_scores.csv rows → one top-level run each. oof_refs: source → (repo path, sha256)."""
    runs, seen = [], set()
    for row in rows:
        name = row["run"].strip()
        if not name or name in seen:
            raise MappingError(f"experiments_scores.csv: empty or duplicate run name {name!r}")
        seen.add(name)
        spec = RunSpec(key=f"scores/{name}", name=name)
        spec.params = {p: row[p] for p in SCORE_PARAMS}
        spec.tags = {t: row[t] for t in SCORE_TAGS if row.get(t)}
        spec.tags[INPUT_SHA_TAG] = input_sha
        if row.get("source") in oof_refs:
            path, sha = oof_refs[row["source"]]
            spec.tags["oof.file"] = path
            spec.tags["oof.sha256"] = sha
        for m in SCORE_METRICS:
            value = to_float(row.get(m, ""), f"experiments_scores.csv {name}.{m}")
            if value is not None:
                spec.metrics[m] = value
        check_no_paths({**spec.params, **spec.tags}, f"run {name}")
        runs.append(spec)
    return runs


def fold_runs(rows: list[dict[str, str]], parents: set[str], input_sha: str) -> list[RunSpec]:
    """training_history.csv rows → one run per (run, seed, fold) with the epochs as step metrics."""
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["run"], row["seed"], row["fold"]), []).append(row)
    runs = []
    for (run, seed, fold), epochs in groups.items():
        if run not in parents:
            raise MappingError(f"training_history.csv: run {run!r} has no row in experiments_scores.csv")
        numbers = [int(r["epoch"]) for r in epochs]
        if len(set(numbers)) != len(numbers):
            raise MappingError(f"training_history.csv: duplicate epoch in {run} fold {fold}")
        spec = RunSpec(key=f"history/{run}/fold{fold}", name=f"{run}-fold{fold}", parent_key=f"scores/{run}")
        spec.params = {"seed": seed, "fold": fold, "epochs": str(len(epochs))}
        spec.tags = {INPUT_SHA_TAG: input_sha, "history.source": "training/results/training_history.csv"}
        for row in sorted(epochs, key=lambda r: int(r["epoch"])):
            for m in EPOCH_METRICS:
                value = to_float(row.get(m, ""), f"training_history.csv {run} fold {fold} epoch {row['epoch']}")
                if value is not None:
                    spec.steps.append((m, value, int(row["epoch"])))
        runs.append(spec)
    return runs


def final_metrics(data: dict) -> tuple[dict[str, float], list[tuple[str, float, int]], dict[str, str]]:
    """final_metrics.json → (metrics, step metrics, params). ci95 pairs become _lo/_hi; fold_scores are steps."""
    metrics: dict[str, float] = {}

    def add(prefix: str, value) -> None:
        if isinstance(value, dict):
            for key, inner in value.items():
                add(f"{prefix}.{metric_key(key)}" if prefix else metric_key(key), inner)
        elif isinstance(value, list) and len(value) == 2 and "ci95" in prefix.split(".")[-2:]:
            # tta.ci95.score = [lo, hi] and seed_ensembling.ci95 = [lo, hi]
            metrics[f"{prefix}_lo"], metrics[f"{prefix}_hi"] = float(value[0]), float(value[1])
        elif isinstance(value, int | float) and not isinstance(value, bool):
            metrics[prefix] = float(value)
        else:
            raise MappingError(f"final_metrics.json: unexpected value at {prefix}")

    body = {k: v for k, v in data.items() if k not in ("model", "fold_scores", "bootstrap")}
    add("", body)
    steps = [("fold_score", float(v), i) for i, v in enumerate(data.get("fold_scores", []))]
    params = {f"bootstrap.{k}": str(v) for k, v in data.get("bootstrap", {}).items()}
    return metrics, steps, params


def runtime(data: dict) -> tuple[dict[str, str], dict[str, float], dict[str, dict[str, float]]]:
    """kaggle_runtime.json → (final-run tags, final-run metrics, per-run metrics by run name). No paths."""
    tags: dict[str, str] = {}
    for source, prefix in ENV_SOURCES.items():
        env = data.get(source, {})
        for key in ENV_FIELDS:
            if key in env:
                value = env[key]
                tags[f"{prefix}.{key}"] = ", ".join(value) if isinstance(value, list) else str(value)
        for package, version in env.get("preinstalled", {}).items():
            tags[f"{prefix}.pkg.{package}"] = str(version)
    reference = data.get("reference_inference", {})
    metrics = {
        f"ref_inference.{k}": float(v)
        for k, v in reference.items()
        if isinstance(v, int | float) and not isinstance(v, bool)
    }
    if "strict_reproduction" in reference:
        tags["ref_inference.strict_reproduction"] = str(reference["strict_reproduction"])
    per_run: dict[str, dict[str, float]] = {}
    for entry in data.get("training_runs", []) + data.get("noiseoff_runs", []):
        per_run[entry["name"]] = {"train_minutes": float(entry["minutes"]), "returncode": float(entry["returncode"])}
    check_no_paths(tags, "kaggle_runtime.json")
    return tags, metrics, per_run


def oof_references(provenance: dict, training_dir: Path) -> dict[str, tuple[str, str]]:
    """provenance.json: original source path → (repo path of the collected OOF file, sha256), checked on disk."""
    refs = {}
    for results_path, entry in provenance.get("files", {}).items():
        if not results_path.startswith("results/oof/") or "source" not in entry:
            continue
        local = training_dir / results_path
        if not local.is_file():
            continue
        if sha256_file(local) != entry["sha256"]:
            raise MappingError(f"{results_path}: sha256 differs from provenance.json")
        refs[entry["source"]] = (f"training/{results_path}", entry["sha256"])
    return refs


def release_info(fetch_script: str) -> dict[str, str]:
    """Release tag, repository and pinned manifest sha256 from scripts/fetch_model.sh (one source of truth)."""
    found = {}
    for name in ("TAG", "REPO", "MANIFEST_SHA256"):
        match = re.search(rf'^{name}="\$\{{[A-Z0-9_]+:-([^}}"]+)\}}"', fetch_script, re.MULTILINE)
        if not match:
            raise MappingError(f"fetch_model.sh: no default for {name}")
        found[name] = match.group(1)
    if not re.fullmatch(r"[0-9a-f]{64}", found["MANIFEST_SHA256"]):
        raise MappingError("fetch_model.sh: MANIFEST_SHA256 is not a sha256")
    base = f"https://github.com/{found['REPO']}/releases"
    return {
        "tag": found["TAG"],
        "url": f"{base}/tag/{found['TAG']}",
        "manifest_url": f"{base}/download/{found['TAG']}/manifest.json",
        "manifest_sha256": found["MANIFEST_SHA256"],
    }


def build(training_dir: Path, config: dict, fetch_script: str) -> History:
    """All specs from a training/ folder; config is configs/final.yaml already parsed."""
    results = training_dir / "results"
    scores_csv, history_csv = results / "experiments_scores.csv", results / "training_history.csv"
    provenance = json.loads((results / "provenance.json").read_text(encoding="utf-8"))
    scores = score_runs(
        read_csv(scores_csv, ("run", *SCORE_PARAMS, *SCORE_METRICS[:1])),
        sha256_file(scores_csv),
        oof_references(provenance, training_dir),
    )
    by_name = {run.name: run for run in scores}
    folds = fold_runs(
        read_csv(history_csv, ("run", "seed", "fold", "epoch", "loss", "score")),
        set(by_name),
        sha256_file(history_csv),
    )

    if FINAL_RUN not in by_name:
        raise MappingError(f"experiments_scores.csv: no {FINAL_RUN} row")
    final = by_name[FINAL_RUN]
    config_params = flatten({k: v for k, v in config.items() if k not in CONFIG_EXCLUDED}, "config.")
    check_no_paths(config_params, "final.yaml")
    final.params.update(config_params)
    metrics, steps, params = final_metrics(json.loads((results / "metrics" / "final_metrics.json").read_text()))
    final.metrics.update({f"final.{k}": v for k, v in metrics.items()})
    final.steps.extend(steps)
    final.params.update(params)
    tags, ref_metrics, per_run = runtime(json.loads((results / "kaggle_runtime.json").read_text(encoding="utf-8")))
    final.tags.update(tags)
    final.metrics.update(ref_metrics)
    for name, values in per_run.items():
        if name in by_name:
            by_name[name].metrics.update(values)
    if "workspace_commit" in provenance:
        final.tags["provenance.workspace_commit"] = str(provenance["workspace_commit"])

    release = release_info(fetch_script)
    oof = metrics  # unprefixed final_metrics.json values
    version_tags = {
        "release": RELEASE_VERSION,
        "release_tag": release["tag"],
        "release_url": release["url"],
        "manifest_url": release["manifest_url"],
        "manifest_sha256": release["manifest_sha256"],
        "format": "ONNX model package, 15 members (3 seeds x 5 folds)",
        "oof_score": str(oof.get("oof_components.score", "")),
        "oof_pauc80_norm": str(oof.get("oof_components.pauc80_norm", "")),
        "oof_macro_auc_11": str(oof.get("oof_components.macro_auc_11", "")),
        "oof_fair_gap": str(oof.get("oof_components.fair_gap", "")),
        "licence": LICENCE,
        "disclaimer": DISCLAIMER,
    }
    version = ModelVersionSpec(
        name=REGISTERED_MODEL,
        source=release["url"],
        run_key=final.key,
        description=(
            f"Release {RELEASE_VERSION} ({release['tag']}): catalogue entry only; the files stay on the GitHub "
            f"Release, verify them against manifest.json (sha256 {release['manifest_sha256']}). {DISCLAIMER} "
            f"{LICENCE}"
        ),
        tags=version_tags,
    )
    experiment_tags = {"mlflow.note.content": f"{NOTE} {DISCLAIMER} {LICENCE}", "licence": LICENCE}
    return History(experiment_tags=experiment_tags, runs=scores + folds, model_version=version)
