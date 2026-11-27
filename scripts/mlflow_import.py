"""Import the finished training run history (training/results/) into a local MLflow store; idempotent.

Run with the MLflow venv: `make mlflow-import` (writes), `make mlflow-import MLFLOW_ARGS=--check` (counts only).
The mapping lives in mlflow_history.py (no MLflow import, tested without MLflow). This script only writes:

- one experiment, one top-level run per experiments_scores.csv row, the fold runs nested under their seed runs;
- one registered model version for release 1.0.0 whose source is the GitHub Release URL (no files are copied).

Every run carries an import key; a second import skips existing runs and fails if an input file changed (reset the
store instead: `rm artifacts/mlflow/mlflow.db`). The artifact root is the scheme `mlflow-artifacts:/` (what
`mlflow server` uses by default), so no local absolute path is written into the store. Telemetry is off.
Read-only towards training/. Not a live tracker: MLflow did not run during training.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sqlite3
import sys
import time
from pathlib import Path

# Before MLflow is imported: no telemetry, and the artifact root `mlflow server` itself sets when it proxies artifacts
# (an internal variable of MLflow 3.18; without it a client writing to SQLite stores <cwd>/mlruns as an absolute path).
os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
os.environ["DO_NOT_TRACK"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"  # its log line prints the install path (home folder)
os.environ["_MLFLOW_SERVER_ARTIFACT_ROOT"] = "mlflow-artifacts:/"

import mlflow_history as history  # noqa: E402  (same folder; stdlib only)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORE = ROOT / "artifacts" / "mlflow" / "mlflow.db"
IMPORT_USER = "milk10k-import"  # recorded as the run user instead of the local login name
BATCH = 100  # params / tags per log_batch call (MLflow limit: 100 params, 100 tags, 1000 metrics)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE, help="SQLite file (default: %(default)s)")
    parser.add_argument("--training-dir", type=Path, default=ROOT / "training")
    parser.add_argument("--fetch-script", type=Path, default=ROOT / "scripts" / "fetch_model.sh")
    parser.add_argument("--check", action="store_true", help="only print the counts and the path scan of the store")
    return parser.parse_args(argv)


def load_history(training_dir: Path, fetch_script: Path) -> history.History:
    import yaml  # a dependency of mlflow-skinny

    config = yaml.safe_load((training_dir / "configs" / "final.yaml").read_text(encoding="utf-8"))
    return history.build(training_dir, config, fetch_script.read_text(encoding="utf-8"))


def _chunks(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def import_history(client, data: history.History) -> dict[str, int]:
    from mlflow.entities import Metric, Param, RunStatus, ViewType
    from mlflow.exceptions import MlflowException

    created = {"runs_created": 0, "runs_skipped": 0, "versions_created": 0, "versions_skipped": 0}
    experiment = client.get_experiment_by_name(history.EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = client.create_experiment(history.EXPERIMENT_NAME, tags=data.experiment_tags)
    else:
        experiment_id = experiment.experiment_id
    existing = {
        run.data.tags[history.IMPORT_KEY_TAG]: run
        for run in client.search_runs([experiment_id], run_view_type=ViewType.ACTIVE_ONLY, max_results=10000)
        if history.IMPORT_KEY_TAG in run.data.tags
    }
    run_ids: dict[str, str] = {}
    for spec in data.runs:  # parents come before their fold runs
        if spec.key in existing:
            run = existing[spec.key]
            if run.info.status != "FINISHED":
                raise SystemExit(f"run {spec.name} is from an interrupted import; reset the store and import again")
            if run.data.tags.get(history.INPUT_SHA_TAG) != spec.tags[history.INPUT_SHA_TAG]:
                raise SystemExit(f"run {spec.name}: its input file changed since the last import; reset the store")
            run_ids[spec.key] = run.info.run_id
            created["runs_skipped"] += 1
            continue
        tags = {
            **spec.tags,
            history.IMPORT_KEY_TAG: spec.key,
            "mlflow.user": IMPORT_USER,
            "mlflow.source.name": "scripts/mlflow_import.py",
            "mlflow.source.type": "LOCAL",
            "mlflow.note.content": history.NOTE,
        }
        if spec.parent_key:
            tags["mlflow.parentRunId"] = run_ids[spec.parent_key]
        run = client.create_run(experiment_id, run_name=spec.name, tags=tags)
        run_id = run.info.run_id
        now = int(time.time() * 1000)
        metrics = [Metric(k, v, now, 0) for k, v in spec.metrics.items()]
        metrics += [Metric(k, v, now, step) for k, v, step in spec.steps]
        params = [Param(k, v) for k, v in spec.params.items()]
        for batch in _chunks(params, BATCH):
            client.log_batch(run_id, params=batch)
        for batch in _chunks(metrics, 1000):
            client.log_batch(run_id, metrics=batch)
        client.set_terminated(run_id, RunStatus.to_string(RunStatus.FINISHED))
        run_ids[spec.key] = run_id
        created["runs_created"] += 1

    version = data.model_version
    try:
        client.get_registered_model(version.name)
    except MlflowException:  # RESOURCE_DOES_NOT_EXIST
        client.create_registered_model(
            version.name,
            tags={"licence": history.LICENCE, "disclaimer": history.DISCLAIMER},
            description=f"MILK10k skin-lesion ensemble. {history.DISCLAIMER} {history.LICENCE}",
        )
    versions = client.search_model_versions(f"name='{version.name}'")
    same = [v for v in versions if v.tags.get("release") == version.tags["release"]]
    if same:
        if same[0].source != version.source:
            raise SystemExit(f"model version for release {version.tags['release']} has another source")
        created["versions_skipped"] += 1
    else:
        client.create_model_version(
            version.name,
            source=version.source,
            run_id=run_ids[version.run_key],
            tags=version.tags,
            description=version.description,
        )
        created["versions_created"] += 1
    return created


def store_counts(client) -> dict[str, int]:
    from mlflow.entities import ViewType

    experiment = client.get_experiment_by_name(history.EXPERIMENT_NAME)
    if experiment is None:
        return {"experiments": 0}
    runs = client.search_runs([experiment.experiment_id], run_view_type=ViewType.ALL, max_results=10000)
    nested = [r for r in runs if "mlflow.parentRunId" in r.data.tags]
    epochs = sum(len(client.get_metric_history(r.info.run_id, "loss")) for r in nested)
    versions = client.search_model_versions(f"name='{history.REGISTERED_MODEL}'")
    return {
        "experiments": 1,
        "runs_total": len(runs),
        "runs_top_level": len(runs) - len(nested),
        "runs_nested": len(nested),
        "epoch_rows": epochs,
        "model_versions": len(versions),
    }


def path_scan(store: Path) -> dict[str, int]:
    """Counts only (never prints the matches): home folder and login name in every text cell, and in the raw file."""
    needles = {"home_path": str(Path.home()), "user_name": getpass.getuser(), "file_uri": "file:/"}
    counts = dict.fromkeys(needles, 0)
    with sqlite3.connect(f"file:{store}?mode=ro", uri=True) as conn:
        for line in conn.iterdump():
            for name, needle in needles.items():
                counts[name] += line.count(needle)
    raw = store.read_bytes()
    counts.update({f"{name}_raw_bytes": raw.count(needle.encode()) for name, needle in needles.items()})
    return counts


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    store = args.store.resolve()
    if args.check and not store.is_file():
        print("no store yet: run make mlflow-import first", file=sys.stderr)
        return 1
    store.parent.mkdir(parents=True, exist_ok=True)
    from mlflow import MlflowClient

    uri = f"sqlite:///{store}"  # the URI is not stored in the database
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    if not args.check:
        result = import_history(client, load_history(args.training_dir, args.fetch_script))
        print("import:", ", ".join(f"{k} {v}" for k, v in result.items()))
    print("store:", ", ".join(f"{k} {v}" for k, v in store_counts(client).items()))
    scan = path_scan(store)
    print("path scan (counts only):", ", ".join(f"{k} {v}" for k, v in scan.items()))
    return 1 if any(scan.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
