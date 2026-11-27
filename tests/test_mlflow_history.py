"""The MLflow import of the training run history (scripts/mlflow_history.py, scripts/mlflow_import.py).

Small fake files only, never the real results. The mapping tests need no MLflow (they run in the serving and Jenkins
test environments); the end-to-end import test is skipped unless MLflow is installed (the MLflow venv).
"""

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


h = _load("mlflow_history")

SCORES_HEADER = (
    "run,phase,environment,kind,change,recorded_decision,score,pauc80_norm,macro_auc_11,fair_gap,pauc80_light,"
    "pauc80_medium,pauc80_dark,fold_mean,fold_std,source,source_sha256"
)
HISTORY_HEADER = (
    "run,seed,fold,epoch,loss,score,pauc80,macro_auc,fair_gap,pauc80_light,pauc80_medium,pauc80_dark,lr_end,seconds"
)
FETCH_SCRIPT = (
    'TAG="${MODEL_RELEASE_TAG:-model-v9.9.9}"\n'
    'REPO="${MODEL_REPO:-someone/some-repo}"\n'
    'MANIFEST_SHA256="${MODEL_MANIFEST_SHA256:-' + "ab" * 32 + '}"\n'
)
CONFIG = {
    "name": "final-P1a-kg",
    "paths": {"data_dir": "data/raw"},
    "recipe": {"lr": 3.0e-4, "pretrained": True},
    "ensemble": {"seeds": [42, 43]},
}


def _scores_row(run: str, kind: str = "train", score: str = "0.6", source: str = "models/x_oof.csv") -> str:
    metrics = f"{score},0.5,0.8,0.1,,0.5,0.5,0.6,0.02"  # pauc80_light empty
    return f'{run},1,Kaggle T4 fp16,{kind},"a change, with a comma",keep,{metrics},{source},{"0" * 64}'


def _history_rows(run: str, seed: int, folds: int, epochs: int) -> list[str]:
    return [
        f"{run},{seed},{f},{e},{2 - e / 10},0.{e},0.4,0.8,0.1,,0.5,0.5,0.0003,50"
        for f in range(folds)
        for e in range(1, epochs + 1)
    ]


@pytest.fixture
def training_dir(tmp_path: Path) -> Path:
    """A fake training/ folder: 3 score rows, 2 seeds x 2 folds x 3 epochs, one collected OOF file."""
    root = tmp_path / "training"
    (root / "results" / "metrics").mkdir(parents=True)
    (root / "results" / "oof").mkdir()
    oof = root / "results" / "oof" / "s42_oof.csv"
    oof.write_text("isic_id,target_binary\nISIC_0000001,0.5\n")
    rows = [
        _scores_row("s42", source="models/s42_oof.csv"),
        _scores_row("s43", score=""),
        _scores_row("final-P1a-kg", kind="ensemble"),
    ]
    (root / "results" / "experiments_scores.csv").write_text("\n".join([SCORES_HEADER, *rows]) + "\n")
    history = _history_rows("s42", 42, 2, 3) + _history_rows("s43", 43, 2, 3)
    (root / "results" / "training_history.csv").write_text("\n".join([HISTORY_HEADER, *history]) + "\n")
    (root / "results" / "metrics" / "final_metrics.json").write_text(
        json.dumps(
            {
                "model": "final-P1a-kg",
                "oof_components": {"score": 0.65, "fair_gap": 0.17},
                "pauc80_by_code_group": {"codes 1-2": 0.72},
                "fold_scores": [0.66, 0.67],
                "tta": {"delta": 0.005, "ci95": {"score": [-0.001, 0.01]}},
                "seed_ensembling": {"delta": 0.02, "ci95": [0.01, 0.03]},
                "bootstrap": {"n_boot": 1000, "seed": 0},
            }
        )
    )
    (root / "results" / "kaggle_runtime.json").write_text(
        json.dumps(
            {
                "training_runs": [{"name": "s42", "seed": 42, "returncode": 0, "minutes": 39.6}],
                "training_env": {
                    "python": "3.12",
                    "gpus": ["T4", "T4"],
                    "preinstalled": {"timm": "1.0"},
                    "paths": {"data": "/kaggle/input/someone/data"},
                },
                "reference_inference": {"minutes": 6.0, "strict_reproduction": "passed"},
            }
        )
    )
    (root / "results" / "provenance.json").write_text(
        json.dumps(
            {
                "files": {
                    "results/oof/s42_oof.csv": {"source": "models/s42_oof.csv", "sha256": h.sha256_file(oof)},
                    "results/training_history.csv": {"parsed_from": [{"source": "/kaggle/x.log"}], "sha256": "0"},
                },
                "workspace_commit": "abc123",
            }
        )
    )
    return root


def _build(training_dir: Path):
    return h.build(training_dir, CONFIG, FETCH_SCRIPT)


@pytest.mark.unit
def test_one_top_level_run_per_score_row_and_one_nested_run_per_fold(training_dir):
    data = _build(training_dir)
    top = [r for r in data.runs if r.parent_key is None]
    nested = [r for r in data.runs if r.parent_key]
    assert [r.name for r in top] == ["s42", "s43", "final-P1a-kg"]
    assert sorted(r.name for r in nested) == ["s42-fold0", "s42-fold1", "s43-fold0", "s43-fold1"]
    assert {r.parent_key for r in nested} == {"scores/s42", "scores/s43"}
    keys = [r.key for r in data.runs]
    assert len(keys) == len(set(keys))
    # parents come first, so the importer can resolve every parent run id
    assert max(keys.index(r.parent_key) for r in nested) < min(keys.index(r.key) for r in nested)


@pytest.mark.unit
def test_epochs_become_step_metrics_and_empty_cells_are_skipped(training_dir):
    fold = next(r for r in _build(training_dir).runs if r.name == "s42-fold1")
    assert fold.params == {"seed": "42", "fold": "1", "epochs": "3"}
    assert sorted({step for _, _, step in fold.steps}) == [1, 2, 3]
    assert ("loss", 1.9, 1) in fold.steps
    assert not [k for k, _, _ in fold.steps if k == "pauc80_light"]  # empty in every row
    assert len(fold.steps) == 3 * (len(h.EPOCH_METRICS) - 1)


@pytest.mark.unit
def test_score_row_maps_params_tags_metrics_and_the_oof_reference(training_dir):
    runs = {r.name: r for r in _build(training_dir).runs}
    s42, s43 = runs["s42"], runs["s43"]
    assert s42.params == {"phase": "1", "environment": "Kaggle T4 fp16", "kind": "train"}
    assert s42.tags["change"] == "a change, with a comma"
    assert s42.tags["oof.file"] == "training/results/oof/s42_oof.csv"
    assert "oof.file" not in s43.tags  # its source was not collected into results/oof/
    assert s42.metrics["score"] == 0.6 and "pauc80_light" not in s42.metrics
    assert "score" not in s43.metrics  # empty cell: no metric, not 0
    assert s42.metrics["train_minutes"] == 39.6  # kaggle_runtime.json training_runs


@pytest.mark.unit
def test_final_run_gets_config_metrics_and_environment_without_paths(training_dir):
    final = next(r for r in _build(training_dir).runs if r.name == "final-P1a-kg")
    assert final.params["config.recipe.lr"] == "0.0003"
    assert final.params["config.recipe.pretrained"] == "true"
    assert final.params["config.ensemble.seeds"] == "[42, 43]"
    assert not [k for k in final.params if k.startswith("config.paths")]
    assert final.metrics["final.oof_components.score"] == 0.65
    assert final.metrics["final.pauc80_by_code_group.codes_1-2"] == 0.72
    assert final.metrics["final.tta.ci95.score_lo"] == -0.001
    assert final.metrics["final.seed_ensembling.ci95_hi"] == 0.03
    assert ("fold_score", 0.67, 1) in final.steps
    assert final.params["bootstrap.n_boot"] == "1000"
    assert final.tags["train_env.gpus"] == "T4, T4"
    assert final.tags["train_env.pkg.timm"] == "1.0"
    assert final.tags["provenance.workspace_commit"] == "abc123"
    assert not [v for v in [*final.tags.values(), *final.params.values()] if "/kaggle" in v]


@pytest.mark.unit
def test_model_version_points_at_the_release_with_the_pinned_manifest(training_dir):
    version = _build(training_dir).model_version
    assert version.source == "https://github.com/someone/some-repo/releases/tag/model-v9.9.9"
    assert version.run_key == "scores/final-P1a-kg"
    assert version.tags["release"] == "1.0.0"
    assert version.tags["manifest_sha256"] == "ab" * 32
    assert version.tags["manifest_url"].endswith("/releases/download/model-v9.9.9/manifest.json")
    assert version.tags["oof_score"] == "0.65"
    assert "not a medical device" in version.description and "MILK study team" in version.description


@pytest.mark.unit
def test_release_info_reads_the_real_fetch_script():
    info = h.release_info((SCRIPTS / "fetch_model.sh").read_text())
    assert info["tag"] == "model-v1.0.0"
    assert info["url"].startswith("https://github.com/") and info["url"].endswith("/releases/tag/model-v1.0.0")


@pytest.mark.unit
@pytest.mark.parametrize(
    "value",
    ["/kaggle/input/x", "data at /srv/someone/x", "path=/srv/x", "file:///tmp/x", "~/data", "C:\\data", "C:/data"],
)
def test_absolute_paths_are_refused(value):
    with pytest.raises(h.MappingError, match="absolute path"):
        h.check_no_paths({"k": value}, "test")


@pytest.mark.unit
@pytest.mark.parametrize("value", ["models/baseline_oof.csv", "3 seeds x 5 folds", "P4b on P4a", "0.0003"])
def test_relative_values_pass_the_path_check(value):
    h.check_no_paths({"k": value}, "test")


@pytest.mark.unit
def test_history_without_a_score_row_fails(training_dir):
    path = training_dir / "results" / "training_history.csv"
    path.write_text(path.read_text() + _history_rows("ghost", 1, 1, 1)[0] + "\n")
    with pytest.raises(h.MappingError, match="ghost"):
        _build(training_dir)


@pytest.mark.unit
def test_duplicate_epoch_fails(training_dir):
    path = training_dir / "results" / "training_history.csv"
    path.write_text(path.read_text() + _history_rows("s42", 42, 1, 1)[0] + "\n")
    with pytest.raises(h.MappingError, match="duplicate epoch"):
        _build(training_dir)


@pytest.mark.unit
def test_changed_oof_file_fails_the_provenance_check(training_dir):
    (training_dir / "results" / "oof" / "s42_oof.csv").write_text("changed\n")
    with pytest.raises(h.MappingError, match="provenance"):
        _build(training_dir)


def test_import_twice_creates_no_duplicates_and_no_local_paths(training_dir, tmp_path):
    pytest.importorskip("mlflow")
    importer = _load("mlflow_import")
    (training_dir / "configs").mkdir()
    (training_dir / "configs" / "final.yaml").write_text("name: final-P1a-kg\nrecipe: {lr: 0.0003}\npaths: {x: y}\n")
    fetch = tmp_path / "fetch_model.sh"
    fetch.write_text(FETCH_SCRIPT)
    store = tmp_path / "store" / "mlflow.db"
    args = ["--store", str(store), "--training-dir", str(training_dir), "--fetch-script", str(fetch)]

    from mlflow import MlflowClient

    uri = f"sqlite:///{store}"
    for _ in range(2):
        assert importer.main(args) == 0  # 0: the store has no home path, login name or file: URI
        counts = importer.store_counts(MlflowClient(tracking_uri=uri, registry_uri=uri))
        assert counts == {
            "experiments": 1,
            "runs_total": 7,
            "runs_top_level": 3,
            "runs_nested": 4,
            "epoch_rows": 12,
            "model_versions": 1,
        }
    with sqlite3.connect(store) as conn:
        text = "\n".join(conn.iterdump())
    assert str(tmp_path) not in text and "file:/" not in text
    assert "mlflow-artifacts:/" in text
