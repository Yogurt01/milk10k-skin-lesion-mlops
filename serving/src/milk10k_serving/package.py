"""Model package: verification against its manifest, ONNX Runtime sessions and the ensemble mean.

A package folder holds `manifest.json` and every file it lists under `files` (15 ONNX members,
`preprocessing.json`, `tabular_encoding.json`, `NOTICE.md`). Notebook 08 writes it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# onnxruntime telemetry (P1-API-13): onnxruntime 1.31 opens an HTTPS connection to its Microsoft collector
# (mobile.events.data.microsoft.com) when a session is created, unless ORT_DISABLE_TELEMETRY=1. It reads the variable
# when onnxruntime is imported (measured: setting it after the import, before the session, still connected), so set
# the default before anything here imports onnxruntime; an explicit value in the environment still wins. If
# onnxruntime was imported earlier without the variable, the default cannot apply: say so loudly.
_TELEMETRY_ALREADY_SET = "ORT_DISABLE_TELEMETRY" in os.environ
ORT_TELEMETRY_DEFAULT_APPLIED = "onnxruntime" not in sys.modules or _TELEMETRY_ALREADY_SET
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
if not ORT_TELEMETRY_DEFAULT_APPLIED:
    logging.getLogger(__name__).error(
        "onnxruntime was imported before milk10k_serving.package and ORT_DISABLE_TELEMETRY was not set: onnxruntime "
        "telemetry is NOT disabled. Import milk10k_serving first or set ORT_DISABLE_TELEMETRY=1 in the environment."
    )

import numpy as np  # noqa: E402

from .preprocess import INPUT_NAMES, PreprocessSpec  # noqa: E402

MANIFEST = "manifest.json"
OUTPUT_NAMES = ("probs_cls", "prob_bin")


class ModelPackageError(RuntimeError):
    """The model package is missing, incomplete or does not match its manifest."""


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def verify_package(model_dir: Path, verify_sha256: bool = True) -> dict:
    """Return the manifest after checking that every listed file exists with the right size (and sha256)."""
    model_dir = Path(model_dir)
    manifest_path = model_dir / MANIFEST
    if not manifest_path.is_file():
        raise ModelPackageError(f"model package not found: {manifest_path} does not exist (set MODEL_DIR)")
    try:
        manifest = json.loads(manifest_path.read_text())
        files = manifest["files"]
        members = manifest["ensemble"]["members"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ModelPackageError(f"invalid manifest {manifest_path}: {exc}") from exc
    problems = []
    for name, info in files.items():
        path = model_dir / name
        if not path.is_file():
            problems.append(f"{name}: missing")
        elif path.stat().st_size != info["bytes"]:
            problems.append(f"{name}: size {path.stat().st_size} != manifest {info['bytes']}")
        elif verify_sha256 and (actual := sha256_file(path)) != info["sha256"]:
            problems.append(f"{name}: sha256 {actual} != manifest {info['sha256']}")
    unlisted = [m["file"] for m in members if m["file"] not in files]
    problems += [f"{name}: ensemble member not listed under files" for name in unlisted]
    if problems:
        raise ModelPackageError(f"model package {model_dir} failed verification: " + "; ".join(problems))
    return manifest


@dataclass
class Ensemble:
    """Equal mean of the members' probabilities, then class renormalisation (as `milk10k.inference`)."""

    sessions: list
    member_files: list[str]

    def predict(self, feeds: dict[str, np.ndarray], stage=None) -> np.ndarray:
        """(batch, 12) float64: column 0 = P(malignant) from the binary head, columns 1..11 = class probabilities.
        `stage` (optional, see `predictor`) wraps each member's run as `ensemble_member` with its index."""
        total = None
        for index, session in enumerate(self.sessions):
            if stage is None:
                probs_cls, prob_bin = session.run(list(OUTPUT_NAMES), feeds)
            else:
                with stage("ensemble_member", member_index=index):
                    probs_cls, prob_bin = session.run(list(OUTPUT_NAMES), feeds)
            cls = probs_cls.astype(np.float64)
            cls /= cls.sum(axis=1, keepdims=True)
            frame = np.column_stack([prob_bin.astype(np.float64).reshape(-1), cls])
            total = frame if total is None else total + frame
        mean = total / len(self.sessions)
        mean[:, 1:] /= mean[:, 1:].sum(axis=1, keepdims=True)
        return mean


@dataclass
class ModelPackage:
    path: Path
    manifest: dict
    spec: PreprocessSpec
    ensemble: Ensemble

    @property
    def name(self) -> str:
        return self.manifest["name"]

    @property
    def version(self) -> str:
        return self.manifest["version"]

    @property
    def classes(self) -> list[str]:
        return list(self.manifest["classes"])


def load_package(model_dir: Path, verify_sha256: bool = True, threads: int = 4) -> ModelPackage:
    """Verify the package, then create one ONNX Runtime CPU session per member. Raises ModelPackageError."""
    import onnxruntime as ort

    model_dir = Path(model_dir)
    manifest = verify_package(model_dir, verify_sha256)
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    # The 15 sessions run one after another; spinning worker threads of the previous session would compete with
    # the next one (measured: 1.58 s -> 0.70 s per lesion at 4 threads with spinning off).
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    sessions, files = [], []
    for member in manifest["ensemble"]["members"]:
        session = ort.InferenceSession(str(model_dir / member["file"]), options, providers=["CPUExecutionProvider"])
        names = tuple(i.name for i in session.get_inputs())
        if names != INPUT_NAMES:
            raise ModelPackageError(f"{member['file']}: inputs {names} != expected {INPUT_NAMES}")
        sessions.append(session)
        files.append(member["file"])
    return ModelPackage(model_dir, manifest, PreprocessSpec.from_package(model_dir), Ensemble(sessions, files))
