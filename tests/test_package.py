import hashlib
import json
import shutil
import subprocess
import sys

import pytest

from milk10k_serving.package import ModelPackageError, load_package, verify_package

from .conftest import ROOT


def _fake_package(tmp_path, files=None):
    files = files or {"m1.onnx": b"weights-1", "m2.onnx": b"weights-2", "preprocessing.json": b"{}"}
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    manifest = {
        "name": "fake",
        "version": "0",
        "classes": [],
        "files": {n: {"sha256": hashlib.sha256(d).hexdigest(), "bytes": len(d)} for n, d in files.items()},
        "ensemble": {"members": [{"file": "m1.onnx"}, {"file": "m2.onnx"}]},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


@pytest.mark.unit
def test_intact_fake_package_verifies(tmp_path):
    assert verify_package(_fake_package(tmp_path))["name"] == "fake"


@pytest.mark.unit
def test_corrupted_file_fails_with_its_name_and_hashes(tmp_path):
    pkg = _fake_package(tmp_path)
    (pkg / "m2.onnx").write_bytes(b"weights-X")  # same size, different content
    with pytest.raises(ModelPackageError, match=r"m2\.onnx: sha256 [0-9a-f]{64} != manifest [0-9a-f]{64}"):
        verify_package(pkg)
    verify_package(pkg, verify_sha256=False)  # the size still matches, so only the hash check catches it


@pytest.mark.unit
def test_missing_and_resized_files_are_all_reported(tmp_path):
    pkg = _fake_package(tmp_path)
    (pkg / "m1.onnx").unlink()
    (pkg / "preprocessing.json").write_bytes(b"{ }")
    with pytest.raises(ModelPackageError, match=r"m1\.onnx: missing.*preprocessing\.json: size 3 != manifest 2"):
        verify_package(pkg, verify_sha256=False)


@pytest.mark.unit
def test_missing_package_points_to_model_dir(tmp_path):
    with pytest.raises(ModelPackageError, match="MODEL_DIR"):
        verify_package(tmp_path / "nowhere")


@pytest.mark.slow
def test_real_package_with_one_corrupted_member_fails_at_startup(model_dir, tmp_path):
    manifest = json.loads((model_dir / "manifest.json").read_text())
    victim = manifest["ensemble"]["members"][-1]["file"]
    for name in [*manifest["files"], "manifest.json"]:
        if name != victim:
            (tmp_path / name).symlink_to(model_dir / name)
    shutil.copyfile(model_dir / victim, tmp_path / victim)
    with open(tmp_path / victim, "r+b") as fh:  # flip one byte in the middle of the weights
        fh.seek(fh.seek(0, 2) // 2)
        byte = fh.read(1)
        fh.seek(-1, 1)
        fh.write(bytes([byte[0] ^ 0xFF]))
    with pytest.raises(ModelPackageError, match=rf"{victim}: sha256"):
        load_package(tmp_path)


@pytest.mark.slow
def test_real_package_loads_all_members(loaded_package):
    members = loaded_package.manifest["ensemble"]["members"]
    assert len(loaded_package.ensemble.sessions) == len(members) == 15


@pytest.mark.unit
@pytest.mark.parametrize(
    ("env_value", "import_ort_first", "expected", "error_logged"),
    [
        (None, False, "1", False),  # default applied before onnxruntime is imported
        ("0", False, "0", False),  # an explicit value in the environment wins
        (None, True, "1", True),  # onnxruntime imported first: the default is set but cannot apply -> error logged
        ("1", True, "1", False),  # imported first, but the variable was already set: it applied
    ],
)
def test_onnxruntime_telemetry_default_in_a_clean_process(env_value, import_ort_first, expected, error_logged):
    """P1-API-13: importing milk10k_serving.package sets ORT_DISABLE_TELEMETRY=1 unless the environment has a value;
    onnxruntime reads it at its own import, so an earlier import without the variable is reported."""
    code = (
        ("import onnxruntime\n" if import_ort_first else "")
        + "import os, sys\n"
        + "from milk10k_serving import package\n"
        + "print(os.environ['ORT_DISABLE_TELEMETRY'], package.ORT_TELEMETRY_DEFAULT_APPLIED)\n"
        + "print('onnxruntime' in sys.modules)\n"
    )
    env = {"PATH": ""} | ({"ORT_DISABLE_TELEMETRY": env_value} if env_value is not None else {})
    result = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    # the package import itself never imports onnxruntime (load_package does), so the default is in place before it
    assert result.stdout.split() == [expected, str(not error_logged), str(import_ort_first)]
    assert ("telemetry is NOT disabled" in result.stderr) is error_logged
