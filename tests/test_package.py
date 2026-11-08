import hashlib
import json
import shutil

import pytest

from milk10k_serving.package import ModelPackageError, load_package, verify_package


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
