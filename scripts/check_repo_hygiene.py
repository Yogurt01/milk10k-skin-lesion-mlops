#!/usr/bin/env python3
"""Repository hygiene: nothing that must stay out of git is tracked or staged (PROJECT_REQUIREMENTS §5).

    python scripts/check_repo_hygiene.py                 # files in the git index (tracked + staged)
    python scripts/check_repo_hygiene.py --untracked     # also untracked files that `git add -A` would add
    python scripts/check_repo_hygiene.py notebooks/x.ipynb ...   # also check these notebooks' outputs

Rules:
1. never tracked: private/, CLAUDE.md, .env files (except .env.example), model weights and exports, archives,
   MILK10k dataset files, credentials and keys, Terraform state and variable files;
2. images only under docs/images/ and training/results/figures/ (plots);
3. no file larger than 5 MiB;
4. notebooks: JPEG outputs (lesion images) only in the SHOW_IMAGES cells of training notebooks 02 and 03 (the
   release's deliberate exception), and no output containing an absolute local path (home or repository folder).
Standard library only. Exit code 1 if any rule is broken.
"""

import argparse
import fnmatch
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 5 * 2**20
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp", ".dcm"}
IMAGE_DIRS = ("docs/images/", "training/results/figures/")
JPEG_NOTEBOOKS = ("training/notebooks/02_eda.ipynb", "training/notebooks/03_preprocessing_and_features.ipynb")
FORBIDDEN = [  # (glob on the repo-relative path or the file name, reason)
    ("private/*", "private working files"),
    ("CLAUDE.md", "local agent instructions"),
    (".env", "environment file with credentials"),
    (".env.*", "environment file with credentials"),
    ("*.pth", "model weights"),
    ("*.pt", "model weights"),
    ("*.ckpt", "model weights"),
    ("*.onnx", "exported model"),
    ("*.safetensors", "model weights"),
    ("*.pkl", "pickled object"),
    ("*.joblib", "pickled object"),
    ("*.h5", "model weights"),
    ("*.zip", "archive"),
    ("*.tar", "archive"),
    ("*.tar.gz", "archive"),
    ("*.tgz", "archive"),
    ("MILK10k_*", "MILK10k release file"),
    ("train_metadata.csv", "dataset file"),
    ("test_metadata.csv", "dataset file"),
    ("sample_submission.csv", "dataset file"),
    ("sample_metadata.csv", "dataset-derived sample metadata"),
    ("*.pem", "key"),
    ("*.key", "key"),
    ("*credentials*.json", "credentials"),
    ("*service-account*.json", "credentials"),
    ("kubeconfig*", "cluster credentials"),
    ("*.tfstate", "Terraform state"),
    ("*.tfstate.*", "Terraform state"),
    ("*.tfvars", "Terraform variables"),
    ("*.tfvars.json", "Terraform variables"),
]
ALLOWED = {".env.example"}


def git_files(untracked: bool) -> list[str]:
    cmd = ["git", "-C", str(ROOT), "ls-files", "-z", "--cached"]
    if untracked:
        cmd += ["--others", "--exclude-standard"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout.decode()
    return sorted({p for p in out.split("\0") if p and (ROOT / p).exists()})


def path_problems(path: str) -> list[str]:
    name = Path(path).name
    problems = []
    if name not in ALLOWED:
        for pattern, reason in FORBIDDEN:
            if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(name, pattern):
                problems.append(f"{path}: {reason} must not be in git ({pattern})")
                break
    if Path(path).suffix.lower() in IMAGE_EXT and not path.startswith(IMAGE_DIRS):
        problems.append(f"{path}: image outside {' or '.join(IMAGE_DIRS)}")
    size = (ROOT / path).stat().st_size
    if size > MAX_BYTES:
        problems.append(f"{path}: {size / 2**20:.1f} MiB is larger than {MAX_BYTES // 2**20} MiB")
    return problems


def notebook_problems(path: str) -> list[str]:
    nb = json.loads((ROOT / path).read_text(encoding="utf-8"))
    needles = {str(Path.home()), str(ROOT)}
    problems = []
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", ""))
        outputs = cell.get("outputs", [])
        has_jpeg = any("image/jpeg" in o.get("data", {}) for o in outputs)
        if has_jpeg and not (path in JPEG_NOTEBOOKS and "SHOW_IMAGES" in source):
            problems.append(f"{path} cell {i}: JPEG output (possible lesion image) outside the allowed cells")
        text = json.dumps(outputs)
        if any(n in text for n in needles):
            problems.append(f"{path} cell {i}: absolute local path in outputs")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--untracked", action="store_true", help="also check untracked, non-ignored files")
    parser.add_argument("notebooks", nargs="*", help="extra notebooks to check (repo-relative paths)")
    args = parser.parse_args()
    files = git_files(args.untracked)
    problems = [p for f in files for p in path_problems(f)]
    notebooks = sorted({f for f in files if f.endswith(".ipynb")} | set(args.notebooks))
    problems += [p for nb in notebooks for p in notebook_problems(nb)]
    scope = "index + untracked" if args.untracked else "index"
    if problems:
        print(f"repo hygiene: {len(problems)} problem(s) in {len(files)} files ({scope}):")
        print("\n".join(f"  - {p}" for p in problems))
        return 1
    print(f"repo hygiene: OK ({len(files)} files in {scope}, {len(notebooks)} notebooks checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
