"""Static checks of the release (no notebook execution, no training, no network).

    MILK10K_DATA_DIR=/path/to/prepared/milk10k python scripts/check_static.py      # needs nbformat

1. every notebook validates against the nbformat schema; without --allow-outputs it must have no outputs, with
   --allow-outputs (executed notebooks) lesion images may appear only in the SHOW_IMAGES cells of notebooks 02 and
   03 (the two deliberate exceptions, README), and no output may contain an absolute local path;
2. every code cell compiles;
3. `import milk10k` and every submodule import;
4. every notebook INPUTS path exists now, or is produced by a known step: scripts/export_results.py (EXPORTS),
   sample/make_sample.py, or an EARLIER notebook's OUTPUTS; "data:" inputs must exist under the data directory;
5. every release-relative path mentioned in the Markdown files resolves the same way.
"""
import ast
import importlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
MAKE_SAMPLE = {"sample/sample_metadata.csv", "sample/images/"}
FLAGGED = {"models/retrain/"}  # notebook 04 writes it only with RUN_FULL_TRAINING = True
USER_DATA = ("data/raw",)  # the user downloads/prepares the dataset here (data/README.md); checked via the data dir
NOTEBOOK_DIRS = ("results/figures/", "results/tables/", "results/metrics/", "results/demo_training/")
TOP = ("src/", "configs/", "data/", "docs/", "models/", "notebooks/", "results/", "sample/", "scripts/")
IMAGE_NOTEBOOKS = ("02_eda.ipynb", "03_preprocessing_and_features.ipynb")  # lesion images allowed in SHOW_IMAGES cells only
SUBMODULES = ["config", "metrics", "evaluation", "splits", "data_prep", "dataset", "models", "train", "inference"]


def literal(tree: ast.AST, name: str) -> dict:
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    return {}


ALLOW_OUTPUTS = "--allow-outputs" in sys.argv


def main() -> int:
    import nbformat
    from export_results import EXPORTS

    from milk10k.config import load_config

    cfg = load_config()
    data = Path(cfg["paths"]["data_dir"])
    problems, report = [], {"notebooks": {}, "imports": {}, "paths": {}, "docs": {}}

    for name in ["milk10k"] + [f"milk10k.{m}" for m in SUBMODULES]:
        try:
            importlib.import_module(name)
            report["imports"][name] = "ok"
        except Exception as e:  # noqa: BLE001
            report["imports"][name] = f"FAIL {type(e).__name__}: {e}"
            problems.append(f"import {name}")

    produced = set(EXPORTS) | MAKE_SAMPLE | FLAGGED
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    for nb_path in notebooks:
        nb = nbformat.read(nb_path, as_version=4)
        info = {"cells": len(nb.cells)}
        try:
            nbformat.validate(nb)
            info["nbformat_valid"] = True
        except nbformat.ValidationError as e:
            info["nbformat_valid"] = False
            problems.append(f"{nb_path.name}: nbformat {e.message}")
        dirty = [i for i, c in enumerate(nb.cells) if c.cell_type == "code" and (c.outputs or c.execution_count)]
        info["cells_with_outputs"] = len(dirty)
        if dirty and not ALLOW_OUTPUTS:
            problems.append(f"{nb_path.name}: outputs present in cells {dirty}")
        for i, c in enumerate(nb.cells):
            if c.cell_type != "code":
                continue
            has_img = any(k.startswith("image/") for o in c.outputs for k in o.get("data", {}))
            if "SHOW_IMAGES" in c.source and has_img and nb_path.name not in IMAGE_NOTEBOOKS:
                problems.append(f"{nb_path.name} cell {i}: lesion images are only allowed in notebooks 02 and 03")
            if any("image/jpeg" in o.get("data", {}) for o in c.outputs) and "SHOW_IMAGES" not in c.source:
                problems.append(f"{nb_path.name} cell {i}: JPEG output outside a SHOW_IMAGES cell")
            blob = json.dumps(c.outputs)
            for needle in (str(Path.home()), str(ROOT)):
                if needle in blob:
                    problems.append(f"{nb_path.name} cell {i}: absolute local path in outputs")
        setup = None
        for i, c in enumerate(nb.cells):
            if c.cell_type != "code":
                continue
            try:
                tree = ast.parse(c.source, filename=f"{nb_path.name}[{i}]")
                compile(tree, f"{nb_path.name}[{i}]", "exec")
            except SyntaxError as e:
                problems.append(f"{nb_path.name} cell {i}: {e}")
                continue
            if setup is None and "INPUTS" in c.source:
                setup = tree
        info["code_cells_compile"] = True
        inputs, outputs = literal(setup, "INPUTS"), literal(setup, "OUTPUTS")
        statuses = {}
        for key, v in inputs.items():
            if v.startswith("data:"):
                ok = (data / v[5:]).exists()
                statuses[key] = "exists (data dir)" if ok else f"MISSING under data dir {data}"
            elif (ROOT / v).exists():
                statuses[key] = "exists"
            elif v in produced:
                statuses[key] = "produced by an earlier step"
            else:
                statuses[key] = "MISSING and not produced by any earlier step"
            if "MISSING" in statuses[key]:
                problems.append(f"{nb_path.name}: input {key} -> {v}: {statuses[key]}")
        bad_out = [v for v in outputs.values() if not v.startswith("results/")]
        if bad_out:
            problems.append(f"{nb_path.name}: outputs outside results/: {bad_out}")
        produced |= set(outputs.values())
        info["inputs"] = statuses
        info["outputs"] = len(outputs)
        report["notebooks"][nb_path.name] = info

    texts = {str(d.relative_to(ROOT)): d.read_text() for d in sorted(ROOT.rglob("*.md"))
             if ".venv" not in d.relative_to(ROOT).parts}
    for nb_path in notebooks:   # markdown cells of the notebooks too
        nb = nbformat.read(nb_path, as_version=4)
        texts[f"notebooks/{nb_path.name} (markdown)"] = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    for doc, text in texts.items():
        refs = set(re.findall(r"\]\(([^)#\s]+)\)", text)) | set(re.findall(r"`([^`\s]+)`", text))
        checked = {}
        for ref in sorted(refs):
            if ref.startswith(("http", "mailto")) or not ref.startswith(TOP) or any(ch in ref for ch in "{}*<>"):
                continue
            rel = ref.rstrip("/")
            ok = (rel.startswith(USER_DATA) or (ROOT / rel).exists() or ref in produced or rel in produced or f"{rel}/" in produced
                  or rel.startswith(tuple(d.rstrip("/") for d in NOTEBOOK_DIRS))
                  or any(p.startswith(rel + "/") for p in produced))
            checked[ref] = "ok" if ok else "UNRESOLVED"
            if not ok:
                problems.append(f"{doc}: path {ref} does not exist and is not produced")
        report["docs"][doc] = checked

    report["problems"] = problems
    report["data_dir"] = str(data)
    print(json.dumps(report, indent=1))
    print(f"\n{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    sys.exit(main())
