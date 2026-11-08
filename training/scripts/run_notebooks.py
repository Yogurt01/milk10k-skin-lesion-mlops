"""Execute the notebooks in order, in place (needs the prepared data, the checkpoints and results/).

    python scripts/run_notebooks.py                 # all seven, 01 -> 07
    python scripts/run_notebooks.py 05 07           # selected ones

Run from any directory inside the requirements.txt environment. Kernels run with the notebook folder as working
directory, progress bars off (TQDM_DISABLE), no Albumentations update check and HF_HUB_OFFLINE=1 (pretrained
weights only from the local Hugging Face cache; notebook 04 reports which). After execution, absolute local
paths that libraries print (e.g. site-packages paths in warnings) are replaced in the saved outputs by
<release>, <data>, <checkpoints> or ~; the number of replacements is reported. Runtimes go to
results/metrics/notebook_runtime.json.
"""
import json
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from milk10k.config import load_config


def redactions() -> list[tuple[str, str]]:
    cfg = load_config()
    pairs = [(str(ROOT), "<release>"), (str(cfg["paths"]["data_dir"]), "<data>"),
             (str(cfg["paths"]["checkpoint_dir"]), "<checkpoints>"), (str(Path.home()), "~")]
    return sorted({(a, b) for a, b in pairs if len(a) > 1}, key=lambda x: -len(x[0]))  # longest first


def redact_outputs(nb, pairs) -> int:
    n = 0

    def fix(text: str) -> str:
        nonlocal n
        for a, b in pairs:
            if a in text:
                n += text.count(a)
                text = text.replace(a, b)
        return text

    for cell in nb.cells:
        for o in cell.get("outputs", []):
            if "text" in o:
                o["text"] = fix(o["text"])
            for key in ("text/plain", "text/html", "text/markdown"):
                if key in o.get("data", {}):
                    o["data"][key] = fix(o["data"][key])
            if "traceback" in o:
                o["traceback"] = [fix(t) for t in o["traceback"]]
    return n


def main() -> None:
    os.environ.update(TQDM_DISABLE="1", NO_ALBUMENTATIONS_UPDATE="1", PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1")
    wanted = sys.argv[1:]
    pairs = redactions()
    runtime_path = ROOT / "results/metrics/notebook_runtime.json"
    runtime = json.loads(runtime_path.read_text()) if runtime_path.is_file() else {}
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        if wanted and not any(path.name.startswith(w) for w in wanted):
            continue
        nb = nbformat.read(path, as_version=4)
        t0 = time.time()
        try:
            NotebookClient(nb, timeout=None, kernel_name="python3",
                           resources={"metadata": {"path": str(path.parent)}}).execute()
        finally:
            n = redact_outputs(nb, pairs)
            nbformat.write(nb, path)
        runtime[path.name] = round(time.time() - t0, 1)
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_path.write_text(json.dumps(runtime, indent=1) + "\n")
        print(f"{path.name}: executed in {runtime[path.name]:.0f}s; {n} absolute-path occurrence(s) redacted; "
              f"{path.stat().st_size / 1024:.0f} KiB", flush=True)


if __name__ == "__main__":
    main()
