"""Rebuild sample/ (images + metadata) from a local MILK10k download; images are never committed.

    python sample/make_sample.py                      # uses configs/final.yaml paths.data_dir or MILK10K_DATA_DIR
    python sample/make_sample.py --data-dir /path/to/prepared/milk10k

Reads sample/sample_ids.csv (six test lesions chosen as the first six isic_id values in sorted order; no image or
prediction was looked at when choosing them), copies their derm/clinical images into sample/images/, and writes
sample/sample_metadata.csv with paths relative to sample/. Scenarios exercise the missing-clinical handling:
    intact                 clinical image present (rows also used for the parity check in notebook 07)
    clinical_blank         clinical_image_path = ""            (the official missing-modality encoding)
    clinical_nan           clinical_image_path = NaN
    clinical_missing_file  clinical_image_path points to a file that does not exist
MILK10k images: ISIC Archive, CC BY-NC 4.0, attribution "MILK study team" (data/README.md).
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from milk10k.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None, help="prepared dataset root (default: config / MILK10K_DATA_DIR)")
    args = ap.parse_args()
    data_dir = Path(args.data_dir or os.environ.get("MILK10K_DATA_DIR") or load_config()["paths"]["data_dir"])
    test = pd.read_csv(data_dir / "test_metadata.csv")
    ids = pd.read_csv(HERE / "sample_ids.csv")
    meta = ids.merge(test, on="isic_id", how="left", validate="1:1")
    if meta.derm_image_path.isna().any():
        raise SystemExit(f"IDs not found in {data_dir / 'test_metadata.csv'}: {meta.isic_id[meta.derm_image_path.isna()].tolist()}")
    out = HERE / "images"
    for col in ("derm_image_path", "clinical_image_path"):
        for rel in meta[col]:
            if isinstance(rel, str) and rel:
                dst = out / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(data_dir / rel, dst)
    meta["derm_image_path"] = "images/" + meta["derm_image_path"]
    meta["clinical_image_path"] = ("images/" + meta["clinical_image_path"]).astype(object)
    meta.loc[meta.scenario == "clinical_blank", "clinical_image_path"] = ""
    meta.loc[meta.scenario == "clinical_nan", "clinical_image_path"] = np.nan
    meta.loc[meta.scenario == "clinical_missing_file", "clinical_image_path"] = "images/test/clinical/DOES_NOT_EXIST.jpg"
    meta.loc[meta.scenario != "intact", "clinical_available"] = 0
    meta.to_csv(HERE / "sample_metadata.csv", index=False)
    print(f"sample: {len(meta)} lesions, {sum(1 for _ in out.rglob('*.jpg'))} images -> sample/images/; metadata -> sample/sample_metadata.csv")


if __name__ == "__main__":
    main()
