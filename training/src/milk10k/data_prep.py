"""Build the per-lesion layout (data/README.md) from the raw ISIC MILK10k release.

Source files (https://challenge.isic-archive.com/data/#milk10k, CC BY-NC 4.0) go in
data/raw/_source/ (see data/README.md):

    MILK10k_Training_Input.zip   MILK10k_Training_Metadata.csv   MILK10k_Training_GroundTruth.csv
    MILK10k_Test_Input.zip       MILK10k_Test_Metadata.csv       MILK10k_Training_Supplement.csv

    python -m milk10k.data_prep --source data/raw/_source --out data/raw

Release-specific choices:
  * No patient_id in MILK10k → patient_id := lesion_id. Grouped CV
    can't stop leakage across lesions of the same patient, so CV is somewhat optimistic.
  * skin_tone_class 0–5 → fitzpatrick_type 1–6 (monotone +1).
  * MILK10k site names → a shared site vocabulary where equivalent; trunk/hand/foot are kept as-is.
  * The official test set has no public labels: test_metadata.csv lists it for inference only.
"""
import argparse
import shutil
import zipfile
from pathlib import Path

import pandas as pd

from .metrics import CLASSES, MALIGNANT

SITE_MAP = {
    "head_neck_face": "head/neck",
    "upper_extremity": "upper extremity",
    "lower_extremity": "lower extremity",
    "genital": "oral/genital",
    "trunk": "trunk",       # MILK10k does not split anterior/posterior/lateral torso
    "hand": "hand",         # hand ≠ palm, so not mapped to palms/soles
    "foot": "foot",
}
COLUMNS = ["isic_id", "lesion_id", "patient_id", "derm_image_path", "clinical_image_path",
           "clinical_available", "age_approx", "sex", "anatomical_site_general",
           "fitzpatrick_type", "diagnosis_confirm_type", "diagnostic_class_11", "target_binary"]


def build_split(meta: pd.DataFrame, split: str, gt: pd.DataFrame | None = None,
                supp: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-image MILK10k metadata → one row per lesion (columns in data/README.md)."""
    is_derm = meta["image_type"].eq("dermoscopic")
    is_clin = meta["image_type"].str.startswith("clinical")
    assert (is_derm | is_clin).all(), f"unknown image_type: {set(meta.image_type)}"
    derm = meta[is_derm].set_index("lesion_id")
    clin = meta[is_clin].set_index("lesion_id")
    assert derm.index.is_unique and clin.index.is_unique, "expected one image per (lesion, modality)"

    clin_id = clin["isic_id"].reindex(derm.index)
    unknown_sites = set(derm["site"].dropna()) - set(SITE_MAP)
    assert not unknown_sites, f"unmapped sites: {unknown_sites}"

    df = pd.DataFrame({
        "isic_id": derm["isic_id"],
        "lesion_id": derm.index,
        "patient_id": derm.index,   # fallback: no patient_id in the release
        "derm_image_path": f"{split}/derm/" + derm["isic_id"] + ".jpg",
        "clinical_image_path": (f"{split}/clinical/" + clin_id + ".jpg").fillna(""),
        "clinical_available": clin_id.notna().astype("int8"),
        "age_approx": derm["age_approx"].astype(float),
        "sex": derm["sex"],
        "anatomical_site_general": derm["site"].map(SITE_MAP),
        "fitzpatrick_type": (derm["skin_tone_class"] + 1).astype("Int8"),
    }).reset_index(drop=True)

    if gt is None:
        return df[[c for c in COLUMNS if c in df.columns
                   and c not in ("diagnosis_confirm_type", "diagnostic_class_11", "target_binary")]]

    gt = gt.set_index("lesion_id")[CLASSES]
    assert (gt.sum(axis=1) == 1).all(), "ground truth is not one-hot"
    df["diagnostic_class_11"] = gt.idxmax(axis=1).reindex(df["lesion_id"]).to_numpy()
    assert df["diagnostic_class_11"].notna().all(), "lesions without ground truth"
    df["target_binary"] = df["diagnostic_class_11"].isin(MALIGNANT).astype("int8")
    df["diagnosis_confirm_type"] = df["isic_id"].map(supp.set_index("isic_id")["diagnosis_confirm_type"])
    return df[COLUMNS]


def unpack_images(zip_path: Path, out: Path, split: str, df: pd.DataFrame) -> None:
    """Extract the release zip and move each image to the path named in the metadata."""
    tmp = out / f"_extract_{split}"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp)
    by_id = {p.stem: p for p in tmp.rglob("*.jpg")}
    wanted = pd.concat([df["derm_image_path"], df["clinical_image_path"]])
    wanted = wanted[wanted != ""]
    missing = [rel for rel in wanted if Path(rel).stem not in by_id]
    if missing:
        raise SystemExit(f"{len(missing)} images from the metadata are not in {zip_path.name}, "
                         f"e.g. {missing[:3]}")
    for rel in wanted:
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(by_id[Path(rel).stem], dst)
    shutil.rmtree(tmp)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="data/raw/_source")
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()
    src, out = Path(args.source), Path(args.out)

    need = ["MILK10k_Training_Input.zip", "MILK10k_Training_Metadata.csv",
            "MILK10k_Training_GroundTruth.csv", "MILK10k_Training_Supplement.csv",
            "MILK10k_Test_Input.zip", "MILK10k_Test_Metadata.csv"]
    absent = [f for f in need if not (src / f).is_file()]
    if absent:
        raise SystemExit(f"Missing in {src}: {absent}\n"
                         "Download from https://challenge.isic-archive.com/data/#milk10k")

    train = build_split(pd.read_csv(src / "MILK10k_Training_Metadata.csv"), "train",
                        gt=pd.read_csv(src / "MILK10k_Training_GroundTruth.csv"),
                        supp=pd.read_csv(src / "MILK10k_Training_Supplement.csv"))
    test = build_split(pd.read_csv(src / "MILK10k_Test_Metadata.csv"), "test")

    for split, df, zname in (("train", train, "Training"), ("test", test, "Test")):
        unpack_images(src / f"MILK10k_{zname}_Input.zip", out, split, df)
        df.to_csv(out / f"{split}_metadata.csv", index=False)
        print(f"{split}: {len(df)} lesions → {out / f'{split}_metadata.csv'}")

    # Uninformative but valid prediction file: training-set class priors.
    prior = train["diagnostic_class_11"].value_counts(normalize=True).reindex(CLASSES, fill_value=0)
    sub = pd.DataFrame({"isic_id": test["isic_id"], "target_binary": prior[list(MALIGNANT)].sum()})
    for c in CLASSES:
        sub[c] = prior[c]
    sub.to_csv(out / "sample_submission.csv", index=False)

    # ---- Integrity checks ------------------------------------------------------
    for df in (train, test):
        paths = pd.concat([df["derm_image_path"], df["clinical_image_path"]])
        assert all((out / p).is_file() for p in paths if p), "image path does not resolve"
    print("Class counts:", train["diagnostic_class_11"].value_counts().to_dict())
    print("Fitzpatrick:", train["fitzpatrick_type"].value_counts(dropna=False).sort_index().to_dict())
    print(f"Malignant rate: {train.target_binary.mean():.3f} | "
          f"sample_submission.csv: {len(sub)} rows")


if __name__ == "__main__":
    main()
