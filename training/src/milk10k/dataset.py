"""Dataset, tabular encoding, augmentations and sampling for MILK10k.

Both image streams get their own Albumentations 2.x pipeline; metadata uses a fixed encoding (nothing fitted).
"""
import os
import random
from typing import Optional, Sequence

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, WeightedRandomSampler

from .metrics import CLASSES

SITES = ["head/neck", "upper extremity", "lower extremity", "anterior torso",
         "posterior torso", "lateral torso", "palms/soles", "oral/genital",
         "trunk", "hand", "foot"]  # appended for the MILK10k release (src/prepare_data.py)
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}

IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
# Better: use each backbone's own stats, e.g.
# timm.data.resolve_data_config({}, model=backbone)["mean"/"std"]


# --------------------------------------------------------------------------- #
# Tabular features & image I/O
# --------------------------------------------------------------------------- #
def encode_tabular(df: pd.DataFrame) -> dict:
    """Integer-encode categoricals (0 = missing) and scale age. No fitting, so no leakage."""
    site = df.anatomical_site_general.map({s: i + 1 for i, s in enumerate(SITES)}).fillna(0)
    sex = df.sex.map({"male": 1, "female": 2}).fillna(0)
    fst = df.fitzpatrick_type.fillna(0)
    return {
        "site": torch.tensor(site.to_numpy(), dtype=torch.long),
        "sex": torch.tensor(sex.to_numpy(), dtype=torch.long),
        "fst": torch.tensor(fst.to_numpy(), dtype=torch.long),
        "age": torch.tensor((df.age_approx.fillna(0) / 100).to_numpy(), dtype=torch.float32),
        "age_missing": torch.tensor(df.age_approx.isna().to_numpy(), dtype=torch.float32),
    }


def read_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def shades_of_gray(img: np.ndarray, p: int = 6) -> np.ndarray:
    """Shades-of-Gray colour constancy. Apply identically at train and test time."""
    x = img.astype(np.float32)
    illum = np.power(np.mean(np.power(x, p), axis=(0, 1)), 1 / p)
    illum = illum / np.sqrt((illum ** 2).sum()) * np.sqrt(3)
    return np.clip(x / illum, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Augmentations
# --------------------------------------------------------------------------- #
class HairAugmentation(A.ImageOnlyTransform):
    """Draw synthetic hairs (thin, dark, curved strokes) to build robustness to hair occlusion."""

    def __init__(self, num_hairs=(5, 30), p=0.3):
        super().__init__(p=p)
        self.num_hairs = num_hairs

    def apply(self, img, **params):
        img = img.copy()
        h, w = img.shape[:2]
        rng = self.random_generator
        for _ in range(rng.integers(*self.num_hairs)):
            p0 = rng.uniform((0, 0), (w, h))
            angle = rng.uniform(0, 2 * np.pi)
            length = rng.uniform(0.15, 0.6) * max(h, w)
            p2 = p0 + length * np.array([np.cos(angle), np.sin(angle)])
            p1 = (p0 + p2) / 2 + rng.normal(0, 0.1 * length, 2)          # bend
            t = np.linspace(0, 1, 40)[:, None]
            curve = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2  # quadratic Bézier
            shade = int(rng.integers(10, 70))
            color = (shade, int(shade * 0.8), int(shade * 0.6))             # dark brown / black
            thickness = int(rng.integers(1, max(2, min(h, w) // 250) + 1))
            cv2.polylines(img, [curve.astype(np.int32)], False, color, thickness, cv2.LINE_AA)
        return img

    def get_transform_init_args_names(self):
        return ("num_hairs",)


def derm_transforms(size: int = 448, train: bool = True,
                    mean: Sequence[float] = IMNET_MEAN, std: Sequence[float] = IMNET_STD):
    """Dermoscopy: rotation/flip invariant, conservative colour changes."""
    if not train:
        return A.Compose([A.Resize(size, size), A.Normalize(mean, std), ToTensorV2()])
    return A.Compose([
        A.RandomResizedCrop(size=(size, size), scale=(0.5, 1.0), ratio=(0.8, 1.25)),
        A.HorizontalFlip(), A.VerticalFlip(), A.RandomRotate90(),
        A.Affine(rotate=(-45, 45), scale=(0.9, 1.1), p=0.5),
        HairAugmentation(p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02, p=0.7),
        A.OneOf([A.GaussianBlur(blur_limit=(3, 5)), A.MotionBlur(blur_limit=5), A.GaussNoise()], p=0.2),
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.05, 0.12),
                        hole_width_range=(0.05, 0.12), p=0.3),
        A.Normalize(mean, std), ToTensorV2(),
    ])


def clinical_transforms(size: int = 384, train: bool = True,
                        mean: Sequence[float] = IMNET_MEAN, std: Sequence[float] = IMNET_STD):
    """Clinical close-ups: meaningful orientation, strong photometric/device variability."""
    if not train:
        return A.Compose([A.LongestMaxSize(size), A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT),
                          A.Normalize(mean, std), ToTensorV2()])
    return A.Compose([
        A.RandomResizedCrop(size=(size, size), scale=(0.35, 1.0), ratio=(0.75, 1.33)),
        A.HorizontalFlip(),
        A.Affine(rotate=(-30, 30), scale=(0.85, 1.15), shear=(-5, 5), p=0.6),
        A.Perspective(scale=(0.02, 0.06), p=0.3),
        A.RandomBrightnessContrast(0.25, 0.25, p=0.7),
        A.ColorJitter(brightness=0, contrast=0, saturation=0.15, hue=0.02, p=0.5),
        A.OneOf([A.MotionBlur(blur_limit=7), A.Defocus(radius=(1, 3)), A.GaussNoise()], p=0.3),
        A.ImageCompression(quality_range=(40, 95), p=0.4),   # smartphone / WhatsApp artefacts
        A.Downscale(scale_range=(0.5, 0.9), p=0.2),
        A.Normalize(mean, std), ToTensorV2(),
    ])


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class MILKDataset(Dataset):
    """One item per lesion: dermoscopic image, clinical image (or zeros + mask), tabular, labels.

    `root` is the dataset root that the `*_image_path` columns are relative to
    (e.g. `dataset/raw`). With `train=True`, the clinical image is dropped with
    probability `p_drop_clin` to simulate the hidden-test missing-modality rows.
    """

    def __init__(self, df: pd.DataFrame, root: str, derm_tf, clin_tf, clin_size: int = 384,
                 train: bool = False, p_drop_clin: float = 0.15, color_constancy: bool = False):
        self.df = df.reset_index(drop=True)
        self.root, self.derm_tf, self.clin_tf = root, derm_tf, clin_tf
        self.clin_size, self.train, self.p_drop_clin = clin_size, train, p_drop_clin
        self.color_constancy = color_constancy
        self.tab = encode_tabular(self.df)
        self.has_labels = "diagnostic_class_11" in self.df

    def __len__(self):
        return len(self.df)

    def _load(self, rel_path: str) -> np.ndarray:
        img = read_rgb(os.path.join(self.root, rel_path))
        return shades_of_gray(img) if self.color_constancy else img

    def __getitem__(self, i):
        r = self.df.iloc[i]
        derm = self.derm_tf(image=self._load(r.derm_image_path))["image"]

        has_clin = isinstance(r.clinical_image_path, str) and r.clinical_image_path != ""
        if has_clin and self.train and random.random() < self.p_drop_clin:
            has_clin = False  # modality dropout: simulate the hidden-test condition
        if has_clin:
            clin = self.clin_tf(image=self._load(r.clinical_image_path))["image"]
        else:
            clin = torch.zeros(3, self.clin_size, self.clin_size)

        item = {"derm": derm, "clin": clin, "clin_mask": torch.tensor(has_clin),
                **{k: v[i] for k, v in self.tab.items()}}
        if self.has_labels:
            item["y_cls"] = torch.tensor(CLS2IDX[r.diagnostic_class_11])
            item["y_bin"] = torch.tensor(float(r.target_binary))
        return item


# --------------------------------------------------------------------------- #
# Imbalance-aware sampling
# --------------------------------------------------------------------------- #
def make_sqrt_balanced_sampler(labels: pd.Series,
                               num_samples: Optional[int] = None) -> WeightedRandomSampler:
    """Sample weight ∝ class_freq^-0.5 (softer than full inverse-frequency)."""
    freq = labels.map(labels.value_counts())
    w = torch.tensor((1.0 / np.sqrt(freq)).to_numpy(), dtype=torch.double)
    return WeightedRandomSampler(w, num_samples=num_samples or len(w), replacement=True)
