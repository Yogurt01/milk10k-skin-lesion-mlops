"""Uploaded bytes + metadata -> prediction (one lesion).

A missing or undecodable dermoscopic image raises `InvalidInput` (the API answers 422, decision P1-DEC-09); the
release's prior-row fallback is not used. A missing or undecodable clinical image is treated as missing, as in the
release (`[NULL]` clinical token, `clin_mask = False`).
"""

from __future__ import annotations

from dataclasses import dataclass

from .package import ModelPackage
from .preprocess import build_inputs, decode_rgb


class InvalidInput(ValueError):
    """The request cannot be scored (e.g. no readable dermoscopic image)."""

    def __init__(self, field: str, message: str):
        super().__init__(f"{field}: {message}")
        self.field = field
        self.message = message


@dataclass(frozen=True)
class Prediction:
    p_malignant: float
    probabilities: dict[str, float]
    diagnosis: str
    clinical_image_used: bool
    model: str
    model_version: str

    def row(self) -> dict[str, float]:
        """The 12 probabilities in the reference CSV's column order (`target_binary`, then the 11 classes)."""
        return {"target_binary": self.p_malignant, **self.probabilities}


class Predictor:
    def __init__(self, package: ModelPackage):
        self.package = package

    def predict(
        self,
        derm: bytes | None,
        clinical: bytes | None = None,
        *,
        age_approx=None,
        sex=None,
        anatomical_site_general=None,
        fitzpatrick_type=None,
    ) -> Prediction:
        derm_img = decode_rgb(derm)
        if derm_img is None:
            reason = "missing" if not derm else "not a decodable image"
            raise InvalidInput("dermoscopic_image", f"dermoscopic image is {reason}")
        clin_img = decode_rgb(clinical)
        feeds = build_inputs(
            self.package.spec,
            derm_img,
            clin_img,
            age_approx=age_approx,
            sex=sex,
            anatomical_site_general=anatomical_site_general,
            fitzpatrick_type=fitzpatrick_type,
        )
        out = self.package.ensemble.predict(feeds)[0]
        classes = self.package.classes
        probabilities = {c: float(p) for c, p in zip(classes, out[1:], strict=True)}
        return Prediction(
            p_malignant=float(out[0]),
            probabilities=probabilities,
            diagnosis=max(probabilities, key=probabilities.get),
            clinical_image_used=clin_img is not None,
            model=self.package.name,
            model_version=self.package.version,
        )
