"""Uploaded bytes + metadata -> prediction (one lesion).

A missing or undecodable dermoscopic image raises `InvalidInput` (the API answers 422, decision P1-DEC-09); the
release's prior-row fallback is not used. A missing or undecodable clinical image is treated as missing, as in the
release (`[NULL]` clinical token, `clin_mask = False`).

`stage` (optional) wraps each step of a prediction: a callable `stage(name, **attributes)` returning a context manager.
The API passes one that opens a tracing span and records the stage duration metric; without it nothing is recorded.
Stages: `decode`, `preprocess`, `inference` (with one `ensemble_member` per member inside). Attributes carry counts
and flags only, never image bytes or metadata values.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass

from .package import ModelPackage
from .preprocess import ImageTooLarge, build_inputs, decode_rgb


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


Stage = Callable[..., AbstractContextManager]


def no_stage(name: str, **attributes) -> AbstractContextManager:
    return nullcontext()


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
        stage: Stage = no_stage,
    ) -> Prediction:
        with stage("decode", clinical_image_provided=bool(clinical)) as span:
            derm_img = _decode(derm, "dermoscopic_image")
            if derm_img is None:
                reason = "missing" if not derm else "not a decodable image"
                raise InvalidInput("dermoscopic_image", f"dermoscopic image is {reason}")
            clin_img = _decode(clinical, "clinical_image")  # undecodable = missing (as the release); too large = 422
            if span is not None:
                span.set_attribute("milk10k.clinical_image_used", clin_img is not None)
        with stage("preprocess"):
            feeds = build_inputs(
                self.package.spec,
                derm_img,
                clin_img,
                age_approx=age_approx,
                sex=sex,
                anatomical_site_general=anatomical_site_general,
                fitzpatrick_type=fitzpatrick_type,
            )
        info = self.package.manifest.get("ensemble", {})
        with stage("inference", members=len(info.get("members", [])), tta=bool(info.get("tta", False))):
            out = self.package.ensemble.predict(feeds, stage=stage)[0]
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


def _decode(data: bytes | None, field: str):
    """decode_rgb, with an image above the decoding cap reported as invalid input (422) for either image: a clinical
    image that is too large is not silently treated as missing."""
    try:
        return decode_rgb(data)
    except ImageTooLarge as exc:
        raise InvalidInput(field, f"{field.replace('_', ' ')} is too large: {exc}") from None
