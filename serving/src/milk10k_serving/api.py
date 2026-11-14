"""FastAPI service for the MILK10k final-P1a-kg model (research model, not a medical device).

Run from the repository root (settings come from environment variables, see `settings.py` and `.env.example`):

    uvicorn milk10k_serving.api:app --host 0.0.0.0 --port 8000

Loading: the lifespan hook starts the model load in a background thread and returns at once, so uvicorn accepts
connections immediately. Uvicorn does not accept connections until lifespan startup finishes, so a blocking load there
would make `/ready` unreachable (connection refused) instead of answering 503. `/health` is always 200; `/ready` is
503 while loading or after a failed load (with the error) and 200 once the predictor is ready. The error shown to
clients has absolute paths redacted (`<MODEL_DIR>/file`, other paths reduced to their file name); the full message
and traceback go to the server log.

Status contract of `POST /predict`: 200 ok; 422 missing or invalid field, or missing / undecodable dermoscopic image
(decision P1-DEC-09); 413 oversized upload (`MAX_UPLOAD_MB`); 503 model not ready; 400 body that is not parseable
multipart; 500 generic message for unexpected errors (no stack trace in the response).
"""

import logging
import re
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import DISCLAIMER, __version__
from .package import load_package
from .predictor import InvalidInput, Prediction, Predictor
from .preprocess import PreprocessSpec
from .settings import Settings, load_settings

logger = logging.getLogger("milk10k_serving.api")
MB = 1024 * 1024
# Field names used by the predictor -> form field names of the API
FORM_FIELDS = {"dermoscopic_image": "derm_image", "clinical_image": "clinical_image"}


# --------------------------------------------------------------------------------------------------------------------
# Response models
# --------------------------------------------------------------------------------------------------------------------
class PredictionResponse(BaseModel):
    p_malignant: float = Field(ge=0, le=1, description="P(malignant) from the model's binary head (`target_binary`).")
    probabilities: dict[str, float] = Field(
        description="11-class diagnosis probabilities keyed by diagnosis code (classes listed in the model package "
        "manifest); they sum to 1."
    )
    diagnosis: str = Field(description="Diagnosis code with the highest probability.")
    clinical_image_used: bool = Field(
        description="False when no clinical image was sent or it could not be decoded (it is then treated as missing)."
    )
    model: str = Field(description="Model package name (manifest `name`).")
    model_version: str = Field(description="Model package version (manifest `version`).")
    disclaimer: str = Field(min_length=1, description="Always present: research model, not a medical device.")


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadyResponse(BaseModel):
    status: Literal["loading", "ready", "failed"]
    model: str | None = None
    model_version: str | None = None
    load_seconds: float | None = None
    error: str | None = Field(
        default=None,
        description="Why loading failed: error type and message (e.g. a sha256 mismatch), absolute paths redacted.",
    )


class ErrorResponse(BaseModel):
    detail: str


PREDICT_RESPONSES: dict[int | str, dict[str, Any]] = {
    200: {"description": "Prediction for one lesion. `disclaimer` is present in every response."},
    400: {"model": ErrorResponse, "description": "The request body is not parseable multipart form data."},
    413: {"model": ErrorResponse, "description": "An uploaded image (or the whole request) exceeds `MAX_UPLOAD_MB`."},
    422: {
        "description": "A form field is missing or invalid (allowed values come from the model package's "
        "`tabular_encoding.json`), or the dermoscopic image is missing, empty or not a decodable image (P1-DEC-09). "
        "A missing or undecodable **clinical** image is not an error: it is treated as missing, as in the release, "
        "and the response says `clinical_image_used: false`."
    },
    503: {"model": ErrorResponse, "description": "The model is still loading (`Retry-After`) or failed to load."},
    500: {"model": ErrorResponse, "description": "Unexpected server error; generic message, no details."},
}


# --------------------------------------------------------------------------------------------------------------------
# Load errors as shown to clients: no absolute server paths
# --------------------------------------------------------------------------------------------------------------------
# An absolute POSIX path: "/" not preceded by a word character, ".", ":", "/" or "<" (so "head/neck", "1/2", URLs and
# the "<MODEL_DIR>/" placeholder are left alone), followed by characters that do not end a path in a message.
_ABSOLUTE_PATH = re.compile(r"(?<![\w.:/<>-])/[^\s'\"`,;:()\[\]{}<>]+")


def redact_paths(message: str, model_dir: Path | None = None) -> str:
    """`message` without absolute paths: MODEL_DIR (as configured and resolved) -> `<MODEL_DIR>`, any other absolute
    path -> `<path>/<file name>`. File names and the rest of the text (e.g. a sha256 mismatch) are kept."""
    if model_dir is not None:
        forms = {str(model_dir), str(Path(model_dir).resolve())}
        for form in sorted((f.rstrip("/") for f in forms if "/" in f), key=len, reverse=True):
            message = message.replace(form, "<MODEL_DIR>")

    def file_name(match: re.Match) -> str:
        name = match.group(0).rstrip("/").rsplit("/", 1)[-1]
        return f"<path>/{name}" if name else "<path>"

    return _ABSOLUTE_PATH.sub(file_name, message)


# --------------------------------------------------------------------------------------------------------------------
# Model state: loaded once, in a background thread started from the lifespan hook
# --------------------------------------------------------------------------------------------------------------------
class ModelState:
    def __init__(self, loader: Callable[[], Predictor], model_dir: Path | None = None):
        self._loader = loader
        self._model_dir = model_dir  # only used to redact it from the error shown to clients
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._thread: threading.Thread | None = None
        self.status: Literal["loading", "ready", "failed"] = "loading"
        self.predictor: Predictor | None = None
        self.error: str | None = None
        self.load_seconds: float | None = None
        self.on_ready: Callable[[], None] | None = None

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._load, name="model-loader", daemon=True)
            self._thread.start()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until loading finished (ready or failed); True if it finished within `timeout`."""
        return self._done.wait(timeout)

    def _load(self) -> None:
        t0 = time.perf_counter()
        try:
            predictor = self._loader()
        except Exception as exc:  # any load failure is reported by /ready, never raised into the server
            detail = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self.error = redact_paths(detail, self._model_dir)
                self.status = "failed"
            logger.error("model failed to load: %s", detail, exc_info=True)  # full detail stays server-side
            self._done.set()
            return
        with self._lock:
            self.predictor = predictor
            self.load_seconds = time.perf_counter() - t0
            self.status = "ready"
        package = predictor.package
        logger.info("model loaded: %s %s in %.1f s", package.name, package.version, self.load_seconds)
        if self.on_ready is not None:
            self.on_ready()
        self._done.set()


def default_loader(settings: Settings) -> Callable[[], Predictor]:
    def load() -> Predictor:
        return Predictor(load_package(settings.model_dir, settings.verify_sha256, settings.ort_threads))

    return load


def get_predictor(request: Request) -> Predictor:
    """FastAPI dependency: the loaded predictor, or 503 while loading / after a failed load. Tests override it."""
    state: ModelState = request.app.state.model
    if state.status == "ready" and state.predictor is not None:
        return state.predictor
    if state.status == "failed":
        raise HTTPException(503, detail=f"model failed to load: {state.error}")
    raise HTTPException(503, detail="model is loading", headers={"Retry-After": "5"})


# --------------------------------------------------------------------------------------------------------------------
# Validation and response assembly (pure functions, unit-tested)
# --------------------------------------------------------------------------------------------------------------------
def allowed_values(spec: PreprocessSpec) -> dict[str, list]:
    """Allowed form values, all taken from the model package's `tabular_encoding.json` (via PreprocessSpec)."""
    return {
        "sex": sorted(spec.sex_codes, key=spec.sex_codes.get),
        "anatomical_site_general": list(spec.sites),
        "fitzpatrick_type": list(spec.fitzpatrick_values),
    }


def metadata_errors(spec: PreprocessSpec, **values) -> list[dict]:
    """Errors (FastAPI's validation-error format) for form values outside the package's vocabulary."""
    errors = []
    for field, allowed in allowed_values(spec).items():
        value = values[field]
        if value not in allowed:
            errors.append(
                {
                    "type": "enum",
                    "loc": ("body", field),
                    "msg": "Input should be one of: " + ", ".join(map(str, allowed)),
                    "input": value,
                }
            )
    return errors


def to_response(prediction: Prediction) -> PredictionResponse:
    return PredictionResponse(
        p_malignant=prediction.p_malignant,
        probabilities=prediction.probabilities,
        diagnosis=prediction.diagnosis,
        clinical_image_used=prediction.clinical_image_used,
        model=prediction.model,
        model_version=prediction.model_version,
        disclaimer=DISCLAIMER,
    )


def read_limited(upload: UploadFile, field: str, limit_bytes: int) -> bytes:
    """The upload's bytes; 413 if it is larger than `limit_bytes`."""
    data = upload.file.read(limit_bytes + 1)
    if len(data) > limit_bytes:
        raise HTTPException(413, detail=f"{field} is larger than {limit_bytes // MB} MB (MAX_UPLOAD_MB)")
    return data


# --------------------------------------------------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------------------------------------------------
router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="Liveness: the process is up (always 200)")
def health() -> HealthResponse:
    return HealthResponse()


@router.get(
    "/ready",
    response_model=ReadyResponse,
    response_model_exclude_none=True,
    summary="Readiness: 200 once the model is loaded, 503 while loading or after a failed load",
    responses={503: {"model": ReadyResponse, "description": "Loading, or loading failed (`error` says why)."}},
)
def ready(request: Request):
    state: ModelState = request.app.state.model
    if state.status == "ready" and state.predictor is not None:
        package = state.predictor.package
        return ReadyResponse(
            status="ready", model=package.name, model_version=package.version, load_seconds=round(state.load_seconds, 2)
        )
    body = ReadyResponse(status=state.status, error=state.error)
    return JSONResponse(status_code=503, content=body.model_dump(exclude_none=True))


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict P(malignant) and the 11-class diagnosis for one lesion",
    description="Multipart form: a dermoscopic image (required), a clinical close-up (optional) and four metadata "
    "fields. Images may be any format OpenCV decodes (e.g. JPEG, PNG); the type is checked by decoding, not by the "
    "Content-Type header. Allowed values of `sex`, `anatomical_site_general` and `fitzpatrick_type` come from the "
    "model package's `tabular_encoding.json` (listed in this schema once the model is loaded). "
    "Research model, not a medical device.",
    responses=PREDICT_RESPONSES,
)
def predict(
    request: Request,
    predictor: Annotated[Predictor, Depends(get_predictor)],
    derm_image: Annotated[UploadFile, File(description="Dermoscopic image (required).")],
    age_approx: Annotated[
        float, Form(ge=0, le=120, allow_inf_nan=False, description="Approximate age in years (0-120).")
    ],
    sex: Annotated[str, Form(description="Sex, as in the model package's tabular encoding.")],
    anatomical_site_general: Annotated[str, Form(description="Anatomical site, as in the tabular encoding.")],
    fitzpatrick_type: Annotated[int, Form(description="Skin-tone code (Fitzpatrick type), per the tabular encoding.")],
    clinical_image: Annotated[
        UploadFile | None,
        File(description="Clinical close-up (optional). Missing or undecodable = treated as missing."),
    ] = None,
) -> PredictionResponse:
    limit = request.app.state.settings.max_upload_mb * MB
    derm = read_limited(derm_image, "derm_image", limit)
    clinical = read_limited(clinical_image, "clinical_image", limit) if clinical_image is not None else None
    errors = metadata_errors(
        predictor.package.spec,
        sex=sex,
        anatomical_site_general=anatomical_site_general,
        fitzpatrick_type=fitzpatrick_type,
    )
    if errors:
        raise RequestValidationError(errors)
    try:
        prediction = predictor.predict(
            derm,
            clinical or None,
            age_approx=age_approx,
            sex=sex,
            anatomical_site_general=anatomical_site_general,
            fitzpatrick_type=fitzpatrick_type,
        )
    except InvalidInput as exc:
        field = FORM_FIELDS.get(exc.field, exc.field)
        raise RequestValidationError(
            [{"type": "value_error", "loc": ("body", field), "msg": exc.message, "input": None}]
        ) from None
    return to_response(prediction)


# --------------------------------------------------------------------------------------------------------------------
# App factory
# --------------------------------------------------------------------------------------------------------------------
def _configure_logging(level: str) -> None:
    package_logger = logging.getLogger("milk10k_serving")
    package_logger.setLevel(level)
    if not package_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        package_logger.addHandler(handler)


def _add_vocabulary(schema: dict, spec: PreprocessSpec) -> None:
    """Show the allowed form values (from the loaded package) as enums in the OpenAPI schema of /predict."""
    for component in schema.get("components", {}).get("schemas", {}).values():
        properties = component.get("properties", {})
        if "derm_image" in properties:
            for field, allowed in allowed_values(spec).items():
                if field in properties:
                    properties[field]["enum"] = allowed


def create_app(settings: Settings | None = None, loader: Callable[[], Predictor] | None = None) -> FastAPI:
    settings = settings or load_settings()
    _configure_logging(settings.log_level)
    state = ModelState(loader or default_loader(settings), model_dir=settings.model_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state.start()  # returns at once; the load runs in a background thread
        yield

    app = FastAPI(
        title="MILK10k skin-lesion API",
        version=__version__,
        description=f"P(malignant) and an 11-class diagnosis from a dermoscopic image, an optional clinical image "
        f"and four metadata fields (ISIC MILK10k, model `final-P1a-kg`, ONNX Runtime on CPU). **{DISCLAIMER}**",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.model = state
    app.include_router(router)

    @app.middleware("http")
    async def limit_request_size(request: Request, call_next):
        """Reject a /predict body larger than two images at the limit (+1 MB of form overhead) before reading it."""
        if request.method == "POST" and request.url.path == "/predict":
            length = request.headers.get("content-length", "")
            if length.isdigit() and int(length) > 2 * settings.max_upload_mb * MB + MB:
                return JSONResponse(
                    status_code=413,
                    content={
                        "detail": f"request body is larger than allowed for two {settings.max_upload_mb} MB "
                        "images (MAX_UPLOAD_MB)"
                    },
                )
        return await call_next(request)

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unexpected error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "internal server error"})

    def openapi() -> dict:
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(title=app.title, version=app.version, description=app.description, routes=app.routes)
        if state.status == "ready" and state.predictor is not None:
            _add_vocabulary(schema, state.predictor.package.spec)
            app.openapi_schema = schema  # cache only the complete schema
        return schema

    app.openapi = openapi
    state.on_ready = lambda: setattr(app, "openapi_schema", None)
    return app


app = create_app()
