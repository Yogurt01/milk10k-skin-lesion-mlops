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
(decision P1-DEC-09). `age_approx` and `anatomical_site_general` are optional (P1-API-11): omitted or empty means
missing, encoded as in the release (age-missing flag, site code 0), and listed in `metadata_missing`; `sex` and
`fitzpatrick_type` are required; 413 oversized upload (`MAX_UPLOAD_MB`); 503 model not ready, or server busy (more
than `MAX_CONCURRENT_PREDICTIONS` running plus `MAX_QUEUED_PREDICTIONS` waiting; `Retry-After`); 400 body that is not
parseable multipart; 500 generic message for unexpected errors (no stack trace in the response).

Concurrency (P1-API-05): `/health`, `/ready` and the `get_predictor` dependency are `async def` and never block, so
they never wait for the threadpool. `/predict` waits for a slot of the `PredictionLimiter` in the event loop (a waiting
request holds no thread) and only then runs the blocking work in the threadpool, so at most
`MAX_CONCURRENT_PREDICTIONS` threads do inference. Overload is not "not ready": `/ready` stays 200.
"""

import asyncio
import logging
import re
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import DISCLAIMER, __version__
from .package import load_package
from .predictor import InvalidInput, Prediction, Predictor
from .preprocess import IMAGE_PIXEL_CAP_APPLIED, PreprocessSpec
from .settings import Settings, load_settings

logger = logging.getLogger("milk10k_serving.api")
MB = 1024 * 1024
# Field names used by the predictor -> form field names of the API
FORM_FIELDS = {"dermoscopic_image": "derm_image", "clinical_image": "clinical_image"}
# Metadata fields that may be omitted (missing in 20 / 31 of 5,240 training lesions and 0 / 6 of 479 test lesions);
# `sex` and `fitzpatrick_type` were never missing, so their missing code is untrained and stays a 422 (P1-API-11).
OPTIONAL_METADATA = ("age_approx", "anatomical_site_general")
RETRY_AFTER_BUSY = "1"  # seconds; one prediction takes ~0.7 s on 4 threads (docs/serving_measurements.md)


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
    metadata_missing: list[str] = Field(
        description="Optional metadata fields that were omitted or empty and were encoded as missing, as in the "
        "release (`age_approx`: age-missing flag; `anatomical_site_general`: site code 0). Empty list when none.",
        examples=[[], ["anatomical_site_general"]],
    )
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
    413: {
        "model": ErrorResponse,
        "description": "The request body is larger than two images at `MAX_UPLOAD_MB` plus 1 MB of form overhead "
        "(checked from `Content-Length` before the body is read), or a single uploaded image is larger than "
        "`MAX_UPLOAD_MB`.",
    },
    422: {
        "description": "`sex` or `fitzpatrick_type` is missing, a form field has an invalid value (allowed values come "
        "from the model package's `tabular_encoding.json`; `age_approx` 0-120), or the dermoscopic image is missing, "
        "empty or not a decodable image (P1-DEC-09). Not errors: a missing or undecodable **clinical** image is "
        "treated as missing, as in the release (`clinical_image_used: false`), and an omitted or empty `age_approx` "
        "or `anatomical_site_general` is encoded as missing (listed in `metadata_missing`)."
    },
    503: {
        "model": ErrorResponse,
        "description": "The model is still loading (`Retry-After`) or failed to load; or the server is busy: "
        "`MAX_CONCURRENT_PREDICTIONS` predictions are running and `MAX_QUEUED_PREDICTIONS` are waiting "
        '(`Retry-After`). Busy is not "not ready": `/ready` stays 200. Model readiness is checked before field '
        "validation, so a request sent while the model is not ready gets 503 even if a field is invalid.",
    },
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
    path -> `<path>/<file name>`. File names and the rest of the text (e.g. a sha256 mismatch) are kept. A MODEL_DIR
    of "/" (or one that resolves to it) is not substituted. Limitation: a path containing spaces is only redacted up
    to the first space."""
    if model_dir is not None:
        forms = {str(model_dir), str(Path(model_dir).resolve())}
        for form in sorted((f.rstrip("/") for f in forms if "/" in f and f.rstrip("/")), key=len, reverse=True):
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


# --------------------------------------------------------------------------------------------------------------------
# Concurrency limit for /predict
# --------------------------------------------------------------------------------------------------------------------
class PredictionLimiter:
    """At most `max_running` predictions at once; up to `max_queued` more wait in the event loop (holding no thread);
    any further request is rejected at once with 503 + `Retry-After`. Used from one event loop, so the counters need
    no lock. Simpler than a threading semaphore with a timeout, whose waiting requests would each hold a threadpool
    thread (the pool is shared with every sync call of the app)."""

    def __init__(self, max_running: int, max_queued: int):
        self.max_running = max_running
        self.max_queued = max_queued
        self.running = 0
        self.waiting = 0
        self._semaphore = asyncio.Semaphore(max_running)

    @asynccontextmanager
    async def slot(self):
        if self.running + self.waiting >= self.max_running + self.max_queued:
            raise HTTPException(
                503,
                detail="server busy: too many predictions in progress, retry later",
                headers={"Retry-After": RETRY_AFTER_BUSY},
            )
        self.waiting += 1
        try:
            await self._semaphore.acquire()
        finally:
            self.waiting -= 1
        self.running += 1
        try:
            yield
        finally:
            self.running -= 1
            self._semaphore.release()


async def get_predictor(request: Request) -> Predictor:
    """FastAPI dependency: the loaded predictor, or 503 while loading / after a failed load. Tests override it.
    `async def` because it never blocks: it must not wait for the threadpool when predictions fill it."""
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
    """Errors (FastAPI's validation-error format) for form values outside the package's vocabulary. A missing (None)
    optional field is not an error."""
    errors = []
    for field, allowed in allowed_values(spec).items():
        value = values[field]
        if value is None and field in OPTIONAL_METADATA:
            continue
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


def to_response(prediction: Prediction, metadata_missing: list[str] | None = None) -> PredictionResponse:
    return PredictionResponse(
        p_malignant=prediction.p_malignant,
        probabilities=prediction.probabilities,
        diagnosis=prediction.diagnosis,
        clinical_image_used=prediction.clinical_image_used,
        model=prediction.model,
        model_version=prediction.model_version,
        metadata_missing=list(metadata_missing or []),
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


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness: the process is up (always 200)",
    description="Always 200 while the process serves requests, also while the model loads, after a failed load and "
    "under overload. Never waits for running predictions. Use it for a liveness probe.",
)
async def health() -> HealthResponse:
    return HealthResponse()


@router.get(
    "/ready",
    response_model=ReadyResponse,
    response_model_exclude_none=True,
    summary="Readiness: 200 once the model is loaded, 503 while loading or after a failed load",
    description="200 with the model name, version and load time once the model package is loaded and verified "
    "(sha256). 503 with `status` `loading`, or `failed` and the reason (absolute paths redacted). Overload does not "
    "change readiness: when `/predict` is busy (503 + `Retry-After`) `/ready` stays 200. Use it for a readiness "
    "probe.",
    responses={503: {"model": ReadyResponse, "description": "Loading, or loading failed (`error` says why)."}},
)
async def ready(request: Request):
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
    "model package's `tabular_encoding.json` (listed in this schema once the model is loaded). `sex` and "
    "`fitzpatrick_type` are required; `age_approx` (0-120) and `anatomical_site_general` are optional: omitted or "
    "empty means missing, encoded as in the release and listed in the response's `metadata_missing`. At most "
    "`MAX_CONCURRENT_PREDICTIONS` predictions run at once and `MAX_QUEUED_PREDICTIONS` more wait; further requests get "
    "503 with `Retry-After` while `/ready` stays 200. Research model, not a medical device.",
    responses=PREDICT_RESPONSES,
)
async def predict(
    request: Request,
    predictor: Annotated[Predictor, Depends(get_predictor)],
    derm_image: Annotated[UploadFile, File(description="Dermoscopic image (required).")],
    sex: Annotated[
        str, Form(description="Sex (required), as in the model package's tabular encoding.", examples=["male"])
    ],
    fitzpatrick_type: Annotated[
        int,
        Form(description="Skin-tone code (Fitzpatrick type, required), per the tabular encoding.", examples=[4]),
    ],
    age_approx: Annotated[
        float | None,
        Form(
            ge=0,
            le=120,
            allow_inf_nan=False,
            description="Approximate age in years, 0-120 (optional). Omitted or empty = missing (age-missing flag, "
            "as in the release); listed in `metadata_missing`.",
            examples=[60],
        ),
    ] = None,
    anatomical_site_general: Annotated[
        str | None,
        Form(
            description="Anatomical site, as in the tabular encoding (optional). Omitted or empty = missing (site "
            "code 0, as in the release); listed in `metadata_missing`.",
            examples=["head/neck"],
        ),
    ] = None,
    clinical_image: Annotated[
        UploadFile | None,
        File(description="Clinical close-up (optional). Missing or undecodable = treated as missing."),
    ] = None,
) -> PredictionResponse:
    limiter: PredictionLimiter = request.app.state.limiter
    async with limiter.slot():  # waits in the event loop; 503 at once if the queue is full
        return await run_in_threadpool(
            _predict_blocking,
            request.app.state.settings,
            predictor,
            derm_image,
            clinical_image,
            age_approx=age_approx,
            sex=sex,
            anatomical_site_general=anatomical_site_general,
            fitzpatrick_type=fitzpatrick_type,
        )


def _predict_blocking(
    settings: Settings,
    predictor: Predictor,
    derm_image: UploadFile,
    clinical_image: UploadFile | None,
    *,
    age_approx: float | None,
    sex: str,
    anatomical_site_general: str | None,
    fitzpatrick_type: int,
) -> PredictionResponse:
    """The blocking part of /predict (file reads, validation against the package, inference); runs in a thread."""
    limit = settings.max_upload_mb * MB
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
    optional = {"age_approx": age_approx, "anatomical_site_general": anatomical_site_general}
    missing = [field for field in OPTIONAL_METADATA if optional[field] is None]
    if missing:
        logger.info("metadata treated as missing: %s", ", ".join(missing))
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
    return to_response(prediction, missing)


# --------------------------------------------------------------------------------------------------------------------
# App factory
# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
# Request ids and logging (P1-PY-07). Log lines carry ids, sizes, status and latency only: never image bytes or
# metadata values (medical data).
# --------------------------------------------------------------------------------------------------------------------
REQUEST_ID_HEADER = "X-Request-ID"
_VALID_REQUEST_ID = re.compile(r"[A-Za-z0-9._:-]{1,64}")
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


def request_id_from(header: str | None) -> str:
    """The client's X-Request-ID if it is 1-64 characters of [A-Za-z0-9._:-], otherwise a new random id."""
    if header and _VALID_REQUEST_ID.fullmatch(header):
        return header
    return uuid.uuid4().hex


class RequestIdFilter(logging.Filter):
    """Adds `request_id` (from the current request's context, "-" outside a request) to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def _configure_logging(level: str) -> None:
    package_logger = logging.getLogger("milk10k_serving")
    package_logger.setLevel(level)
    if not package_logger.handlers:
        handler = logging.StreamHandler()
        handler.addFilter(RequestIdFilter())
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s"))
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
    if not IMAGE_PIXEL_CAP_APPLIED:
        logger.error("decoded-image cap not applied: cv2 was imported before milk10k_serving.preprocess")
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
    app.state.limiter = PredictionLimiter(settings.max_concurrent_predictions, settings.max_queued_predictions)
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

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        """Outermost: request id (accepted from X-Request-ID or generated, returned in the response header and set
        for every log line of the request), one access line (method, path, status, latency, body size), and the
        generic 500 for any unexpected error."""
        request_id = request_id_from(request.headers.get(REQUEST_ID_HEADER))
        token = request_id_var.set(request_id)
        request.state.request_id = request_id
        t0 = time.perf_counter()
        try:
            try:
                response = await call_next(request)
            except Exception:
                logger.exception(
                    "unexpected error on %s %s (request_id=%s)", request.method, request.url.path, request_id
                )
                response = JSONResponse(status_code=500, content={"detail": "internal server error"})
            response.headers[REQUEST_ID_HEADER] = request_id
            level = logging.DEBUG if request.url.path in ("/health", "/ready") else logging.INFO  # probes: quiet
            logger.log(
                level,
                "request_id=%s method=%s path=%s status=%d duration_ms=%.1f request_bytes=%s",
                request_id,
                request.method,
                request.url.path,
                response.status_code,
                (time.perf_counter() - t0) * 1000,
                request.headers.get("content-length", "-"),
            )
            return response
        finally:
            request_id_var.reset(token)

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        """Fallback for errors outside `request_context` (the middleware handles the rest)."""
        request_id = getattr(request.state, "request_id", "-")
        logger.exception("unexpected error on %s %s (request_id=%s)", request.method, request.url.path, request_id)
        return JSONResponse(
            status_code=500, content={"detail": "internal server error"}, headers={REQUEST_ID_HEADER: request_id}
        )

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
