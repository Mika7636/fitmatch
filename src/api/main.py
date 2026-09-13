"""The FastAPI application: lifespan, CORS, error shape, request logging.

Run it::

    run_api.bat
    python -m src.api
    python -m uvicorn src.api.main:app --reload --port 8000

Then open http://127.0.0.1:8000/docs.

The one thing this file must get right is the lifespan handler.  Fitting
TF-IDF over 10,000 products takes about a second; doing it per request would
put that second on every recommendation.  So the cascade is loaded once, here,
before the first request is served, and the load is timed and logged.  Every
route reaches it through the ``get_service`` dependency and none of them may
build their own.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.logging_setup import configure_logging
from src.api.routes import router
from src.api.service import RecommenderService
from src.api.vocabulary import get_vocabulary
from src.recommender.knowledge_based import DEFAULT_MIN_RESULTS
from src.recommender.pipeline import CACHE_PATH

LOGGER = logging.getLogger("fitmatch.api")

# The Vite dev server. 127.0.0.1 is listed as well as localhost because a
# browser treats them as different origins and Vite prints whichever it feels
# like.  Override with FITMATCH_CORS_ORIGINS="http://a,http://b".
DEFAULT_CORS_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")

API_TITLE = "FitMatch API"
API_VERSION = "0.3.0"
API_DESCRIPTION = """
HTTP access to the FitMatch two-technique recommender.

* **Knowledge-based constraint filtering (Ch7)** decides which products are
  admissible, relaxing soft constraints until enough candidates remain.
* **Content-based filtering, TF-IDF + cosine (Ch3)** decides the order.

Two measured facts from session 2 shape every `/api/recommend` response:
relaxation is the normal path (the median user has **no** products matching
every stated preference, and colour is relaxed for 98.7% of users), and 85.2%
of users are ranked by the cold-start profile-only mode.  Both are reported
per request in `constraints` and `scoring` -- they are part of the contract,
not debug output, because a UI that hides them is misreporting the result.
""".strip()


def _cors_origins() -> list[str]:
    raw = os.environ.get("FITMATCH_CORS_ORIGINS", "")
    if not raw.strip():
        return list(DEFAULT_CORS_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        LOGGER.warning("ignoring non-numeric %s=%r", name, os.environ.get(name))
        return default


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the recommender once, before the first request.

    A failure here does not kill the process: the app stays up so
    ``/api/health`` can report *why* it is not ready, which is more useful
    than a container that exits before anyone can read the traceback.  Every
    other route answers 503 until it loads.
    """
    configure_logging()
    app.state.service = None
    app.state.startup_error = None
    app.state.cache_path = str(CACHE_PATH)

    min_results = _env_int("FITMATCH_MIN_RESULTS", DEFAULT_MIN_RESULTS)
    rebuild = os.environ.get("FITMATCH_REBUILD_CACHE", "").lower() in {"1", "true", "yes"}

    try:
        get_vocabulary()  # prime, so the first request does not pay for the read
        app.state.service = RecommenderService.load(
            min_results=min_results, rebuild_cache=rebuild
        )
    except Exception as error:  # noqa: BLE001 -- reported, not swallowed
        app.state.startup_error = "{}: {}".format(type(error).__name__, error)
        LOGGER.exception(
            "startup failed: the API is running but not ready",
            extra={"event": "startup_failed"},
        )

    try:
        yield
    finally:
        LOGGER.info("shutting down", extra={"event": "shutdown"})
        app.state.service = None


def create_app(*, cors_origins: list[str] | None = None) -> FastAPI:
    """Build the application. A function, not a module-level side effect, so
    the tests can build a fresh one without re-importing the package."""
    configure_logging()

    app = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins if cors_origins is not None else _cors_origins(),
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """One structured line per request: method, path, status, latency."""
        request_id = uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            LOGGER.exception(
                "request failed",
                extra={
                    "event": "request_error",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
                },
            )
            raise
        latency_ms = (time.perf_counter() - started) * 1000.0
        LOGGER.info(
            "%s %s -> %s",
            request.method,
            request.url.path,
            response.status_code,
            extra={
                "event": "request",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "query": str(request.url.query) or None,
                "status": response.status_code,
                "latency_ms": round(latency_ms, 2),
            },
        )
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-ms"] = "{:.2f}".format(latency_ms)
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        """422 that names the offending field and what would have been valid.

        The message comes from the validator, and the vocabulary validators
        list their options, so a bad ``primary_sport`` comes back naming the
        ten sports rather than just saying no.
        """
        items = [
            {
                "field": _field_name(item.get("loc", ())),
                "message": _clean_message(item.get("msg", "")),
                "type": str(item.get("type", "")),
            }
            for item in error.errors()
        ]
        LOGGER.info(
            "422 %s %s",
            request.method,
            request.url.path,
            extra={
                "event": "validation_error",
                "request_id": getattr(request.state, "request_id", None),
                "path": request.url.path,
                "fields": [item["field"] for item in items],
            },
        )
        return JSONResponse(
            # The literal, not status.HTTP_422_*: starlette renamed the constant
            # and deprecated the old spelling, so naming it pins us to a version.
            status_code=422,
            content={
                "detail": "; ".join(
                    "{}: {}".format(item["field"], item["message"]) for item in items
                )
                or "invalid request",
                "error": "validation_error",
                "errors": items,
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error(
        request: Request, error: StarletteHTTPException
    ) -> JSONResponse:
        """Give every error the same body shape: detail, error, hint."""
        detail: Any = error.detail
        if isinstance(detail, dict) and "detail" in detail:
            content = {
                "detail": detail.get("detail"),
                "error": detail.get("error", "error"),
                "hint": detail.get("hint"),
            }
        else:
            content = {
                "detail": str(detail),
                "error": _error_code(error.status_code),
                "hint": None,
            }
        return JSONResponse(
            status_code=error.status_code,
            content=content,
            headers=getattr(error, "headers", None),
        )

    app.include_router(router)

    @app.get("/", include_in_schema=False)
    def index() -> dict[str, Any]:
        return {
            "name": API_TITLE,
            "version": API_VERSION,
            "docs": "/docs",
            "endpoints": [
                "POST /api/recommend",
                "GET /api/products",
                "GET /api/products/{product_id}",
                "GET /api/users/{user_id}",
                "GET /api/meta",
                "GET /api/health",
            ],
        }

    return app


def _field_name(loc: Any) -> str:
    """``("body", "budget_min")`` -> ``"budget_min"``; keep list indices."""
    parts = [str(part) for part in loc if part not in {"body", "query", "path"}]
    return ".".join(parts) if parts else "request"


def _clean_message(message: str) -> str:
    """Drop Pydantic's ``Value error, `` prefix; the field name carries it."""
    prefix = "Value error, "
    return message[len(prefix) :] if message.startswith(prefix) else message


def _error_code(status_code: int) -> str:
    return {
        400: "bad_request",
        404: "not_found",
        405: "method_not_allowed",
        422: "validation_error",
        503: "not_ready",
    }.get(status_code, "error")


app = create_app()
