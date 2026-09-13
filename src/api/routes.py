"""The six endpoints.

Each one is thin on purpose: validate (Pydantic, against the measured
vocabularies), call :class:`~src.api.service.RecommenderService`, return.  The
recommendation route additionally writes the structured per-request log line,
because it is the only route whose interesting fields are not visible from the
HTTP envelope.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, status

from src.api.schemas import (
    ErrorResponse,
    HealthResponse,
    MetaResponse,
    ProductDetail,
    ProductPage,
    ProductQuery,
    RecommendRequest,
    RecommendResponse,
    UserProfile,
)
from src.api.service import ProductNotFound, RecommenderService, UserNotFound

LOGGER = logging.getLogger("fitmatch.api.request")

router = APIRouter(prefix="/api")

NOT_FOUND = {404: {"model": ErrorResponse, "description": "No such record."}}
NOT_READY = {
    503: {"model": ErrorResponse, "description": "The recommender is still loading."}
}


def get_service(request: Request) -> RecommenderService:
    """The one loaded cascade, put on ``app.state`` by the lifespan handler.

    A 503 rather than a 500 when it is missing: the recommender either failed
    to load or is still loading, and neither is the caller's fault.
    """
    service: RecommenderService | None = getattr(request.app.state, "service", None)
    if service is None:
        detail = getattr(request.app.state, "startup_error", None)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "detail": detail
                or "The recommender is still loading; retry in a moment.",
                "error": "not_ready",
                "hint": "GET /api/health reports load state and cache warmth.",
            },
        )
    return service


ServiceDep = Annotated[RecommenderService, Depends(get_service)]


# ----------------------------------------------------------------------
# Recommendation
# ----------------------------------------------------------------------
@router.post(
    "/recommend",
    response_model=RecommendResponse,
    responses={**NOT_FOUND, **NOT_READY},
    summary="Rank the catalogue for one profile",
)
def recommend(
    payload: RecommendRequest, service: ServiceDep, request: Request
) -> RecommendResponse:
    """Run the two-technique cascade: constraint filter, then TF-IDF ranking.

    The response is three blocks, and the last two are not optional extras.
    ``constraints`` says what was tried and which preferences had to be set
    aside -- the median user in this dataset has *no* products matching every
    stated preference, so this is the ordinary case.  ``scoring`` says which of
    the two content modes ranked the results, since 85.2% of users fall back to
    the cold-start one.
    """
    try:
        outcome = service.recommend(
            profile=payload.to_profile(),
            user_id=payload.user_id,
            top_n=payload.top_n,
            min_results=payload.min_results,
        )
    except UserNotFound as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "detail": str(error),
                "error": "user_not_found",
                "hint": "Omit user_id to score the posted profile instead.",
            },
        ) from error

    LOGGER.info(
        "recommend",
        extra={
            "event": "recommend",
            "request_id": getattr(request.state, "request_id", None),
            "source": "user_id" if payload.user_id else "profile",
            "profile": payload.profile_summary(),
            "top_n": payload.top_n,
            **outcome.log_fields,
        },
    )
    return RecommendResponse.model_validate(outcome.body)


# ----------------------------------------------------------------------
# Catalogue
# ----------------------------------------------------------------------
@router.get(
    "/products",
    response_model=ProductPage,
    responses=NOT_READY,
    summary="Browse the catalogue",
)
def list_products(
    query: Annotated[ProductQuery, Query()], service: ServiceDep
) -> ProductPage:
    """A filtered, paginated page of products.

    Ordered by a fixed brand interleave, so the first page shows the
    catalogue's real 52% Nike / 40% Adidas / 8% Jordan mix instead of the CSV's
    write order, which opened on twelve consecutive Adidas products.  The order
    is computed once from the catalogue and never randomised: the same page
    always returns the same products, and paging through it cannot repeat or
    skip one.  Deliberately still not ``sales_rank``: neither recommender
    technique reads popularity, so a browse view that sorted by it would show a
    ranking the recommender does not believe in.
    """
    return ProductPage.model_validate(
        service.browse_products(
            sport=query.sport,
            brand=query.brand,
            category=query.category,
            price_min=query.price_min,
            price_max=query.price_max,
            page=query.page,
            page_size=query.page_size,
        )
    )


@router.get(
    "/products/{product_id}",
    response_model=ProductDetail,
    responses={**NOT_FOUND, **NOT_READY},
    summary="One product in full",
)
def get_product(
    product_id: Annotated[str, Path(min_length=1, max_length=32)],
    service: ServiceDep,
) -> ProductDetail:
    """Every column ``products.csv`` holds for one product, description included."""
    try:
        return ProductDetail.model_validate(service.product_detail(product_id))
    except ProductNotFound as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "detail": str(error),
                "error": "product_not_found",
                "hint": "GET /api/products lists valid ids.",
            },
        ) from error


# ----------------------------------------------------------------------
# Users
# ----------------------------------------------------------------------
@router.get(
    "/users/{user_id}",
    response_model=UserProfile,
    responses={**NOT_FOUND, **NOT_READY},
    summary="Load an existing profile",
)
def get_user(
    user_id: Annotated[str, Path(min_length=1, max_length=32)],
    service: ServiceDep,
) -> UserProfile:
    """A users.csv row, normalised for the demo button.

    ``42``, ``"42"`` and ``"U00042"`` all resolve to the same user. The
    response is shaped so it can be posted straight back to
    ``/api/recommend`` -- drop ``user_id`` first if the user is allowed to edit
    it, since ``user_id`` means "score the stored row, not this form".
    """
    try:
        return UserProfile.model_validate(service.user_profile(user_id))
    except UserNotFound as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "detail": str(error),
                "error": "user_not_found",
                "hint": "Ids run {} to {}; GET /api/meta reports the range.".format(
                    service.vocabulary.user_id_min, service.vocabulary.user_id_max
                ),
            },
        ) from error


# ----------------------------------------------------------------------
# Meta and health
# ----------------------------------------------------------------------
@router.get(
    "/meta",
    response_model=MetaResponse,
    responses=NOT_READY,
    summary="Every dropdown the frontend needs",
)
def get_meta(service: ServiceDep) -> MetaResponse:
    """The controlled vocabularies and numeric ranges, measured at startup.

    Nothing here is hardcoded: every list is read from ``products.csv`` and
    ``users.csv``, which is also what the request validators check against, so
    a value this endpoint offers can never be a value ``/api/recommend``
    rejects.
    """
    return MetaResponse.model_validate(service.meta())


@router.get("/health", response_model=HealthResponse, summary="Readiness")
def get_health(request: Request) -> HealthResponse:
    """Readiness, cold-start time and whether the joblib TF-IDF cache is warm.

    Returns 200 while starting or failed, with ``ready: false`` and a reason --
    a health check that cannot answer is worse than one that answers "not yet".
    """
    service: RecommenderService | None = getattr(request.app.state, "service", None)
    if service is not None:
        return HealthResponse.model_validate(service.health())

    error = getattr(request.app.state, "startup_error", None)
    cache_path = getattr(request.app.state, "cache_path", "")
    return HealthResponse(
        status="error" if error else "starting",
        ready=False,
        cache={
            "warm": False,
            "hit_on_startup": False,
            "path": str(cache_path),
            "size_bytes": None,
        },
        cold_start_ms=None,
        uptime_s=0.0,
        catalogue_size=0,
        n_users=0,
        n_vocabulary_terms=None,
        min_results_default=0,
        detail=error or "The recommender is still loading.",
    )
