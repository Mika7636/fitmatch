"""Session 3: the HTTP layer.

These tests use FastAPI's ``TestClient`` as a context manager, which is what
runs the lifespan handler -- without the ``with``, nothing loads and every
route answers 503.  The client is session-scoped so the recommender is loaded
once for the whole file, the same way the real process loads it once at
startup.

The interesting cases are not the happy paths.  Session 2 measured that the
median user has *no* products matching every stated preference and that 85.2%
of users get the cold-start ranker, so the tests that matter are the ones that
check the API says so: heavy relaxation, and a request whose hard constraints
leave nothing at all.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.main import create_app  # noqa: E402
from src.data.common import PRODUCTS_CSV, USERS_CSV  # noqa: E402
from src.recommender.knowledge_based import HARD_CONSTRAINTS  # noqa: E402

# A budget band above the most expensive product in the catalogue. Nothing can
# satisfy it, and budget is a *hard* constraint, so no amount of relaxation
# rescues it -- which is exactly the case the response has to explain rather
# than return an unexplained empty list for.
IMPOSSIBLE_BUDGET = {"budget_min": 1650.0, "budget_max": 1699.0}

# Narrow budget + yoga + a specific colour: the profile session 2 predicted
# would lose colour (98.7% of users do) and most of the other soft constraints.
HEAVY_RELAXATION_PROFILE = {
    "age": 31,
    "gender": "female",
    "height_cm": 166.0,
    "weight_kg": 58.0,
    "shoe_size": 7.5,
    "apparel_size": "S",
    "foot_arch_type": "high",
    "primary_sport": "yoga",
    "fitness_level": "intermediate",
    "workouts_per_week": 5,
    "indoor_or_outdoor": "indoor",
    "budget_min": 40.0,
    "budget_max": 65.0,
    "preferred_brands": ["Jordan"],
    "style_preference": "athleisure",
    "color_preference": "purple",
    "material_preference": "gore-tex",
    "climate": "tropical",
}


@pytest.fixture(scope="session")
def client() -> TestClient:
    """The app with its lifespan run, so the recommender is loaded once."""
    for path, produced_by in (
        (PRODUCTS_CSV, "src.data.build_products"),
        (USERS_CSV, "src.data.build_users"),
    ):
        if not path.exists():
            pytest.skip(
                "{} is missing -- run build_data.bat first".format(path.name),
                allow_module_level=True,
            )
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def meta(client: TestClient) -> dict:
    return client.get("/api/meta").json()


@pytest.fixture(scope="session")
def a_product_id(client: TestClient) -> str:
    return client.get("/api/products", params={"page_size": 1}).json()["items"][0][
        "product_id"
    ]


@pytest.fixture(scope="session")
def centroid_user_id() -> str:
    """A user with enough 4+ ratings to reach content mode (b).

    Found from the data rather than hardcoded: only 14.8% of users qualify, and
    which ones they are depends on the interaction build.
    """
    from src.data.common import INTERACTIONS_CSV

    if not INTERACTIONS_CSV.exists():
        pytest.skip("interactions.csv is missing")
    interactions = pd.read_csv(INTERACTIONS_CSV)
    ratings = interactions[interactions["event_type"] == "rating"]
    liked = ratings[pd.to_numeric(ratings["rating"], errors="coerce") >= 4]
    counts = liked.groupby("user_id").size().sort_values(ascending=False)
    if counts.empty or counts.iloc[0] < 3:
        pytest.skip("no user has three ratings of 4 or better")
    return str(counts.index[0])


# ----------------------------------------------------------------------
# /api/health
# ----------------------------------------------------------------------
def test_health_reports_ready_and_cache_state(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["ready"] is True
    assert body["catalogue_size"] == 10000
    assert body["n_users"] == 5000
    # The TF-IDF model is fitted, so the vocabulary is non-empty...
    assert body["n_vocabulary_terms"] > 0
    # ...and the joblib cache exists on disk, whether this run built it or
    # reused one. `hit_on_startup` distinguishes the two.
    assert body["cache"]["warm"] is True
    assert isinstance(body["cache"]["hit_on_startup"], bool)
    assert body["cold_start_ms"] is not None


def test_index_lists_the_endpoints(client: TestClient) -> None:
    body = client.get("/").json()
    assert "POST /api/recommend" in body["endpoints"]


# ----------------------------------------------------------------------
# /api/meta
# ----------------------------------------------------------------------
def test_meta_matches_the_catalogue_vocabularies(meta: dict) -> None:
    """The sizes session 1 built, read back off the CSVs rather than declared."""
    assert len(meta["sport_type"]) == 10
    assert len(meta["brand"]) == 3
    assert len(meta["color_preference"]) == 14
    assert len(meta["material_preference"]) == 10
    assert len(meta["style_preference"]) == 4
    assert len(meta["climate"]) == 5
    assert len(meta["category"]) == 3
    assert len(meta["foot_arch_type"]) == 3
    assert len(meta["apparel_size"]) == 6
    assert meta["brand"] == ["Adidas", "Jordan", "Nike"]
    assert meta["apparel_size"] == ["XS", "S", "M", "L", "XL", "XXL"]


def test_meta_carries_the_numeric_ranges(meta: dict) -> None:
    assert 0 < meta["price"]["min"] < meta["price"]["max"]
    assert meta["shoe_size"]["min"] >= meta["limits"]["shoe_size"]["min"]
    assert meta["shoe_size"]["max"] <= meta["limits"]["shoe_size"]["max"]
    assert meta["shoe_size"]["step"] == 0.5
    assert meta["limits"]["age"] == {"min": 13, "max": 100}
    assert meta["user_id_range"]["min"] == "U00001"


def test_meta_is_read_from_the_data_not_hardcoded(meta: dict) -> None:
    """Every advertised value must actually occur in products.csv."""
    products = pd.read_csv(PRODUCTS_CSV)
    assert set(meta["sport_type"]) == set(products["sport_type"].unique())
    assert set(meta["color_preference"]) == set(products["color"].unique())
    assert set(meta["material_preference"]) == set(products["material"].unique())


# ----------------------------------------------------------------------
# /api/products
# ----------------------------------------------------------------------
def test_products_browse_paginates(client: TestClient) -> None:
    first = client.get("/api/products", params={"page": 1, "page_size": 5}).json()
    second = client.get("/api/products", params={"page": 2, "page_size": 5}).json()
    assert len(first["items"]) == 5
    assert first["total"] == 10000
    assert first["total_pages"] == 2000
    assert first["has_next"] is True
    assert first["has_previous"] is False
    assert second["has_previous"] is True
    assert {item["product_id"] for item in first["items"]}.isdisjoint(
        item["product_id"] for item in second["items"]
    )


def test_products_browse_page_one_is_not_one_brand(client: TestClient) -> None:
    """The catalogue must not open on a single-brand block.

    ``products.csv`` is written brand by brand and the 4,000 Adidas rows are
    appended last, so any ordering that follows the file gave page 1 as twelve
    consecutive Adidas products.  The endpoint interleaves instead, and a
    12-row page should carry the catalogue's own mix rather than one brand.
    """
    items = client.get(
        "/api/products", params={"page": 1, "page_size": 12}
    ).json()["items"]
    brands = [item["brand"] for item in items]
    assert len(set(brands)) > 1, brands

    # Not just "more than one": the proportions should be the catalogue's.
    # 5,219 / 4,000 / 781 of 10,000 over 12 rows is 6 / 5 / 1.
    counts = Counter(brands)
    assert counts["Nike"] > counts["Adidas"] > 0
    assert counts["Jordan"] > 0


def test_products_browse_order_is_stable(client: TestClient) -> None:
    """Two identical requests return the identical product_id sequence.

    The interleave is arithmetic, not a shuffle, so there is no seed to drift
    and no per-request randomness: a link to page 7 has to keep meaning the
    same twelve products.
    """
    params = {"page": 7, "page_size": 12}
    first = client.get("/api/products", params=params).json()["items"]
    second = client.get("/api/products", params=params).json()["items"]
    assert [item["product_id"] for item in first] == [
        item["product_id"] for item in second
    ]
    assert len(first) == 12


def test_products_browse_paging_loses_nothing(client: TestClient) -> None:
    """Paging is a partition: no product repeats, none goes missing.

    Checked on the accessory slice rather than all 10,000 rows so the test
    stays quick; the ordering is one permutation of the catalogue, so what
    holds for a filtered subset of it holds for the whole.
    """
    params = {"category": "accessory", "page_size": 100}
    first = client.get("/api/products", params={**params, "page": 1}).json()
    total = first["total"]

    seen: list[str] = []
    for page in range(1, first["total_pages"] + 1):
        body = client.get("/api/products", params={**params, "page": page}).json()
        seen.extend(item["product_id"] for item in body["items"])

    assert len(seen) == total
    assert len(set(seen)) == total


def test_products_browse_filters(client: TestClient) -> None:
    body = client.get(
        "/api/products",
        params={
            "sport": "yoga",
            "brand": "Nike",
            "category": "apparel",
            "price_min": 30,
            "price_max": 90,
            "page_size": 50,
        },
    ).json()
    assert body["total"] > 0
    for item in body["items"]:
        assert item["sport_type"] == "yoga"
        assert item["brand"] == "Nike"
        assert item["category"] == "apparel"
        assert 30 <= item["price"] <= 90
    assert body["filters"]["sport"] == "yoga"


def test_products_browse_rejects_an_unknown_brand(client: TestClient) -> None:
    response = client.get("/api/products", params={"brand": "Reebok"})
    assert response.status_code == 422
    assert "Nike" in response.json()["detail"]


def test_product_detail_returns_every_column(
    client: TestClient, a_product_id: str
) -> None:
    body = client.get("/api/products/{}".format(a_product_id)).json()
    assert body["product_id"] == a_product_id
    assert body["product_description"]
    for field in ("arch_support", "cushioning_level", "breathability", "sales_rank"):
        assert field in body


def test_unknown_product_id_is_404(client: TestClient) -> None:
    response = client.get("/api/products/NOPE12345")
    assert response.status_code == 404
    body = response.json()
    assert body["error"] == "product_not_found"
    assert "NOPE12345" in body["detail"]


# ----------------------------------------------------------------------
# /api/users
# ----------------------------------------------------------------------
def test_user_profile_loads_and_can_be_posted_back(client: TestClient) -> None:
    body = client.get("/api/users/42").json()
    assert body["user_id"] == "U00042"
    assert body["primary_sport"]
    assert isinstance(body["preferred_brands"], list)
    assert body["expected_scoring_mode"] in {"profile_only", "profile_plus_centroid"}

    # The demo button's round trip: load a profile, drop user_id so the form is
    # scored rather than the stored row, post it.
    profile = {
        key: value
        for key, value in body.items()
        if key not in {"user_id", "qualifying_ratings", "expected_scoring_mode"}
        and value is not None
    }
    echoed = client.post("/api/recommend", json=profile)
    assert echoed.status_code == 200, echoed.text


def test_user_id_accepts_bare_numbers_and_padded_ids(client: TestClient) -> None:
    padded = client.get("/api/users/U00042").json()
    bare = client.get("/api/users/42").json()
    assert padded == bare


def test_unknown_user_id_is_404(client: TestClient) -> None:
    response = client.get("/api/users/99999")
    assert response.status_code == 404
    body = response.json()
    assert body["error"] == "user_not_found"
    assert "U05000" in body["hint"]


# ----------------------------------------------------------------------
# /api/recommend -- the happy path
# ----------------------------------------------------------------------
def test_recommend_returns_ranked_products_with_both_kinds_of_explanation(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/recommend",
        json={
            "age": 28,
            "gender": "male",
            "shoe_size": 10.0,
            "apparel_size": "L",
            "primary_sport": "running",
            "budget_min": 50,
            "budget_max": 200,
            "climate": "temperate",
            "top_n": 5,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert len(body["recommendations"]) == 5
    assert [item["rank"] for item in body["recommendations"]] == [1, 2, 3, 4, 5]
    scores = [item["score"] for item in body["recommendations"]]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= score <= 1.0 for score in scores)

    first = body["recommendations"][0]
    assert first["product"]["price"] <= 200
    assert first["product"]["stock_status"] == "in_stock"
    sources = {line["source"] for line in first["explanations"]}
    assert sources == {"knowledge_based", "content_based"}
    assert body["latency_ms"] > 0


def test_recommend_honours_top_n(client: TestClient) -> None:
    body = client.post(
        "/api/recommend", json={"primary_sport": "tennis", "top_n": 3}
    ).json()
    assert len(body["recommendations"]) == 3


def test_recommend_by_user_id_scores_the_stored_row(client: TestClient) -> None:
    body = client.post("/api/recommend", json={"user_id": "42", "top_n": 4}).json()
    assert body["user_id"] == "U00042"
    assert len(body["recommendations"]) == 4


def test_recommend_reports_the_hard_filter(client: TestClient) -> None:
    body = client.post(
        "/api/recommend", json={"primary_sport": "training", "gender": "female"}
    ).json()
    constraints = body["constraints"]
    assert constraints["hard_applied"] == list(HARD_CONSTRAINTS)
    assert set(constraints["hard_excluded"]) == set(HARD_CONSTRAINTS)
    assert constraints["catalogue_size"] == 10000
    assert (
        constraints["candidates_after_hard"]
        + sum(constraints["hard_excluded"].values())
        == 10000
    )
    assert constraints["final_candidate_count"] >= len(body["recommendations"])


# ----------------------------------------------------------------------
# /api/recommend -- relaxation, which is the normal path
# ----------------------------------------------------------------------
def test_heavy_relaxation_is_reported_as_first_class_data(client: TestClient) -> None:
    """Narrow budget + yoga + a specific colour: the session-2 worst case.

    Session 2 measured the median feasible set with every soft constraint
    applied at 0, and colour relaxed for 98.7% of users. This profile is built
    to land there, and the test is that the response *says so* -- with the
    dropped constraint named, the reason given, and a message fit to show the
    user.
    """
    body = client.post("/api/recommend", json=HEAVY_RELAXATION_PROFILE).json()
    constraints = body["constraints"]

    assert constraints["candidates_after_soft"] < constraints["candidates_after_hard"]
    assert constraints["relaxed"], "this profile is supposed to force relaxation"

    dropped = {item["constraint"] for item in constraints["relaxed"]}
    assert "color_preference" in dropped

    colour = next(
        item for item in constraints["relaxed"] if item["constraint"] == "color_preference"
    )
    assert colour["label"] == "colour"
    assert colour["requested"] == "purple"
    assert colour["reason"]
    assert "purple" in colour["message"]

    # A relaxed constraint is not also reported as still applied.
    assert dropped.isdisjoint(constraints["still_applied"])
    assert constraints["final_candidate_count"] >= constraints["candidates_after_soft"]
    assert "10,000" in constraints["summary"]
    assert constraints["log"][0]["action"] == "applied"


def test_relaxation_order_is_cheapest_first(client: TestClient) -> None:
    """Colour goes before size: a shoe that does not fit is not a result."""
    body = client.post("/api/recommend", json=HEAVY_RELAXATION_PROFILE).json()
    dropped = [item["constraint"] for item in body["constraints"]["relaxed"]]
    if "color_preference" in dropped and "size" in dropped:
        assert dropped.index("color_preference") < dropped.index("size")
    assert [item["order"] for item in body["constraints"]["relaxed"]] == list(
        range(len(dropped))
    )


def test_unstated_preferences_are_relaxed_without_claiming_a_loss(
    client: TestClient,
) -> None:
    """Relaxing a preference the user never gave must not read as a loss."""
    body = client.post(
        "/api/recommend", json={"primary_sport": "swimming", "budget_max": 45}
    ).json()
    for item in body["constraints"]["relaxed"]:
        if item["requested"] is None:
            assert item["candidates_before"] == item["candidates_after"]
            assert "did not state" in item["message"]


def test_zero_results_still_explains_what_was_tried(client: TestClient) -> None:
    """The hard constraints are never relaxed, so they can leave nothing.

    An empty list on its own would be a dead end for the UI. The contract is
    that ``constraints`` explains it: which filters ran, which one did the
    damage, and what the user could change.
    """
    body = client.post(
        "/api/recommend", json={"primary_sport": "yoga", **IMPOSSIBLE_BUDGET}
    ).json()

    assert body["recommendations"] == []
    constraints = body["constraints"]
    assert constraints["candidates_after_hard"] == 0
    assert constraints["final_candidate_count"] == 0
    assert constraints["exhausted"] is True
    assert constraints["hard_excluded"]["budget"] > 0
    assert "budget" in constraints["summary"].lower()
    assert constraints["summary"].endswith(".")
    # Still a well-formed answer, not an error.
    assert body["scoring"]["mode"] == "profile_only"


# ----------------------------------------------------------------------
# /api/recommend -- scoring mode
# ----------------------------------------------------------------------
def test_scoring_reports_the_cold_start_fallback(client: TestClient) -> None:
    """An anonymous profile has no history, so it takes the 85.2% path."""
    body = client.post(
        "/api/recommend", json={"primary_sport": "basketball", "top_n": 3}
    ).json()
    scoring = body["scoring"]
    assert scoring["mode"] == "profile_only"
    assert scoring["qualifying_ratings"] == 0
    assert scoring["min_ratings_for_centroid"] == 3
    assert "cold-start" in scoring["explanation"]
    assert scoring["label"]


def test_scoring_reports_the_centroid_mode_for_a_user_with_history(
    client: TestClient, centroid_user_id: str
) -> None:
    body = client.post(
        "/api/recommend", json={"user_id": centroid_user_id, "top_n": 3}
    ).json()
    scoring = body["scoring"]
    assert scoring["mode"] == "profile_plus_centroid"
    assert scoring["qualifying_ratings"] >= scoring["min_ratings_for_centroid"]
    assert "blended" in scoring["explanation"]


def test_scoring_mode_is_consistent_with_the_user_endpoint(
    client: TestClient, centroid_user_id: str
) -> None:
    predicted = client.get("/api/users/{}".format(centroid_user_id)).json()
    actual = client.post("/api/recommend", json={"user_id": centroid_user_id}).json()
    assert predicted["expected_scoring_mode"] == actual["scoring"]["mode"]
    assert predicted["qualifying_ratings"] == actual["scoring"]["qualifying_ratings"]


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
def test_budget_violation_is_422_naming_both_values(client: TestClient) -> None:
    response = client.post(
        "/api/recommend",
        json={"primary_sport": "running", "budget_min": 200, "budget_max": 50},
    )
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "validation_error"
    assert "budget_min" in body["detail"] and "budget_max" in body["detail"]


def test_equal_budgets_are_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/recommend", json={"budget_min": 100, "budget_max": 100}
    )
    assert response.status_code == 422


def test_invalid_sport_type_is_422_naming_the_valid_values(
    client: TestClient, meta: dict
) -> None:
    response = client.post(
        "/api/recommend", json={"primary_sport": "quidditch", "budget_max": 200}
    )
    assert response.status_code == 422
    body = response.json()
    assert body["errors"][0]["field"] == "primary_sport"
    message = body["errors"][0]["message"]
    for sport in meta["sport_type"]:
        assert sport in message, "the 422 must name every valid sport"


def test_invalid_colour_is_422_naming_the_fourteen_colours(
    client: TestClient, meta: dict
) -> None:
    response = client.post("/api/recommend", json={"color_preference": "teal"})
    assert response.status_code == 422
    message = response.json()["errors"][0]["message"]
    assert all(colour in message for colour in meta["color_preference"])


def test_invalid_brand_in_the_list_is_422(client: TestClient) -> None:
    response = client.post(
        "/api/recommend", json={"preferred_brands": ["Nike", "Reebok"]}
    )
    assert response.status_code == 422
    assert response.json()["errors"][0]["field"] == "preferred_brands.1"


@pytest.mark.parametrize(
    "field, value",
    [
        ("age", 12),
        ("age", 101),
        ("shoe_size", 2.5),
        ("shoe_size", 20.5),
        ("height_cm", 129),
        ("height_cm", 231),
        ("weight_kg", 29),
        ("weight_kg", 251),
    ],
)
def test_numeric_bounds_are_enforced(
    client: TestClient, field: str, value: float
) -> None:
    response = client.post("/api/recommend", json={field: value})
    assert response.status_code == 422
    assert response.json()["errors"][0]["field"] == field


@pytest.mark.parametrize(
    "field, boundary",
    [("age", 13), ("age", 100), ("shoe_size", 3), ("shoe_size", 20), ("height_cm", 130)],
)
def test_numeric_bounds_are_inclusive(
    client: TestClient, field: str, boundary: float
) -> None:
    response = client.post("/api/recommend", json={field: boundary})
    assert response.status_code == 200, response.text


def test_vocabulary_values_are_case_insensitive(client: TestClient) -> None:
    """A form posting `Yoga` and `nike` means the same as the CSV's spelling."""
    response = client.post(
        "/api/recommend",
        json={
            "primary_sport": "Yoga",
            "preferred_brands": ["nike"],
            "apparel_size": "m",
            "top_n": 2,
        },
    )
    assert response.status_code == 200, response.text


def test_unknown_field_is_rejected(client: TestClient) -> None:
    """A typo must not be silently ignored into a wrong recommendation."""
    response = client.post(
        "/api/recommend", json={"primary_sport": "yoga", "colour_preference": "navy"}
    )
    assert response.status_code == 422


def test_unknown_user_id_on_recommend_is_404(client: TestClient) -> None:
    response = client.post("/api/recommend", json={"user_id": "U99999"})
    assert response.status_code == 404
    assert response.json()["error"] == "user_not_found"


def test_top_n_is_bounded(client: TestClient) -> None:
    assert client.post("/api/recommend", json={"top_n": 0}).status_code == 422
    assert client.post("/api/recommend", json={"top_n": 1000}).status_code == 422


# ----------------------------------------------------------------------
# Cross-cutting
# ----------------------------------------------------------------------
def test_cors_allows_the_vite_dev_server(client: TestClient) -> None:
    response = client.options(
        "/api/meta",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert (
        response.headers["access-control-allow-origin"] == "http://localhost:5173"
    )


def test_every_response_carries_a_request_id(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.headers["X-Request-ID"]
    assert float(response.headers["X-Response-Time-ms"]) >= 0


def test_recommendations_are_deterministic(client: TestClient) -> None:
    """Neither technique draws a random number; the API must not add one."""
    first = client.post("/api/recommend", json=HEAVY_RELAXATION_PROFILE).json()
    second = client.post("/api/recommend", json=HEAVY_RELAXATION_PROFILE).json()
    assert [item["product_id"] for item in _products(first)] == [
        item["product_id"] for item in _products(second)
    ]
    assert first["constraints"]["relaxed"] == second["constraints"]["relaxed"]


def test_min_results_changes_how_far_relaxation_goes(client: TestClient) -> None:
    """A bigger candidate target means more soft constraints have to go."""
    small = client.post(
        "/api/recommend", json={**HEAVY_RELAXATION_PROFILE, "min_results": 5}
    ).json()
    large = client.post(
        "/api/recommend", json={**HEAVY_RELAXATION_PROFILE, "min_results": 500}
    ).json()
    assert len(large["constraints"]["relaxed"]) >= len(small["constraints"]["relaxed"])
    assert (
        large["constraints"]["final_candidate_count"]
        >= small["constraints"]["final_candidate_count"]
    )


def test_openapi_schema_is_generated(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    for path in (
        "/api/recommend",
        "/api/products",
        "/api/products/{product_id}",
        "/api/users/{user_id}",
        "/api/meta",
        "/api/health",
    ):
        assert path in schema["paths"]


def _products(body: dict) -> list[dict]:
    return [item["product"] for item in body["recommendations"]]
