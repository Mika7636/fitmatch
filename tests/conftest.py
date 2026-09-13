"""Shared fixtures for the session-2 recommender tests.

The three processed CSVs and the two fitted models are session-scoped: loading
and fitting them once takes a couple of seconds, and every test below is
read-only, so there is nothing to isolate between tests.

The models are also built *separately* here -- a bare
:class:`ConstraintBasedRecommender` and a bare
:class:`ContentBasedRecommender` alongside the full cascade -- because the
whole point of the two-module layout is that each technique can be tested
without the other.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.common import (  # noqa: E402
    INTERACTIONS_CSV,
    PRODUCTS_CSV,
    SEED,
    SPORTS,
    USERS_CSV,
)
from src.recommender.content_based import ContentBasedRecommender  # noqa: E402
from src.recommender.knowledge_based import ConstraintBasedRecommender  # noqa: E402
from src.recommender.pipeline import FitMatchRecommender  # noqa: E402

pytest.register_assert_rewrite("tests")


def _require(path: Path, produced_by: str) -> None:
    if not path.exists():
        pytest.skip(
            "{} is missing -- run `python -m {}` (or build_data.bat) first".format(
                path.name, produced_by
            ),
            allow_module_level=True,
        )


@pytest.fixture(scope="session")
def products() -> pd.DataFrame:
    _require(PRODUCTS_CSV, "src.data.build_products")
    return pd.read_csv(PRODUCTS_CSV)


@pytest.fixture(scope="session")
def users() -> pd.DataFrame:
    _require(USERS_CSV, "src.data.build_users")
    return pd.read_csv(USERS_CSV)


@pytest.fixture(scope="session")
def interactions() -> pd.DataFrame:
    _require(INTERACTIONS_CSV, "src.data.build_interactions")
    return pd.read_csv(INTERACTIONS_CSV)


@pytest.fixture(scope="session")
def knowledge(products: pd.DataFrame) -> ConstraintBasedRecommender:
    """Chapter 7 on its own -- no content model anywhere near it."""
    return ConstraintBasedRecommender(products)


@pytest.fixture(scope="session")
def content(
    products: pd.DataFrame, interactions: pd.DataFrame
) -> ContentBasedRecommender:
    """Chapter 3 on its own -- no constraint filter anywhere near it."""
    return ContentBasedRecommender(products, interactions).fit()


@pytest.fixture(scope="session")
def engine(
    products: pd.DataFrame, users: pd.DataFrame, interactions: pd.DataFrame
) -> FitMatchRecommender:
    """The cascade. Built from frames rather than `load()` so the tests never
    touch, or depend on, the joblib cache on disk."""
    return FitMatchRecommender(products, users, interactions)


@pytest.fixture(scope="session")
def median_profiles(users: pd.DataFrame) -> dict[str, dict]:
    """One representative profile per sport: the median of that sport's users.

    Numeric fields take the median, categorical fields the mode. This is the
    "median profile" the coverage tests are specified against -- a real,
    typical user of each sport rather than a hand-picked easy case.
    """
    profiles: dict[str, dict] = {}
    for sport in SPORTS:
        segment = users[users["primary_sport"] == sport]
        if segment.empty:
            segment = users
        profiles[sport] = {
            "user_id": None,
            "primary_sport": sport,
            "gender": segment["gender"].mode().iloc[0],
            "shoe_size": float(segment["shoe_size"].median()),
            "apparel_size": segment["apparel_size"].mode().iloc[0],
            "foot_arch_type": segment["foot_arch_type"].mode().iloc[0],
            "fitness_level": segment["fitness_level"].mode().iloc[0],
            "indoor_or_outdoor": segment["indoor_or_outdoor"].mode().iloc[0],
            "budget_min": float(segment["budget_min"].median()),
            "budget_max": float(segment["budget_max"].median()),
            "preferred_brands": segment["preferred_brands"].mode().iloc[0],
            "style_preference": segment["style_preference"].mode().iloc[0],
            "color_preference": segment["color_preference"].mode().iloc[0],
            "material_preference": segment["material_preference"].mode().iloc[0],
            "climate": segment["climate"].mode().iloc[0],
        }
    return profiles


@pytest.fixture(scope="session")
def seed() -> int:
    return SEED
