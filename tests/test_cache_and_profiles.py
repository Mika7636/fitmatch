"""The joblib TF-IDF cache, and the profile contract both techniques share.

The cache exists so the session-3 API never refits TF-IDF per request, so the
thing worth testing is that a cached model gives *identical* answers to a
freshly fitted one, and that a stale cache is rebuilt rather than trusted.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.data.common import APPAREL_SIZES, INTERACTIONS_CSV, PRODUCTS_CSV, USERS_CSV
from src.recommender.pipeline import FitMatchRecommender, _catalogue_signature
from src.recommender.profiles import (
    PROFILE_DEFAULTS,
    adjacent_apparel_sizes,
    gender_targets,
    normalize_profile,
)


# ----------------------------------------------------------------------
# The joblib cache
# ----------------------------------------------------------------------
def test_cache_is_written_then_reused(tmp_path, engine):
    """First load fits and writes; second load reads back and agrees exactly."""
    cache = tmp_path / "tfidf.joblib"
    assert not cache.exists()

    cold = FitMatchRecommender.load(cache_path=cache)
    assert cache.exists(), "the first load should have written the cache"
    warm = FitMatchRecommender.load(cache_path=cache)

    for user_id in ("U00042", "U01234"):
        pd.testing.assert_frame_equal(
            cold.recommend(user_id, top_n=10, explain=False).items,
            warm.recommend(user_id, top_n=10, explain=False).items,
        )
        # And both must agree with a model that never saw a cache at all.
        pd.testing.assert_frame_equal(
            warm.recommend(user_id, top_n=10, explain=False).items,
            engine.recommend(user_id, top_n=10, explain=False).items,
        )


def test_a_stale_cache_is_rebuilt_not_trusted(tmp_path):
    """A cache whose signature does not match the catalogue must be discarded."""
    import joblib

    cache = tmp_path / "tfidf.joblib"
    FitMatchRecommender.load(cache_path=cache)

    payload = joblib.load(cache)
    payload["signature"] = ("stale", 0, 0, 0, 0)
    joblib.dump(payload, cache)

    model = FitMatchRecommender.load(cache_path=cache)
    assert model.content.is_fitted
    assert joblib.load(cache)["signature"] != ("stale", 0, 0, 0, 0), (
        "a stale cache should have been overwritten with a fresh one"
    )


def test_a_corrupt_cache_is_survivable(tmp_path):
    """A half-written cache file must not take the whole recommender down."""
    cache = tmp_path / "tfidf.joblib"
    cache.write_bytes(b"this is not a joblib file")
    model = FitMatchRecommender.load(cache_path=cache)
    assert model.content.is_fitted
    assert len(model.recommend("U00042", top_n=5, explain=False).items) == 5


def test_cache_can_be_disabled(tmp_path):
    cache = tmp_path / "tfidf.joblib"
    model = FitMatchRecommender.load(cache_path=None)
    assert model.content.is_fitted
    assert not cache.exists()


def test_rebuild_cache_forces_a_refit(tmp_path):
    cache = tmp_path / "tfidf.joblib"
    FitMatchRecommender.load(cache_path=cache)
    first = cache.stat().st_mtime_ns
    model = FitMatchRecommender.load(cache_path=cache, rebuild_cache=True)
    assert model.content.is_fitted
    assert cache.stat().st_mtime_ns >= first


def test_signature_changes_with_the_catalogue():
    signature = _catalogue_signature(PRODUCTS_CSV, 10_000)
    assert signature != _catalogue_signature(PRODUCTS_CSV, 9_999)


def test_load_names_the_missing_build_script(tmp_path):
    with pytest.raises(SystemExit, match="build_products"):
        FitMatchRecommender.load(products_path=tmp_path / "nope.csv")


# ----------------------------------------------------------------------
# The profile contract
# ----------------------------------------------------------------------
def test_a_users_csv_row_normalises_cleanly(users):
    profile = normalize_profile(users.iloc[0].to_dict())
    assert set(PROFILE_DEFAULTS) <= set(profile)
    assert isinstance(profile["preferred_brands"], tuple)
    assert profile["budget_min"] < profile["budget_max"]


def test_missing_fields_take_their_documented_defaults():
    profile = normalize_profile({})
    assert profile["budget_min"] == 0.0
    assert profile["budget_max"] == math.inf
    assert profile["primary_sport"] == "lifestyle"
    assert profile["preferred_brands"] == ()
    assert profile["gender"] is None
    assert profile["color_preference"] is None


@pytest.mark.parametrize(
    "value,expected",
    [
        ("Nike|Adidas", ("Nike", "Adidas")),
        (["Nike", "Adidas"], ("Nike", "Adidas")),
        ("Nike", ("Nike",)),
        ("", ()),
        (None, ()),
        (float("nan"), ()),
    ],
)
def test_preferred_brands_accepts_every_spelling(value, expected):
    assert normalize_profile({"preferred_brands": value})["preferred_brands"] == expected


def test_numeric_strings_from_a_web_form_are_coerced():
    profile = normalize_profile(
        {"budget_min": "20", "budget_max": "90", "shoe_size": "9.5", "age": "31"}
    )
    assert profile["budget_min"] == 20.0
    assert profile["budget_max"] == 90.0
    assert profile["shoe_size"] == 9.5
    assert profile["age"] == 31.0


def test_a_reversed_budget_is_repaired_not_rejected():
    profile = normalize_profile({"budget_min": 200, "budget_max": 50})
    assert (profile["budget_min"], profile["budget_max"]) == (50.0, 200.0)


def test_vocabularies_are_case_normalised():
    profile = normalize_profile(
        {"gender": " Female ", "primary_sport": "YOGA", "apparel_size": " xl "}
    )
    assert profile["gender"] == "female"
    assert profile["primary_sport"] == "yoga"
    assert profile["apparel_size"] == "XL"


def test_normalising_twice_changes_nothing(users):
    once = normalize_profile(users.iloc[3].to_dict())
    assert normalize_profile(once) == once


def test_normalize_rejects_a_non_mapping():
    with pytest.raises(TypeError):
        normalize_profile(["not", "a", "mapping"])


def test_unknown_keys_are_carried_through():
    profile = normalize_profile({"session_token": "abc123"})
    assert profile["session_token"] == "abc123"


@pytest.mark.parametrize(
    "gender,expected",
    [
        ("male", ("men", "unisex")),
        ("female", ("women", "unisex")),
        (None, ("men", "women", "unisex")),
    ],
)
def test_gender_targets(gender, expected):
    assert gender_targets(normalize_profile({"gender": gender})) == expected


@pytest.mark.parametrize(
    "size,expected",
    [
        ("M", ("S", "M", "L")),
        ("XS", ("XS", "S")),
        ("XXL", ("XL", "XXL")),
        ("L", ("M", "L", "XL")),
    ],
)
def test_adjacent_apparel_sizes(size, expected):
    assert adjacent_apparel_sizes(size) == expected


def test_an_unknown_apparel_size_does_not_narrow_anything():
    assert adjacent_apparel_sizes("XXXL") == tuple(APPAREL_SIZES)
    assert adjacent_apparel_sizes(None) == tuple(APPAREL_SIZES)


def test_every_real_user_normalises(users):
    """5,000 rows, no exceptions, no lost fields."""
    for profile in users.to_dict("records"):
        normalized = normalize_profile(profile)
        assert normalized["gender"] in ("male", "female")
        assert normalized["preferred_brands"]
        assert normalized["budget_min"] < normalized["budget_max"]
