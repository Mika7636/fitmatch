"""Technique 1 in isolation: Knowledge-Based Constraint-Based Recommendation.

Nothing in this file imports the content-based module. If TF-IDF broke
entirely, every test here would still pass -- which is the point of keeping
the two techniques in separate modules.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pandas as pd
import pytest

from src.data.common import SPORTS
from src.recommender.knowledge_based import (
    ARCH_NEED,
    CLIMATE_SEASON,
    HARD_CONSTRAINTS,
    RELAXATION_ORDER,
    ConstraintBasedRecommender,
)
from src.recommender.profiles import (
    GENDER_TO_TARGET,
    adjacent_apparel_sizes,
    normalize_profile,
)


# ----------------------------------------------------------------------
# The hard constraints are hard
# ----------------------------------------------------------------------
@pytest.mark.parametrize("index", [0, 41, 500, 1234, 2999, 4999])
def test_hard_constraints_hold_for_real_users(knowledge, users, index):
    """Budget, gender and stock are never violated, whatever gets relaxed."""
    profile = users.iloc[index].to_dict()
    feasible, report = knowledge.filter(profile)

    assert len(feasible) > 0, "every real user should get a non-empty feasible set"
    assert (feasible["price"] <= profile["budget_max"]).all()
    assert (feasible["price"] >= profile["budget_min"]).all()
    assert (feasible["stock_status"] == "in_stock").all()

    allowed = {GENDER_TO_TARGET[profile["gender"]], "unisex"}
    assert set(feasible["gender_target"]) <= allowed


def test_footwear_is_sport_strict_apparel_is_not(knowledge, users):
    """Spec (b): only footwear is filtered on sport; apparel and accessories
    are sport-flexible whatever their labelled sport_type."""
    profile = users[users["primary_sport"] == "yoga"].iloc[0].to_dict()
    feasible, _ = knowledge.filter(profile)

    footwear = feasible[feasible["category"] == "footwear"]
    assert set(footwear["sport_type"]) <= {"yoga", "lifestyle"}

    apparel = feasible[feasible["category"] != "footwear"]
    assert len(set(apparel["sport_type"])) > 2, (
        "apparel must not be collapsed to the user's sport -- 744 yoga users "
        "face 150 yoga products"
    )


def test_out_of_stock_and_low_stock_are_both_excluded(knowledge, users):
    """`stock_status == "in_stock"` is the hard constraint, so `low_stock` is
    excluded too. Deliberate: 81.6% of the catalogue qualifies."""
    profile = users.iloc[7].to_dict()
    feasible, _ = knowledge.filter(profile)
    assert "low_stock" not in set(feasible["stock_status"])
    assert "out_of_stock" not in set(feasible["stock_status"])


def test_hard_constraints_survive_impossible_soft_preferences(knowledge, users):
    """Even a profile whose soft preferences match nothing keeps its hard
    guarantees -- relaxation must never reach into the hard set."""
    profile = users.iloc[3].to_dict()
    profile.update(
        color_preference="chartreuse",       # not in the catalogue vocabulary
        material_preference="unobtainium",
        preferred_brands="Reebok",           # not one of the three brands
    )
    feasible, report = knowledge.filter(profile)

    assert (feasible["stock_status"] == "in_stock").all()
    assert (feasible["price"] <= profile["budget_max"]).all()
    assert report.n_feasible <= report.n_after_hard
    assert set(HARD_CONSTRAINTS) == set(report.hard_excluded)


# ----------------------------------------------------------------------
# Soft constraints and the relaxation policy
# ----------------------------------------------------------------------
def test_relaxation_follows_the_declared_order(knowledge, users):
    """Whatever gets dropped, it is dropped in RELAXATION_ORDER."""
    relaxed_anywhere = []
    for index in range(0, 400, 7):
        _, report = knowledge.filter(users.iloc[index].to_dict())
        positions = [RELAXATION_ORDER.index(name) for name in report.relaxed]
        assert positions == sorted(positions)
        assert positions == list(range(len(positions))), (
            "relaxation must drop a prefix of RELAXATION_ORDER, not skip around"
        )
        relaxed_anywhere.extend(report.relaxed)
    assert relaxed_anywhere, "some user in 57 should have needed a relaxation"


def test_relaxation_stops_as_soon_as_it_is_big_enough(knowledge, users):
    """It relaxes to reach min_results, not past it."""
    for index in range(0, 200, 11):
        feasible, report = knowledge.filter(users.iloc[index].to_dict())
        if not report.relaxed:
            continue
        # The step before the last relaxation was below target; the last one
        # is the first to reach it.
        assert report.steps[-1].n_before < knowledge.min_results
        assert len(feasible) >= knowledge.min_results or len(report.relaxed) == len(
            RELAXATION_ORDER
        )


def test_relaxation_only_grows_the_feasible_set(knowledge, users):
    """Dropping a conjunct can never remove a row."""
    for index in range(0, 300, 13):
        _, report = knowledge.filter(users.iloc[index].to_dict())
        for step in report.steps:
            if step.action == "relaxed":
                assert step.n_after >= step.n_before


def test_relaxation_log_is_json_ready(knowledge, users):
    """Session 3 will serialise this straight into an HTTP response."""
    _, report = knowledge.filter(users.iloc[41].to_dict())
    payload = report.as_dicts()
    assert payload and isinstance(payload, list)
    for step in payload:
        assert set(step) == {"order", "constraint", "action", "n_before",
                             "n_after", "reason"}
        assert isinstance(step["n_before"], int)
    assert list(report), "ConstraintReport should iterate as its steps"


# ----------------------------------------------------------------------
# Size, the constraint spec (a) is about
# ----------------------------------------------------------------------
def test_soft_size_respects_the_tolerance_rules(knowledge, products):
    """Footwear within +/- 0.5 US sizes, apparel at or one step from the
    user's size, one-size always."""
    profile = {
        "gender": "female",
        "primary_sport": "running",
        "budget_min": 0,
        "budget_max": 10_000,
        "shoe_size": 8.0,
        "apparel_size": "M",
    }
    mask = knowledge._size_mask(normalize_profile(profile))
    kept = products.loc[mask]

    shoe = kept["size"].str.extract(r"US\s*([\d.]+)")[0].astype(float)
    footwear = kept[shoe.notna()]
    assert (
        (footwear["size"].str.extract(r"US\s*([\d.]+)")[0].astype(float) - 8.0)
        .abs()
        .le(0.5)
        .all()
    )

    letters = kept[shoe.isna() & (kept["size"].str.lower() != "one-size")]
    assert set(letters["size"].str.upper()) <= set(adjacent_apparel_sizes("M"))
    assert "US 12" not in set(kept["size"])


def test_one_size_always_matches(knowledge, products):
    profile = {"shoe_size": 5.0, "apparel_size": "XS"}
    mask = knowledge._size_mask(normalize_profile(profile))
    one_size = products["size"].str.strip().str.lower() == "one-size"
    assert mask[one_size.to_numpy()].all()


def test_size_is_soft_not_hard(knowledge, users):
    """A profile whose size is rare must still get results -- the whole reason
    size is soft (spec (a))."""
    profile = users.iloc[0].to_dict()
    profile["shoe_size"] = 14.0     # the top of the users.csv range
    profile["apparel_size"] = "XXL"
    feasible, _ = knowledge.filter(profile)
    assert len(feasible) >= knowledge.min_results


def test_profile_with_no_sizes_is_not_filtered_on_size(knowledge, users):
    """A web form that skips the sizing questions must not lose the catalogue."""
    stated = users.iloc[10].to_dict()
    unstated = dict(stated, shoe_size=None, apparel_size=None)
    assert len(knowledge.filter(unstated)[0]) >= len(knowledge.filter(stated)[0])


# ----------------------------------------------------------------------
# Coverage: every sport, including the thin segments
# ----------------------------------------------------------------------
@pytest.mark.parametrize("sport", SPORTS)
def test_every_sport_has_a_workable_feasible_set(knowledge, median_profiles, sport):
    feasible, report = knowledge.filter(median_profiles[sport])
    assert len(feasible) >= 10, "sport {} got {} products".format(sport, len(feasible))
    assert report.n_feasible == len(feasible)


@pytest.mark.parametrize("sport", ["yoga", "swimming", "skateboarding"])
def test_thin_catalogue_segments_still_get_candidates(knowledge, users, sport):
    """yoga has 150 products for 744 users, swimming 109 for 125. Every real
    user of those sports must still get a usable candidate pool."""
    segment = users[users["primary_sport"] == sport]
    assert len(segment) > 0
    for profile in segment.head(40).to_dict("records"):
        feasible, _ = knowledge.filter(profile)
        assert len(feasible) >= 10, "{} user {} got {}".format(
            sport, profile["user_id"], len(feasible)
        )


# ----------------------------------------------------------------------
# Cold products
# ----------------------------------------------------------------------
def test_zero_interaction_products_stay_reachable(knowledge, products, interactions):
    """Session 1 left 1,284 products untouched on purpose. Every one that is
    in stock must survive the hard filter for somebody -- the ones that are
    not in stock are excluded by a hard constraint doing its job."""
    touched = set(interactions["product_id"].unique())
    cold = products[~products["product_id"].isin(touched)]
    assert len(cold) == 1284

    cold_in_stock = set(cold.loc[cold["stock_status"] == "in_stock", "product_id"])
    assert len(cold_in_stock) > 900

    reachable: set[str] = set()
    for sport in SPORTS:
        for gender in ("male", "female"):
            for low, high in ((0, 60), (60, 150), (150, 2000)):
                feasible, _ = knowledge.filter(
                    {"primary_sport": sport, "gender": gender,
                     "budget_min": low, "budget_max": high}
                )
                reachable.update(feasible["product_id"])

    missed = cold_in_stock - reachable
    assert not missed, "{} cold in-stock products are unreachable: {}".format(
        len(missed), sorted(missed)[:5]
    )


# ----------------------------------------------------------------------
# Explanations
# ----------------------------------------------------------------------
def test_explain_constraints_only_claims_what_is_true(knowledge, users, products):
    profile = users.iloc[41].to_dict()
    feasible, _ = knowledge.filter(profile)
    product = feasible.iloc[0]
    reasons = knowledge.explain_constraints(profile, product)

    assert reasons
    assert all(isinstance(reason, str) and reason for reason in reasons)
    assert any("budget" in reason for reason in reasons)
    assert any(reason == "in stock" for reason in reasons)

    # A product the user cannot afford must not be told it is in budget.
    too_expensive = products[products["price"] > profile["budget_max"]].iloc[0]
    assert not any(
        "budget" in reason for reason in knowledge.explain_constraints(profile, too_expensive)
    )


def test_explain_constraints_accepts_a_dict_or_a_series(knowledge, users, products):
    profile = users.iloc[2].to_dict()
    product = products.iloc[0]
    assert knowledge.explain_constraints(profile, product) == \
        knowledge.explain_constraints(profile, product.to_dict())


def test_arch_support_is_only_claimed_for_footwear(knowledge, users, products):
    """Spec (c): arch_support is "medium" on 80.1% of rows because non-footwear
    carries a neutral value, so it must not be offered as a reason there."""
    profile = dict(users.iloc[0].to_dict(), foot_arch_type="normal")
    apparel = products[
        (products["category"] == "apparel") & (products["arch_support"] == "medium")
    ].iloc[0]
    reasons = knowledge.explain_constraints(profile, apparel)
    assert not any("arch support" in reason for reason in reasons)


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------
def test_rejects_a_catalogue_missing_columns(products):
    with pytest.raises(ValueError, match="missing columns"):
        ConstraintBasedRecommender(products.drop(columns=["price"]))


def test_rejects_a_nonsense_min_results(products):
    with pytest.raises(ValueError):
        ConstraintBasedRecommender(products, min_results=0)


def test_climate_and_arch_maps_cover_their_vocabularies(users):
    assert set(users["climate"]) <= set(CLIMATE_SEASON)
    assert set(users["foot_arch_type"]) <= set(ARCH_NEED)


def test_the_two_techniques_are_independent_modules():
    """The professor's brief requires two SEPARATE, independently testable
    techniques. Neither module may import the other -- if one did, neither
    could be graded on its own."""
    import src.recommender.content_based as ch3
    import src.recommender.knowledge_based as ch7

    ch7_source = pathlib.Path(ch7.__file__).read_text(encoding="utf-8")
    ch3_source = pathlib.Path(ch3.__file__).read_text(encoding="utf-8")

    assert "import" in ch7_source  # sanity: we really read the source
    for line in ch7_source.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            assert "content_based" not in stripped, line
    for line in ch3_source.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            assert "knowledge_based" not in stripped, line
