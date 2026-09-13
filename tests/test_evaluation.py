"""Session 5: the evaluation harness itself.

The numbers in ``reports/technical_summary.md`` are only worth what the code
that produced them is worth, so the metric functions are tested against
hand-computed values rather than against themselves, and the split is tested
for the two properties the write-up depends on: that it leaks nothing forward
in time, and that it is a partition.

Nothing here touches the recommender.  ``tests/test_pipeline.py`` and friends
already cover that; this file covers the ruler, not the thing being measured.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluate import (  # noqa: E402
    CONFIG_ORDER,
    average_precision_at_k,
    leave_latest_out,
    ndcg_at_k,
    precision_at_k,
    reachable_relevant,
    recall_at_k,
)
from src.evaluation.cold_start import (  # noqa: E402
    SEGMENTS,
    collaborative_reach,
    segment_of,
)
from src.recommender.knowledge_based import ConstraintBasedRecommender  # noqa: E402


# ----------------------------------------------------------------------
# Metrics -- checked against values worked out by hand
# ----------------------------------------------------------------------
def test_precision_divides_by_k_not_by_list_length() -> None:
    """A short list is penalised for the slots it could not fill.

    This is the choice documented in the function: returning three items when
    twenty were asked for is not precision 1.0 because all three were right.
    """
    assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == 1.0
    assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, 10) == pytest.approx(0.3)


def test_precision_counts_only_the_top_k() -> None:
    assert precision_at_k(["x", "y", "a"], {"a"}, 2) == 0.0
    assert precision_at_k(["x", "y", "a"], {"a"}, 3) == pytest.approx(1 / 3)


def test_recall_is_hits_over_relevant() -> None:
    assert recall_at_k(["a", "x"], {"a", "b"}, 5) == pytest.approx(0.5)
    assert recall_at_k(["a", "b"], {"a", "b"}, 5) == 1.0
    # No relevant items means the user is unevaluable, not perfectly served.
    assert recall_at_k(["a"], set(), 5) == 0.0


def test_ndcg_rewards_putting_the_hit_first() -> None:
    first = ndcg_at_k(["a", "x", "y"], {"a"}, 3)
    third = ndcg_at_k(["x", "y", "a"], {"a"}, 3)
    assert first == 1.0
    assert third == pytest.approx(1.0 / np.log2(4))
    assert first > third


def test_ndcg_is_one_when_every_relevant_item_leads() -> None:
    assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, 3) == pytest.approx(1.0)


def test_average_precision_denominator_is_capped_at_k() -> None:
    """A user with more relevant items than K can still score 1.0.

    Without the cap, three relevant items and K=2 would be capped at 2/3 for a
    purely arithmetic reason rather than because the ranking was worse.
    """
    assert average_precision_at_k(["a", "b"], {"a", "b", "c"}, 2) == pytest.approx(1.0)
    # 1/1 then 2/3, over min(2, 2) = 2.
    assert average_precision_at_k(["a", "x", "b"], {"a", "b"}, 3) == pytest.approx(
        (1.0 + 2.0 / 3.0) / 2
    )


def test_metrics_are_zero_on_an_empty_ranking() -> None:
    for metric in (precision_at_k, recall_at_k, ndcg_at_k, average_precision_at_k):
        assert metric([], {"a"}, 10) == 0.0


# ----------------------------------------------------------------------
# The split
# ----------------------------------------------------------------------
def _toy_log() -> pd.DataFrame:
    """Two users, ten events each, timestamps strictly increasing."""
    rows = []
    for user in ("U1", "U2"):
        for index in range(10):
            rows.append(
                {
                    "user_id": user,
                    "product_id": "P{}{}".format(user[-1], index),
                    "event_type": "rating" if index >= 8 else "view",
                    "timestamp": "2024-01-{:02d} 00:00:00".format(index + 1),
                    "rating": 5.0 if index >= 8 else np.nan,
                    "review_text": None,
                }
            )
    return pd.DataFrame(rows)


def test_split_holds_out_the_latest_fifth() -> None:
    split = leave_latest_out(_toy_log(), test_fraction=0.2)
    assert len(split.train) == 16
    assert len(split.test) == 4
    # The held-out rows are the last two per user, by timestamp.
    assert set(split.test["product_id"]) == {"P18", "P19", "P28", "P29"}


def test_split_never_leaks_backwards_in_time() -> None:
    """Every training event precedes every held-out event, per user.

    The property the whole evaluation rests on: if it failed, the recommender
    would be ranking a user's present using their future.
    """
    split = leave_latest_out(_toy_log(), test_fraction=0.2)
    for user in ("U1", "U2"):
        latest_train = split.train.loc[
            split.train["user_id"] == user, "timestamp"
        ].max()
        earliest_test = split.test.loc[
            split.test["user_id"] == user, "timestamp"
        ].min()
        assert latest_train < earliest_test


def test_split_is_a_partition() -> None:
    log = _toy_log()
    split = leave_latest_out(log, test_fraction=0.2)
    assert len(split.train) + len(split.test) == len(log)
    combined = pd.concat([split.train, split.test])
    assert len(combined.drop_duplicates()) == len(log)


def test_split_keeps_a_training_row_and_a_held_out_row_for_everyone() -> None:
    """Even at an extreme fraction, no user ends up with an empty half."""
    for fraction in (0.01, 0.2, 0.99):
        split = leave_latest_out(_toy_log(), test_fraction=fraction)
        for user in ("U1", "U2"):
            assert (split.train["user_id"] == user).sum() >= 1
            assert (split.test["user_id"] == user).sum() >= 1


def test_split_drops_users_with_a_single_interaction() -> None:
    """One event cannot be both trained on and held out, so the user is dropped."""
    log = pd.DataFrame(
        [
            {
                "user_id": "U9",
                "product_id": "P1",
                "event_type": "rating",
                "timestamp": "2024-01-01 00:00:00",
                "rating": 5.0,
                "review_text": None,
            }
        ]
    )
    split = leave_latest_out(log)
    assert split.train.empty
    assert split.test.empty
    assert split.evaluable_users == []


def test_relevance_is_a_rating_of_four_or_better() -> None:
    log = _toy_log()
    log.loc[log["product_id"] == "P18", "rating"] = 3.0
    split = leave_latest_out(log, test_fraction=0.2)
    assert split.relevant["U1"] == ["P19"]
    assert split.relevant["U2"] == ["P28", "P29"]


def test_evaluable_users_are_ordered_so_the_run_is_reproducible() -> None:
    """The random baseline draws in user order, so that order must be fixed."""
    split = leave_latest_out(_toy_log(), test_fraction=0.2)
    assert split.evaluable_users == sorted(split.evaluable_users)


# ----------------------------------------------------------------------
# Reachability -- the ceiling the report quotes
# ----------------------------------------------------------------------
def test_reachable_relevant_drops_items_the_hard_filter_blocks(
    products: pd.DataFrame,
) -> None:
    """An out-of-stock item is unreachable however highly the user rated it."""
    knowledge = ConstraintBasedRecommender(products)
    row_of = {str(pid): index for index, pid in enumerate(products["product_id"])}

    in_stock = products[products["stock_status"] == "in_stock"].iloc[0]
    out_of_stock = products[products["stock_status"] == "out_of_stock"].iloc[0]
    profile = {
        "gender": None,
        "primary_sport": str(in_stock["sport_type"]),
        "budget_min": 0.0,
        "budget_max": 1e9,
    }

    reachable = reachable_relevant(
        knowledge,
        profile,
        [str(in_stock["product_id"]), str(out_of_stock["product_id"])],
        row_of,
    )
    assert str(out_of_stock["product_id"]) not in reachable


def test_the_measured_ceiling_matches_the_published_one(
    products: pd.DataFrame, users: pd.DataFrame, interactions: pd.DataFrame
) -> None:
    """The recall ceiling is close to the 57.2%/60.3% in feasibility_report.md.

    Not equal to either: those are measured over all interactions and over all
    ratings, while this is measured over held-out items rated 4+. The write-up
    quotes all three and explains the gap, so this test guards the claim that
    they are the same order of magnitude rather than a coincidence.
    """
    split = leave_latest_out(interactions)
    knowledge = ConstraintBasedRecommender(products)
    row_of = {str(pid): index for index, pid in enumerate(products["product_id"])}
    profiles = {str(row["user_id"]): row for row in users.to_dict("records")}

    shares = []
    for user_id in split.evaluable_users[:400]:
        profile = profiles.get(user_id)
        if profile is None:
            continue
        relevant = split.relevant[user_id]
        reachable = reachable_relevant(knowledge, profile, relevant, row_of)
        shares.append(len(reachable) / len(relevant))

    ceiling = float(np.mean(shares))
    assert 0.50 < ceiling < 0.72, ceiling


# ----------------------------------------------------------------------
# Cold-start segmentation
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "count,expected",
    [(0, "0 ratings"), (1, "1-2 ratings"), (2, "1-2 ratings"),
     (3, "3-5 ratings"), (5, "3-5 ratings"), (6, "6+ ratings"), (99, "6+ ratings")],
)
def test_segments_partition_the_rating_counts(count: int, expected: str) -> None:
    assert segment_of(count) == expected


def test_segment_boundary_falls_where_the_scoring_mode_changes() -> None:
    """The first two segments must be exactly the cold-start population.

    If a boundary moved, the report's "85.2% take the profile-only path" would
    stop lining up with the segment table printed beneath it.
    """
    from src.recommender.content_based import MIN_RATINGS_FOR_CENTROID

    cold = [name for name, low, _ in SEGMENTS if low < MIN_RATINGS_FOR_CENTROID]
    assert cold == ["0 ratings", "1-2 ratings"]


def test_population_segments_reproduce_the_published_cold_start_share(
    users: pd.DataFrame, interactions: pd.DataFrame
) -> None:
    """Independently re-derives the 85.2% from session 2."""
    from src.evaluation.cold_start import population_segments

    population = population_segments(users, interactions)
    assert int(population["users"].sum()) == len(users)
    cold = population[population["mode"].str.startswith("profile only")]
    assert float(cold["share"].sum()) == pytest.approx(0.852, abs=0.005)


def test_collaborative_reach_counts_the_preconditions(
    products: pd.DataFrame, users: pd.DataFrame, interactions: pd.DataFrame
) -> None:
    """The numbers behind "CF would have returned nothing" for a third of users."""
    split = leave_latest_out(interactions)
    reach = collaborative_reach(users, products, split.train)

    assert reach["users_with_ratings"] + reach["users_without_ratings"] == len(users)
    assert (
        reach["products_with_ratings"] + reach["products_without_ratings"]
        == len(products)
    )
    # A user with no neighbour is a superset of a user with no rating.
    assert reach["users_without_neighbour"] >= reach["users_without_ratings"]
    # Co-rated products are a subset of rated products.
    assert reach["co_rated_products"] <= reach["products_with_ratings"]
    # The headline claims, loosely bounded so a data rebuild does not break them.
    assert reach["density"] < 0.001
    assert reach["users_without_ratings"] > len(users) * 0.25
    assert reach["products_without_ratings"] > len(products) * 0.5


# ----------------------------------------------------------------------
# Wiring
# ----------------------------------------------------------------------
def test_every_configuration_has_a_runner() -> None:
    """CONFIG_ORDER drives the loop, the table and the charts; a typo in it
    would silently drop a configuration from all three."""
    from src.evaluation.evaluate import CONFIG_COLORS, CONFIG_LABELS, Configurations

    for name in CONFIG_ORDER:
        assert hasattr(Configurations, name), name
        assert name in CONFIG_LABELS
        assert name in CONFIG_COLORS
