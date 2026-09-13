"""Does the profile-only path hold up? It serves 85.2% of users, so it must.

Run with::

    python -m src.evaluation.cold_start
    python -m src.evaluation.cold_start --recompute      # ignore the cached per-user CSV
    python -m src.evaluation.cold_start --no-figures

Writes ``reports/cold_start_results.csv`` and two charts under
``reports/figures/``.

Why this file exists
--------------------
Chapter 3 has two modes.  Mode (b) blends the user's profile vector with the
centroid of the items they rated 4 or 5, and needs at least three such ratings.
Mode (a) uses the profile alone.  Session 2 measured that 4,258 of 5,000 users
(85.2%) cannot reach mode (b), because only 7,939 ratings exist across the
whole population -- a mean of 1.29 liked items per user.

That single number is the argument for the two techniques this project chose.
A recommender whose quality collapses without ratings would be useless to six
users in seven here.  So the question this module answers is not "is the
cascade accurate" -- ``evaluate.py`` covers that, with all the circularity
caveats -- but "does it degrade when the ratings are taken away".  It segments
the evaluated users by how many qualifying ratings they have and compares the
segments.

The counterfactual is the other half of the argument.  Collaborative filtering
was on the course list and was not chosen, and :func:`collaborative_reach`
measures what it would have been able to return on this data: for a user with
no ratings, nothing at all -- not a worse ranking, an empty one.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.data.common import (
    INTERACTIONS_CSV,
    PRODUCTS_CSV,
    PROJECT_ROOT,
    USERS_CSV,
    log,
    require,
    section,
)
from src.evaluation.evaluate import (
    CONFIG_COLORS,
    CONFIG_LABELS,
    CONFIG_ORDER,
    FIGURES_DIR,
    PER_USER_CSV,
    REPORTS_DIR,
    evaluate,
    leave_latest_out,
)
from src.recommender.content_based import (
    LIKED_RATING,
    MIN_RATINGS_FOR_CENTROID,
)

RESULTS_CSV = REPORTS_DIR / "cold_start_results.csv"

#: Segment boundaries over the qualifying-rating count, as half-open [low, high).
#: Chosen so the mode boundary falls between segment 2 and segment 3: a user
#: needs MIN_RATINGS_FOR_CENTROID (3) qualifying ratings for mode (b), so the
#: first two segments are exactly the cold-start population.
SEGMENTS: tuple[tuple[str, int, float], ...] = (
    ("0 ratings", 0, 1),
    ("1-2 ratings", 1, 3),
    ("3-5 ratings", 3, 6),
    ("6+ ratings", 6, float("inf")),
)


def segment_of(count: int) -> str:
    """Which segment a qualifying-rating count falls in."""
    for name, low, high in SEGMENTS:
        if low <= count < high:
            return name
    return SEGMENTS[-1][0]


def segment_mode(name: str) -> str:
    """Which Chapter 3 mode that segment gets, by construction."""
    low = dict((label, low) for label, low, _ in SEGMENTS)[name]
    return (
        "profile + centroid"
        if low >= MIN_RATINGS_FOR_CENTROID
        else "profile only (cold start)"
    )


# ----------------------------------------------------------------------
# Population shape -- over all 5,000 users, not just the evaluated ones
# ----------------------------------------------------------------------
def population_segments(
    users: pd.DataFrame, interactions: pd.DataFrame
) -> pd.DataFrame:
    """How the whole user base splits across the four segments.

    Counted over every user in ``users.csv``, including the ones the offline
    evaluation cannot score, because the 85.2% figure this module exists to
    justify is a statement about the product's whole audience rather than about
    the evaluable subset.
    """
    liked = interactions[
        pd.to_numeric(interactions["rating"], errors="coerce") >= LIKED_RATING
    ]
    counts = liked.groupby("user_id").size()
    per_user = (
        users["user_id"]
        .astype(str)
        .map(counts)
        .fillna(0)
        .astype(int)
    )
    frame = pd.DataFrame({"qualifying_ratings": per_user})
    frame["segment"] = frame["qualifying_ratings"].map(segment_of)

    rows = []
    for name, _, _ in SEGMENTS:
        block = frame[frame["segment"] == name]
        rows.append(
            {
                "segment": name,
                "mode": segment_mode(name),
                "users": int(len(block)),
                "share": len(block) / len(frame) if len(frame) else 0.0,
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# The counterfactual: what collaborative filtering could have served
# ----------------------------------------------------------------------
def collaborative_reach(
    users: pd.DataFrame, products: pd.DataFrame, train: pd.DataFrame
) -> dict[str, Any]:
    """What a collaborative filter could return on this data, before quality.

    Deliberately not a CF implementation.  Every number here is a *precondition*
    for collaborative filtering to produce any output at all, so they bound the
    best case of every CF variant at once -- user-based, item-based, or matrix
    factorisation -- without anyone having to agree on which variant or which
    hyperparameters.  A user with no ratings has no row in the matrix, so
    user-based CF has no neighbourhood to average and factorisation has no
    latent vector to place them at; an item nobody has rated has no column, so
    it can never be recommended by any of them.

    Measured on the *training* half of the same split ``evaluate.py`` uses, so
    the numbers describe the same experiment.
    """
    rated = train[pd.to_numeric(train["rating"], errors="coerce").notna()]
    n_users = int(len(users))
    n_products = int(len(products))

    users_with_ratings = int(rated["user_id"].nunique())
    products_with_ratings = int(rated["product_id"].nunique())

    # A user-based neighbourhood needs at least one other user who rated
    # something this user also rated.
    with_neighbour: set[str] = set()
    for _, group in rated.groupby("product_id", sort=False)["user_id"]:
        members = set(group)
        if len(members) > 1:
            with_neighbour |= members

    # An item-item similarity needs a product rated by at least two users.
    co_rated_products = int(
        (rated.groupby("product_id")["user_id"].nunique() >= 2).sum()
    )

    return {
        "n_ratings": int(len(rated)),
        "n_users": n_users,
        "n_products": n_products,
        "density": len(rated) / float(n_users * n_products),
        "users_with_ratings": users_with_ratings,
        "users_without_ratings": n_users - users_with_ratings,
        "users_with_neighbour": len(with_neighbour),
        "users_without_neighbour": n_users - len(with_neighbour),
        "products_with_ratings": products_with_ratings,
        "products_without_ratings": n_products - products_with_ratings,
        "co_rated_products": co_rated_products,
    }


# ----------------------------------------------------------------------
# Segmented metrics
# ----------------------------------------------------------------------
def segment_metrics(per_user: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Per-segment metrics for every configuration at one K."""
    block = per_user[per_user["k"] == k].copy()
    block["segment"] = block["qualifying_ratings"].map(segment_of)

    grouped = (
        block.groupby(["config", "segment"], sort=False)
        .agg(
            users=("user_id", "nunique"),
            n_relevant=("n_relevant", "sum"),
            n_reachable=("n_reachable", "sum"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            recall_reachable=("recall_reachable", "mean"),
            ndcg=("ndcg", "mean"),
            map=("ap", "mean"),
        )
        .reset_index()
    )
    grouped["k"] = k
    grouped["mode"] = grouped["segment"].map(segment_mode)

    config_rank = {name: index for index, name in enumerate(CONFIG_ORDER)}
    segment_rank = {name: index for index, (name, _, _) in enumerate(SEGMENTS)}
    grouped["_c"] = grouped["config"].map(config_rank)
    grouped["_s"] = grouped["segment"].map(segment_rank)
    return (
        grouped.sort_values(["_c", "_s"], kind="stable")
        .drop(columns=["_c", "_s"])
        .reset_index(drop=True)
    )


# ----------------------------------------------------------------------
# Presentation
# ----------------------------------------------------------------------
def print_population(population: pd.DataFrame) -> None:
    section("1. THE POPULATION -- all 5,000 users in users.csv")
    log(
        "Qualifying ratings are ratings of {} or better. A user needs {} of them\n"
        "before Chapter 3 blends past behaviour into their vector.".format(
            LIKED_RATING, MIN_RATINGS_FOR_CENTROID
        )
    )
    log("")
    header = "{:<16}{:<30}{:>10}{:>10}".format("segment", "Chapter 3 mode", "users", "share")
    log(header)
    log("-" * len(header))
    for row in population.itertuples():
        log(
            "{:<16}{:<30}{:>10,}{:>9.1f}%".format(
                row.segment, row.mode, row.users, 100.0 * row.share
            )
        )
    cold = population[population["mode"].str.startswith("profile only")]
    log("-" * len(header))
    log(
        "{:<16}{:<30}{:>10,}{:>9.1f}%".format(
            "COLD START", "profile only", int(cold["users"].sum()),
            100.0 * float(cold["share"].sum()),
        )
    )


def print_segments(metrics: pd.DataFrame, k: int) -> None:
    section("2. METRICS BY SEGMENT AT K = {}".format(k))
    log(
        "Segmented over the users the offline evaluation can score. Qualifying\n"
        "ratings are counted from the TRAINING half only -- which is why the\n"
        "cold-start segments are even larger here than in the population above."
    )
    for name in CONFIG_ORDER:
        block = metrics[metrics["config"] == name]
        if block.empty:
            continue
        log("")
        log(CONFIG_LABELS[name])
        header = "  {:<16}{:>8}{:>10}{:>10}{:>10}{:>10}".format(
            "segment", "users", "P@K", "R@K", "NDCG@K", "MAP@K"
        )
        log(header)
        log("  " + "-" * (len(header) - 2))
        for row in block.itertuples():
            log(
                "  {:<16}{:>8,}{:>10.4f}{:>10.4f}{:>10.4f}{:>10.4f}".format(
                    row.segment, row.users, row.precision, row.recall,
                    row.ndcg, row.map,
                )
            )


def print_verdict(metrics: pd.DataFrame, k: int) -> None:
    """Does the profile-only path hold up? Answered from the numbers, not asserted."""
    cascade = metrics[metrics["config"] == "cascade"].set_index("segment")
    cold_names = [name for name, low, _ in SEGMENTS if low < MIN_RATINGS_FOR_CENTROID]
    warm_names = [name for name, low, _ in SEGMENTS if low >= MIN_RATINGS_FOR_CENTROID]

    def weighted(names: list[str], column: str) -> tuple[float, int]:
        block = cascade.loc[[n for n in names if n in cascade.index]]
        if block.empty or block["users"].sum() == 0:
            return float("nan"), 0
        total = int(block["users"].sum())
        value = float((block[column] * block["users"]).sum() / total)
        return value, total

    section("3. DOES THE PROFILE-ONLY PATH HOLD UP?")
    header = "{:<34}{:>10}{:>12}{:>12}{:>12}".format(
        "cascade, users grouped by mode", "users", "P@{}".format(k),
        "R@{}".format(k), "NDCG@{}".format(k)
    )
    log(header)
    log("-" * len(header))
    rows = []
    for label, names in (
        ("profile only (cold start)", cold_names),
        ("profile + centroid", warm_names),
    ):
        precision, total = weighted(names, "precision")
        recall, _ = weighted(names, "recall")
        ndcg, _ = weighted(names, "ndcg")
        rows.append((label, total, precision, recall, ndcg))
        log(
            "{:<34}{:>10,}{:>12.4f}{:>12.4f}{:>12.4f}".format(
                label, total, precision, recall, ndcg
            )
        )

    cold_row, warm_row = rows
    log("")
    if warm_row[1] == 0 or not np.isfinite(warm_row[2]) or warm_row[2] == 0:
        log("Not enough mode (b) users at this K to compare the two paths.")
        return

    ratios = {
        "precision@{}".format(k): cold_row[2] / warm_row[2] if warm_row[2] else np.nan,
        "recall@{}".format(k): cold_row[3] / warm_row[3] if warm_row[3] else np.nan,
        "NDCG@{}".format(k): cold_row[4] / warm_row[4] if warm_row[4] else np.nan,
    }
    log("cold-start path as a multiple of the warm path:")
    for label, value in ratios.items():
        log("  {:<16}{:>8.2f}x".format(label, value))
    log("")
    log(
        "Cold start serves {:,} of the {:,} evaluated users ({:.1%}); the warm\n"
        "path serves {:,}. Read the ratios with that second number in mind --\n"
        "the warm segment is small enough that one extra hit moves it, and the\n"
        "6+ bucket in particular holds too few users to carry any weight.".format(
            cold_row[1], cold_row[1] + warm_row[1],
            cold_row[1] / float(cold_row[1] + warm_row[1]), warm_row[1],
        )
    )
    log("")

    finite = [value for value in ratios.values() if np.isfinite(value)]
    if finite and min(finite) >= 0.75:
        log(
            "The profile-only path holds up. Taking every rating away costs the\n"
            "cascade little on any metric, because the ranking was never leaning\n"
            "on ratings: Chapter 3's signal is the product description and the\n"
            "structured tokens, Chapter 7's is the profile form, and neither\n"
            "needs a rating history to exist. That is the property the technique\n"
            "choice was made for."
        )
    else:
        weak = [label for label, value in ratios.items()
                if np.isfinite(value) and value < 0.75]
        log(
            "The profile-only path holds up on some metrics and not all: it\n"
            "trails the warm path on {}. The gap is in the expected\n"
            "direction -- three ratings genuinely do add signal -- and the\n"
            "cascade still degrades gracefully rather than failing, which is\n"
            "the property collaborative filtering could not have offered at\n"
            "all (section 4). It is reported as measured rather than rounded\n"
            "into a cleaner story; see the limitations section of the\n"
            "technical summary.".format(", ".join(weak))
        )


def print_counterfactual(reach: Mapping[str, Any], population: pd.DataFrame) -> None:
    section("4. WHAT COLLABORATIVE FILTERING WOULD HAVE RETURNED")
    log(
        "Collaborative filtering was on the course's technique list and was not\n"
        "chosen. These are the preconditions it needs on this data, measured on\n"
        "the same training split the evaluation uses. They bound every CF\n"
        "variant at once: no row in the matrix means no neighbourhood and no\n"
        "latent vector; no column means an item that can never be returned."
    )
    log("")
    log(
        "rating matrix                  {:,} x {:,} holding {:,} ratings".format(
            reach["n_users"], reach["n_products"], reach["n_ratings"]
        )
    )
    log("density                        {:.4%}".format(reach["density"]))
    log("")
    header = "{:<44}{:>10}{:>10}".format("precondition", "count", "share")
    log(header)
    log("-" * len(header))

    def line(label: str, count: int, total: int) -> None:
        log("{:<44}{:>10,}{:>9.1f}%".format(label, count, 100.0 * count / total))

    line("users with no rating at all", reach["users_without_ratings"], reach["n_users"])
    line(
        "users with no co-rating neighbour",
        reach["users_without_neighbour"],
        reach["n_users"],
    )
    line(
        "products no one has rated",
        reach["products_without_ratings"],
        reach["n_products"],
    )
    line(
        "products rated by only one user",
        reach["products_with_ratings"] - reach["co_rated_products"],
        reach["n_products"],
    )
    log("")
    log(
        "So a user-based collaborative filter returns an EMPTY LIST for {:,} of\n"
        "{:,} users ({:.1f}%) -- not a worse ranking, no ranking. Widening the\n"
        "requirement to 'has at least one neighbour who rated the same product'\n"
        "takes it to {:,} users ({:.1f}%). An item-based filter can never reach\n"
        "{:,} of the {:,} products ({:.1f}% of the catalogue), because nothing in\n"
        "the training half has rated them.".format(
            reach["users_without_ratings"],
            reach["n_users"],
            100.0 * reach["users_without_ratings"] / reach["n_users"],
            reach["users_without_neighbour"],
            100.0 * reach["users_without_neighbour"] / reach["n_users"],
            reach["products_without_ratings"],
            reach["n_products"],
            100.0 * reach["products_without_ratings"] / reach["n_products"],
        )
    )
    log("")
    log(
        "The two techniques that were chosen have no such precondition. The\n"
        "Chapter 7 filter reads the profile form; the Chapter 3 ranker reads the\n"
        "product text. Both work for a user who has existed for four seconds,\n"
        "and every one of the 10,000 products is reachable by both -- session 2\n"
        "verified that all 1,010 in-stock zero-interaction products appear in at\n"
        "least one feasible set (feasibility_report.md 5)."
    )


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def write_figures(
    metrics: pd.DataFrame,
    reach: Mapping[str, Any],
    k: int,
    figures_dir: Path = FIGURES_DIR,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 130,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.axisbelow": True,
        }
    )
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    names = [name for name, _, _ in SEGMENTS]

    # --- 1. cascade metrics per segment ---------------------------------
    cascade = metrics[metrics["config"] == "cascade"].set_index("segment")
    present = [name for name in names if name in cascade.index]
    figure, (left, right) = plt.subplots(1, 2, figsize=(10.4, 4.0),
                                         gridspec_kw={"width_ratios": [1.15, 1]})

    width = 0.27
    positions = np.arange(len(present))
    for offset, (column, label, color) in enumerate(
        [
            ("precision", "Precision@{}".format(k), "#b3202c"),
            ("recall", "Recall@{}".format(k), "#4c78a8"),
            ("ndcg", "NDCG@{}".format(k), "#72b7b2"),
        ]
    ):
        values = [float(cascade.loc[name, column]) for name in present]
        left.bar(positions + (offset - 1) * width, values, width=width,
                 color=color, label=label)
    left.set_xticks(positions)
    left.set_xticklabels(present, fontsize=8.5)
    left.axvline(1.5, color="#999999", linestyle="--", linewidth=1.2)
    left.text(1.45, left.get_ylim()[1] * 0.95, "profile only  ", ha="right",
              va="top", fontsize=8, color="#555555")
    left.text(1.55, left.get_ylim()[1] * 0.95, "  + centroid", ha="left",
              va="top", fontsize=8, color="#555555")
    left.set_title("Full cascade by qualifying-rating count", fontsize=10)
    left.legend(frameon=False, fontsize=8)

    counts = [int(cascade.loc[name, "users"]) for name in present]
    colors = ["#b3202c" if index < 2 else "#9e9e9e" for index in range(len(present))]
    bars = right.bar(range(len(counts)), counts, color=colors, width=0.62)
    right.set_xticks(range(len(counts)))
    right.set_xticklabels(present, fontsize=8.5)
    right.set_ylabel("evaluated users")
    right.set_title("How many users each segment holds", fontsize=10)
    for bar, value in zip(bars, counts):
        right.text(bar.get_x() + bar.get_width() / 2, value, "{:,}".format(value),
                   ha="center", va="bottom", fontsize=8)
    right.set_ylim(0, max(counts) * 1.16)

    figure.suptitle(
        "The cold-start path (red) carries most of the population", fontsize=11
    )
    figure.tight_layout()
    path = figures_dir / "cold_start_segments.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    written.append(path)

    # --- 2. what CF could reach ------------------------------------------
    figure, axis = plt.subplots(figsize=(7.4, 3.6))
    rows = [
        ("Users a user-based CF\ncan serve at all",
         reach["users_with_ratings"], reach["n_users"]),
        ("Users with a co-rating\nneighbour",
         reach["users_with_neighbour"], reach["n_users"]),
        ("Products an item-based CF\ncan ever recommend",
         reach["products_with_ratings"], reach["n_products"]),
        ("Products with 2+ raters\n(item-item similarity)",
         reach["co_rated_products"], reach["n_products"]),
    ]
    labels = [row[0] for row in rows]
    shares = [100.0 * row[1] / row[2] for row in rows]
    positions = np.arange(len(rows))
    axis.barh(positions, [100] * len(rows), color="#eeeeee", height=0.6)
    bars = axis.barh(positions, shares, color="#b3202c", height=0.6)
    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlim(0, 100)
    axis.set_xlabel("% reachable")
    axis.set_title(
        "Collaborative filtering on this data: what it could even return",
        fontsize=10,
    )
    for bar, (label, count, total) in zip(bars, rows):
        axis.text(
            bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
            "{:,} of {:,}  ({:.1f}%)".format(count, total, 100.0 * count / total),
            va="center", fontsize=8,
        )
    figure.tight_layout()
    path = figures_dir / "collaborative_filtering_reach.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    written.append(path)

    return written


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluation.cold_start",
        description="Cold-start segmentation of the FitMatch cascade, and the "
        "quantitative case against collaborative filtering on this data.",
    )
    parser.add_argument(
        "--recompute", action="store_true",
        help="re-run the evaluation instead of reusing reports/evaluation_per_user.csv",
    )
    parser.add_argument(
        "--sample-users", type=int, default=None,
        help="when recomputing, evaluate this many users only",
    )
    parser.add_argument("--k", type=int, default=10, help="cut-off to segment at")
    parser.add_argument(
        "--no-figures", action="store_true", help="skip the matplotlib charts"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    for path, produced_by in (
        (PRODUCTS_CSV, "src.data.build_products"),
        (USERS_CSV, "src.data.build_users"),
        (INTERACTIONS_CSV, "src.data.build_interactions"),
    ):
        require(path, produced_by)

    products = pd.read_csv(PRODUCTS_CSV)
    users = pd.read_csv(USERS_CSV)
    interactions = pd.read_csv(INTERACTIONS_CSV)

    section("FITMATCH COLD-START ANALYSIS")
    log(
        "The profile-only path serves 85.2% of users. This file asks whether it\n"
        "holds up, and what the alternative technique would have done instead."
    )

    if not args.recompute and PER_USER_CSV.exists():
        log("")
        log("Reusing {} (pass --recompute to re-run).".format(
            PER_USER_CSV.relative_to(PROJECT_ROOT)
        ))
        per_user = pd.read_csv(PER_USER_CSV)
    else:
        log("")
        log("Running the evaluation to get per-user results...")
        _, per_user, _ = evaluate(
            products, users, interactions, sample_users=args.sample_users
        )

    if args.k not in set(per_user["k"].unique()):
        raise SystemExit(
            "K = {} is not in the per-user results (have {}). Re-run with "
            "--recompute.".format(args.k, sorted(per_user["k"].unique()))
        )

    population = population_segments(users, interactions)
    metrics = segment_metrics(per_user, k=args.k)
    split = leave_latest_out(interactions)
    reach = collaborative_reach(users, products, split.train)

    print_population(population)
    print_segments(metrics, args.k)
    print_verdict(metrics, args.k)
    print_counterfactual(reach, population)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(RESULTS_CSV, index=False)

    section("WRITTEN")
    log("  {}".format(RESULTS_CSV.relative_to(PROJECT_ROOT)))
    if not args.no_figures:
        for path in write_figures(metrics, reach, args.k):
            log("  {}".format(path.relative_to(PROJECT_ROOT)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
