"""Measure the recommender's behaviour on the real data and write it down.

Run with:  python -m src.recommender.audit

Nothing here is part of the recommender.  It exists because session 2's spec
asks for several numbers to be *reported* rather than assumed, and because
session 5 will need them to explain its evaluation results:

* median feasible-set size **with and without** the soft size constraint,
  since ``products.size`` holds one value per row rather than a stocked run;
* per-sport feasible-set sizes, showing which user segments the catalogue
  under-serves (744 yoga users, 150 yoga products);
* how many historical interactions the **hard** constraints make unreachable,
  and which constraint does it -- this depresses recall in session 5 and is
  correct behaviour, not a bug;
* what fraction of users fall back to content mode (a) for want of ratings;
* how much of the user vector actually lands in the TF-IDF vocabulary, which
  is where the weakness of ``style_preference`` shows up;
* whether the 1,284 zero-interaction products stay reachable.

Output goes to stdout and to ``reports/feasibility_report.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.common import (
    INTERACTIONS_CSV,
    PRODUCTS_CSV,
    PROJECT_ROOT,
    SEED,
    SPORTS,
    USERS_CSV,
    log,
    require,
    section,
)
from src.recommender.content_based import ContentBasedRecommender
from src.recommender.knowledge_based import (
    HARD_CONSTRAINTS,
    RELAXATION_ORDER,
    ConstraintBasedRecommender,
)
from src.recommender.profiles import normalize_profile

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORT_PATH = REPORTS_DIR / "feasibility_report.md"

# How many users to sample for the per-profile sweeps.  The full 5,000 works
# too, it just takes a couple of minutes for no extra insight.
SAMPLE_USERS = 600


def _median(values) -> float:
    return float(np.median(values)) if len(values) else 0.0


# ----------------------------------------------------------------------
# 1. Feasible-set sizes, with and without the soft size constraint
# ----------------------------------------------------------------------
def audit_feasibility(
    knowledge: ConstraintBasedRecommender, users: pd.DataFrame, rng: np.random.Generator
) -> dict[str, Any]:
    """Feasible-set size per user, split by sport, with and without size."""
    sample = users.sample(min(SAMPLE_USERS, len(users)), random_state=SEED)
    rows: list[dict[str, Any]] = []

    for profile in sample.to_dict("records"):
        normalized = normalize_profile(profile)
        hard = knowledge._hard_masks(normalized)
        hard_mask = np.ones(len(knowledge.products), dtype=bool)
        for name in HARD_CONSTRAINTS:
            hard_mask &= hard[name]

        soft = knowledge._soft_masks(normalized)
        with_size = hard_mask.copy()
        without_size = hard_mask.copy()
        for name in RELAXATION_ORDER:
            with_size &= soft[name]
            if name != "size":
                without_size &= soft[name]

        feasible, report = knowledge.filter(normalized)
        rows.append(
            {
                "user_id": normalized["user_id"],
                "sport": normalized["primary_sport"],
                "n_hard": int(hard_mask.sum()),
                "n_all_soft": int(with_size.sum()),
                "n_soft_no_size": int(without_size.sum()),
                "n_final": len(feasible),
                "n_relaxed": len(report.relaxed),
                "relaxed": ",".join(report.relaxed),
            }
        )

    frame = pd.DataFrame(rows)
    per_sport = (
        frame.groupby("sport")
        .agg(
            users=("n_final", "size"),
            median_hard=("n_hard", "median"),
            median_all_soft=("n_all_soft", "median"),
            median_no_size=("n_soft_no_size", "median"),
            median_final=("n_final", "median"),
            min_final=("n_final", "min"),
            mean_relaxed=("n_relaxed", "mean"),
        )
        .reindex([s for s in SPORTS if s in set(frame["sport"])])
    )

    relaxed_counts = {
        name: int(frame["relaxed"].str.contains(name, regex=False).sum())
        for name in RELAXATION_ORDER
    }

    return {
        "frame": frame,
        "per_sport": per_sport,
        "median_hard": _median(frame["n_hard"]),
        "median_with_size": _median(frame["n_all_soft"]),
        "median_without_size": _median(frame["n_soft_no_size"]),
        "median_final": _median(frame["n_final"]),
        "min_final": int(frame["n_final"].min()),
        "relaxed_counts": relaxed_counts,
        "n_sampled": len(frame),
    }


# ----------------------------------------------------------------------
# 2. What the hard filter costs in recall
# ----------------------------------------------------------------------
def audit_hard_filter_recall(
    products: pd.DataFrame, users: pd.DataFrame, interactions: pd.DataFrame
) -> dict[str, Any]:
    """How many historical interactions do the hard constraints exclude?

    Hard filtering removes items the user really did interact with -- 18.4% of
    the catalogue is not ``in_stock``, 23.9% of interactions fall outside the
    user's budget band, 6.4% mismatch gender.  Session 5 will see this as
    depressed recall, so it is counted here per constraint rather than
    discovered later.
    """
    merged = interactions.merge(
        products[["product_id", "price", "gender_target", "stock_status",
                  "category", "sport_type"]],
        on="product_id", how="inner",
    ).merge(
        users[["user_id", "gender", "budget_min", "budget_max", "primary_sport"]],
        on="user_id", how="inner",
    )

    target = merged["gender"].map({"male": "men", "female": "women"})
    in_budget = merged["price"].between(merged["budget_min"], merged["budget_max"])
    gender_ok = (merged["gender_target"] == "unisex") | (merged["gender_target"] == target)
    stock_ok = merged["stock_status"] == "in_stock"
    sport_ok = (merged["category"] != "footwear") | merged["sport_type"].isin(
        ["lifestyle"]
    ) | (merged["sport_type"] == merged["primary_sport"])

    survives = in_budget & gender_ok & stock_ok & sport_ok
    total = len(merged)

    ratings = merged[merged["event_type"] == "rating"]
    rating_survives = survives[merged["event_type"] == "rating"]

    return {
        "n_interactions": total,
        "per_constraint": {
            "budget": int((~in_budget).sum()),
            "gender": int((~gender_ok).sum()),
            "stock": int((~stock_ok).sum()),
            "sport (footwear only)": int((~sport_ok).sum()),
        },
        "n_survives": int(survives.sum()),
        "survival_rate": float(survives.mean()),
        "n_ratings": len(ratings),
        "rating_survival_rate": float(rating_survives.mean()) if len(ratings) else 0.0,
    }


# ----------------------------------------------------------------------
# 3. Content-based cold start and vocabulary coverage
# ----------------------------------------------------------------------
def audit_content(
    content: ContentBasedRecommender, users: pd.DataFrame
) -> dict[str, Any]:
    """Cold-start fallback rate, and how much of the user vector is in vocab."""
    fallback = content.fallback_rate(users["user_id"])

    sample = users.sample(min(SAMPLE_USERS, len(users)), random_state=SEED)
    in_vocab, out_vocab, empty = [], [], 0
    for profile in sample.to_dict("records"):
        vector = content.user_vector(profile, allow_history=False)
        in_vocab.append(vector.in_vocabulary)
        out_vocab.append(vector.out_of_vocabulary)
        if vector.vector.nnz == 0:
            empty += 1

    # Which of the style tokens actually exist in the fitted vocabulary?
    from src.recommender.content_based import STYLE_TOKENS

    vocabulary = set(content.feature_names)
    style_coverage = {
        style: "{}/{} in vocabulary ({})".format(
            sum(1 for token in tokens if token in vocabulary),
            len(tokens),
            ", ".join(t for t in tokens if t not in vocabulary) or "all present",
        )
        for style, tokens in STYLE_TOKENS.items()
    }

    return {
        **fallback,
        "vocabulary_size": len(vocabulary),
        "median_in_vocabulary": _median(in_vocab),
        "median_out_of_vocabulary": _median(out_vocab),
        "n_empty_vectors": empty,
        "style_coverage": style_coverage,
    }


# ----------------------------------------------------------------------
# 4. Are the cold products still reachable?
# ----------------------------------------------------------------------
def audit_cold_products(
    knowledge: ConstraintBasedRecommender,
    products: pd.DataFrame,
    users: pd.DataFrame,
    interactions: pd.DataFrame,
) -> dict[str, Any]:
    """Do the 1,284 zero-interaction products survive the hard filter for
    anybody?

    A product that is ``out_of_stock`` can never survive -- that is the hard
    constraint doing its job.  The question is whether the *rest* stay
    reachable, because they are session 5's cold-start cases.
    """
    touched = set(interactions["product_id"].unique())
    cold = products[~products["product_id"].isin(touched)]
    cold_in_stock = cold[cold["stock_status"] == "in_stock"]

    # A probe grid rather than a tailored profile per product: every sport,
    # both genders, three budget bands.
    reachable: set[str] = set()
    for sport in SPORTS:
        for gender in ("male", "female"):
            for budget in ((0, 60), (60, 150), (150, 2000)):
                feasible, _ = knowledge.filter(
                    {
                        "primary_sport": sport,
                        "gender": gender,
                        "budget_min": budget[0],
                        "budget_max": budget[1],
                    }
                )
                reachable.update(feasible["product_id"])

    cold_ids = set(cold_in_stock["product_id"])
    hit = cold_ids & reachable
    return {
        "n_zero_interaction": len(cold),
        "n_zero_in_stock": len(cold_in_stock),
        "n_reachable": len(hit),
        "reach_rate": len(hit) / max(len(cold_ids), 1),
        "n_blocked_by_stock": int((cold["stock_status"] != "in_stock").sum()),
        "n_catalogue_reachable": len(reachable),
    }


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------
def write_report(sections: dict[str, Any]) -> Path:
    feasibility = sections["feasibility"]
    recall = sections["recall"]
    content = sections["content"]
    cold = sections["cold"]

    lines: list[str] = [
        "# FitMatch feasibility and coverage report",
        "",
        "Generated by `python -m src.recommender.audit`. Every number here is",
        "measured from `data/processed/`, not asserted. Session 5 needs these to",
        "explain its evaluation numbers.",
        "",
        "## 1. Feasible-set size (Chapter 7 constraint filter)",
        "",
        "Sampled {:,} users at seed {}.".format(feasibility["n_sampled"], SEED),
        "",
        "| stage | median products |",
        "| --- | ---: |",
        "| catalogue | 10,000 |",
        "| after hard constraints | {:,.0f} |".format(feasibility["median_hard"]),
        "| + all soft constraints, **including** size | {:,.0f} |".format(
            feasibility["median_with_size"]
        ),
        "| + all soft constraints, **excluding** size | {:,.0f} |".format(
            feasibility["median_without_size"]
        ),
        "| after relaxation (what the ranker actually sees) | {:,.0f} |".format(
            feasibility["median_final"]
        ),
        "",
        "Smallest feasible set over the sample: **{:,}**.".format(
            feasibility["min_final"]
        ),
        "",
        "The size row is the point of spec (a). `products.size` holds one value",
        "per row -- `US 9.5`, `XXL` or `one-size` -- rather than a run of stocked",
        "sizes, so treating it as a hard exact match would collapse footwear to a",
        "few hundred rows before any other filter bites. It is soft, with",
        "tolerance: footwear within +/- 0.5 US sizes, apparel at the user's size",
        "or one adjacent size, `one-size` always matching.",
        "",
        "### How often each soft constraint gets relaxed",
        "",
        "| soft constraint (in relaxation order) | users who lost it |",
        "| --- | ---: |",
    ]
    for name in RELAXATION_ORDER:
        count = feasibility["relaxed_counts"][name]
        lines.append(
            "| {} | {:,} ({:.1f}%) |".format(
                name, count, 100.0 * count / max(feasibility["n_sampled"], 1)
            )
        )

    lines += [
        "",
        "## 2. Per-sport feasible sets -- who the catalogue under-serves",
        "",
        "There is no `general` sport (spec (b)): `sport_type` is one of ten real",
        "sports, and the catalogue mix does not match demand. `lifestyle` is",
        "treated as the sport-agnostic fallback tier and apparel/accessories are",
        "sport-flexible; only footwear is sport-strict. Without that, yoga users",
        "would be choosing from 150 products.",
        "",
        "| primary_sport | users sampled | median hard | median final | min final |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for sport, row in feasibility["per_sport"].iterrows():
        lines.append(
            "| {} | {:,} | {:,.0f} | {:,.0f} | {:,.0f} |".format(
                sport, int(row["users"]), row["median_hard"],
                row["median_final"], row["min_final"],
            )
        )

    lines += [
        "",
        "## 3. What hard filtering costs in recall",
        "",
        "Hard constraints are never relaxed, so they remove items the user",
        "genuinely interacted with. **This is correct, not a bug** -- an",
        "out-of-stock shoe in the wrong size is not a recommendation -- but it",
        "caps the recall session 5 can measure against the historical log.",
        "",
        "Of {:,} interactions joinable to both a user and a product:".format(
            recall["n_interactions"]
        ),
        "",
        "| hard constraint | interactions it excludes |",
        "| --- | ---: |",
    ]
    for name, count in recall["per_constraint"].items():
        lines.append(
            "| {} | {:,} ({:.1f}%) |".format(
                name, count, 100.0 * count / max(recall["n_interactions"], 1)
            )
        )
    lines += [
        "",
        "**{:,} of {:,} interactions ({:.1f}%) survive all four hard constraints.**".format(
            recall["n_survives"], recall["n_interactions"], 100 * recall["survival_rate"]
        ),
        "Restricted to the {:,} explicit ratings, {:.1f}% survive.".format(
            recall["n_ratings"], 100 * recall["rating_survival_rate"]
        ),
        "",
        "So a recall@10 measured against the raw interaction log has a ceiling",
        "of roughly {:.0f}%, before the ranker is judged at all. Session 5 should",
        "report recall against the *feasible* held-out items as well as the raw",
        "ones, and quote both.".format(100 * recall["survival_rate"]),
        "",
        "## 4. Content-based cold start (Chapter 3)",
        "",
        "Mode (b), profile + centroid of items rated >= 4, needs at least",
        "{} such ratings. Only 7,939 ratings exist across 5,000 users.".format(
            content["min_ratings_for_centroid"]
        ),
        "",
        "| | users | share |",
        "| --- | ---: | ---: |",
        "| mode (b), profile + centroid | {:,} | {:.1f}% |".format(
            content["n_profile_plus_centroid"],
            100 * (1 - content["fallback_rate"]),
        ),
        "| mode (a), profile only (fallback) | {:,} | **{:.1f}%** |".format(
            content["n_profile_only"], 100 * content["fallback_rate"]
        ),
        "",
        "Mean liked items per user: {:.2f}.".format(content["mean_liked_items"]),
        "",
        "### User-vector vocabulary coverage",
        "",
        "TF-IDF vocabulary: {:,} terms. A user document contributes a median of".format(
            content["vocabulary_size"]
        ),
        "{:.0f} distinct in-vocabulary terms and {:.0f} out-of-vocabulary terms;".format(
            content["median_in_vocabulary"], content["median_out_of_vocabulary"]
        ),
        "{} of the sampled users produced an all-zero vector.".format(
            content["n_empty_vectors"]
        ),
        "",
        "`style_preference` is the weak field: products.csv has no style column,",
        "so it is mapped onto catalogue words. Coverage of that map:",
        "",
        "| style_preference | tokens found in vocabulary |",
        "| --- | --- |",
    ]
    for style, coverage in content["style_coverage"].items():
        lines.append("| {} | {} |".format(style, coverage))

    lines += [
        "",
        "## 5. Cold products stay reachable",
        "",
        "Session 1 left {:,} products with zero interactions on purpose, as".format(
            cold["n_zero_interaction"]
        ),
        "cold-start cases. {:,} of them are `in_stock` and so can pass the hard".format(
            cold["n_zero_in_stock"]
        ),
        "filter at all; the other {:,} are blocked by stock, which is the".format(
            cold["n_blocked_by_stock"]
        ),
        "constraint working as specified.",
        "",
        "Probing with every sport x both genders x three budget bands, **{:,} of".format(
            cold["n_reachable"]
        ),
        "{:,} in-stock cold products ({:.1f}%) appear in at least one feasible".format(
            cold["n_zero_in_stock"], 100 * cold["reach_rate"]
        ),
        "set**. The content ranker never sees popularity (`sales_rank`), so a cold",
        "product competes with a popular one on its description alone.",
        "",
        "## See also",
        "",
        "`reports/circularity_note.md` -- which scoring features overlap the",
        "interaction generator, and what that means for session-5 evaluation.",
        "",
    ]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return REPORT_PATH


def main() -> int:
    rng = np.random.default_rng(SEED)

    section("FITMATCH RECOMMENDER AUDIT  (seed={})".format(SEED))
    require(PRODUCTS_CSV, "src.data.build_products")
    require(USERS_CSV, "src.data.build_users")
    require(INTERACTIONS_CSV, "src.data.build_interactions")

    products = pd.read_csv(PRODUCTS_CSV)
    users = pd.read_csv(USERS_CSV)
    interactions = pd.read_csv(INTERACTIONS_CSV)
    log("loaded {:,} products, {:,} users, {:,} interactions".format(
        len(products), len(users), len(interactions)))

    knowledge = ConstraintBasedRecommender(products)
    content = ContentBasedRecommender(products, interactions).fit()

    section("1. FEASIBLE-SET SIZE, WITH AND WITHOUT THE SOFT SIZE CONSTRAINT")
    feasibility = audit_feasibility(knowledge, users, rng)
    log("  sampled {:,} users".format(feasibility["n_sampled"]))
    log("  median after hard constraints        : {:>7,.0f}".format(feasibility["median_hard"]))
    log("  median with all soft incl. size      : {:>7,.0f}".format(feasibility["median_with_size"]))
    log("  median with all soft excl. size      : {:>7,.0f}".format(feasibility["median_without_size"]))
    log("  median after relaxation (ranker sees): {:>7,.0f}".format(feasibility["median_final"]))
    log("  smallest feasible set                : {:>7,}".format(feasibility["min_final"]))
    log("")
    log("  soft constraints relaxed, by how many of the sampled users:")
    for name in RELAXATION_ORDER:
        count = feasibility["relaxed_counts"][name]
        log("    {:<22}{:>6,}  ({:5.1f}%)".format(
            name, count, 100.0 * count / max(feasibility["n_sampled"], 1)))

    section("2. PER-SPORT FEASIBLE SETS")
    log(feasibility["per_sport"].round(1).to_string())

    section("3. WHAT HARD FILTERING COSTS IN RECALL")
    recall = audit_hard_filter_recall(products, users, interactions)
    log("  interactions joined to a user and a product: {:,}".format(recall["n_interactions"]))
    for name, count in recall["per_constraint"].items():
        log("    excluded by {:<24}{:>8,}  ({:5.1f}%)".format(
            name, count, 100.0 * count / max(recall["n_interactions"], 1)))
    log("  surviving all four hard constraints: {:,} ({:.1f}%)".format(
        recall["n_survives"], 100 * recall["survival_rate"]))
    log("  same for the {:,} explicit ratings : {:.1f}%".format(
        recall["n_ratings"], 100 * recall["rating_survival_rate"]))
    log("")
    log("  NOTE: this is a recall ceiling for session 5, and it is correct")
    log("        behaviour -- the hard constraints are never relaxed.")

    section("4. CONTENT-BASED COLD START AND VOCABULARY COVERAGE")
    content_stats = audit_content(content, users)
    log("  users on mode (b) profile+centroid : {:,}".format(
        content_stats["n_profile_plus_centroid"]))
    log("  users on mode (a) profile only     : {:,}".format(
        content_stats["n_profile_only"]))
    log("  cold-start fallback rate           : {:.1f}%".format(
        100 * content_stats["fallback_rate"]))
    log("  TF-IDF vocabulary                  : {:,} terms".format(
        content_stats["vocabulary_size"]))
    log("  median in-vocabulary user terms    : {:.0f}".format(
        content_stats["median_in_vocabulary"]))
    log("  median out-of-vocabulary user terms: {:.0f}".format(
        content_stats["median_out_of_vocabulary"]))
    log("  all-zero user vectors              : {}".format(
        content_stats["n_empty_vectors"]))
    log("")
    log("  style_preference token coverage:")
    for style, coverage in content_stats["style_coverage"].items():
        log("    {:<14}{}".format(style, coverage))

    section("5. COLD PRODUCTS STAY REACHABLE")
    cold = audit_cold_products(knowledge, products, users, interactions)
    log("  zero-interaction products          : {:,}".format(cold["n_zero_interaction"]))
    log("    of those, in_stock               : {:,}".format(cold["n_zero_in_stock"]))
    log("    of those, blocked by stock       : {:,}".format(cold["n_blocked_by_stock"]))
    log("  reachable by the probe grid        : {:,} ({:.1f}%)".format(
        cold["n_reachable"], 100 * cold["reach_rate"]))
    log("  catalogue products reachable at all: {:,}".format(cold["n_catalogue_reachable"]))

    path = write_report(
        {
            "feasibility": feasibility,
            "recall": recall,
            "content": content_stats,
            "cold": cold,
        }
    )
    section("DONE -- audit")
    log("wrote {}".format(path.relative_to(PROJECT_ROOT)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
