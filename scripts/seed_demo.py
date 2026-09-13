"""Three contrasting profiles for the live demo, with the script to read aloud.

Run with::

    python scripts/seed_demo.py
    python scripts/seed_demo.py --top-n 10
    python scripts/seed_demo.py --json          # the same three as API request bodies

The point of the demo is not that the recommendations are good -- offline
accuracy is ``src/evaluation/`` territory, and ``reports/circularity_note.md``
explains why that number means less than it looks like.  The point is that the
system is *legible*: three different people put three different things into the
form and get three visibly different answers, and it says out loud which of
their preferences it could not honour and which of the two scoring modes ranked
the list.

The three profiles are chosen to exercise different parts of the cascade rather
than to flatter it:

* **Budget-conscious beginner runner.**  A narrow band at the cheap end of the
  catalogue, so the hard budget constraint bites hard and the feasible set is
  small before any soft constraint is considered.  No brand loyalty, so the
  brand constraint never fires.
* **Advanced outdoor athlete.**  A high budget that the budget constraint
  barely touches, and a gore-tex preference against a catalogue holding only
  191 gore-tex products, so the *material* constraint is the one that has to be
  relaxed.
* **Intermediate indoor gym user.**  Strong Nike loyalty and a performance
  style, so the brand constraint stays applied and the Chapter 3 ranker is
  working inside a brand-filtered set.

All three are anonymous profiles with no ``user_id``, which means all three
take the profile-only scoring path -- the one 85.2% of real users in this
dataset take.  That is the honest thing to demo.  To show the centroid path
instead, run the pipeline against a stored user with enough ratings::

    python -m src.recommender.pipeline --user-id 42 --top-n 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.common import log, section  # noqa: E402
from src.recommender.knowledge_based import RELAXATION_ORDER  # noqa: E402
from src.recommender.pipeline import FitMatchRecommender, Recommendation  # noqa: E402

# Human names for the machine names, matching src/api/service.py so the demo
# and the UI say the same words about the same thing.
CONSTRAINT_LABELS: dict[str, str] = {
    "color_preference": "colour",
    "material_preference": "material",
    "arch_support": "arch support",
    "seasonality": "seasonal suitability",
    "preferred_brands": "preferred brands",
    "size": "size",
}

MODE_LABELS: dict[str, str] = {
    "profile_only": "profile only (cold start) -- the path 85.2% of users take",
    "profile_plus_centroid": "profile + centroid of items rated 4+",
}


class DemoProfile:
    """One demo persona: a profile, a name, and the line to say about it."""

    def __init__(
        self, key: str, name: str, pitch: str, profile: Mapping[str, Any]
    ) -> None:
        self.key = key
        self.name = name
        self.pitch = pitch
        self.profile = dict(profile)


#: Priced against the measured catalogue: the median product is $60.80, the
#: 10th percentile $31.69 and the 90th $133.29, so 35-70 really is the cheap
#: end and 150-400 really is the top.
DEMO_PROFILES: tuple[DemoProfile, ...] = (
    DemoProfile(
        key="budget_runner",
        name="Budget-conscious beginner runner",
        pitch=(
            "Just started running, wants shoes that fit and will not spend more "
            "than $70. No brand opinion at all."
        ),
        profile={
            "age": 24,
            "gender": "female",
            "height_cm": 167.0,
            "weight_kg": 61.0,
            "shoe_size": 8.0,
            "apparel_size": "M",
            "foot_arch_type": "normal",
            "primary_sport": "running",
            "fitness_level": "beginner",
            "workouts_per_week": 2,
            "indoor_or_outdoor": "outdoor",
            "budget_min": 35,
            "budget_max": 70,
            # Deliberately empty: no brand loyalty, so the brand constraint
            # never fires and never needs relaxing.
            "preferred_brands": [],
            "style_preference": "casual",
            "color_preference": "blue",
            "material_preference": "mesh",
            "location": "Portland, OR",
            "climate": "temperate",
        },
    ),
    DemoProfile(
        key="outdoor_athlete",
        name="Advanced outdoor athlete",
        pitch=(
            "Trains outdoors year round in a cold climate, budget is not the "
            "constraint, wants gore-tex and waterproofing."
        ),
        profile={
            "age": 38,
            "gender": "male",
            "height_cm": 183.0,
            "weight_kg": 82.0,
            "shoe_size": 11.5,
            "apparel_size": "L",
            "foot_arch_type": "high",
            "primary_sport": "outdoor",
            "fitness_level": "advanced",
            "workouts_per_week": 6,
            "indoor_or_outdoor": "outdoor",
            "budget_min": 150,
            "budget_max": 400,
            "preferred_brands": [],
            "style_preference": "performance",
            "color_preference": "green",
            # Only 191 of 10,000 products are gore-tex, so this is the
            # constraint the demo watches get relaxed.
            "material_preference": "gore-tex",
            "location": "Denver, CO",
            "climate": "cold",
        },
    ),
    DemoProfile(
        key="nike_gym",
        name="Intermediate indoor gym user, Nike loyalist",
        pitch=(
            "Trains indoors three or four times a week, will wear Nike and "
            "nothing else, wants performance kit rather than lifestyle."
        ),
        profile={
            "age": 29,
            "gender": "male",
            "height_cm": 178.0,
            "weight_kg": 76.0,
            "shoe_size": 10.0,
            "apparel_size": "M",
            "foot_arch_type": "low",
            "primary_sport": "training",
            "fitness_level": "intermediate",
            "workouts_per_week": 4,
            "indoor_or_outdoor": "indoor",
            "budget_min": 60,
            "budget_max": 180,
            "preferred_brands": ["Nike"],
            "style_preference": "performance",
            "color_preference": "black",
            "material_preference": "synthetic",
            "location": "Chicago, IL",
            "climate": "continental",
        },
    ),
)


# ----------------------------------------------------------------------
# Printing -- written to be read aloud
# ----------------------------------------------------------------------
def describe_profile(demo: DemoProfile) -> None:
    log(demo.pitch)
    log("")
    profile = demo.profile
    brands = profile.get("preferred_brands") or []
    log(
        "  {} {}, {}, {} {}   |   sport: {}   |   {} training".format(
            profile["age"],
            profile["gender"],
            profile["fitness_level"],
            profile["workouts_per_week"],
            "session/week" if profile["workouts_per_week"] == 1 else "sessions/week",
            profile["primary_sport"],
            profile["indoor_or_outdoor"],
        )
    )
    log(
        "  budget ${:,}-${:,}   |   shoe {}   |   apparel {}   |   {} arch".format(
            profile["budget_min"],
            profile["budget_max"],
            profile["shoe_size"],
            profile["apparel_size"],
            profile["foot_arch_type"],
        )
    )
    log(
        "  wants {} {} in {}   |   brands: {}   |   {}, {} climate".format(
            profile["style_preference"],
            profile["material_preference"],
            profile["color_preference"],
            ", ".join(brands) if brands else "no preference",
            profile["location"],
            profile["climate"],
        )
    )


def describe_constraints(result: Recommendation) -> None:
    report = result.relaxation_log
    log("")
    log(
        "  Constraint filter (Ch7):  10,000 -> {:,} after the four hard "
        "constraints -> {:,} feasible".format(report.n_after_hard, report.n_feasible)
    )
    excluded = ", ".join(
        "{} {:,}".format(name, count)
        for name, count in report.hard_excluded.items()
        if count
    )
    log("    hard constraints removed: {}".format(excluded or "nothing"))

    if report.relaxed:
        log(
            "    RELAXED, in order: {}".format(
                ", ".join(CONSTRAINT_LABELS.get(name, name) for name in report.relaxed)
            )
        )
        for step in report.steps:
            if step.action != "relaxed":
                continue
            log(
                "      gave up {:<22} {:,} -> {:,} products".format(
                    CONSTRAINT_LABELS.get(step.constraint, step.constraint),
                    step.n_before,
                    step.n_after,
                )
            )
    else:
        log("    RELAXED: nothing -- every stated preference was honoured")

    still = [
        CONSTRAINT_LABELS.get(name, name)
        for name in RELAXATION_ORDER
        if name in report.applied
    ]
    log("    still applied: {}".format(", ".join(still) if still else "none"))
    log(
        "  Scoring mode (Ch3):       {}".format(
            MODE_LABELS.get(result.content_mode, result.content_mode)
        )
    )


def _clip(value: Any, width: int) -> str:
    """Truncate with an ellipsis, so a cut-off word does not read as a typo."""
    text = str(value)
    return text if len(text) <= width else text[: width - 1] + "…"


def describe_results(result: Recommendation, top_n: int) -> None:
    log("")
    log(
        "  Top {} -- cosine similarity against the profile's TF-IDF vector:".format(
            min(top_n, len(result.items))
        )
    )
    log("")
    header = "   {:>2}  {:<12}{:>8}  {:<10}{:<26}{:<14}{:<10}".format(
        "#", "product_id", "score", "brand", "subcategory", "sport", "price"
    )
    log(header)
    log("   " + "-" * (len(header) - 3))
    for row in result.items.head(top_n).to_dict("records"):
        log(
            "   {:>2}  {:<12}{:>8.4f}  {:<10}{:<26}{:<14}${:<9,.2f}".format(
                int(row["rank"]),
                str(row["product_id"]),
                float(row["score"]),
                _clip(row.get("brand", ""), 10),
                _clip(row.get("subcategory", ""), 25),
                _clip(row.get("sport_type", ""), 13),
                float(row["price"]),
            )
        )

    # One worked example, so the demo can answer "but why that one?"
    if len(result.items):
        first = str(result.items.iloc[0]["product_id"])
        why = result.explanations.get(first, {})
        if why:
            log("")
            log("   why #1 ({}):".format(first))
            for reason in why.get("constraints", [])[:3]:
                log("     [Ch7] " + reason)
            for reason in why.get("content", [])[:3]:
                log("     [Ch3] " + reason)


# ----------------------------------------------------------------------
# Divergence check
# ----------------------------------------------------------------------
def check_divergence(
    results: Mapping[str, Recommendation], top_n: int
) -> tuple[bool, list[str]]:
    """Do the three profiles actually return different products?

    A demo whose three personas return overlapping lists proves nothing, so
    this is asserted rather than hoped for.  Checked pairwise on the top-N
    product ids, and reported as an overlap count so a partial collision is
    visible rather than being rounded to pass or fail.
    """
    keys = list(results)
    top: dict[str, list[str]] = {
        key: [str(pid) for pid in results[key].items["product_id"].head(top_n)]
        for key in keys
    }

    lines: list[str] = []
    all_distinct = True
    for index, left in enumerate(keys):
        for right in keys[index + 1 :]:
            shared = set(top[left]) & set(top[right])
            ok = not shared
            all_distinct &= ok
            lines.append(
                "  {:<38}{:>3} shared of top {}   {}".format(
                    "{} vs {}".format(left, right),
                    len(shared),
                    top_n,
                    "OK" if ok else "OVERLAP: " + ", ".join(sorted(shared)),
                )
            )

    union = {pid for ids in top.values() for pid in ids}
    lines.append("")
    lines.append(
        "  {} distinct products across {} profiles x top {} = {} slots".format(
            len(union), len(keys), top_n, len(keys) * top_n
        )
    )
    return all_distinct, lines


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/seed_demo.py",
        description="Three contrasting demo profiles, their recommendations, "
        "their relaxation logs and their scoring modes.",
    )
    parser.add_argument("--top-n", type=int, default=5, help="how many per profile")
    parser.add_argument(
        "--json", action="store_true",
        help="print the three profiles as POST /api/recommend bodies and exit",
    )
    parser.add_argument(
        "--rebuild-cache", action="store_true",
        help="refit TF-IDF instead of loading data/cache/tfidf_model.joblib",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.json:
        payload = {
            demo.key: {**demo.profile, "top_n": args.top_n}
            for demo in DEMO_PROFILES
        }
        print(json.dumps(payload, indent=2))
        return 0

    section("FITMATCH DEMO -- THREE PROFILES, THREE ANSWERS")
    log(
        "Each profile goes through the same two steps: the Chapter 7 constraint\n"
        "filter decides what is admissible, then the Chapter 3 TF-IDF ranker\n"
        "decides the order. Nothing below is random -- the same profile gives\n"
        "the same list on every machine, every run."
    )

    model = FitMatchRecommender.load(rebuild_cache=args.rebuild_cache)
    results: dict[str, Recommendation] = {}

    for index, demo in enumerate(DEMO_PROFILES, start=1):
        section("{}. {}".format(index, demo.name.upper()))
        describe_profile(demo)
        result = model.recommend(demo.profile, top_n=max(args.top_n, 5))
        results[demo.key] = result
        describe_constraints(result)
        describe_results(result, args.top_n)

    section("DO THE THREE ACTUALLY DIVERGE?")
    distinct, lines = check_divergence(results, args.top_n)
    for line in lines:
        log(line)
    log("")
    if distinct:
        log(
            "PASS -- no product appears in more than one profile's top {}. The\n"
            "three personas are genuinely being served different catalogues, not\n"
            "the same list in a different order.".format(args.top_n)
        )
        return 0

    log(
        "FAIL -- at least two profiles share a recommendation in their top {}.\n"
        "That is worth knowing before the demo rather than during it: either the\n"
        "personas are too similar, or the constraint filter is not separating\n"
        "them as intended.".format(args.top_n)
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
