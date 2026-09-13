"""Offline evaluation of the FitMatch cascade against the held-out interaction log.

Run with::

    python -m src.evaluation.evaluate
    python -m src.evaluation.evaluate --sample-users 300   # quick pass
    python -m src.evaluation.evaluate --no-figures

Writes ``reports/evaluation_results.csv``, ``reports/evaluation_per_user.csv``
and four charts under ``reports/figures/``.

What is being measured, and what it is worth
--------------------------------------------
Read ``reports/circularity_note.md`` first.  ``interactions.csv`` is synthetic:
``src/data/build_interactions.py`` scored every (user, product) pair on a
hand-written fit function and sampled from it.  The Chapter 7 filter reads all
ten of the fields that generator used, and the Chapter 3 ranker reads five of
its eight user fields.  So precision and recall here largely measure how far
two hand-written rule sets agree with each other.  They are reported because
the assignment asks for them and because the *relative* ordering of the five
configurations is still informative; they are not evidence that the
recommender would work on real traffic.

Two structural facts bound every number below, both measured rather than
assumed:

* **The recall ceiling.**  The four hard constraints are never relaxed, so they
  exclude items the user demonstrably interacted with.  Only 57.2% of all
  interactions and 60.3% of all ratings survive them (``feasibility_report.md``
  section 3).  On this evaluation's own held-out set -- latest 20% per user,
  relevance = rating >= 4 -- the measured figure is around 0.61.  Recall@K is
  therefore reported three ways: raw, divided by that ceiling, and computed
  only over the relevant items that are actually reachable.
* **Popularity is a strong baseline here, and that is bad news, not good.**
  Spearman between ``sales_rank`` and realised interaction count is -0.557.
  ``sales_rank`` is a *rank*, where 1 is the best seller, so a negative
  correlation means the best-selling products drew the most interactions --
  popularity is strongly and positively predictive of this log.  It has to be:
  ``build_interactions.py`` put a Zipf popularity prior on ``sales_rank`` as
  its single largest term: the 100 best-selling products average 241
  interactions each, the 100 worst-selling 2.2.  The popularity row therefore
  beats the cascade on every
  accuracy metric below, at a catalogue coverage of about 0.1%.  Neither
  recommender technique reads ``sales_rank`` at all, by design and for the
  reason in ``circularity_note.md``: ranking by popularity is what produced
  the ground truth, so scoring well on it is a measure of agreement with the
  generator rather than of recommendation quality.

Methodology
-----------
*Split.*  Leave-latest-out per user: interactions sorted by timestamp, the most
recent 20% held out, the rest kept for training.  Users with a single
interaction are skipped -- they cannot be split.  The split is by *time*, not
at random, so nothing from a user's future is used to rank their present.

*Relevance.*  A held-out item is relevant when its ``rating`` is 4 or 5.  Only
7,939 of 100,003 interactions carry a rating at all, so most held-out items are
unlabelled and count as neither relevant nor irrelevant -- they simply do not
score.  A user with no relevant held-out item cannot be evaluated and is
excluded, which leaves roughly 1,700 of 5,000 users.

*Leakage.*  The Chapter 3 centroid mode blends in items the user rated >= 4.
The model used here is fitted on the **training** interactions only, so a
held-out rating can never enter the vector that ranks it.  Nothing else in
either technique reads the interaction log: the Chapter 7 filter never touches
it, and Chapter 3 ranks on TF-IDF over product text.  A user's training rows
therefore cannot leak into the ranking of their held-out rows.

*Why training items are not excluded.*  The usual convention is to drop items
the user has already seen from the candidate pool.  It is wrong for this log
and was measured to be wrong before being dropped.  ``build_interactions.py``
emits a **funnel** over a single product -- all 7,939 ratings arrive as
``(view, add_to_cart, purchase, rating)`` on the same ``(user, product)`` pair
-- so the earlier events land in the training half while the rating lands in
the held-out half.  Excluding training items removes 69.4% of all relevant
held-out items outright and leaves 1,176 of 1,742 evaluable users with nothing
retrievable at all, which drives every configuration to a precision of exactly
zero.  That is an artifact of the generator's event model, not a property of
any recommender, so the exclusion is off by default.  ``--exclude-seen``
restores it for comparison.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.data.common import (
    INTERACTIONS_CSV,
    PRODUCTS_CSV,
    PROJECT_ROOT,
    SEED,
    USERS_CSV,
    log,
    require,
    section,
)
from src.recommender.content_based import LIKED_RATING, ContentBasedRecommender
from src.recommender.knowledge_based import (
    HARD_CONSTRAINTS,
    ConstraintBasedRecommender,
)
from src.recommender.profiles import normalize_profile

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_CSV = REPORTS_DIR / "evaluation_results.csv"
PER_USER_CSV = REPORTS_DIR / "evaluation_per_user.csv"

#: Cut-offs every metric is reported at.
K_VALUES: tuple[int, ...] = (5, 10, 20)

#: Fraction of each user's timeline held out, most recent first.
TEST_FRACTION = 0.20

#: The ceilings quoted in feasibility_report.md section 3, over *all*
#: interactions and over the rated subset.  The evaluation measures its own,
#: over held-out rating>=4 items only; the three differ and the report says why.
PUBLISHED_INTERACTION_CEILING = 0.572
PUBLISHED_RATING_CEILING = 0.603

#: Order the five configurations are always reported in: baselines, then each
#: technique alone, then both together.
CONFIG_ORDER: tuple[str, ...] = (
    "random",
    "popularity",
    "knowledge_only",
    "content_only",
    "cascade",
)

CONFIG_LABELS: dict[str, str] = {
    "random": "(a) Random",
    "popularity": "(b) Popularity",
    "knowledge_only": "(c) Knowledge-based only (Ch7)",
    "content_only": "(d) Content-based only (Ch3)",
    "cascade": "(e) Full cascade (Ch7 -> Ch3)",
}


# ----------------------------------------------------------------------
# Split
# ----------------------------------------------------------------------
@dataclass
class Split:
    """One leave-latest-out split, plus everything derived from it."""

    train: pd.DataFrame
    test: pd.DataFrame
    #: user_id -> held-out product_ids with rating >= LIKED_RATING
    relevant: dict[str, list[str]]
    #: user_id -> product_ids seen in training, removed from every candidate pool
    seen: dict[str, set[str]]

    @property
    def evaluable_users(self) -> list[str]:
        """Users with at least one relevant held-out item, in a fixed order."""
        return sorted(self.relevant)


def leave_latest_out(
    interactions: pd.DataFrame, test_fraction: float = TEST_FRACTION
) -> Split:
    """Split each user's timeline, holding out the most recent ``test_fraction``.

    Sorted by ``(timestamp, product_id)`` rather than timestamp alone: the
    generator emits minute-resolution timestamps and ties are common, so
    without the second key the split would depend on row order in the CSV.

    Every user keeps at least one training interaction and gives up at least
    one held-out interaction; a user with exactly one interaction is dropped,
    because there is no way to do both.
    """
    frame = interactions.sort_values(
        ["user_id", "timestamp", "product_id"], kind="stable"
    ).reset_index(drop=True)

    grouped = frame.groupby("user_id", sort=False)
    position = grouped.cumcount().to_numpy()
    size = grouped["product_id"].transform("size").to_numpy()

    n_test = np.rint(size * test_fraction).astype(int)
    n_test = np.clip(n_test, 1, np.maximum(size - 1, 0))
    is_test = position >= (size - n_test)
    # size == 1 leaves nothing to hold out; drop those users from both halves.
    splittable = size > 1
    is_test &= splittable

    test = frame.loc[is_test]
    train = frame.loc[~is_test & splittable]

    liked = test[pd.to_numeric(test["rating"], errors="coerce") >= LIKED_RATING]
    relevant = {
        str(user): sorted(set(group["product_id"].astype(str)))
        for user, group in liked.groupby("user_id", sort=False)
    }
    seen = {
        str(user): set(group["product_id"].astype(str))
        for user, group in train.groupby("user_id", sort=False)
    }
    return Split(train=train, test=test, relevant=relevant, seen=seen)


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def precision_at_k(recommended: Sequence[str], relevant: set[str], k: int) -> float:
    """Share of the top-K that is relevant.

    Divided by ``k``, not by the length of the list: a configuration that
    returns eight items where twenty were asked for is being penalised for the
    twelve it could not fill, which is the honest reading of "precision at 20".
    """
    if k <= 0:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / float(k)


def recall_at_k(recommended: Sequence[str], relevant: set[str], k: int) -> float:
    """Share of the user's relevant held-out items that appear in the top-K."""
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / float(len(relevant))


def ndcg_at_k(recommended: Sequence[str], relevant: set[str], k: int) -> float:
    """Normalised discounted cumulative gain, binary gain.

    Gain is 1 for a relevant item and 0 otherwise rather than being graded by
    star rating.  Relevance here is already a threshold (``rating >= 4``), and
    grading 5 above 4 would make NDCG answer a different question from the
    precision and recall reported beside it.
    """
    if not relevant:
        return 0.0
    gains = [1.0 if item in relevant else 0.0 for item in recommended[:k]]
    dcg = sum(gain / np.log2(index + 2) for index, gain in enumerate(gains))
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(index + 2) for index in range(ideal_hits))
    return float(dcg / idcg) if idcg else 0.0


def average_precision_at_k(
    recommended: Sequence[str], relevant: set[str], k: int
) -> float:
    """Average precision over the top-K, divided by the reachable hit count.

    The denominator is ``min(|relevant|, k)``, so a user with three relevant
    items can still score 1.0 at K=5 by putting all three first -- rather than
    being capped at 0.6 for the arithmetic reason that K exceeds their history.
    """
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for index, item in enumerate(recommended[:k]):
        if item in relevant:
            hits += 1
            total += hits / float(index + 1)
    denominator = min(len(relevant), k)
    return total / denominator if denominator else 0.0


# ----------------------------------------------------------------------
# The five configurations
# ----------------------------------------------------------------------
class Configurations:
    """The five ranked lists, built over one catalogue and one fitted model.

    Each method returns at most ``k`` product ids, best first, with the user's
    training items already removed.  They share the exclusion and the cut-off
    so that the only thing differing between configurations is the ranking
    principle itself.
    """

    def __init__(
        self,
        products: pd.DataFrame,
        knowledge: ConstraintBasedRecommender,
        content: ContentBasedRecommender,
        rng: np.random.Generator,
    ) -> None:
        self.products = products
        self.knowledge = knowledge
        self.content = content
        self.rng = rng

        self.product_ids = products["product_id"].astype(str).to_numpy()
        # (b) popularity: sales_rank 1 is the best seller.  Precomputed once,
        # so the baseline costs nothing per user.
        self.by_popularity = (
            products.sort_values(["sales_rank", "product_id"], kind="stable")[
                "product_id"
            ]
            .astype(str)
            .to_numpy()
        )

    # -- (a) -----------------------------------------------------------
    def random(
        self, profile: Mapping[str, Any], exclude: set[str], k: int
    ) -> list[str]:
        """Uniform sample of the catalogue. The floor every other row is read against."""
        pool = self.product_ids[~np.isin(self.product_ids, list(exclude))]
        if pool.size == 0:
            return []
        size = min(k, pool.size)
        return [str(item) for item in self.rng.choice(pool, size=size, replace=False)]

    # -- (b) -----------------------------------------------------------
    def popularity(
        self, profile: Mapping[str, Any], exclude: set[str], k: int
    ) -> list[str]:
        """The k best-selling products by ``sales_rank``, same list for everyone.

        Not personalised at all, and it wins anyway.  The generator's dominant
        term was a Zipf prior on ``sales_rank``, so the best sellers are
        genuinely over-represented in the log (Spearman -0.557 against a rank
        where 1 is best, i.e. strongly positive popularity signal; the 100
        best sellers average 241 interactions each against 2.2 for the 100
        worst).  This row is the control
        that shows how much of the ground truth is explained by popularity
        alone -- which is why the cascade, which never reads ``sales_rank``,
        loses to it on accuracy while reaching a hundred times more of the
        catalogue.
        """
        out: list[str] = []
        for product_id in self.by_popularity:
            if product_id in exclude:
                continue
            out.append(str(product_id))
            if len(out) == k:
                break
        return out

    # -- (c) -----------------------------------------------------------
    def knowledge_only(
        self, profile: Mapping[str, Any], exclude: set[str], k: int
    ) -> list[str]:
        """Chapter 7's feasible set, unranked -- the first k in catalogue order.

        This is the configuration that isolates the constraint filter.  The
        filter returns a set, not a ranking, so "first k" has to mean
        *something* arbitrary; catalogue order is the honest choice because it
        is the order ``products.csv`` happens to sit in, and it is not sorted by
        price, rating or ``sales_rank``.  Any lift over random here is the
        filter's alone -- no similarity is computed.
        """
        feasible, _ = self.knowledge.filter(profile)
        ids = feasible["product_id"].astype(str)
        return [product_id for product_id in ids if product_id not in exclude][:k]

    # -- (d) -----------------------------------------------------------
    def content_only(
        self, profile: Mapping[str, Any], exclude: set[str], k: int
    ) -> list[str]:
        """Chapter 3 over the **whole catalogue**, with no constraint filter.

        Free to recommend an out-of-stock, over-budget item in the wrong
        gender, which is exactly the point of running it: the gap between this
        row and the cascade is what the constraint filter is worth, and its
        recall is not bounded by the hard-filter ceiling.
        """
        ranked = self.content.rank(profile, None, top_n=k + len(exclude))
        ids = ranked["product_id"].astype(str)
        return [product_id for product_id in ids if product_id not in exclude][:k]

    # -- (e) -----------------------------------------------------------
    def cascade(
        self, profile: Mapping[str, Any], exclude: set[str], k: int
    ) -> list[str]:
        """The shipped system: Chapter 7 filters, then Chapter 3 ranks."""
        feasible, _ = self.knowledge.filter(profile)
        if feasible.empty:
            return []
        ranked = self.content.rank(profile, feasible, top_n=k + len(exclude))
        ids = ranked["product_id"].astype(str)
        return [product_id for product_id in ids if product_id not in exclude][:k]

    def runner(
        self, name: str
    ) -> Callable[[Mapping[str, Any], set[str], int], list[str]]:
        return getattr(self, name)


# ----------------------------------------------------------------------
# Reachability -- the recall ceiling, measured rather than quoted
# ----------------------------------------------------------------------
def reachable_relevant(
    knowledge: ConstraintBasedRecommender,
    profile: Mapping[str, Any],
    relevant: Iterable[str],
    row_of: Mapping[str, int],
) -> set[str]:
    """Which of a user's relevant held-out items survive the four hard constraints.

    These are the only items the cascade could possibly retrieve.  An item
    outside this set is one the user really did interact with and rate highly,
    that the recommender refuses to show because it is out of stock, over
    budget, for the wrong gender, or footwear for another sport.  That refusal
    is the specified behaviour; the cost of it is this function.
    """
    masks = knowledge._hard_masks(dict(profile))
    mask = np.ones(len(knowledge.products), dtype=bool)
    for name in HARD_CONSTRAINTS:
        mask &= masks[name]
    return {
        str(product_id)
        for product_id in relevant
        if product_id in row_of and bool(mask[row_of[product_id]])
    }


# ----------------------------------------------------------------------
# The evaluation loop
# ----------------------------------------------------------------------
def evaluate(
    products: pd.DataFrame,
    users: pd.DataFrame,
    interactions: pd.DataFrame,
    k_values: Sequence[int] = K_VALUES,
    sample_users: int | None = None,
    progress_every: int = 250,
    exclude_seen: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Score all five configurations.

    Returns ``(results, per_user, meta)``: the aggregate table, one row per
    (user, configuration, K) with its hit counts, and the split statistics the
    write-up quotes.

    ``exclude_seen`` removes each user's training items from every candidate
    pool.  Off by default -- see the module docstring for the measurement that
    settled it.
    """
    split = leave_latest_out(interactions)
    evaluable = split.evaluable_users
    if sample_users is not None and sample_users < len(evaluable):
        picked = np.random.default_rng(SEED).choice(
            len(evaluable), size=sample_users, replace=False
        )
        evaluable = [evaluable[index] for index in sorted(picked)]

    # The model sees training interactions only -- see the module docstring.
    knowledge = ConstraintBasedRecommender(products)
    content = ContentBasedRecommender(products, split.train).fit()
    configurations = Configurations(
        products, knowledge, content, np.random.default_rng(SEED)
    )

    profiles = {
        str(row["user_id"]): normalize_profile(row)
        for row in users.to_dict("records")
    }
    row_of = {str(pid): index for index, pid in enumerate(products["product_id"])}
    max_k = max(k_values)

    records: list[dict[str, Any]] = []
    recommended_sets: dict[tuple[str, int], set[str]] = {
        (name, k): set() for name in CONFIG_ORDER for k in k_values
    }

    log(
        "Scoring {:,} evaluable users x {} configurations...".format(
            len(evaluable), len(CONFIG_ORDER)
        )
    )

    for index, user_id in enumerate(evaluable, start=1):
        profile = profiles.get(user_id)
        if profile is None:
            continue
        relevant = set(split.relevant[user_id])
        exclude = split.seen.get(user_id, set()) if exclude_seen else set()
        reachable = reachable_relevant(knowledge, profile, relevant, row_of)
        n_liked_train = int(content.user_vector(profile).n_liked)

        for name in CONFIG_ORDER:
            ranked = configurations.runner(name)(profile, exclude, max_k)
            for k in k_values:
                recommended_sets[(name, k)].update(ranked[:k])
                records.append(
                    {
                        "user_id": user_id,
                        "config": name,
                        "k": k,
                        "n_relevant": len(relevant),
                        "n_reachable": len(reachable),
                        "n_returned": len(ranked[:k]),
                        "qualifying_ratings": n_liked_train,
                        "precision": precision_at_k(ranked, relevant, k),
                        "recall": recall_at_k(ranked, relevant, k),
                        "recall_reachable": (
                            recall_at_k(ranked, reachable, k) if reachable else np.nan
                        ),
                        "ndcg": ndcg_at_k(ranked, relevant, k),
                        "ap": average_precision_at_k(ranked, relevant, k),
                    }
                )

        if progress_every and index % progress_every == 0:
            log("  {:>5,} / {:,} users".format(index, len(evaluable)))

    per_user = pd.DataFrame.from_records(records)
    if per_user.empty:
        raise SystemExit("no evaluable users -- is interactions.csv populated?")

    unique_users = per_user.drop_duplicates("user_id")
    ceiling = float(
        (unique_users["n_reachable"] / unique_users["n_relevant"]).mean()
    )

    # How much the discarded "exclude items already seen" convention would have
    # cost, quoted in the write-up as the reason it is not used.
    relevant_in_train = sum(
        1
        for user_id in evaluable
        for product_id in split.relevant[user_id]
        if product_id in split.seen.get(user_id, set())
    )
    total_relevant = sum(len(split.relevant[user_id]) for user_id in evaluable)

    aggregate = (
        per_user.groupby(["config", "k"], sort=False)
        .agg(
            users=("user_id", "nunique"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            recall_reachable=("recall_reachable", "mean"),
            ndcg=("ndcg", "mean"),
            map=("ap", "mean"),
        )
        .reset_index()
    )
    aggregate["recall_normalised"] = aggregate["recall"] / ceiling
    aggregate["coverage"] = [
        len(recommended_sets[(row.config, row.k)]) / len(products)
        for row in aggregate.itertuples()
    ]
    aggregate["label"] = aggregate["config"].map(CONFIG_LABELS)

    rank_of = {name: index for index, name in enumerate(CONFIG_ORDER)}
    aggregate["_order"] = aggregate["config"].map(rank_of)
    aggregate = (
        aggregate.sort_values(["k", "_order"], kind="stable")
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    meta = {
        "n_train": int(len(split.train)),
        "n_test": int(len(split.test)),
        "n_evaluable_users": int(len(evaluable)),
        "n_relevant_items": int(unique_users["n_relevant"].sum()),
        "n_reachable_items": int(unique_users["n_reachable"].sum()),
        "recall_ceiling": ceiling,
        "users_with_zero_reachable": int((unique_users["n_reachable"] == 0).sum()),
        "cold_start_share_evaluable": float(
            (unique_users["qualifying_ratings"] < 3).mean()
        ),
        "exclude_seen": bool(exclude_seen),
        "relevant_also_in_train": int(relevant_in_train),
        "relevant_also_in_train_share": (
            relevant_in_train / total_relevant if total_relevant else 0.0
        ),
        "k_values": list(k_values),
    }
    return aggregate, per_user, meta


# ----------------------------------------------------------------------
# Presentation
# ----------------------------------------------------------------------
def print_table(results: pd.DataFrame, meta: Mapping[str, Any]) -> None:
    """The results table, one block per K."""
    for k in meta["k_values"]:
        block = results[results["k"] == k]
        section("METRICS AT K = {}".format(k))
        header = "{:<32}{:>10}{:>10}{:>12}{:>10}{:>10}{:>10}".format(
            "configuration", "P@K", "R@K", "R@K/ceil", "NDCG@K", "MAP@K", "Cov@K"
        )
        log(header)
        log("-" * len(header))
        for row in block.itertuples():
            log(
                "{:<32}{:>10.4f}{:>10.4f}{:>12.4f}{:>10.4f}{:>10.4f}{:>9.1f}%".format(
                    row.label,
                    row.precision,
                    row.recall,
                    row.recall_normalised,
                    row.ndcg,
                    row.map,
                    100.0 * row.coverage,
                )
            )

    section("RECALL AGAINST REACHABLE ITEMS ONLY")
    log(
        "Recall computed over the relevant held-out items that actually survive\n"
        "each user's hard constraints. Users with none are excluded, so the\n"
        "hard-filter ceiling is divided out per user rather than globally."
    )
    log("")
    header = "{:<32}" + "{:>12}" * len(meta["k_values"])
    log(header.format("configuration", *["K = {}".format(k) for k in meta["k_values"]]))
    log("-" * (32 + 12 * len(meta["k_values"])))
    for name in CONFIG_ORDER:
        values = [
            float(
                results.loc[
                    (results["config"] == name) & (results["k"] == k),
                    "recall_reachable",
                ].iloc[0]
            )
            for k in meta["k_values"]
        ]
        log(
            ("{:<32}" + "{:>12.4f}" * len(values)).format(CONFIG_LABELS[name], *values)
        )


def print_context(meta: Mapping[str, Any]) -> None:
    section("SPLIT AND CEILING")
    log("training interactions          {:>10,}".format(meta["n_train"]))
    log("held-out interactions          {:>10,}".format(meta["n_test"]))
    log("evaluable users                {:>10,}".format(meta["n_evaluable_users"]))
    log("relevant held-out items        {:>10,}".format(meta["n_relevant_items"]))
    log("  of which reachable           {:>10,}".format(meta["n_reachable_items"]))
    log("")
    log(
        "measured recall ceiling        {:>10.3f}   "
        "(mean per-user share of relevant items the hard filter admits)".format(
            meta["recall_ceiling"]
        )
    )
    log(
        "published ceiling, all events  {:>10.3f}   "
        "(feasibility_report.md 3, over all 100,003 interactions)".format(
            PUBLISHED_INTERACTION_CEILING
        )
    )
    log(
        "published ceiling, ratings     {:>10.3f}   "
        "(same source, restricted to the 7,939 rated interactions)".format(
            PUBLISHED_RATING_CEILING
        )
    )
    log("")
    log(
        "users whose every relevant item is blocked by a hard constraint: "
        "{:,} of {:,} ({:.1f}%).".format(
            meta["users_with_zero_reachable"],
            meta["n_evaluable_users"],
            100.0 * meta["users_with_zero_reachable"] / meta["n_evaluable_users"],
        )
    )
    log(
        "Their maximum achievable recall is 0.000 for every configuration that\n"
        "respects the hard constraints, however good the ranking is."
    )
    log("")
    log(
        "cold-start share among evaluable users: {:.1%}, against 85.2% over the\n"
        "full 5,000-user population (feasibility_report.md 4). It is HIGHER\n"
        "here, not lower, and the split is why: qualifying ratings are counted\n"
        "from the training half only, and leave-latest-out moves each user's\n"
        "most recent ratings into the held-out half. An evaluable user is one\n"
        "who rated something recently, which is exactly the rating the split\n"
        "takes away. The profile-only path therefore carries even more of this\n"
        "evaluation than it carries in production.".format(
            meta["cold_start_share_evaluable"]
        )
    )
    log("")
    log(
        "{:,} of {:,} relevant held-out items ({:.1%}) also appear in their own\n"
        "user's training rows: the generator emits view -> add_to_cart ->\n"
        "purchase -> rating on one product, so the funnel straddles the split.\n"
        "Training items are therefore NOT excluded from the candidate pools\n"
        "({}). Excluding them would delete most of the ground truth.".format(
            meta["relevant_also_in_train"],
            meta["n_relevant_items"],
            meta["relevant_also_in_train_share"],
            "--exclude-seen was passed, so they ARE excluded here"
            if meta["exclude_seen"]
            else "pass --exclude-seen to see what that convention costs",
        )
    )


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
#: Colourblind-safe, and ordered so the two baselines read as muted and the
#: cascade as the emphasis.
CONFIG_COLORS: dict[str, str] = {
    "random": "#9e9e9e",
    "popularity": "#c2a25a",
    "knowledge_only": "#4c78a8",
    "content_only": "#72b7b2",
    "cascade": "#b3202c",
}


def _short_labels() -> list[str]:
    return [
        CONFIG_LABELS[name]
        .split(") ", 1)[-1]
        .replace(" (Ch7)", "")
        .replace(" (Ch3)", "")
        for name in CONFIG_ORDER
    ]


def write_figures(
    results: pd.DataFrame, meta: Mapping[str, Any], figures_dir: Path = FIGURES_DIR
) -> list[Path]:
    """Four charts. Returns the paths written."""
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
            "grid.linestyle": "-",
            "axes.axisbelow": True,
        }
    )

    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    short = _short_labels()
    colors = [CONFIG_COLORS[name] for name in CONFIG_ORDER]

    def at(name: str, k: int, column: str) -> float:
        return float(
            results.loc[
                (results["config"] == name) & (results["k"] == k), column
            ].iloc[0]
        )

    # --- 1. metric comparison -------------------------------------------
    metrics = [
        ("precision", "Precision@10"),
        ("recall", "Recall@10"),
        ("ndcg", "NDCG@10"),
        ("map", "MAP@10"),
    ]
    # Two rows. The popularity baseline is an order of magnitude above the
    # rest, so on a shared axis the four rows that are actually being compared
    # become invisible; the second row drops it and rescales.
    zoomed = [name for name in CONFIG_ORDER if name != "popularity"]
    zoom_short = [
        label
        for name, label in zip(CONFIG_ORDER, short)
        if name != "popularity"
    ]
    zoom_colors = [CONFIG_COLORS[name] for name in zoomed]

    figure, axes = plt.subplots(2, 4, figsize=(11, 6.6))
    for axis, (column, title) in zip(axes[0], metrics):
        values = [at(name, 10, column) for name in CONFIG_ORDER]
        axis.bar(range(len(values)), values, color=colors, width=0.7)
        axis.set_title(title, fontsize=10)
        axis.set_xticks(range(len(values)))
        axis.set_xticklabels(short, rotation=45, ha="right", fontsize=7.5)
        axis.set_ylim(0, (max(values) or 1.0) * 1.25)
        for index, value in enumerate(values):
            axis.text(
                index, value, "{:.3f}".format(value), ha="center",
                va="bottom", fontsize=7,
            )
    for axis, (column, title) in zip(axes[1], metrics):
        values = [at(name, 10, column) for name in zoomed]
        axis.bar(range(len(values)), values, color=zoom_colors, width=0.7)
        axis.set_title(title + "  (popularity dropped)", fontsize=9)
        axis.set_xticks(range(len(values)))
        axis.set_xticklabels(zoom_short, rotation=45, ha="right", fontsize=7.5)
        axis.set_ylim(0, (max(values) or 1.0) * 1.3)
        for index, value in enumerate(values):
            axis.text(
                index, value, "{:.4f}".format(value), ha="center",
                va="bottom", fontsize=7,
            )
    figure.suptitle(
        "FitMatch offline metrics at K = 10  ({:,} evaluable users)\n"
        "Top: all five. Bottom: rescaled without the popularity baseline, "
        "which is the ground truth's own generator term.".format(
            meta["n_evaluable_users"]
        ),
        fontsize=10.5,
    )
    figure.tight_layout()
    path = figures_dir / "metric_comparison.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    written.append(path)

    # --- 2. precision vs K ----------------------------------------------
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    for name in CONFIG_ORDER:
        values = [at(name, k, "precision") for k in meta["k_values"]]
        axis.plot(
            meta["k_values"], values, marker="o", color=CONFIG_COLORS[name],
            label=CONFIG_LABELS[name],
            linewidth=2.4 if name == "cascade" else 1.5,
        )
    axis.set_xlabel("K")
    axis.set_ylabel("Precision@K")
    axis.set_title(
        "Precision falls with K: most users have exactly one relevant item",
        fontsize=10,
    )
    axis.set_xticks(list(meta["k_values"]))
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    path = figures_dir / "precision_at_k.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    written.append(path)

    # --- 3. coverage -----------------------------------------------------
    figure, axis = plt.subplots(figsize=(6.6, 3.8))
    values = [100.0 * at(name, 10, "coverage") for name in CONFIG_ORDER]
    bars = axis.barh(range(len(values)), values, color=colors, height=0.62)
    axis.set_yticks(range(len(values)))
    axis.set_yticklabels(short, fontsize=8.5)
    axis.invert_yaxis()
    axis.set_xlabel("Share of the 10,000-product catalogue reached at K = 10 (%)")
    axis.set_title("Catalogue coverage", fontsize=10)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_width() + max(values) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            "{:.1f}%".format(value), va="center", fontsize=8,
        )
    axis.set_xlim(0, max(values) * 1.18)
    figure.tight_layout()
    path = figures_dir / "coverage.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    written.append(path)

    # --- 4. recall against the ceiling -----------------------------------
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    values = [at(name, 10, "recall") for name in CONFIG_ORDER]
    axis.bar(range(len(values)), values, color=colors, width=0.66)
    for index, value in enumerate(values):
        axis.text(
            index, value, "{:.3f}".format(value), ha="center",
            va="bottom", fontsize=8,
        )

    ceiling = meta["recall_ceiling"]
    axis.axhline(ceiling, color="#b3202c", linestyle="--", linewidth=1.6)
    axis.text(
        len(values) - 0.4, ceiling, "measured ceiling {:.3f}".format(ceiling),
        va="bottom", ha="right", fontsize=8, color="#b3202c",
    )
    axis.axhline(
        PUBLISHED_INTERACTION_CEILING, color="#555555", linestyle=":", linewidth=1.4
    )
    # Sits below its own line: the two ceilings are only 0.04 apart and the
    # labels collide if both hang above.
    axis.text(
        len(values) - 0.4, PUBLISHED_INTERACTION_CEILING,
        "all-interaction ceiling {:.3f}".format(PUBLISHED_INTERACTION_CEILING),
        va="top", ha="right", fontsize=8, color="#555555",
    )
    axis.axhline(1.0, color="#222222", linewidth=1.0)
    axis.text(
        len(values) - 0.4, 1.0, "theoretical maximum 1.000",
        va="bottom", ha="right", fontsize=8, color="#222222",
    )

    axis.set_xticks(range(len(values)))
    axis.set_xticklabels(short, rotation=20, ha="right", fontsize=8.5)
    axis.set_ylabel("Recall@10")
    axis.set_ylim(0, 1.14)
    axis.set_title(
        "Recall@10 cannot reach 1.0: hard constraints exclude {:.0f}% of "
        "relevant items".format(100.0 * (1.0 - ceiling)),
        fontsize=10,
    )
    figure.tight_layout()
    path = figures_dir / "recall_ceiling.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    written.append(path)

    return written


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluation.evaluate",
        description="Offline evaluation of the FitMatch cascade: five "
        "configurations at K = 5/10/20 against a leave-latest-out "
        "split of the interaction log.",
    )
    parser.add_argument(
        "--sample-users", type=int, default=None,
        help="evaluate this many users instead of all of them (quick pass)",
    )
    parser.add_argument(
        "--no-figures", action="store_true", help="skip the matplotlib charts"
    )
    parser.add_argument(
        "--exclude-seen", action="store_true",
        help="drop each user's training items from the candidate pools. Off by "
             "default because this log's view/cart/purchase/rating funnel puts "
             "69%% of the relevant held-out items into training as well.",
    )
    parser.add_argument(
        "--k", type=int, nargs="+", default=list(K_VALUES),
        help="cut-offs to report at (default: 5 10 20)",
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

    section("FITMATCH OFFLINE EVALUATION")
    log(
        "seed {} -- deterministic. The recommender draws no random numbers; the".format(
            SEED
        )
    )
    log("random baseline is the only thing here that touches the generator.")
    log("")
    log("Read reports/circularity_note.md before reading these numbers. The")
    log("interaction log is synthetic and shares its scoring fields with the")
    log("recommender, so offline accuracy measures self-consistency as much as")
    log("recommendation quality.")

    products = pd.read_csv(PRODUCTS_CSV)
    users = pd.read_csv(USERS_CSV)
    interactions = pd.read_csv(INTERACTIONS_CSV)

    results, per_user, meta = evaluate(
        products,
        users,
        interactions,
        k_values=tuple(args.k),
        sample_users=args.sample_users,
        exclude_seen=args.exclude_seen,
    )

    print_context(meta)
    print_table(results, meta)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    columns = [
        "config", "label", "k", "users", "precision", "recall",
        "recall_normalised", "recall_reachable", "ndcg", "map", "coverage",
    ]
    results[columns].to_csv(RESULTS_CSV, index=False)
    per_user.to_csv(PER_USER_CSV, index=False)

    section("WRITTEN")
    log("  {}".format(RESULTS_CSV.relative_to(PROJECT_ROOT)))
    log("  {}".format(PER_USER_CSV.relative_to(PROJECT_ROOT)))

    if not args.no_figures:
        for path in write_figures(results, meta):
            log("  {}".format(path.relative_to(PROJECT_ROOT)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
