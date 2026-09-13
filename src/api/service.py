"""The layer between HTTP and the session-2 cascade.

Everything here is translation.  The service holds one
:class:`~src.recommender.pipeline.FitMatchRecommender`, loaded once at
startup, and turns its ``Recommendation`` / ``ConstraintReport`` objects into
the JSON shapes in :mod:`src.api.schemas`.  No constraint is evaluated here,
no score is computed here, and no number in a response is derived from
anything but the recommender's own report -- the one thing this module adds is
English, because "colour was relaxed" has to reach the user as a sentence.

The prose matters more than it looks.  Session 2 measured the median user's
feasible set at 0 with every soft constraint applied, and colour relaxed for
98.7% of users.  A UI that quietly returns ten products the user did not ask
for is lying by omission; the ``message`` strings built in
:func:`_describe_relaxation` are what stops that.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from src.api.vocabulary import Vocabulary, get_vocabulary
from src.recommender.content_based import LIKED_RATING
from src.recommender.knowledge_based import (
    DEFAULT_MIN_RESULTS,
    HARD_CONSTRAINTS,
    ConstraintReport,
)
from src.recommender.pipeline import (
    CACHE_PATH,
    RESULT_COLUMNS,
    FitMatchRecommender,
    Recommendation,
    _canonical_user_id,
)

LOGGER = logging.getLogger("fitmatch.api.service")

# Session-2 findings quoted back to the user in the scoring explanation, so a
# cold-start label reads as "this is normal" rather than "something failed".
# Source: reports/feasibility_report.md sections 1 and 4.
COLD_START_SHARE = "85.2%"
COLOR_RELAXED_SHARE = "98.7%"

# Human names for the machine names in RELAXATION_ORDER.
CONSTRAINT_LABELS: dict[str, str] = {
    "color_preference": "colour",
    "material_preference": "material",
    "arch_support": "arch support",
    "seasonality": "seasonal suitability",
    "preferred_brands": "preferred brands",
    "size": "size",
}

# What each hard constraint means, for the zero-result explanation.
HARD_LABELS: dict[str, str] = {
    "budget": "your budget",
    "gender": "the gender the product is made for",
    "stock": "stock availability",
    "sport": "the sport the footwear is built for",
}

HARD_FIXES: dict[str, str] = {
    "budget": "widen the budget range",
    "gender": "there is little to do about this one -- it is the catalogue's mix",
    "stock": "nothing to change; 18.4% of the catalogue is not in stock",
    "sport": "footwear is filtered on sport, so try a different sport or a wider budget",
}


class ProductNotFound(LookupError):
    """No such product_id in the catalogue."""


class UserNotFound(LookupError):
    """No such user_id in users.csv."""


@dataclass
class CacheInfo:
    """Whether the joblib TF-IDF cache was there, and whether it was used."""

    path: Path
    warm: bool = False
    hit_on_startup: bool = False
    size_bytes: Optional[int] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "warm": self.warm,
            "hit_on_startup": self.hit_on_startup,
            "path": str(self.path),
            "size_bytes": self.size_bytes,
        }


@dataclass
class RecommendOutcome:
    """One recommendation call: the response body plus what to log about it."""

    body: dict[str, Any]
    log_fields: dict[str, Any] = field(default_factory=dict)


class RecommenderService:
    """Holds the loaded cascade and the catalogue lookups the API needs.

    Built once by the lifespan handler.  Everything on it is read-only after
    construction, which is what makes it safe to share across requests without
    a lock: neither technique mutates state when it scores.
    """

    def __init__(
        self,
        model: FitMatchRecommender,
        vocabulary: Vocabulary,
        cache: CacheInfo,
        cold_start_ms: float,
        min_results: int = DEFAULT_MIN_RESULTS,
    ) -> None:
        self.model = model
        self.vocabulary = vocabulary
        self.cache = cache
        self.cold_start_ms = cold_start_ms
        self.min_results = int(min_results)
        self.started_at = time.time()

        self.products = model.products
        # Keyed upper-case so a lower-cased id in a URL still resolves; the
        # catalogue spells them NK.../AD... but a hand-typed URL may not.
        self._product_rows: dict[str, dict[str, Any]] = {
            str(row["product_id"]).upper(): row
            for row in self.products.to_dict("records")
        }
        self._users = model.users
        # Computed once: the catalogue is read-only after construction, so the
        # browse order is too.  Held as positional indices rather than as a
        # column because ``model.products`` is the frame the recommender scores
        # against and nothing here may reshape it.
        self._browse_order = _interleaved_brand_order(self.products)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def load(
        cls,
        min_results: int = DEFAULT_MIN_RESULTS,
        cache_path: Path = CACHE_PATH,
        rebuild_cache: bool = False,
    ) -> "RecommenderService":
        """Load the cascade, timing the cold start and noting the cache.

        Whether the joblib cache was *used* is not something
        ``FitMatchRecommender.load`` reports, and session 3 is not allowed to
        change it to.  The file's mtime answers the question just as well: a
        rebuilt cache is a rewritten file.
        """
        cache_path = Path(cache_path)
        before = cache_path.stat().st_mtime_ns if cache_path.exists() else None

        started = time.perf_counter()
        model = FitMatchRecommender.load(
            min_results=min_results,
            cache_path=cache_path,
            rebuild_cache=rebuild_cache,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        after = cache_path.stat() if cache_path.exists() else None
        cache = CacheInfo(
            path=cache_path,
            warm=after is not None,
            hit_on_startup=(
                before is not None and after is not None and after.st_mtime_ns == before
            ),
            size_bytes=int(after.st_size) if after is not None else None,
        )

        LOGGER.info(
            "cold start: recommender ready in %.0f ms (%s products, %s users, "
            "TF-IDF cache %s)",
            elapsed_ms,
            len(model.products),
            0 if model.users is None else len(model.users),
            "reused" if cache.hit_on_startup else "rebuilt",
            extra={
                "event": "startup",
                "cold_start_ms": round(elapsed_ms, 1),
                "cache_hit": cache.hit_on_startup,
                "cache_path": str(cache_path),
                "n_products": int(len(model.products)),
                "n_vocabulary_terms": int(len(model.content.feature_names)),
            },
        )
        return cls(
            model=model,
            vocabulary=get_vocabulary(),
            cache=cache,
            cold_start_ms=elapsed_ms,
            min_results=min_results,
        )

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------
    def recommend(
        self,
        profile: Mapping[str, Any] | None = None,
        user_id: str | None = None,
        top_n: int = 10,
        min_results: int | None = None,
    ) -> RecommendOutcome:
        """Run the cascade for one request and shape the response.

        ``user_id`` wins over ``profile``: it means "score this users.csv row",
        which is also the only path on which the content ranker can reach mode
        (b), since an anonymous web profile has no rating history to blend in.
        """
        started = time.perf_counter()
        target = self.min_results if min_results is None else int(min_results)

        if user_id is not None:
            key = _canonical_user_id(user_id)
            try:
                resolved: Any = self.model.resolve_profile(key)
            except KeyError as error:
                raise UserNotFound(str(error.args[0])) from error
        else:
            resolved = dict(profile or {})

        result = self.model.recommend(
            resolved, top_n=top_n, explain=True, min_results=target
        )
        latency_ms = (time.perf_counter() - started) * 1000.0

        scoring = self._scoring_block(result)
        constraints = self._constraints_block(result, target)
        body = {
            "user_id": result.user_id,
            "recommendations": self._recommendation_items(result),
            "constraints": constraints,
            "scoring": scoring,
            "latency_ms": round(latency_ms, 2),
        }
        log_fields = {
            "user_id": result.user_id,
            "n_results": len(body["recommendations"]),
            "candidates_after_hard": constraints["candidates_after_hard"],
            "candidates_after_soft": constraints["candidates_after_soft"],
            "final_candidate_count": constraints["final_candidate_count"],
            "relaxed": [item["constraint"] for item in constraints["relaxed"]],
            "scoring_mode": scoring["mode"],
            "qualifying_ratings": scoring["qualifying_ratings"],
            "min_results_target": target,
            "latency_ms": round(latency_ms, 2),
        }
        return RecommendOutcome(body=body, log_fields=log_fields)

    # ------------------------------------------------------------------
    # Catalogue
    # ------------------------------------------------------------------
    def product_detail(self, product_id: str) -> dict[str, Any]:
        row = self._product_rows.get(str(product_id).strip().upper())
        if row is None:
            raise ProductNotFound(
                "no product {!r} in the catalogue of {:,} products".format(
                    product_id, len(self.products)
                )
            )
        return _json_safe(row)

    def browse_products(
        self,
        sport: str | None = None,
        brand: str | None = None,
        category: str | None = None,
        price_min: float | None = None,
        price_max: float | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        """One page of the catalogue, filtered.

        Ordered by the brand interleave built in
        :func:`_interleaved_brand_order`, not by ``product_id`` and not by
        ``sales_rank``.  ``product_id`` order followed the CSV write order, and
        the 4,000 Adidas rows were appended as a block, so page 1 was twelve
        consecutive Adidas products and the catalogue read as single-brand
        until you paged deep or filtered.  ``sales_rank`` is still refused:
        neither technique reads popularity (see
        ``reports/circularity_note.md``), so the browse view must not quietly
        introduce it as a ranking.
        """
        frame = self.products
        mask = np.ones(len(frame), dtype=bool)
        if sport is not None:
            mask &= (frame["sport_type"] == sport).to_numpy()
        if brand is not None:
            mask &= (frame["brand"] == brand).to_numpy()
        if category is not None:
            mask &= (frame["category"] == category).to_numpy()
        if price_min is not None:
            mask &= (frame["price"] >= price_min).to_numpy()
        if price_max is not None:
            mask &= (frame["price"] <= price_max).to_numpy()

        # Take the precomputed order and drop the rows the filters excluded.
        # Subsetting an interleaved sequence keeps it interleaved, so a
        # filtered page is mixed for whatever brands survive the filter, and a
        # single-brand filter degrades to that brand's own product_id order.
        order = self._browse_order[mask[self._browse_order]]
        total = int(order.size)
        total_pages = max(1, math.ceil(total / page_size)) if total else 0
        start = (page - 1) * page_size
        window = frame.iloc[order[start : start + page_size]]

        columns = [c for c in RESULT_COLUMNS if c in window.columns]
        return {
            "items": [_json_safe(row) for row in window[columns].to_dict("records")],
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": total_pages,
            "has_next": start + page_size < total,
            "has_previous": page > 1 and total > 0,
            "filters": {
                "sport": sport,
                "brand": brand,
                "category": category,
                "price_min": price_min,
                "price_max": price_max,
            },
        }

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------
    def user_profile(self, user_id: str) -> dict[str, Any]:
        """A users.csv row, normalised, ready to drop into the demo form."""
        key = _canonical_user_id(user_id)
        try:
            profile = self.model.resolve_profile(key)
        except KeyError as error:
            raise UserNotFound(str(error.args[0])) from error

        qualifying = self._qualifying_ratings(profile)
        payload = _json_safe(dict(profile))
        payload["user_id"] = key
        payload["preferred_brands"] = list(profile.get("preferred_brands") or ())
        payload["qualifying_ratings"] = qualifying
        payload["expected_scoring_mode"] = (
            "profile_plus_centroid"
            if qualifying >= self.model.content.min_ratings_for_centroid
            else "profile_only"
        )
        # normalize_profile reports "no budget ceiling" as +inf, which is not
        # JSON; _json_safe turned it into None, which is what "no ceiling"
        # looks like on a form.
        return payload

    # ------------------------------------------------------------------
    # Meta and health
    # ------------------------------------------------------------------
    def meta(self) -> dict[str, Any]:
        return self.vocabulary.as_meta()

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "ready": True,
            "cache": self.cache.as_dict(),
            "cold_start_ms": round(self.cold_start_ms, 1),
            "uptime_s": round(time.time() - self.started_at, 1),
            "catalogue_size": int(len(self.products)),
            "n_users": 0 if self._users is None else int(len(self._users)),
            "n_vocabulary_terms": int(len(self.model.content.feature_names)),
            "min_results_default": self.min_results,
            "detail": None,
        }

    # ------------------------------------------------------------------
    # Response shaping
    # ------------------------------------------------------------------
    def _recommendation_items(self, result: Recommendation) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        product_columns = [
            column for column in RESULT_COLUMNS if column in result.items.columns
        ]
        for row in result.items.to_dict("records"):
            product_id = str(row["product_id"])
            why = result.explanations.get(product_id, {})
            explanations = [
                {"source": "knowledge_based", "chapter": "Ch7", "text": text}
                for text in why.get("constraints", [])
            ] + [
                {"source": "content_based", "chapter": "Ch3", "text": text}
                for text in why.get("content", [])
            ]
            items.append(
                {
                    "rank": int(row["rank"]),
                    "score": float(row["score"]),
                    "product": _json_safe(
                        {column: row[column] for column in product_columns}
                    ),
                    "explanations": explanations,
                }
            )
        return items

    def _scoring_block(self, result: Recommendation) -> dict[str, Any]:
        minimum = self.model.content.min_ratings_for_centroid
        qualifying = self._qualifying_ratings(result.profile)
        mode = result.content_mode

        if mode == "profile_plus_centroid":
            label = "Profile + your history"
            explanation = (
                "Ranked on your stated profile blended with the {} item{} you "
                "rated {} or better.".format(
                    qualifying, "" if qualifying == 1 else "s", LIKED_RATING
                )
            )
        else:
            label = "Profile only (cold start)"
            explanation = (
                "Ranked on your stated profile alone. You have {} rating{} of {} "
                "or better and {} are needed before past ratings are blended in, "
                "so this is the cold-start path -- the one {} of users in this "
                "dataset take.".format(
                    qualifying,
                    "" if qualifying == 1 else "s",
                    LIKED_RATING,
                    minimum,
                    COLD_START_SHARE,
                )
            )
        return {
            "mode": mode,
            "qualifying_ratings": qualifying,
            "min_ratings_for_centroid": minimum,
            "label": label,
            "explanation": explanation,
        }

    def _qualifying_ratings(self, profile: Mapping[str, Any]) -> int:
        """How many ratings of 4+ this profile's user has.

        An anonymous profile has none by definition, and asking costs a TF-IDF
        transform, so it is only asked for a profile that names a user.
        """
        if not profile.get("user_id"):
            return 0
        return int(self.model.content.user_vector(profile).n_liked)

    def _constraints_block(
        self, result: Recommendation, target: int
    ) -> dict[str, Any]:
        report: ConstraintReport = result.relaxation_log
        steps = report.as_dicts()

        after_soft = next(
            (
                step["n_after"]
                for step in steps
                if step["constraint"] == "SOFT (all applied)"
            ),
            report.n_after_hard,
        )
        relaxed_steps = [step for step in steps if step["action"] == "relaxed"]
        relaxed = [
            _describe_relaxation(step, result.profile, target, order)
            for order, step in enumerate(relaxed_steps)
        ]
        exhausted = any(step["action"] == "exhausted" for step in steps)

        return {
            "hard_applied": list(HARD_CONSTRAINTS),
            "hard_excluded": {
                name: int(count) for name, count in report.hard_excluded.items()
            },
            "candidates_after_hard": int(report.n_after_hard),
            "candidates_after_soft": int(after_soft),
            "relaxed": relaxed,
            "still_applied": list(report.applied),
            "final_candidate_count": int(report.n_feasible),
            "min_results_target": int(target),
            "exhausted": exhausted,
            "catalogue_size": int(report.n_catalogue),
            "summary": _summarise(report, result, target, after_soft),
            "log": steps,
        }


# ----------------------------------------------------------------------
# English
# ----------------------------------------------------------------------
def _interleaved_brand_order(frame: pd.DataFrame) -> np.ndarray:
    """Positional indices of ``frame`` in the default browse order.

    The problem it solves: ``products.csv`` is written brand by brand, so any
    ordering that follows the file — ``product_id`` included, because the ids
    are brand-prefixed and near-sequential — puts 4,000 Adidas rows in one
    unbroken run.  Page 1 was twelve Adidas products in a row.

    The fix is a stratified interleave, not a shuffle.  Each brand's rows are
    put in ``product_id`` order and given a position in ``[0, 1)``::

        key = (rank_within_brand + 0.5) / products_of_that_brand

    which spreads every brand evenly across the whole range regardless of how
    many rows it has.  Sorting on that key therefore reproduces the catalogue's
    own 52% / 40% / 8% mix in *any* prefix of the result, including the first
    page: 12 rows come out roughly 6 Nike, 5 Adidas, 1 Jordan.

    Three properties this has that a random sort would not:

    * **Deterministic.**  No RNG and no seed to remember.  The key is a pure
      function of the catalogue, so the same page is the same page on every
      process, machine and run — which is what makes a screenshot of page 3
      mean anything.
    * **Total.**  Ties are broken by brand then ``product_id``, so the order is
      a permutation of the catalogue: paging through it can neither repeat a
      product nor skip one.
    * **Not a ranking.**  Position says nothing about quality or popularity.
      It is the same refusal to smuggle in ``sales_rank`` that
      ``browse_products`` documents, held one level down.
    """
    keys = pd.DataFrame(
        {
            "position": np.arange(len(frame)),
            "brand": frame["brand"].astype(str).to_numpy(),
            "product_id": frame["product_id"].astype(str).to_numpy(),
        }
    ).sort_values(["brand", "product_id"], kind="stable")

    grouped = keys.groupby("brand", sort=False)["product_id"]
    # +0.5 centres each brand's rows in its own slice, so the smallest brand
    # lands mid-interval rather than always first or always last.
    keys["key"] = (grouped.cumcount() + 0.5) / grouped.transform("size")

    ordered = keys.sort_values(["key", "brand", "product_id"], kind="stable")
    return ordered["position"].to_numpy()


def _requested_value(constraint: str, profile: Mapping[str, Any]) -> str | None:
    """What the user actually asked for on the constraint being dropped."""
    if constraint == "preferred_brands":
        brands = profile.get("preferred_brands") or ()
        return ", ".join(brands) if brands else None
    if constraint == "arch_support":
        arch = profile.get("foot_arch_type")
        return "{} arches".format(arch) if arch else None
    if constraint == "seasonality":
        climate = profile.get("climate")
        return "a {} climate".format(climate) if climate else None
    if constraint == "size":
        parts = []
        shoe = profile.get("shoe_size")
        if shoe is not None:
            parts.append("US {:g}".format(float(shoe)))
        if profile.get("apparel_size"):
            parts.append(str(profile["apparel_size"]))
        return " / ".join(parts) if parts else None
    value = profile.get(constraint)
    return str(value) if value else None


def _describe_relaxation(
    step: Mapping[str, Any], profile: Mapping[str, Any], target: int, order: int
) -> dict[str, Any]:
    """Turn one relaxed step of the log into something a user can read."""
    constraint = str(step["constraint"])
    label = CONSTRAINT_LABELS.get(constraint, constraint.replace("_", " "))
    requested = _requested_value(constraint, profile)

    sentence_label = label[:1].upper() + label[1:]

    if requested is None:
        message = (
            "You did not state {}, so relaxing it changed nothing -- it was "
            "never narrowing anything.".format(label)
        )
    elif step["n_after"] > step["n_before"]:
        message = (
            "We could not keep to {} ({}): with it applied only {:,} product{} "
            "qualified, short of the {:,} candidates the ranker needs, so it was "
            "set aside and {:,} qualified instead.".format(
                label,
                requested,
                step["n_before"],
                "" if step["n_before"] == 1 else "s",
                target,
                step["n_after"],
            )
        )
    elif step["n_after"] == 0:
        message = (
            "{} ({}) could not be honoured: with it applied nothing qualified at "
            "all, and setting it aside was not enough on its own -- another "
            "preference was ruling everything out too.".format(
                sentence_label, requested
            )
        )
    else:
        message = (
            "{} ({}) was set aside on the way to {:,} candidates, but it was not "
            "what narrowed the results: {:,} products qualified either way.".format(
                sentence_label, requested, target, step["n_after"]
            )
        )

    if constraint == "color_preference" and requested is not None:
        message += (
            " Colour is the first preference the filter gives up, and {} of "
            "profiles in this dataset lose it.".format(COLOR_RELAXED_SHARE)
        )

    return {
        "constraint": constraint,
        "label": label,
        "requested": requested,
        "reason": str(step["reason"]),
        "message": message,
        "candidates_before": int(step["n_before"]),
        "candidates_after": int(step["n_after"]),
        "order": order,
    }


def _summarise(
    report: ConstraintReport,
    result: Recommendation,
    target: int,
    after_soft: int,
) -> str:
    """The whole filter, in one sentence the UI can print above the results."""
    n_shown = len(result.items)

    if report.n_after_hard == 0:
        worst = max(report.hard_excluded.items(), key=lambda pair: pair[1])
        return (
            "Nothing in the catalogue of {:,} products meets your hard "
            "requirements, so there was nothing to rank. These four are never "
            "relaxed: {}. {} ruled out the most ({:,} products) -- {}.".format(
                report.n_catalogue,
                ", ".join(HARD_LABELS[name] for name in HARD_CONSTRAINTS),
                HARD_LABELS[worst[0]][:1].upper() + HARD_LABELS[worst[0]][1:],
                worst[1],
                HARD_FIXES[worst[0]],
            )
        )

    relaxed_names = [
        CONSTRAINT_LABELS.get(name, name) for name in report.relaxed
        if _requested_value(name, result.profile) is not None
    ]
    head = (
        "{:,} products narrowed to {:,} by the four constraints that are never "
        "relaxed (budget, gender, stock and sport), then to {:,} with all your "
        "preferences applied.".format(
            report.n_catalogue, report.n_after_hard, after_soft
        )
    )
    if relaxed_names:
        middle = (
            " That was below the {:,} candidates the ranker needs, so {} {} set "
            "aside, leaving {:,}.".format(
                target,
                _join(relaxed_names),
                "was" if len(relaxed_names) == 1 else "were",
                report.n_feasible,
            )
        )
    elif report.relaxed:
        middle = (
            " Preferences you did not state were formally relaxed, which changed "
            "nothing; {:,} candidates remained.".format(report.n_feasible)
        )
    else:
        middle = " Every preference you stated was honoured."

    tail = (
        " The top {} of those {:,} {} ranked by content similarity.".format(
            n_shown, report.n_feasible, "was" if n_shown == 1 else "were"
        )
        if n_shown
        else " No product survived to be ranked."
    )
    return head + middle + tail


def _join(values: list[str]) -> str:
    """`["a","b","c"]` -> `"a, b and c"`."""
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    return ", ".join(values[:-1]) + " and " + values[-1]


# ----------------------------------------------------------------------
# JSON safety
# ----------------------------------------------------------------------
def _json_safe(value: Any) -> Any:
    """numpy scalars to Python, NaN and +/-inf to None, recursively.

    ``normalize_profile`` uses ``math.inf`` for "no budget ceiling" and pandas
    uses ``NaN`` for a missing number; neither is valid JSON, and both mean
    "not set", which is what ``null`` means.
    """
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not math.isfinite(number) else number
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, str):
        return value
    if pd.api.types.is_scalar(value) and pd.isna(value):
        return None
    return value


__all__ = [
    "CacheInfo",
    "ProductNotFound",
    "RecommendOutcome",
    "RecommenderService",
    "UserNotFound",
]
