"""Technique 1 of 2: Knowledge-Based, Constraint-Based Recommendation (Ch7).

A constraint-based recommender does not learn from ratings.  It carries
explicit domain knowledge -- "a size 8 foot does not fit a size 12 shoe",
"do not show an out-of-stock item" -- as a filter over the catalogue, and it
*relaxes* the negotiable parts of that filter when the result set gets too
thin to be useful.  That is the whole technique: no training, no similarity,
just a knowledge base and a relaxation policy.

This module answers **which products are admissible**.  It never ranks them;
ordering is Chapter 3's job (:mod:`src.recommender.content_based`).  The two
modules do not import each other.

Constraint layout
-----------------
HARD -- never relaxed, no matter how small the result set gets:

* ``price`` inside ``[budget_min, budget_max]``
* ``gender_target`` matches the user's gender, or is ``unisex``
* ``stock_status == "in_stock"``
* footwear only: ``sport_type`` is the user's ``primary_sport`` or
  ``lifestyle``

SOFT -- applied first, then dropped one at a time, cheapest first, until the
feasible set reaches ``min_results``.  See :data:`RELAXATION_ORDER`.

Two catalogue realities shape the design (both measured in session 1, both
re-measured by ``python -m src.recommender.audit``):

* **Size is a single value per row, not a stocked run.**  ``products.size``
  holds one of ``"US 9.5"``, ``"XXL"`` or ``"one-size"``.  An exact match on
  it collapses footwear to a few hundred rows before any other filter bites,
  so size is a *soft* constraint with tolerance: footwear within +/- 0.5 US
  sizes, apparel at the user's size or one adjacent size, ``one-size`` always
  matches.
* **There is no ``general`` sport.**  ``sport_type`` is one of ten real
  sports and the catalogue's mix does not match user demand -- 744 yoga users
  face 150 yoga products.  So ``lifestyle`` is treated as the sport-agnostic
  fallback tier, and apparel/accessories are treated as sport-flexible
  whatever their labelled ``sport_type``.  Only footwear is sport-strict.

Circularity warning
-------------------
``build_interactions.py`` synthesised the historical interactions with a fit
score over sport, gender, budget, brand, colour, material, arch support, size,
climate and stock.  Nine of the constraints below read the fields that same
generator used.  Session-5 evaluation against those interactions therefore
measures self-consistency as much as recommendation quality.  The overlap is
enumerated field by field in ``reports/circularity_note.md``.

Hard filtering also *removes* items the user historically interacted with --
only 81.6% of the catalogue is in stock, 76.1% of interactions fall inside
budget and 93.6% match gender.  That is correct behaviour, not a bug, but it
depresses recall.  ``python -m src.recommender.audit`` counts exactly how many
held-out interactions each hard constraint makes unreachable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.data.common import PRODUCTS_CSV
from src.recommender.profiles import (
    adjacent_apparel_sizes,
    gender_targets,
    normalize_profile,
)

# How many admissible products the constraint filter tries to leave behind.
# Below this it starts relaxing soft constraints.  It is a *candidate pool*
# for the Chapter 3 ranker, not a result count, so it sits well above top_n.
DEFAULT_MIN_RESULTS = 50

# Footwear must be for the user's sport; `lifestyle` is the sport-agnostic
# tier that any user may be shown.
SPORT_AGNOSTIC = "lifestyle"

# Apparel and accessories are sport-flexible: a training tee is a fine
# recommendation for a yoga user even though the catalogue labelled it
# `sport_type == "training"`.  Only footwear is sport-strict.
SPORT_STRICT_CATEGORIES = ("footwear",)

# Climate -> the seasonality that suits it.  `all-season` always suits.
CLIMATE_SEASON: dict[str, str] = {
    "hot": "summer",
    "tropical": "summer",
    "cold": "winter",
    "continental": "winter",
    "temperate": "all-season",
}

# A low arch wants a supportive shoe, a high arch wants a neutral one.
ARCH_NEED: dict[str, str] = {"low": "high", "normal": "medium", "high": "low"}

# Soft constraints in **relaxation order**: index 0 is dropped first.  The
# ordering is a domain judgement, stated once here so it can be inspected and
# tested rather than buried in control flow.  Cosmetic preferences go first;
# size goes last because a shoe that does not fit is not a recommendation.
RELAXATION_ORDER: tuple[str, ...] = (
    "color_preference",     # cosmetic -- the cheapest thing to give up
    "material_preference",  # cosmetic, mild comfort implication
    "arch_support",         # only bites on footwear anyway
    "seasonality",          # a winter jacket in Miami is odd, not unwearable
    "preferred_brands",     # a real preference, but a whole brand is a lot to lose
    "size",                 # last: the only soft constraint that is about fit
)

HARD_CONSTRAINTS: tuple[str, ...] = ("budget", "gender", "stock", "sport")

REQUIRED_COLUMNS = frozenset(
    {
        "product_id", "category", "sport_type", "gender_target", "price",
        "stock_status", "size", "brand", "color", "material", "seasonality",
        "arch_support",
    }
)


@dataclass
class RelaxationStep:
    """One line of the relaxation log: what was dropped, and what it bought."""

    order: int
    constraint: str
    action: str           # "applied" | "relaxed" | "exhausted"
    n_before: int
    n_after: int
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "constraint": self.constraint,
            "action": self.action,
            "n_before": self.n_before,
            "n_after": self.n_after,
            "reason": self.reason,
        }

    def __str__(self) -> str:
        return "{:>2}. {:<38} {:<10} {:>6,} -> {:>6,}  {}".format(
            self.order, self.constraint, self.action,
            self.n_before, self.n_after, self.reason,
        )


@dataclass
class ConstraintReport:
    """Everything the filter learned on one call: the relaxation log plus the
    hard-filter exclusion counts session 5 needs to explain its recall."""

    n_catalogue: int
    n_after_hard: int
    n_feasible: int
    hard_excluded: dict[str, int] = field(default_factory=dict)
    steps: list[RelaxationStep] = field(default_factory=list)
    relaxed: list[str] = field(default_factory=list)
    applied: list[str] = field(default_factory=list)

    def as_dicts(self) -> list[dict[str, Any]]:
        """The log as plain dicts, for JSON responses in session 3."""
        return [step.as_dict() for step in self.steps]

    def __iter__(self):
        """A ConstraintReport iterates as its steps, so a caller that only
        wants the relaxation log can treat it as the list of steps it is."""
        return iter(self.steps)

    def __len__(self) -> int:
        return len(self.steps)

    def __str__(self) -> str:
        head = "catalogue {:,} -> hard {:,} -> feasible {:,}".format(
            self.n_catalogue, self.n_after_hard, self.n_feasible
        )
        hard = "  hard filter removed: " + ", ".join(
            "{} {:,}".format(name, count) for name, count in self.hard_excluded.items()
        )
        return "\n".join([head, hard] + ["  " + str(step) for step in self.steps])


class ConstraintBasedRecommender:
    """Knowledge-Based Constraint-Based Recommendation (Chapter 7).

    Parameters
    ----------
    products:
        The catalogue -- ``data/processed/products.csv``, or any frame with
        the same columns.  Use :meth:`from_csv` to load it from disk.
    min_results:
        Target size of the feasible set.  Soft constraints are relaxed, in
        :data:`RELAXATION_ORDER`, until the set reaches this many rows or
        every soft constraint has been dropped.

    Notes
    -----
    Stateless and training-free: the same profile always produces the same
    feasible set, which is what makes the determinism test in
    ``tests/test_determinism.py`` meaningful.
    """

    def __init__(
        self, products: pd.DataFrame, min_results: int = DEFAULT_MIN_RESULTS
    ) -> None:
        if min_results < 1:
            raise ValueError("min_results must be >= 1")
        missing = REQUIRED_COLUMNS - set(products.columns)
        if missing:
            raise ValueError("products is missing columns: {}".format(sorted(missing)))
        self.products = products.reset_index(drop=True)
        self.min_results = int(min_results)
        self._precompute()

    @classmethod
    def from_csv(
        cls, path=PRODUCTS_CSV, min_results: int = DEFAULT_MIN_RESULTS
    ) -> "ConstraintBasedRecommender":
        """Load the catalogue from disk and build the recommender."""
        return cls(pd.read_csv(path), min_results=min_results)

    # ------------------------------------------------------------------
    # Precomputation -- the catalogue is static, so parse it once
    # ------------------------------------------------------------------
    def _precompute(self) -> None:
        products = self.products
        self._price = products["price"].to_numpy(dtype=float)
        self._gender_target = products["gender_target"].to_numpy()
        self._stock = products["stock_status"].to_numpy()
        self._category = products["category"].to_numpy()
        self._sport = products["sport_type"].to_numpy()
        self._brand = products["brand"].to_numpy()
        self._color = products["color"].to_numpy()
        self._material = products["material"].to_numpy()
        self._season = products["seasonality"].to_numpy()
        self._arch = products["arch_support"].to_numpy()

        self._is_footwear = np.isin(self._category, SPORT_STRICT_CATEGORIES)

        size = products["size"].astype(str)
        # "US 10.5" -> 10.5; NaN on apparel, accessories and "one-size".
        self._shoe_size = pd.to_numeric(
            size.str.extract(r"US\s*([\d.]+)")[0], errors="coerce"
        ).to_numpy(dtype=float)
        self._has_us_size = ~np.isnan(self._shoe_size)
        self._letter_size = size.str.strip().str.upper().to_numpy()
        self._one_size = size.str.strip().str.lower().eq("one-size").to_numpy()
        self._all_true = np.ones(len(products), dtype=bool)

    # ------------------------------------------------------------------
    # Hard constraints -- never relaxed
    # ------------------------------------------------------------------
    def _hard_masks(self, profile: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """One boolean mask per hard constraint."""
        budget = (self._price >= profile["budget_min"]) & (
            self._price <= profile["budget_max"]
        )
        gender = np.isin(self._gender_target, gender_targets(profile))
        stock = self._stock == "in_stock"

        # Footwear is sport-strict; apparel and accessories are sport-flexible
        # (spec (b): there is no `general` sport and the catalogue's sport mix
        # does not match user demand, so apparel must not be filtered on it).
        wanted_sports = [profile["primary_sport"], SPORT_AGNOSTIC]
        sport = ~self._is_footwear | np.isin(self._sport, wanted_sports)

        return {"budget": budget, "gender": gender, "stock": stock, "sport": sport}

    # ------------------------------------------------------------------
    # Soft constraints -- relaxed in RELAXATION_ORDER
    # ------------------------------------------------------------------
    def _size_mask(self, profile: Mapping[str, Any]) -> np.ndarray:
        """Soft size with tolerance -- spec (a).

        ``one-size`` matches everyone.  A row carrying a US size matches
        within +/- 0.5 of ``shoe_size``.  Every other row matches
        ``apparel_size`` or one adjacent size on the XS..XXL ladder.  A
        profile that states neither size is not filtered on size at all.
        """
        shoe = profile.get("shoe_size")
        apparel = profile.get("apparel_size")
        if shoe is None and apparel is None:
            return self._all_true

        if shoe is not None:
            with np.errstate(invalid="ignore"):
                footwear_ok = np.abs(self._shoe_size - float(shoe)) <= 0.5
            footwear_ok = np.where(self._has_us_size, footwear_ok, False)
        else:
            # No stated shoe size: do not rule footwear out on size.
            footwear_ok = self._all_true

        letters = adjacent_apparel_sizes(apparel) if apparel is not None else ()
        apparel_ok = np.isin(self._letter_size, letters) if letters else self._all_true

        return self._one_size | np.where(self._has_us_size, footwear_ok, apparel_ok)

    def _soft_masks(self, profile: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """One boolean mask per soft constraint.

        A constraint the profile says nothing about is not a constraint: it
        gets an all-True mask, so relaxing it later is a visible no-op in the
        log rather than a silent lie about what was given up.
        """
        brands = profile.get("preferred_brands") or ()
        brand_mask = np.isin(self._brand, list(brands)) if brands else self._all_true

        color = profile.get("color_preference")
        color_mask = (self._color == color) if color else self._all_true

        material = profile.get("material_preference")
        material_mask = (self._material == material) if material else self._all_true

        climate = profile.get("climate")
        wanted_season = CLIMATE_SEASON.get(climate) if climate else None
        season_mask = (
            (self._season == wanted_season) | (self._season == "all-season")
            if wanted_season
            else self._all_true
        )

        arch_type = profile.get("foot_arch_type")
        wanted_arch = ARCH_NEED.get(arch_type) if arch_type else None
        # arch_support is "medium" on 80.1% of rows because non-footwear
        # carries a neutral value (spec (c)), so the constraint is only
        # meaningful on footwear.  Off footwear it passes everything.
        arch_mask = (
            ~self._is_footwear | (self._arch == wanted_arch)
            if wanted_arch
            else self._all_true
        )

        return {
            "size": self._size_mask(profile),
            "preferred_brands": brand_mask,
            "color_preference": color_mask,
            "material_preference": material_mask,
            "seasonality": season_mask,
            "arch_support": arch_mask,
        }

    # ------------------------------------------------------------------
    # The technique
    # ------------------------------------------------------------------
    def filter(
        self, user_profile: Mapping[str, Any], min_results: int | None = None
    ) -> tuple[pd.DataFrame, ConstraintReport]:
        """Run the constraint-based recommender on one profile.

        Returns ``(feasible_df, relaxation_log)``.  ``feasible_df`` is a slice
        of the catalogue in catalogue order -- deliberately unranked, because
        ranking is Chapter 3's job.  ``relaxation_log`` is a
        :class:`ConstraintReport`; it iterates as its list of
        :class:`RelaxationStep` and also carries the hard-filter exclusion
        counts.
        """
        profile = normalize_profile(user_profile)
        target = self.min_results if min_results is None else int(min_results)

        hard = self._hard_masks(profile)
        hard_mask = self._all_true.copy()
        excluded: dict[str, int] = {}
        for name in HARD_CONSTRAINTS:
            before = int(hard_mask.sum())
            hard_mask &= hard[name]
            excluded[name] = before - int(hard_mask.sum())

        report = ConstraintReport(
            n_catalogue=len(self.products),
            n_after_hard=int(hard_mask.sum()),
            n_feasible=0,
            hard_excluded=excluded,
        )
        report.steps.append(
            RelaxationStep(
                order=0,
                constraint="HARD (budget, gender, stock, sport)",
                action="applied",
                n_before=len(self.products),
                n_after=int(hard_mask.sum()),
                reason="never relaxed",
            )
        )

        # Apply every soft constraint, then drop them in RELAXATION_ORDER
        # until the feasible set is big enough.  Recomputing the conjunction
        # from scratch each round keeps this obviously correct, and the masks
        # are 10k booleans, so the cost is nothing.
        soft = self._soft_masks(profile)
        active = list(RELAXATION_ORDER)
        relaxed: list[str] = []

        def combine(names: Sequence[str]) -> np.ndarray:
            mask = hard_mask.copy()
            for name in names:
                mask &= soft[name]
            return mask

        mask = combine(active)
        report.steps.append(
            RelaxationStep(
                order=1,
                constraint="SOFT (all applied)",
                action="applied",
                n_before=int(hard_mask.sum()),
                n_after=int(mask.sum()),
                reason=", ".join(active),
            )
        )

        order = 2
        while int(mask.sum()) < target and active:
            dropped = active.pop(0)          # cheapest first
            before = int(mask.sum())
            mask = combine(active)
            relaxed.append(dropped)
            report.steps.append(
                RelaxationStep(
                    order=order,
                    constraint=dropped,
                    action="relaxed",
                    n_before=before,
                    n_after=int(mask.sum()),
                    reason="feasible set was below min_results={}".format(target),
                )
            )
            order += 1

        if int(mask.sum()) < target:
            report.steps.append(
                RelaxationStep(
                    order=order,
                    constraint="(no soft constraints left)",
                    action="exhausted",
                    n_before=int(mask.sum()),
                    n_after=int(mask.sum()),
                    reason="the hard constraints alone leave fewer than {}".format(target),
                )
            )

        report.relaxed = relaxed
        report.applied = list(active)
        report.n_feasible = int(mask.sum())

        return self.products.loc[mask].copy(), report

    # ------------------------------------------------------------------
    # Explanation -- the point of a knowledge-based recommender
    # ------------------------------------------------------------------
    def explain_constraints(
        self, user: Mapping[str, Any], product: Mapping[str, Any] | pd.Series
    ) -> list[str]:
        """Which constraints this product satisfies for this user, in words.

        A knowledge-based recommender is expected to be able to say *why*, and
        this is that: one readable line per satisfied constraint, hard ones
        first.  A constraint the profile says nothing about is omitted rather
        than claimed as satisfied.
        """
        profile = normalize_profile(user)
        if isinstance(product, pd.Series):
            product = product.to_dict()

        reasons: list[str] = []

        price = float(product["price"])
        if profile["budget_min"] <= price <= profile["budget_max"]:
            reasons.append(
                "price ${:.2f} is inside your ${:,.0f}-${:,.0f} budget".format(
                    price, profile["budget_min"], profile["budget_max"]
                )
            )

        target = str(product["gender_target"])
        if target in gender_targets(profile):
            reasons.append(
                "unisex, so it fits any profile"
                if target == "unisex"
                else "targeted at {}, matching your profile".format(target)
            )

        if str(product["stock_status"]) == "in_stock":
            reasons.append("in stock")

        category = str(product["category"])
        sport = str(product["sport_type"])
        if category in SPORT_STRICT_CATEGORIES:
            if sport == profile["primary_sport"]:
                reasons.append("{} footwear, your primary sport".format(sport))
            elif sport == SPORT_AGNOSTIC:
                reasons.append("lifestyle footwear, which suits any sport")
        elif sport == profile["primary_sport"]:
            reasons.append("built for {}, your primary sport".format(sport))
        else:
            reasons.append(
                "{} is sport-flexible, so its {} labelling does not rule it out".format(
                    category, sport
                )
            )

        reasons.extend(self._explain_size(profile, str(product["size"]).strip()))

        brands = profile["preferred_brands"]
        if brands and str(product["brand"]) in brands:
            reasons.append("{} is one of your preferred brands".format(product["brand"]))
        if profile["color_preference"] and str(product["color"]) == profile["color_preference"]:
            reasons.append("{} is your preferred colour".format(product["color"]))
        if (
            profile["material_preference"]
            and str(product["material"]) == profile["material_preference"]
        ):
            reasons.append("{} is your preferred material".format(product["material"]))

        wanted_season = CLIMATE_SEASON.get(profile["climate"] or "")
        season = str(product["seasonality"])
        if wanted_season and season in (wanted_season, "all-season"):
            reasons.append("{} wear suits a {} climate".format(season, profile["climate"]))

        wanted_arch = ARCH_NEED.get(profile["foot_arch_type"] or "")
        if (
            wanted_arch
            and category in SPORT_STRICT_CATEGORIES
            and str(product["arch_support"]) == wanted_arch
        ):
            reasons.append(
                "{} arch support, which is what a {} arch needs".format(
                    wanted_arch, profile["foot_arch_type"]
                )
            )
        return reasons

    @staticmethod
    def _explain_size(profile: Mapping[str, Any], size: str) -> list[str]:
        """The size half of :meth:`explain_constraints`, split out because the
        three sizing regimes make it the fiddliest part."""
        if size.lower() == "one-size":
            return ["one-size, so sizing is not a constraint"]

        us = pd.to_numeric(
            pd.Series([size]).str.extract(r"US\s*([\d.]+)")[0], errors="coerce"
        ).iloc[0]
        if not pd.isna(us):
            if profile["shoe_size"] is not None and abs(us - profile["shoe_size"]) <= 0.5:
                return [
                    "size {} is within half a size of your US {:g}".format(
                        size, profile["shoe_size"]
                    )
                ]
            return []

        apparel = profile["apparel_size"]
        if apparel is None:
            return []
        if size.upper() == apparel:
            return ["size {} is your apparel size".format(size)]
        if size.upper() in adjacent_apparel_sizes(apparel):
            return ["size {} is one step from your usual {}".format(size, apparel)]
        return []
