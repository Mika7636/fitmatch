"""Every dropdown the frontend needs, read off the CSVs rather than hardcoded.

``src/data/common.py`` holds the vocabularies the *build* scripts agreed on.
This module deliberately does not import them as the answer: it reads what is
actually in ``data/processed/products.csv`` and ``users.csv``, so a catalogue
rebuilt with a different mix cannot leave the API advertising options that no
longer exist.  The constants in ``common.py`` are used only as a cross-check,
and a disagreement is logged rather than hidden.

Loaded once, at startup, by the lifespan handler.  The Pydantic request models
also read it -- that is what lets a bad ``color_preference`` be a 422 naming
the fourteen real colours instead of a 500 three layers down.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.data import common
from src.data.common import PRODUCTS_CSV, USERS_CSV

LOGGER = logging.getLogger("fitmatch.api.vocabulary")

# The numeric bounds the API accepts on a posted profile.  These are request
# validation, not domain knowledge: `profiles.normalize_profile` is happy with
# anything coercible, and the constraint filter simply returns an empty
# feasible set for a nonsense body.  Rejecting early gives the UI a usable
# error instead of an empty page.
FIELD_LIMITS: dict[str, dict[str, float]] = {
    "age": {"min": 13, "max": 100},
    "height_cm": {"min": 130, "max": 230},
    "weight_kg": {"min": 30, "max": 250},
    "shoe_size": {"min": 3, "max": 20},
    "workouts_per_week": {"min": 0, "max": 21},
    "budget_min": {"min": 0, "max": 100000},
    "budget_max": {"min": 0, "max": 100000},
}

# US half sizes are what the catalogue carries.
SHOE_SIZE_STEP = 0.5


@dataclass(frozen=True)
class NumericRange:
    min: float
    max: float
    step: float | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"min": self.min, "max": self.max}
        if self.step is not None:
            out["step"] = self.step
        return out


@dataclass(frozen=True)
class Vocabulary:
    """The controlled vocabularies and numeric ranges, as measured.

    Every list is sorted, so ``/api/meta`` is stable between restarts and a
    frontend can render it without sorting again.
    """

    # From products.csv
    sports: tuple[str, ...]
    brands: tuple[str, ...]
    colors: tuple[str, ...]
    materials: tuple[str, ...]
    categories: tuple[str, ...]
    subcategories: tuple[str, ...]
    gender_targets: tuple[str, ...]
    stock_statuses: tuple[str, ...]
    seasonality: tuple[str, ...]
    price: NumericRange
    shoe_sizes: NumericRange
    n_products: int
    n_shoe_sizes_out_of_range: int

    # From users.csv
    genders: tuple[str, ...]
    style_preferences: tuple[str, ...]
    climates: tuple[str, ...]
    foot_arch_types: tuple[str, ...]
    apparel_sizes: tuple[str, ...]
    fitness_levels: tuple[str, ...]
    indoor_or_outdoor: tuple[str, ...]
    budget: NumericRange
    n_users: int
    user_id_min: str
    user_id_max: str

    notes: tuple[str, ...] = ()

    # ------------------------------------------------------------------
    # Lookup used by the request models
    # ------------------------------------------------------------------
    def options(self, field_name: str) -> tuple[str, ...]:
        """The valid values for one profile field, by the field's own name."""
        return _FIELD_TO_OPTIONS[field_name](self)

    def canonical(self, field_name: str, value: str) -> str | None:
        """Match ``value`` case-insensitively; return the catalogue's spelling.

        Case folding happens here rather than in the caller because the
        catalogue is not consistent about case -- colours are lower-case,
        brands are title-case, apparel sizes are upper-case -- and a web form
        posting ``"Yoga"`` or ``"nike"`` means exactly what the CSV spells
        ``"yoga"`` and ``"Nike"``.  ``profiles.normalize_profile`` folds the
        same way, so this only ever agrees with it.
        """
        wanted = str(value).strip().casefold()
        for option in self.options(field_name):
            if option.casefold() == wanted:
                return option
        return None

    def as_meta(self) -> dict[str, Any]:
        """The /api/meta payload."""
        return {
            "sport_type": list(self.sports),
            "brand": list(self.brands),
            "color_preference": list(self.colors),
            "material_preference": list(self.materials),
            "category": list(self.categories),
            "subcategory": list(self.subcategories),
            "gender_target": list(self.gender_targets),
            "stock_status": list(self.stock_statuses),
            "seasonality": list(self.seasonality),
            "gender": list(self.genders),
            "style_preference": list(self.style_preferences),
            "climate": list(self.climates),
            "foot_arch_type": list(self.foot_arch_types),
            "apparel_size": list(self.apparel_sizes),
            "fitness_level": list(self.fitness_levels),
            "indoor_or_outdoor": list(self.indoor_or_outdoor),
            "price": self.price.as_dict(),
            "shoe_size": self.shoe_sizes.as_dict(),
            "budget": self.budget.as_dict(),
            "limits": {name: dict(bounds) for name, bounds in FIELD_LIMITS.items()},
            "counts": {
                "products": self.n_products,
                "users": self.n_users,
                "sport_type": len(self.sports),
                "brand": len(self.brands),
                "color_preference": len(self.colors),
                "material_preference": len(self.materials),
                "style_preference": len(self.style_preferences),
                "climate": len(self.climates),
                "category": len(self.categories),
                "foot_arch_type": len(self.foot_arch_types),
                "apparel_size": len(self.apparel_sizes),
            },
            "user_id_range": {"min": self.user_id_min, "max": self.user_id_max},
            "notes": list(self.notes),
        }


# field name -> how to get its option list off a Vocabulary.  Keyed by the
# name the *request* uses, which is why `preferred_brands` and `brand` both
# point at the same three brands.
_FIELD_TO_OPTIONS = {
    "primary_sport": lambda v: v.sports,
    "sport_type": lambda v: v.sports,
    "sport": lambda v: v.sports,
    "brand": lambda v: v.brands,
    "preferred_brands": lambda v: v.brands,
    "color_preference": lambda v: v.colors,
    "color": lambda v: v.colors,
    "material_preference": lambda v: v.materials,
    "material": lambda v: v.materials,
    "category": lambda v: v.categories,
    "subcategory": lambda v: v.subcategories,
    "gender_target": lambda v: v.gender_targets,
    "stock_status": lambda v: v.stock_statuses,
    "seasonality": lambda v: v.seasonality,
    "gender": lambda v: v.genders,
    "style_preference": lambda v: v.style_preferences,
    "climate": lambda v: v.climates,
    "foot_arch_type": lambda v: v.foot_arch_types,
    "apparel_size": lambda v: v.apparel_sizes,
    "fitness_level": lambda v: v.fitness_levels,
    "indoor_or_outdoor": lambda v: v.indoor_or_outdoor,
}

# What session 1 said each vocabulary should contain.  Used only to notice
# drift; the CSVs remain the source of truth.
_EXPECTED = {
    "sports": common.SPORTS,
    "brands": common.BRANDS,
    "colors": common.COLORS,
    "materials": common.MATERIALS,
    "categories": common.CATEGORIES,
    "style_preferences": common.STYLE_PREFERENCES,
    "climates": common.CLIMATES,
    "foot_arch_types": common.FOOT_ARCH_TYPES,
    "apparel_sizes": common.APPAREL_SIZES,
    "fitness_levels": common.FITNESS_LEVELS,
    "indoor_or_outdoor": common.INDOOR_OUTDOOR,
}


def _unique(series: pd.Series) -> tuple[str, ...]:
    return tuple(sorted({str(value) for value in series.dropna().unique()}))


def _ordered(values: Iterable[str], order: Iterable[str]) -> tuple[str, ...]:
    """Keep ``order``'s ordering for the values that are present.

    Apparel sizes must come back XS..XXL, not alphabetically -- a size
    dropdown sorted ``L, M, S, XL, XS, XXL`` looks broken.  Values not in
    ``order`` are appended, sorted, so a new size still appears.
    """
    present = set(values)
    known = [value for value in order if value in present]
    extra = sorted(present - set(known))
    return tuple(known + extra)


def build_vocabulary(products: pd.DataFrame, users: pd.DataFrame) -> Vocabulary:
    """Measure every vocabulary off the two frames."""
    us_sizes = pd.to_numeric(
        products["size"].astype(str).str.extract(r"US\s*([\d.]+)")[0], errors="coerce"
    ).dropna()
    low = FIELD_LIMITS["shoe_size"]["min"]
    high = FIELD_LIMITS["shoe_size"]["max"]
    in_range = us_sizes[(us_sizes >= low) & (us_sizes <= high)]
    out_of_range = int(len(us_sizes) - len(in_range))

    notes: list[str] = [
        "Vocabularies are measured from data/processed/*.csv at startup, not "
        "hardcoded; rebuild the data and they change with it."
    ]
    if out_of_range:
        notes.append(
            "{} of {} sized footwear rows carry a US size outside the accepted "
            "{:g}-{:g} input range and are excluded from shoe_size; they are "
            "non-US size runs left over from the Nike source data.".format(
                out_of_range, len(us_sizes), low, high
            )
        )

    user_ids = sorted(str(value) for value in users["user_id"])

    vocabulary = Vocabulary(
        sports=_unique(products["sport_type"]),
        brands=_unique(products["brand"]),
        colors=_unique(products["color"]),
        materials=_unique(products["material"]),
        categories=_ordered(_unique(products["category"]), common.CATEGORIES),
        subcategories=_unique(products["subcategory"]),
        gender_targets=_ordered(
            _unique(products["gender_target"]), common.GENDER_TARGETS
        ),
        stock_statuses=_ordered(
            _unique(products["stock_status"]), common.STOCK_STATUSES
        ),
        seasonality=_ordered(_unique(products["seasonality"]), common.SEASONALITY),
        price=NumericRange(
            min=float(products["price"].min()), max=float(products["price"].max())
        ),
        shoe_sizes=NumericRange(
            min=float(in_range.min()) if len(in_range) else low,
            max=float(in_range.max()) if len(in_range) else high,
            step=SHOE_SIZE_STEP,
        ),
        n_products=int(len(products)),
        n_shoe_sizes_out_of_range=out_of_range,
        genders=_unique(users["gender"]),
        style_preferences=_unique(users["style_preference"]),
        climates=_unique(users["climate"]),
        foot_arch_types=_ordered(
            _unique(users["foot_arch_type"]), common.FOOT_ARCH_TYPES
        ),
        apparel_sizes=_ordered(_unique(users["apparel_size"]), common.APPAREL_SIZES),
        fitness_levels=_ordered(_unique(users["fitness_level"]), common.FITNESS_LEVELS),
        indoor_or_outdoor=_ordered(
            _unique(users["indoor_or_outdoor"]), common.INDOOR_OUTDOOR
        ),
        budget=NumericRange(
            min=float(users["budget_min"].min()), max=float(users["budget_max"].max())
        ),
        n_users=int(len(users)),
        user_id_min=user_ids[0] if user_ids else "",
        user_id_max=user_ids[-1] if user_ids else "",
        notes=tuple(notes),
    )
    _warn_on_drift(vocabulary)
    return vocabulary


def _warn_on_drift(vocabulary: Vocabulary) -> None:
    """Log where the measured vocabulary differs from session 1's constants."""
    measured = asdict(vocabulary)
    for name, expected in _EXPECTED.items():
        found = set(measured[name])
        if found != set(expected):
            LOGGER.warning(
                "vocabulary drift: %s measured as %s, src.data.common declares %s",
                name,
                sorted(found),
                sorted(expected),
            )


@lru_cache(maxsize=1)
def get_vocabulary(
    products_path: Path = PRODUCTS_CSV, users_path: Path = USERS_CSV
) -> Vocabulary:
    """The process-wide vocabulary, read once.

    Cached because the Pydantic validators call it on every request; the
    lifespan handler primes it at startup so the first request does not pay
    for the read.
    """
    for path, produced_by in (
        (products_path, "src.data.build_products"),
        (users_path, "src.data.build_users"),
    ):
        if not Path(path).exists():
            raise RuntimeError(
                "{} is missing. Run `python -m {}` (or build_data.bat) before "
                "starting the API.".format(path, produced_by)
            )
    return build_vocabulary(pd.read_csv(products_path), pd.read_csv(users_path))


def reset_vocabulary() -> None:
    """Drop the cache -- for tests that swap the CSVs underneath."""
    get_vocabulary.cache_clear()
