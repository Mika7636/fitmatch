"""The user-profile contract shared by both recommender techniques.

Both :mod:`src.recommender.knowledge_based` (Ch7) and
:mod:`src.recommender.content_based` (Ch3) accept the same thing: a plain
``dict``.  It may be a row of ``users.csv`` turned into a dict, or a raw
profile posted by a web form with half the fields missing.  This module is the
one place that decides what a missing field means, so the two techniques can
never disagree about it.

Nothing here scores or ranks anything -- it only normalises.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import pandas as pd

from src.data.common import APPAREL_SIZES

# users.gender is male/female; products.gender_target is men/women/unisex.
GENDER_TO_TARGET: dict[str, str] = {"male": "men", "female": "women"}

# Every field a fully-specified profile carries, with the value that means
# "the user did not tell us".  A None here disables the constraint that reads
# it rather than inventing an answer -- a web form that omits `color_preference`
# must not be treated as a user who wants black.
PROFILE_DEFAULTS: dict[str, Any] = {
    "user_id": None,
    "age": None,
    "gender": None,
    "height_cm": None,
    "weight_kg": None,
    "bmi": None,
    "shoe_size": None,
    "apparel_size": None,
    "foot_arch_type": None,
    "primary_sport": "lifestyle",
    "fitness_level": None,
    "workouts_per_week": None,
    "indoor_or_outdoor": None,
    "budget_min": 0.0,
    "budget_max": math.inf,
    "preferred_brands": (),
    "style_preference": None,
    "color_preference": None,
    "material_preference": None,
    "location": None,
    "climate": None,
}

_NUMERIC = ("age", "height_cm", "weight_kg", "bmi", "shoe_size",
            "workouts_per_week", "budget_min", "budget_max")


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if value is pd.NaT:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _as_brand_tuple(value: Any) -> tuple[str, ...]:
    """`"Nike|Adidas"`, `["Nike", "Adidas"]` and `"Nike"` all mean the same."""
    if _is_missing(value):
        return ()
    if isinstance(value, str):
        parts: Iterable[str] = value.split("|")
    elif isinstance(value, Iterable):
        parts = [str(part) for part in value]
    else:
        parts = [str(value)]
    return tuple(part.strip() for part in parts if part and part.strip())


def normalize_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Return a complete profile dict: every key in :data:`PROFILE_DEFAULTS`,
    missing values replaced by the default, strings lower-cased where the
    vocabulary is lower-case, `preferred_brands` always a tuple.

    Unknown extra keys are kept untouched so callers can carry their own
    metadata through the pipeline.
    """
    if not isinstance(profile, Mapping):
        raise TypeError(
            "profile must be a mapping, got {}".format(type(profile).__name__)
        )

    out: dict[str, Any] = dict(profile)
    for key, default in PROFILE_DEFAULTS.items():
        if key not in out or _is_missing(out[key]):
            out[key] = default

    out["preferred_brands"] = _as_brand_tuple(profile.get("preferred_brands"))

    for key in _NUMERIC:
        if out[key] is not None and not isinstance(out[key], float):
            try:
                out[key] = float(out[key])
            except (TypeError, ValueError):
                out[key] = PROFILE_DEFAULTS[key]

    # Lower-case the controlled vocabularies. apparel_size stays upper-case
    # ("XL"), and preferred_brands stay title-case ("Nike"), because that is
    # how products.csv spells them.
    for key in ("gender", "foot_arch_type", "primary_sport", "fitness_level",
                "indoor_or_outdoor", "style_preference", "color_preference",
                "material_preference", "climate"):
        if isinstance(out[key], str):
            out[key] = out[key].strip().lower()
    if isinstance(out["apparel_size"], str):
        out["apparel_size"] = out["apparel_size"].strip().upper()

    if out["budget_min"] is None:
        out["budget_min"] = 0.0
    if out["budget_max"] is None:
        out["budget_max"] = math.inf
    if out["budget_max"] < out["budget_min"]:
        out["budget_min"], out["budget_max"] = out["budget_max"], out["budget_min"]

    return out


def gender_targets(profile: Mapping[str, Any]) -> tuple[str, ...]:
    """The `products.gender_target` values this profile may be shown.

    A profile with no stated gender is shown everything -- a web form that
    skips the question must not silently become a men's-only store.
    """
    gender = profile.get("gender")
    target = GENDER_TO_TARGET.get(gender) if isinstance(gender, str) else None
    if target is None:
        return ("men", "women", "unisex")
    return (target, "unisex")


def adjacent_apparel_sizes(size: str | None) -> tuple[str, ...]:
    """`"L"` -> `("M", "L", "XL")`.  Session-2 spec (a): apparel matches the
    user's size *or one adjacent size*, because `products.size` holds a single
    value per row rather than a run of stocked sizes.
    """
    if not isinstance(size, str) or size.upper() not in APPAREL_SIZES:
        return tuple(APPAREL_SIZES)
    index = APPAREL_SIZES.index(size.upper())
    low = max(0, index - 1)
    high = min(len(APPAREL_SIZES), index + 2)
    return tuple(APPAREL_SIZES[low:high])
