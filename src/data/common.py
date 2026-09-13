"""Shared helpers for the FitMatch data layer.

Holds the paths, the controlled vocabularies that the three build scripts must
agree on, the provenance tracker (source vs. derived vs. synthetic) and the
end-of-script summary printer.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 42

# Nike source data is priced in GBP, Adidas in USD. Everything downstream is
# USD, converted at this fixed rate so runs stay reproducible.
GBP_TO_USD = 1.27

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PRODUCTS_CSV = PROCESSED_DIR / "products.csv"
USERS_CSV = PROCESSED_DIR / "users.csv"
INTERACTIONS_CSV = PROCESSED_DIR / "interactions.csv"


def find_raw(*patterns: str) -> Path:
    """Return the first file in data/raw/ matching any of the glob patterns.

    Kaggle download names drift, so each source is looked up by pattern rather
    than by an exact filename.
    """
    for pattern in patterns:
        matches = sorted(RAW_DIR.glob(pattern))
        if matches:
            return matches[0]
    raise SystemExit(
        "\nFATAL: no file in {} matches any of {}.\n"
        "Download the Kaggle datasets into data/raw/ first (see README.md).".format(
            RAW_DIR, list(patterns)
        )
    )


# --------------------------------------------------------------------------
# Controlled vocabularies -- shared across products, users and interactions
# --------------------------------------------------------------------------
CATEGORIES = ["footwear", "apparel", "accessory"]
GENDER_TARGETS = ["men", "women", "unisex"]
STOCK_STATUSES = ["in_stock", "low_stock", "out_of_stock"]
ARCH_SUPPORT = ["low", "medium", "high"]
FOOT_ARCH_TYPES = ["low", "normal", "high"]
SEASONALITY = ["summer", "winter", "all-season"]

# Only brands that actually exist in the catalogue, so a user's
# preferred_brands always has something to match against.
BRANDS = ["Nike", "Adidas", "Jordan"]

SPORTS = [
    "running",
    "training",
    "basketball",
    "football",
    "tennis",
    "yoga",
    "outdoor",
    "lifestyle",
    "swimming",
    "skateboarding",
]

COLORS = [
    "black",
    "white",
    "grey",
    "blue",
    "navy",
    "red",
    "pink",
    "green",
    "orange",
    "yellow",
    "purple",
    "brown",
    "beige",
    "multi",
]

MATERIALS = [
    "mesh",
    "knit",
    "leather",
    "synthetic",
    "cotton",
    "polyester",
    "fleece",
    "nylon",
    "gore-tex",
    "rubber",
]

APPAREL_SIZES = ["XS", "S", "M", "L", "XL", "XXL"]
STYLE_PREFERENCES = ["performance", "casual", "athleisure", "retro"]
FITNESS_LEVELS = ["beginner", "intermediate", "advanced"]
INDOOR_OUTDOOR = ["indoor", "outdoor", "both"]

EVENT_TYPES = ["view", "add_to_cart", "purchase", "rating"]

# City -> (state, climate). The cities are the US metros that appear in the
# Adidas sales data; climate drives seasonality preferences downstream.
LOCATIONS = {
    "New York": ("NY", "continental"),
    "Boston": ("MA", "continental"),
    "Philadelphia": ("PA", "temperate"),
    "Chicago": ("IL", "continental"),
    "Minneapolis": ("MN", "cold"),
    "Detroit": ("MI", "continental"),
    "Denver": ("CO", "cold"),
    "Seattle": ("WA", "temperate"),
    "Portland": ("OR", "temperate"),
    "San Francisco": ("CA", "temperate"),
    "Los Angeles": ("CA", "hot"),
    "San Diego": ("CA", "hot"),
    "Phoenix": ("AZ", "hot"),
    "Las Vegas": ("NV", "hot"),
    "Dallas": ("TX", "hot"),
    "Houston": ("TX", "tropical"),
    "Miami": ("FL", "tropical"),
    "Orlando": ("FL", "tropical"),
    "Atlanta": ("GA", "temperate"),
    "Charlotte": ("NC", "temperate"),
}
CLIMATES = sorted({climate for _, climate in LOCATIONS.values()})


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
def log(message: str = "") -> None:
    print(message, flush=True)


def section(title: str) -> None:
    log("")
    log("=" * 78)
    log(title)
    log("=" * 78)


# --------------------------------------------------------------------------
# Provenance: how did each column get its values?
# --------------------------------------------------------------------------
class Provenance:
    """Counts, per column, how many values came straight from the source file,
    how many were derived by rule from source text, and how many were
    synthesised."""

    KINDS = ("source", "derived", "synthetic")

    def __init__(self, label: str) -> None:
        self.label = label
        self._counts: dict[str, Counter] = {}

    def note(self, field: str, kind: str, count: int) -> None:
        if kind not in self.KINDS:
            raise ValueError("unknown provenance kind: " + str(kind))
        self._counts.setdefault(field, Counter())[kind] += int(count)

    def flags(self, field: str, flags) -> None:
        """Record an array of per-row 'source'/'derived'/'synthetic' labels."""
        for kind, count in Counter(np.asarray(flags).tolist()).items():
            self.note(field, kind, count)

    def merge(self, other: "Provenance") -> None:
        for field, counter in other._counts.items():
            for kind, count in counter.items():
                self.note(field, kind, count)

    def report(self) -> None:
        section("PROVENANCE -- {} (values per column)".format(self.label))
        header = "{:<22}{:>10}{:>10}{:>12}{:>17}".format(
            "column", "source", "derived", "synthetic", "derived+synth"
        )
        log(header)
        log("-" * 78)
        for field, counter in self._counts.items():
            total = sum(counter.values()) or 1
            pct = 100.0 * (counter["derived"] + counter["synthetic"]) / total
            log(
                "{:<22}{:>10,}{:>10,}{:>12,}{:>16.1f}%".format(
                    field,
                    counter["source"],
                    counter["derived"],
                    counter["synthetic"],
                    pct,
                )
            )
        totals: Counter = Counter()
        for counter in self._counts.values():
            totals.update(counter)
        grand = sum(totals.values()) or 1
        log("-" * 78)
        log(
            "{:<22}{:>10,}{:>10,}{:>12,}{:>16.1f}%".format(
                "TOTAL",
                totals["source"],
                totals["derived"],
                totals["synthetic"],
                100.0 * (totals["derived"] + totals["synthetic"]) / grand,
            )
        )


# --------------------------------------------------------------------------
# Summary printing
# --------------------------------------------------------------------------
def summarize(
    df: pd.DataFrame,
    name: str,
    dist_cols=(),
    numeric_cols=(),
    top_n: int = 12,
) -> None:
    section("SUMMARY -- {}".format(name))
    log("rows: {:,}    columns: {}".format(len(df), len(df.columns)))
    log("memory: {:.1f} MB".format(df.memory_usage(deep=True).sum() / 1e6))

    log("")
    log("nulls per column:")
    nulls = df.isna().sum()
    width = max(len(c) for c in df.columns) + 2
    for col in df.columns:
        pct = 100.0 * nulls[col] / max(len(df), 1)
        marker = "  <-- NULLS" if nulls[col] else ""
        log("  {:<{w}}{:>10,}  ({:5.2f}%){}".format(col, nulls[col], pct, marker, w=width))

    if numeric_cols:
        cols = [c for c in numeric_cols if c in df.columns]
        log("")
        log("numeric distributions:")
        desc = df[cols].describe().T
        desc = desc[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
        log(desc.round(2).to_string())

    for col in dist_cols:
        if col not in df.columns:
            continue
        counts = df[col].value_counts(dropna=False)
        log("")
        log("{}  ({:,} distinct)".format(col, counts.size))
        for value, count in counts.head(top_n).items():
            bar = "#" * int(40 * count / counts.iloc[0])
            log(
                "  {:<32}{:>9,} ({:5.2f}%) {}".format(
                    str(value)[:30], count, 100.0 * count / len(df), bar
                )
            )
        if counts.size > top_n:
            log("  ... and {:,} more values".format(counts.size - top_n))


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log("")
    log(
        "wrote {}  ({:,} rows, {:.1f} MB)".format(
            path.relative_to(PROJECT_ROOT), len(df), path.stat().st_size / 1e6
        )
    )


def require(path: Path, produced_by: str) -> None:
    if not path.exists():
        raise SystemExit(
            "\nFATAL: {} is missing. Run `python -m {}` first.".format(path, produced_by)
        )
