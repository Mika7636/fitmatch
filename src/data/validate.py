"""Validate the three processed CSVs and fail loudly if anything is wrong.

Run with:  python -m src.data.validate

Exits 0 when every check passes, 1 when any check fails. The referential
integrity checks -- every interaction pointing at a user_id and a product_id
that actually exist -- are the ones this file exists for; the rest are schema,
vocabulary and range checks that would otherwise bite three sessions later.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.data.build_interactions import SCHEMA as INTERACTIONS_SCHEMA
from src.data.build_products import SCHEMA as PRODUCTS_SCHEMA
from src.data.build_users import SCHEMA as USERS_SCHEMA
from src.data.common import (
    APPAREL_SIZES,
    BRANDS,
    CATEGORIES,
    CLIMATES,
    COLORS,
    EVENT_TYPES,
    FITNESS_LEVELS,
    FOOT_ARCH_TYPES,
    GENDER_TARGETS,
    INDOOR_OUTDOOR,
    INTERACTIONS_CSV,
    MATERIALS,
    PRODUCTS_CSV,
    SEASONALITY,
    SPORTS,
    STOCK_STATUSES,
    STYLE_PREFERENCES,
    USERS_CSV,
    ARCH_SUPPORT,
    log,
    require,
    section,
)


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []
        self.passed = 0

    def check(self, ok: bool, description: str, detail: str = "") -> bool:
        if ok:
            self.passed += 1
            log("  PASS  {}".format(description))
        else:
            self.failures.append(description + ((" -- " + detail) if detail else ""))
            log("  FAIL  {}{}".format(description, ("  --  " + detail) if detail else ""))
        return ok

    def warn(self, ok: bool, description: str, detail: str = "") -> None:
        if ok:
            self.passed += 1
            log("  PASS  {}".format(description))
        else:
            self.warnings.append(description + ((" -- " + detail) if detail else ""))
            log("  WARN  {}{}".format(description, ("  --  " + detail) if detail else ""))


def sample(values, limit: int = 8) -> str:
    values = list(values)
    shown = ", ".join(repr(v) for v in values[:limit])
    return "{} of them, e.g. {}".format(len(values), shown)


def check_columns(report: Report, df: pd.DataFrame, expected: list[str], name: str) -> None:
    report.check(
        list(df.columns) == expected,
        "{}: columns match the declared schema".format(name),
        "got {}".format(list(df.columns)) if list(df.columns) != expected else "",
    )


def check_domain(report: Report, df: pd.DataFrame, column: str, allowed, name: str) -> None:
    if column not in df.columns:
        return
    bad = set(df[column].dropna().unique()) - set(allowed)
    report.check(
        not bad,
        "{}.{} only uses the allowed vocabulary".format(name, column),
        "unexpected values: {}".format(sorted(bad)[:10]) if bad else "",
    )


def check_range(report: Report, df: pd.DataFrame, column: str, low, high, name: str) -> None:
    if column not in df.columns:
        return
    values = pd.to_numeric(df[column], errors="coerce")
    outside = values.notna() & ((values < low) | (values > high))
    report.check(
        not outside.any(),
        "{}.{} lies in [{}, {}]".format(name, column, low, high),
        "{:,} rows outside, e.g. {}".format(
            int(outside.sum()), values[outside].head(5).tolist()) if outside.any() else "",
    )


def check_no_nulls(report: Report, df: pd.DataFrame, name: str, exempt=()) -> None:
    nulls = df.isna().sum()
    offenders = {c: int(n) for c, n in nulls.items() if n and c not in exempt}
    report.check(
        not offenders,
        "{}: no nulls outside the columns allowed to have them".format(name),
        str(offenders) if offenders else "",
    )


# ==========================================================================
# Per-file checks
# ==========================================================================
def validate_products(report: Report, products: pd.DataFrame) -> None:
    section("PRODUCTS -- {:,} rows".format(len(products)))
    check_columns(report, products, PRODUCTS_SCHEMA, "products")
    check_no_nulls(report, products, "products")

    report.check(products["product_id"].is_unique, "products.product_id is unique",
                 sample(products.loc[products["product_id"].duplicated(), "product_id"])
                 if not products["product_id"].is_unique else "")

    check_domain(report, products, "category", CATEGORIES, "products")
    check_domain(report, products, "gender_target", GENDER_TARGETS, "products")
    check_domain(report, products, "stock_status", STOCK_STATUSES, "products")
    check_domain(report, products, "arch_support", ARCH_SUPPORT, "products")
    check_domain(report, products, "seasonality", SEASONALITY, "products")
    check_domain(report, products, "sport_type", SPORTS, "products")
    check_domain(report, products, "color", COLORS, "products")
    check_domain(report, products, "material", MATERIALS, "products")
    check_domain(report, products, "brand", BRANDS, "products")

    check_range(report, products, "price", 0.01, 5000, "products")
    check_range(report, products, "discount", 0.0, 1.0, "products")
    check_range(report, products, "cushioning_level", 1, 5, "products")
    check_range(report, products, "breathability", 1, 5, "products")
    check_range(report, products, "avg_rating", 1.0, 5.0, "products")
    check_range(report, products, "review_count", 0, 10 ** 6, "products")
    check_range(report, products, "weight", 1, 10000, "products")

    ranks = products["sales_rank"]
    report.check(
        ranks.is_unique and ranks.min() == 1 and ranks.max() == len(products),
        "products.sales_rank is a dense 1..N permutation",
        "min {}, max {}, unique {}".format(ranks.min(), ranks.max(), ranks.nunique()),
    )

    report.check(
        products["waterproof"].isin([True, False]).all(),
        "products.waterproof is boolean",
    )

    # Cushioning and arch support are only meaningful for footwear.
    non_shoe = products[products["category"] != "footwear"]
    report.warn(
        (non_shoe["cushioning_level"] == 1).all(),
        "products: non-footwear carries the neutral cushioning value",
    )

    report.check(
        products["product_description"].str.len().gt(20).all(),
        "products.product_description is non-trivial on every row",
    )


def validate_users(report: Report, users: pd.DataFrame) -> None:
    section("USERS -- {:,} rows".format(len(users)))
    check_columns(report, users, USERS_SCHEMA, "users")
    check_no_nulls(report, users, "users")

    report.check(users["user_id"].is_unique, "users.user_id is unique")

    check_domain(report, users, "gender", ["male", "female"], "users")
    check_domain(report, users, "apparel_size", APPAREL_SIZES, "users")
    check_domain(report, users, "foot_arch_type", FOOT_ARCH_TYPES, "users")
    check_domain(report, users, "primary_sport", SPORTS, "users")
    check_domain(report, users, "fitness_level", FITNESS_LEVELS, "users")
    check_domain(report, users, "indoor_or_outdoor", INDOOR_OUTDOOR, "users")
    check_domain(report, users, "style_preference", STYLE_PREFERENCES, "users")
    check_domain(report, users, "color_preference", COLORS, "users")
    check_domain(report, users, "material_preference", MATERIALS, "users")
    check_domain(report, users, "climate", CLIMATES, "users")

    check_range(report, users, "age", 13, 100, "users")
    check_range(report, users, "height_cm", 130, 230, "users")
    check_range(report, users, "weight_kg", 30, 250, "users")
    check_range(report, users, "bmi", 12, 60, "users")
    check_range(report, users, "shoe_size", 3, 20, "users")
    check_range(report, users, "workouts_per_week", 0, 7, "users")
    check_range(report, users, "budget_min", 1, 5000, "users")
    check_range(report, users, "budget_max", 1, 5000, "users")

    report.check(
        (users["budget_min"] < users["budget_max"]).all(),
        "users.budget_min is always below budget_max",
        "{:,} rows violate it".format(int((users["budget_min"] >= users["budget_max"]).sum())),
    )

    recomputed = users["weight_kg"] / (users["height_cm"] / 100.0) ** 2
    drift = (recomputed - users["bmi"]).abs()
    report.check(
        drift.max() < 0.15,
        "users.bmi agrees with weight_kg / height_m^2",
        "max drift {:.3f}".format(drift.max()),
    )

    brands = users["preferred_brands"].str.split("|")
    unknown = {b for row in brands for b in row} - set(BRANDS)
    report.check(
        not unknown,
        "users.preferred_brands only names brands that exist in the catalogue",
        "unknown: {}".format(sorted(unknown)) if unknown else "",
    )
    report.check(
        brands.map(lambda row: len(row) == len(set(row))).all(),
        "users.preferred_brands has no repeated brand within a row",
    )

    # Shoe sizes are gendered; a women's US 14 would not join to anything.
    women = users[users["gender"] == "female"]["shoe_size"]
    men = users[users["gender"] == "male"]["shoe_size"]
    report.warn(
        women.mean() < men.mean(),
        "users: women's mean shoe size is below men's",
        "women {:.2f} vs men {:.2f}".format(women.mean(), men.mean()),
    )


def validate_interactions(report: Report, interactions: pd.DataFrame) -> None:
    section("INTERACTIONS -- {:,} rows".format(len(interactions)))
    check_columns(report, interactions, INTERACTIONS_SCHEMA, "interactions")
    check_domain(report, interactions, "event_type", EVENT_TYPES, "interactions")
    check_no_nulls(report, interactions, "interactions", exempt=("rating", "review_text"))

    timestamps = pd.to_datetime(interactions["timestamp"], errors="coerce")
    report.check(
        timestamps.notna().all(),
        "interactions.timestamp parses on every row",
        "{:,} unparseable".format(int(timestamps.isna().sum())),
    )

    is_rating = interactions["event_type"] == "rating"
    report.check(
        interactions.loc[is_rating, "rating"].notna().all(),
        "every rating event carries a rating value",
        "{:,} rating events with no value".format(
            int(interactions.loc[is_rating, "rating"].isna().sum())),
    )
    report.check(
        interactions.loc[~is_rating, "rating"].isna().all(),
        "no non-rating event carries a rating value",
        "{:,} offenders".format(int(interactions.loc[~is_rating, "rating"].notna().sum())),
    )
    check_range(report, interactions, "rating", 1, 5, "interactions")

    report.check(
        interactions.loc[~is_rating, "review_text"].isna().all(),
        "review_text only appears on rating events",
    )
    report.warn(
        interactions.loc[is_rating, "review_text"].notna().mean() > 0.95,
        "at least 95% of rating events carry review text",
        "{:.1f}% do".format(100 * interactions.loc[is_rating, "review_text"].notna().mean()),
    )

    duplicates = interactions.duplicated(subset=["user_id", "product_id", "event_type", "timestamp"])
    report.check(
        not duplicates.any(),
        "no duplicate (user, product, event, timestamp) rows",
        "{:,} duplicates".format(int(duplicates.sum())),
    )


# ==========================================================================
# The point of this file: referential integrity
# ==========================================================================
def validate_referential_integrity(
    report: Report,
    products: pd.DataFrame,
    users: pd.DataFrame,
    interactions: pd.DataFrame,
) -> None:
    section("REFERENTIAL INTEGRITY")

    known_users = set(users["user_id"])
    known_products = set(products["product_id"])

    orphan_users = set(interactions["user_id"]) - known_users
    report.check(
        not orphan_users,
        "every interaction.user_id exists in users.csv",
        sample(sorted(orphan_users)) if orphan_users else "",
    )
    if orphan_users:
        rows = interactions["user_id"].isin(orphan_users).sum()
        log("        {:,} interaction rows reference a user that does not exist".format(rows))

    orphan_products = set(interactions["product_id"]) - known_products
    report.check(
        not orphan_products,
        "every interaction.product_id exists in products.csv",
        sample(sorted(orphan_products)) if orphan_products else "",
    )
    if orphan_products:
        rows = interactions["product_id"].isin(orphan_products).sum()
        log("        {:,} interaction rows reference a product that does not exist".format(rows))

    # Coverage is not an error -- a long tail of untouched products is the
    # point -- but a collapsed catalogue or a wall of silent users is worth
    # knowing about.
    covered_products = interactions["product_id"].nunique()
    covered_users = interactions["user_id"].nunique()
    log("")
    log("  coverage: {:,}/{:,} products ({:.1f}%) and {:,}/{:,} users ({:.1f}%) appear".format(
        covered_products, len(products), 100.0 * covered_products / len(products),
        covered_users, len(users), 100.0 * covered_users / len(users)))
    report.warn(covered_products / len(products) > 0.25,
                "at least a quarter of the catalogue has been interacted with")
    report.warn(covered_users / len(users) > 0.90,
                "at least 90% of users have at least one interaction")


def main() -> int:
    section("VALIDATE PROCESSED DATA")

    require(PRODUCTS_CSV, "src.data.build_products")
    require(USERS_CSV, "src.data.build_users")
    require(INTERACTIONS_CSV, "src.data.build_interactions")

    products = pd.read_csv(PRODUCTS_CSV)
    users = pd.read_csv(USERS_CSV)
    interactions = pd.read_csv(INTERACTIONS_CSV)
    for path, frame in ((PRODUCTS_CSV, products), (USERS_CSV, users), (INTERACTIONS_CSV, interactions)):
        log("  loaded {:<20} {:>8,} rows x {:>2} columns".format(path.name, len(frame), len(frame.columns)))

    report = Report()
    validate_products(report, products)
    validate_users(report, users)
    validate_interactions(report, interactions)
    validate_referential_integrity(report, products, users, interactions)

    section("RESULT")
    log("  {:,} checks passed".format(report.passed))
    if report.warnings:
        log("  {:,} warnings:".format(len(report.warnings)))
        for warning in report.warnings:
            log("    - {}".format(warning))

    if report.failures:
        log("")
        log("!" * 78)
        log("VALIDATION FAILED -- {} broken check(s):".format(len(report.failures)))
        for failure in report.failures:
            log("  * {}".format(failure))
        log("!" * 78)
        return 1

    log("")
    log("  ALL CHECKS PASSED -- products.csv, users.csv and interactions.csv are consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
