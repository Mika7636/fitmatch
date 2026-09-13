"""Build data/processed/interactions.csv.

Run with:  python -m src.data.build_interactions
Requires products.csv and users.csv to exist first.

What makes these interactions non-uniform
-----------------------------------------
Two effects are layered:

1. Popularity. Each product gets a weight of 1/rank^ALPHA from its sales_rank,
   which is a Zipf law -- a small head of products takes a large share of all
   traffic and most of the catalogue sees very little. Reported at the end as
   the share of interactions held by the top 1% / 10% of products, plus a Gini
   coefficient.

2. Fit. On top of popularity each user scores every product on sport match,
   gender target, budget, brand, colour, material, arch support, weather and
   stock, and then samples without replacement from that distribution (Gumbel
   top-k, which is exactly weighted sampling without replacement). So a
   marathon runner on a $60 budget sees mostly running shoes she can afford,
   not a uniform slice of the catalogue.

Ratings and review text come from the Amazon Sports & Outdoors reviews: the
star distribution is the real one (heavily skewed to 5), and each rating event
borrows the text of a real review with that star count, nudged up or down by
how well the product actually fits the user.

One deliberate departure from reality: the cart -> purchase -> review
conversion rates below are more generous than a real storefront's (where a
review follows maybe 5% of purchases). Real rates would leave roughly 1,500
explicit ratings in 100,000 rows, which is too sparse to fit a collaborative
filter on later. Views still dominate the file, as they should.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from src.data.common import (
    INTERACTIONS_CSV,
    PRODUCTS_CSV,
    SEED,
    USERS_CSV,
    find_raw,
    log,
    require,
    section,
    summarize,
    write_csv,
)

TARGET_ROWS = 100_000

# Zipf exponent on sales_rank. Higher = more concentrated head.
POPULARITY_ALPHA = 0.85

# Mean number of distinct products a user engages with. Tuned so that the
# funnel below produces roughly TARGET_ROWS rows.
PRODUCTS_PER_USER_MEAN = 12.6

# Cap on how many review texts to hold per star rating.
TEXTS_PER_STAR = 20_000
REVIEW_TEXT_MAX_CHARS = 600

CATALOGUE_START = pd.Timestamp("2024-03-01")
CATALOGUE_END = pd.Timestamp("2025-09-01")

SCHEMA = ["user_id", "product_id", "event_type", "timestamp", "rating", "review_text"]


# ==========================================================================
# Amazon reviews -- rating distribution and review text
# ==========================================================================
def load_reviews(rng: np.random.Generator):
    path = find_raw("Sports_and_Outdoors*.json", "*sports*outdoor*review*.json", "*review*.json")
    section("AMAZON REVIEWS -- {}".format(path.name))

    stars: Counter = Counter()
    texts: dict[int, list[str]] = defaultdict(list)
    skipped = 0

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            star = int(record.get("overall", 0))
            if not 1 <= star <= 5:
                skipped += 1
                continue
            stars[star] += 1
            if len(texts[star]) < TEXTS_PER_STAR:
                body = str(record.get("reviewText", "")).strip()
                if 20 <= len(body):
                    texts[star].append(" ".join(body.split())[:REVIEW_TEXT_MAX_CHARS])

    total = sum(stars.values())
    log("read {:,} reviews ({:,} unusable lines skipped)".format(total, skipped))
    log("")
    log("real star distribution (used verbatim as the rating prior):")
    for star in range(1, 6):
        share = stars[star] / total
        log("  {} star  {:>8,}  ({:5.2f}%) {}  [{:,} texts kept]".format(
            star, stars[star], 100 * share, "#" * int(share * 60), len(texts[star])))

    star_values = np.array([1, 2, 3, 4, 5])
    star_probs = np.array([stars[s] for s in star_values], dtype=float)
    star_probs /= star_probs.sum()

    # Shuffle each bucket once so texts are handed out in a seeded random order.
    text_pools = {}
    for star in star_values:
        pool = np.array(texts[star], dtype=object)
        rng.shuffle(pool)
        text_pools[int(star)] = pool
    return star_values, star_probs, text_pools


# ==========================================================================
# Affinity scoring
# ==========================================================================
class Catalogue:
    """Product features flattened into numpy arrays for fast per-user scoring."""

    def __init__(self, products: pd.DataFrame):
        self.product_id = products["product_id"].to_numpy()
        self.n = len(products)
        self.price = products["price"].to_numpy(dtype=float)

        self.is_men = products["gender_target"].eq("men").to_numpy()
        self.is_women = products["gender_target"].eq("women").to_numpy()
        self.is_unisex = products["gender_target"].eq("unisex").to_numpy()

        self.sport = pd.Categorical(products["sport_type"])
        self.is_lifestyle = products["sport_type"].eq("lifestyle").to_numpy()
        self.brand = pd.Categorical(products["brand"])
        self.color = pd.Categorical(products["color"])
        self.material = pd.Categorical(products["material"])
        self.season = products["seasonality"].to_numpy()
        self.arch = products["arch_support"].to_numpy()
        self.category = products["category"].to_numpy()
        self.waterproof = products["waterproof"].to_numpy(dtype=bool)
        self.out_of_stock = products["stock_status"].eq("out_of_stock").to_numpy()

        self.is_footwear = products["category"].eq("footwear").to_numpy()
        # "US 10.5" -> 10.5; NaN for apparel and accessories.
        shoe = products["size"].astype(str).str.extract(r"US\s+([\d.]+)")[0]
        self.shoe_size = pd.to_numeric(shoe, errors="coerce").to_numpy(dtype=float)
        self.apparel_size = products["size"].to_numpy()

        # Zipf popularity prior from sales_rank.
        rank = products["sales_rank"].to_numpy(dtype=float)
        self.log_pop = -POPULARITY_ALPHA * np.log(rank)


# Weights applied to each match signal, in log-odds space.
W_SPORT_MATCH = 2.40
W_SPORT_LIFESTYLE = 0.25
W_GENDER_MATCH = 0.90
W_GENDER_MISMATCH = -2.20
W_IN_BUDGET = 0.85
W_BRAND = 0.75
W_COLOR = 0.35
W_MATERIAL = 0.30
W_ARCH = 0.30
W_SIZE_MATCH = 0.55
W_SEASON = 0.25
W_WATERPROOF_OUTDOOR = 0.30
W_OUT_OF_STOCK = -0.60

# climate -> the seasonality that sells there.
CLIMATE_SEASON = {
    "hot": "summer", "tropical": "summer",
    "cold": "winter", "continental": "winter",
    "temperate": "all-season",
}


def score_user(cat: Catalogue, user, brand_codes, sport_code, color_code, material_code):
    """Log-affinity of every product for one user."""
    score = cat.log_pop.copy()

    if sport_code >= 0:
        score += W_SPORT_MATCH * (cat.sport.codes == sport_code)
    score += W_SPORT_LIFESTYLE * cat.is_lifestyle

    wanted = cat.is_men if user.gender == "male" else cat.is_women
    score += W_GENDER_MATCH * (wanted | cat.is_unisex)
    score += W_GENDER_MISMATCH * ~(wanted | cat.is_unisex)

    in_budget = (cat.price >= user.budget_min) & (cat.price <= user.budget_max)
    score += W_IN_BUDGET * in_budget
    # Outside the budget, interest falls off with the size of the overshoot
    # rather than to zero -- people do window-shop above their budget.
    over = np.maximum(cat.price / max(user.budget_max, 1.0), 1.0)
    score -= 1.4 * np.log(over)

    if len(brand_codes):
        score += W_BRAND * np.isin(cat.brand.codes, brand_codes)
    if color_code >= 0:
        score += W_COLOR * (cat.color.codes == color_code)
    if material_code >= 0:
        score += W_MATERIAL * (cat.material.codes == material_code)

    # Low arches want support, high arches want a neutral or cushioned shoe.
    wanted_arch = {"low": "high", "normal": "medium", "high": "low"}[user.foot_arch_type]
    score += W_ARCH * (cat.arch == wanted_arch)

    # Stocked in the user's size.
    size_ok = np.where(
        cat.is_footwear,
        np.abs(cat.shoe_size - user.shoe_size) <= 0.5,
        cat.apparel_size == user.apparel_size,
    )
    score += W_SIZE_MATCH * size_ok

    season = CLIMATE_SEASON.get(user.climate, "all-season")
    score += W_SEASON * ((cat.season == season) | (cat.season == "all-season"))

    if user.indoor_or_outdoor in ("outdoor", "both"):
        score += W_WATERPROOF_OUTDOOR * cat.waterproof

    score += W_OUT_OF_STOCK * cat.out_of_stock
    return score


def top_k_without_replacement(score: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Weighted sampling without replacement via the Gumbel top-k trick."""
    keys = score + rng.gumbel(size=score.shape[0])
    if k >= score.shape[0]:
        return np.argsort(-keys)
    picked = np.argpartition(-keys, k)[:k]
    return picked[np.argsort(-keys[picked])]


# ==========================================================================
# Main
# ==========================================================================
def main() -> None:
    rng = np.random.default_rng(SEED)

    section("BUILD INTERACTIONS  (seed={}, target={:,} rows)".format(SEED, TARGET_ROWS))

    require(PRODUCTS_CSV, "src.data.build_products")
    require(USERS_CSV, "src.data.build_users")
    products = pd.read_csv(PRODUCTS_CSV)
    users = pd.read_csv(USERS_CSV)
    log("loaded {:,} products and {:,} users".format(len(products), len(users)))

    star_values, star_probs, text_pools = load_reviews(rng)

    section("SAMPLING INTERACTIONS")
    cat = Catalogue(products)

    sport_lookup = {value: code for code, value in enumerate(cat.sport.categories)}
    brand_lookup = {value: code for code, value in enumerate(cat.brand.categories)}
    color_lookup = {value: code for code, value in enumerate(cat.color.categories)}
    material_lookup = {value: code for code, value in enumerate(cat.material.categories)}

    # How many distinct products each user engages with: lognormal, so a few
    # heavy users do a lot of the browsing.
    n_users = len(users)
    counts = rng.lognormal(mean=np.log(PRODUCTS_PER_USER_MEAN) - 0.5 * 0.7 ** 2,
                           sigma=0.7, size=n_users)
    counts = np.maximum(1, np.round(counts)).astype(int)
    log("products per user: mean {:.1f}, median {}, max {} (lognormal)".format(
        counts.mean(), int(np.median(counts)), counts.max()))

    # Each user joined at some point in the window and shops from then on.
    span_days = (CATALOGUE_END - CATALOGUE_START).days
    joined = rng.integers(0, int(span_days * 0.85), n_users)

    sessions = []  # one entry per (user, product) engagement
    for i, user in enumerate(users.itertuples(index=False)):
        score = score_user(
            cat,
            user,
            np.array([brand_lookup[b] for b in str(user.preferred_brands).split("|")
                      if b in brand_lookup], dtype=int),
            sport_lookup.get(user.primary_sport, -1),
            color_lookup.get(user.color_preference, -1),
            material_lookup.get(user.material_preference, -1),
        )
        picks = top_k_without_replacement(score, int(counts[i]), rng)

        # Affinity relative to this user's best match, used to drive how far
        # down the funnel each engagement goes and how well it is rated.
        chosen = score[picks]
        affinity = np.exp(np.clip(chosen - chosen.max(), -6, 0))

        offsets = joined[i] + rng.integers(0, max(span_days - joined[i], 1), len(picks))
        for product_idx, aff, day in zip(picks, affinity, offsets):
            sessions.append((i, int(product_idx), float(aff), int(day)))

        if (i + 1) % 1000 == 0:
            log("  scored {:,}/{:,} users".format(i + 1, n_users))

    log("built {:,} user-product engagements".format(len(sessions)))

    # Expand each engagement into a view -> cart -> purchase -> rating funnel,
    # trimming to the row target by dropping whole engagements at random.
    order = rng.permutation(len(sessions))
    user_ids = users["user_id"].to_numpy()
    rows = []
    dropped = 0

    for position in order:
        if len(rows) >= TARGET_ROWS:
            dropped += 1
            continue
        user_idx, product_idx, affinity, day = sessions[position]
        uid = user_ids[user_idx]
        pid = cat.product_id[product_idx]
        oos = bool(cat.out_of_stock[product_idx])

        base = CATALOGUE_START + pd.Timedelta(days=int(day))
        viewed = base + pd.Timedelta(
            hours=int(rng.integers(7, 23)), minutes=int(rng.integers(0, 60))
        )
        rows.append((uid, pid, "view", viewed, np.nan, ""))

        p_cart = (0.28 + 0.30 * affinity) * (0.5 if oos else 1.0)
        if rng.random() >= p_cart:
            continue
        carted = viewed + pd.Timedelta(minutes=int(rng.integers(1, 25)))
        rows.append((uid, pid, "add_to_cart", carted, np.nan, ""))

        p_purchase = 0.0 if oos else 0.40 + 0.30 * affinity
        if rng.random() >= p_purchase:
            continue
        purchased = carted + pd.Timedelta(minutes=int(rng.integers(2, 90)))
        rows.append((uid, pid, "purchase", purchased, np.nan, ""))

        if rng.random() >= 0.85:
            continue
        rated = purchased + pd.Timedelta(
            days=int(rng.integers(1, 22)), hours=int(rng.integers(0, 24))
        )
        star = int(rng.choice(star_values, p=star_probs))
        # A good fit pushes the rating up, a poor one drags it down.
        if affinity > 0.6 and star < 5 and rng.random() < 0.35:
            star += 1
        elif affinity < 0.2 and star > 1 and rng.random() < 0.35:
            star -= 1
        pool = text_pools[star]
        text = str(pool[rng.integers(len(pool))]) if len(pool) else ""
        rows.append((uid, pid, "rating", rated, float(star), text))

    log("dropped {:,} engagements to land on the row target".format(dropped))

    df = pd.DataFrame(rows, columns=SCHEMA)
    df = df.sort_values(["timestamp", "user_id"]).reset_index(drop=True)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df["review_text"] = df["review_text"].replace("", np.nan)
    df["rating"] = df["rating"].astype("Int64")

    write_csv(df, INTERACTIONS_CSV)

    # ----------------------------------------------------------------------
    # Distribution checks
    # ----------------------------------------------------------------------
    summarize(
        df,
        "interactions.csv",
        dist_cols=["event_type", "rating"],
        numeric_cols=["rating"],
    )
    log("")
    log("NOTE: rating and review_text are null on view/add_to_cart/purchase rows")
    log("      by design -- only 'rating' events carry them.")

    section("POPULARITY TAIL")
    per_product = df["product_id"].value_counts()
    covered = per_product.size
    shares = per_product.to_numpy(dtype=float)
    shares = shares / shares.sum()
    for pct in (1, 5, 10, 25, 50):
        head = max(1, int(len(products) * pct / 100))
        log("  top {:>2}% of the catalogue ({:>5,} products) -> {:5.1f}% of interactions".format(
            pct, head, 100.0 * shares[:head].sum()))
    log("  products with at least one interaction: {:,} of {:,} ({:.1f}%)".format(
        covered, len(products), 100.0 * covered / len(products)))

    sorted_counts = np.sort(per_product.to_numpy(dtype=float))
    padded = np.concatenate([np.zeros(len(products) - covered), sorted_counts])
    index = np.arange(1, len(padded) + 1)
    gini = (2 * (index * padded).sum()) / (len(padded) * padded.sum()) - (len(padded) + 1) / len(padded)
    log("  Gini coefficient of product popularity: {:.3f}  (0 = uniform, 1 = winner-take-all)".format(gini))

    section("PER-USER ACTIVITY")
    per_user = df["user_id"].value_counts()
    log("  users with at least one interaction: {:,} of {:,}".format(per_user.size, len(users)))
    log("  interactions per user: min {}, median {:.0f}, mean {:.1f}, max {}".format(
        per_user.min(), per_user.median(), per_user.mean(), per_user.max()))

    section("DOES THE MATCHING ACTUALLY BITE?")
    merged = df.merge(products[["product_id", "sport_type", "gender_target", "price"]],
                      on="product_id", how="left")
    merged = merged.merge(
        users[["user_id", "primary_sport", "gender", "budget_min", "budget_max"]],
        on="user_id", how="left")
    sport_hit = (merged["sport_type"] == merged["primary_sport"]).mean()
    gender_ok = (
        (merged["gender_target"] == "unisex")
        | ((merged["gender"] == "male") & (merged["gender_target"] == "men"))
        | ((merged["gender"] == "female") & (merged["gender_target"] == "women"))
    ).mean()
    in_budget = (
        (merged["price"] >= merged["budget_min"]) & (merged["price"] <= merged["budget_max"])
    ).mean()
    baseline_sport = 1.0 / products["sport_type"].nunique()
    log("  interactions matching the user's primary_sport: {:5.1f}%  "
        "(uniform random would be ~{:.1f}%)".format(100 * sport_hit, 100 * baseline_sport))
    log("  interactions with a compatible gender_target:   {:5.1f}%".format(100 * gender_ok))
    log("  interactions inside the user's budget band:     {:5.1f}%".format(100 * in_budget))

    section("DONE -- interactions")
    log("{:,} interactions written to {}".format(len(df), INTERACTIONS_CSV))


if __name__ == "__main__":
    main()
