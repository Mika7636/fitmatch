"""Build data/processed/products.csv from the Nike and Adidas raw datasets.

Run with:  python -m src.data.build_products

Source reality check
--------------------
The two sources are shaped very differently:

* Nike Sportswear Product Dataset -- a real product catalogue (~229k size
  variants). Rich text (PRODUCT_NAME / TITLE), real prices, real colourways,
  real availability. Almost every catalogue field comes from here directly or
  is derived from the product text by keyword rule.

* Adidas US Sales Dataset -- NOT a catalogue. It is 9,648 sales transactions
  covering only six product segments ("Men's Street Footwear", "Women's
  Apparel", ...). It carries no product names, so product identities cannot be
  read out of it. What is real in it, and what this script uses, is the
  per-segment price distribution, the per-segment gender/category split and the
  units-sold volume that drives popularity. Product line names are drawn from a
  curated Adidas vocabulary so that the same keyword rules used on Nike text
  have something to bite on.

Every column reports whether its values were taken from source, derived by rule
or synthesised; see the provenance table printed at the end.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.data.common import (
    APPAREL_SIZES,
    GBP_TO_USD,
    PRODUCTS_CSV,
    SEED,
    Provenance,
    find_raw,
    log,
    section,
    summarize,
    write_csv,
)

# Roughly 10,000 products, split between the two brands.
NIKE_TARGET = 6000
ADIDAS_TARGET = 4000

SCHEMA = [
    "product_id",
    "brand",
    "category",
    "subcategory",
    "sport_type",
    "gender_target",
    "price",
    "discount",
    "stock_status",
    "size",
    "color",
    "material",
    "weight",
    "cushioning_level",
    "arch_support",
    "breathability",
    "waterproof",
    "seasonality",
    "avg_rating",
    "review_count",
    "sales_rank",
    "product_description",
]


# ==========================================================================
# Generic rule engine
# ==========================================================================
def apply_rules(blob: pd.Series, rules, fallback, prov: Provenance, field: str):
    """Apply (regex, value) rules in priority order to a text blob.

    Rows matched by a rule are tagged 'derived'; rows that fall through are
    filled by `fallback(n)` and tagged 'synthetic'.
    """
    values = pd.Series(np.full(len(blob), None, dtype=object), index=blob.index)
    for pattern, value in rules:
        unset = values.isna()
        if not unset.any():
            break
        hit = unset & blob.str.contains(pattern, case=False, regex=True, na=False)
        values[hit] = value
    missing = values.isna()
    flags = np.where(missing.to_numpy(), "synthetic", "derived")
    if missing.any():
        values[missing] = fallback(int(missing.sum()))
    prov.flags(field, flags)
    return values


def weighted_choice(rng: np.random.Generator, options, weights, size):
    weights = np.asarray(weights, dtype=float)
    return rng.choice(options, size=size, p=weights / weights.sum())


# ==========================================================================
# Keyword rules -- shared by both brands, they run on "name + title" text
# ==========================================================================
SPORT_RULES = [
    (r"terrex|trail|hike|hiking|\bacg\b|gore-?tex|all condition", "outdoor"),
    (r"basketball|jordan|lebron|kyrie|\bkd\b|harden|\bdame\b|trae|d\.o\.n|giannis", "basketball"),
    (r"football|soccer|mercurial|phantom|tiempo|predator|copa|crazyfast|jersey|\bkits?\b|\bfc\b|f\.c\.|united|saint-germain|academy|squadra|entrada|tiro", "football"),
    (r"tennis|barricade|ubersonic|vapor pro|nikecourt|court shoe", "tennis"),
    (r"yoga|studio|pilates|barre", "yoga"),
    (r"swim|aqua|slide|board short|water short", "swimming"),
    (r"skate|\bsb\b|skateboard", "skateboarding"),
    (r"run|pegasus|vaporfly|alphafly|zoomx|infinity|invincible|streakfly|structure|vomero|ultraboost|adizero|supernova|solarglide|solarboost|duramo|runfalcon|singlet", "running"),
    (r"train|gym|metcon|free metcon|powerlift|dropset|adipower|techfit|pro dri-?fit|weightlift|hiit|crossfit", "training"),
    (r"sportswear|club|essentials|heritage|lifestyle|stan smith|samba|gazelle|superstar|campus|forum|nmd|ozweego|air force|dunk|blazer|air max|continental|grand court", "lifestyle"),
]

MATERIAL_RULES = [
    (r"gore-?tex|rain\.rdy|storm-?fit", "gore-tex"),
    (r"flyknit|knit|primeknit", "knit"),
    (r"fleece|sherpa|therma|winterized", "fleece"),
    (r"leather|air force|stan smith|superstar|forum|blazer|premium", "leather"),
    (r"dri-?fit|aeroready|climacool|polyester|tech pack|repel|windrunner", "polyester"),
    (r"nylon|woven|packable|windbreaker|track jacket", "nylon"),
    (r"cotton|jersey tee|t-shirt|\btee\b|hoodie|crew", "cotton"),
    (r"mesh|breathe|air zoom|vaporfly|pegasus|adizero|ultraboost", "mesh"),
    (r"sock|slide|sandal|rubber|\bball\b", "rubber"),
]

# Cushioning 1-5. Max-stack daily trainers and Boost/Air Max at the top,
# flat court shoes and racing spikes at the bottom.
CUSHION_RULES = [
    (r"invincible|infinity|air max|ultraboost|zoomx|vaporfly|alphafly|solarboost|max cushion", 5),
    (r"air zoom|pegasus|react|boost|zoom |supernova|solarglide|renew|winflo|vomero|structure|adizero boston", 4),
    (r"stan smith|samba|gazelle|superstar|campus|forum|blazer|dunk|air force|continental|grand court|slide|sandal|\bflat\b", 1),
    (r"\bfree\b|adios pro|streakfly|racing|spike|metcon|powerlift|adipower|dropset|barricade|ubersonic|nikecourt", 2),
    (r"duramo|runfalcon|grand court|revolution|downshifter|flex|legend|precision", 3),
]

ARCH_RULES = [
    (r"structure|vomero|supernova|solarglide|stability|arch support|motion control|guiderails|terrex|hike", "high"),
    (r"\bfree\b|racing|adios pro|streakfly|vaporfly|minimal|\bflat\b|slide|sandal|samba|gazelle|stan smith", "low"),
    (r"pegasus|invincible|infinity|ultraboost|air zoom|boost|react|duramo|solarboost", "medium"),
]

# Breathability 1-5.
BREATHABILITY_RULES = [
    (r"fleece|therma|winterized|sherpa|cold\.rdy|\bdown\b|puffer|insulated|parka", 1),
    (r"gore-?tex|rain\.rdy|storm-?fit|windrunner|shield|repel|jacket|waterproof", 2),
    (r"dri-?fit adv|aeroready|climacool|breathe|mesh|flyknit|singlet|tank|\bvent\b", 5),
    (r"dri-?fit|primeblue|air zoom|adizero|running short|\btee\b|t-shirt|polo", 4),
    (r"hoodie|sweatshirt|jogger|track pant|tights|leggings|leather", 3),
]

WATERPROOF_RULES = [
    (r"gore-?tex|rain\.rdy|waterproof|storm-?fit|shield|\bacg\b|free hiker|terrex swift", True),
    (r"repel|water-?repellent|\bdwr\b|windrunner|\brain\b", True),
    (r"\btee\b|t-shirt|tank|singlet|short|sock|mesh|slide|knit|fleece|hoodie|legging|tight|jersey"
     r"|shoe|trainer|sneaker|boot|cleat|\bbra\b|\bcap\b|\bhat\b|bag|backpack|polo|dress|skirt", False),
]

SEASON_RULES = [
    (r"therma|fleece|sherpa|winterized|cold\.rdy|\bdown\b|puffer|insulated|parka|beanie|gore-?tex|winter", "winter"),
    (r"tank|singlet|swim|aqua|slide|sandal|board short|breathe|climacool|running short|polo|skirt|dress", "summer"),
]

# Keywords that legitimately move a product's weight away from its category base.
LIGHT_RE = r"racing|adios pro|vaporfly|streakfly|singlet|tank|\bshort\b|shorts"
HEAVY_RE = r"\bboot\b|boots|hiker|gore-?tex|parka|puffer|fleece|hoodie|jacket|backpack"
TINY_RE = r"sock|\bcap\b|\bhat\b|headband|slide|wristband"

# Nike SUBCATEGORY values that carry no information -- these fall through to
# the keyword rules instead.
GENERIC_SUBCATEGORIES = {"all clothing", "all shoes", "all accessories and equipment", "nike by you"}


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text).strip().lower()).strip("-")


# ==========================================================================
# Nike
# ==========================================================================
# Letter-size aliases seen in the Nike data, folded into the six-value
# vocabulary that users.csv also uses.
LETTER_ALIASES = {
    "XXS": "XS", "2XS": "XS",
    "2XL": "XXL", "XXL": "XXL", "3XL": "XXL", "XXXL": "XXL", "4XL": "XXL",
    "0X": "XL", "1X": "XXL", "2X": "XXL", "3X": "XXL", "4X": "XXL",
}
# Waist inches -> letter size, for the trouser rows sized "32" or "32/32".
WAIST_BANDS = [(28, "XS"), (30, "S"), (33, "M"), (36, "L"), (39, "XL")]


def us_shoe_size(uk: float, gender: str) -> str:
    """Nike UK footwear sizes -> US, the convention users.csv is built in."""
    us = uk + (2.5 if gender == "women" else 1.0)
    return "US {:g}".format(round(us * 2) / 2)


def normalize_size(raw: str, category: str, gender: str) -> str:
    """Fold the 224 raw size strings into one small, joinable vocabulary."""
    text = str(raw).strip().upper()
    if text in {"", "NAN", "ONE SIZE", "1SIZE", "PRO", "MISC"}:
        return "one-size"

    # Letter sizes, possibly followed by a fit note: "XS (UK 4-6)", "3X TALL".
    token = re.split(r"[ (/\-]", text)[0]
    token = LETTER_ALIASES.get(token, token)
    if token in APPAREL_SIZES:
        return token

    # Numbers, or a number range like "9-10.5" (take the midpoint).
    numbers = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]
    if not numbers:
        return "one-size"

    if category == "footwear":
        return us_shoe_size(sum(numbers[:2]) / len(numbers[:2]), gender)

    # Apparel sized by waist inches ("32", "32/32"); anything else (cm chest
    # measurements, age codes like "24M") is not a size we can align on.
    waist = numbers[0]
    if 24 <= waist <= 46:
        for limit, letter in WAIST_BANDS:
            if waist < limit:
                return letter
        return "XXL"
    return "one-size"


def normalize_color(raw: str) -> str:
    """Nike colourways look like 'Light Menta/Barely Green'; keep a base colour."""
    text = str(raw).lower()
    if text.count("/") >= 3:
        return "multi"
    primary = text.split("/")[0]
    palette = [
        ("black", "black"), ("white", "white"), ("cream", "beige"), ("sail", "beige"),
        ("bone", "beige"), ("khaki", "beige"), ("beige", "beige"), ("tan", "brown"),
        ("brown", "brown"), ("grey", "grey"), ("gray", "grey"), ("smoke", "grey"),
        ("charcoal", "grey"), ("silver", "grey"), ("platinum", "grey"),
        ("navy", "navy"), ("midnight", "navy"), ("obsidian", "navy"),
        ("blue", "blue"), ("aqua", "blue"), ("teal", "blue"), ("cyan", "blue"),
        ("red", "red"), ("crimson", "red"), ("maroon", "red"), ("burgundy", "red"),
        ("pink", "pink"), ("rose", "pink"), ("fuchsia", "pink"), ("magenta", "pink"),
        ("green", "green"), ("mint", "green"), ("menta", "green"), ("olive", "green"),
        ("lime", "green"), ("orange", "orange"), ("coral", "orange"), ("peach", "orange"),
        ("yellow", "yellow"), ("gold", "yellow"), ("citron", "yellow"), ("lemon", "yellow"),
        ("purple", "purple"), ("violet", "purple"), ("lilac", "purple"), ("grape", "purple"),
    ]
    for needle, base in palette:
        if needle in primary:
            return base
    return "multi"


def load_nike(rng: np.random.Generator, prov: Provenance) -> pd.DataFrame:
    path = find_raw("Nike*.csv", "nike*.csv", "*nike*product*.csv")
    section("NIKE -- {}".format(path.name))

    usecols = [
        "DEPARTMENT", "CATEGORY", "SUBCATEGORY", "SKU", "PRODUCT_NAME", "TITLE",
        "PRODUCT_TYPE", "PRODUCT_SIZE", "LABEL", "IS_BESTSELLER", "COLOR",
        "BRAND", "AVAILABILITY", "PRICE_CURRENT", "PRICE_RETAIL",
    ]
    # The file contains a handful of malformed bytes in the size strings.
    raw = pd.read_csv(path, usecols=usecols, encoding="utf-8", encoding_errors="replace")
    log("loaded {:,} size-variant rows".format(len(raw)))

    kids = raw["DEPARTMENT"].eq("Kids")
    raw = raw[~kids]
    log("dropped {:,} Kids rows (schema only allows men|women|unisex, and the "
        "generated users are all adults)".format(int(kids.sum())))

    raw = raw.dropna(subset=["SKU", "PRODUCT_NAME", "TITLE"])

    # A row is one size of one colourway. Collapse to colourway level: that is
    # the unit a catalogue actually lists.
    n_sku = raw["SKU"].nunique()
    log("{:,} distinct colourways (SKU) available".format(n_sku))
    take = min(NIKE_TARGET, n_sku)
    if take < NIKE_TARGET:
        log("WARNING: only {:,} colourways available, wanted {:,}".format(take, NIKE_TARGET))
    chosen = rng.choice(raw["SKU"].unique(), size=take, replace=False)
    raw = raw[raw["SKU"].isin(set(chosen))]

    grouped = raw.groupby("SKU", sort=False)
    df = grouped.agg(
        department=("DEPARTMENT", "first"),
        nike_category=("CATEGORY", "first"),
        subcategory_raw=("SUBCATEGORY", "first"),
        product_name=("PRODUCT_NAME", "first"),
        title=("TITLE", "first"),
        product_type=("PRODUCT_TYPE", "first"),
        label=("LABEL", "first"),
        bestseller=("IS_BESTSELLER", "max"),
        color_raw=("COLOR", "first"),
        brand_raw=("BRAND", "first"),
        availability=("AVAILABILITY", "first"),
        price_gbp=("PRICE_CURRENT", "median"),
        retail_gbp=("PRICE_RETAIL", "median"),
        sizes=("PRODUCT_SIZE", lambda s: sorted(set(s.astype(str)))),
    ).reset_index()
    log("collapsed to {:,} colourway-level products".format(len(df)))

    out = pd.DataFrame(index=df.index)
    out["product_id"] = "NK" + df["SKU"].astype("int64").astype(str)

    # Jordan is a distinct brand in users' minds; everything else is Nike.
    out["brand"] = np.where(df["brand_raw"].astype(str).str.contains("Jordan"), "Jordan", "Nike")
    prov.note("brand", "source", len(df))

    out["category"] = df["product_type"].map(
        {"FOOTWEAR": "footwear", "APPAREL": "apparel", "EQUIPMENT": "accessory", "ACCESSORIES": "accessory"}
    ).fillna("accessory")
    prov.note("category", "source", len(df))

    # gender_target: DEPARTMENT, overridden when the title says Unisex.
    gender = df["department"].map({"Men": "men", "Women": "women"}).fillna("unisex")
    unisex = df["title"].str.contains("Unisex", case=False, na=False)
    gender[unisex] = "unisex"
    out["gender_target"] = gender.to_numpy()
    prov.note("gender_target", "source", len(df))

    out["price"] = (df["price_gbp"] * GBP_TO_USD).round(2)
    prov.note("price", "source", len(df))
    discount = np.where(
        df["retail_gbp"] > 0,
        (df["retail_gbp"] - df["price_gbp"]) / df["retail_gbp"].replace(0, np.nan),
        0.0,
    )
    out["discount"] = np.clip(np.nan_to_num(discount), 0, 0.9).round(3)
    prov.note("discount", "source", len(df))

    # stock_status: real availability, plus a seeded low_stock slice of the
    # in-stock items (the source has no stock depth).
    status = df["availability"].map(
        {"IN_STOCK": "in_stock", "SOLD_OUT": "out_of_stock", "COMING_SOON": "out_of_stock"}
    ).fillna("in_stock")
    in_stock = status.eq("in_stock").to_numpy()
    low = in_stock & (rng.random(len(df)) < 0.12)
    status[low] = "low_stock"
    out["stock_status"] = status.to_numpy()
    prov.note("stock_status", "source", int((~low).sum()))
    prov.note("stock_status", "synthetic", int(low.sum()))

    # One representative size per colourway, picked from its real size run.
    sizes = []
    for size_list, cat, gen in zip(df["sizes"], out["category"], out["gender_target"]):
        pick = size_list[rng.integers(len(size_list))] if size_list else "one-size"
        sizes.append(normalize_size(pick, cat, gen))
    out["size"] = sizes
    prov.note("size", "source", len(df))

    out["color"] = [normalize_color(c) for c in df["color_raw"]]
    prov.note("color", "source", len(df))

    out["_blob"] = (df["product_name"].astype(str) + " " + df["title"].astype(str)
                    + " " + df["subcategory_raw"].astype(str))
    out["_name"] = df["product_name"].astype(str)
    out["_title"] = df["title"].astype(str)
    out["_bestseller"] = df["bestseller"].astype(str).str.lower().eq("true").to_numpy() | df["label"].eq("BEST_SELLER").to_numpy()

    # subcategory: the source value unless it is one of the generic buckets.
    sub = df["subcategory_raw"].astype(str)
    generic = sub.str.lower().isin(GENERIC_SUBCATEGORIES)
    out["subcategory"] = [slug(s) for s in sub]
    prov.note("subcategory", "source", int((~generic).sum()))
    if generic.any():
        derived_sub = apply_rules(
            out.loc[generic, "_blob"],
            [(r"hoodie|sweatshirt", "hoodies-and-sweatshirts"),
             (r"jacket|gilet|vest|windrunner", "jackets"),
             (r"tee\b|t-shirt|top\b|tank|singlet|polo", "tops-and-t-shirts"),
             (r"short\b|shorts", "shorts"),
             (r"legging|tight", "leggings"),
             (r"trouser|pant|jogger", "trousers"),
             (r"sports bra|bra\b", "sports-bras"),
             (r"jersey|kit\b", "kits-and-jerseys"),
             (r"dress|skirt", "skirts-and-dresses"),
             (r"sock", "socks"),
             (r"bag|backpack|holdall", "bags-and-backpacks"),
             (r"hat|cap|beanie|headband", "hats"),
             (r"tracksuit", "tracksuits"),
             (r"shoe|trainer|sneaker|boot|cleat|slide", "shoes")],
            lambda n: np.where(
                out.loc[generic, "category"].to_numpy()[:n] == "footwear", "shoes", "clothing"
            ),
            prov,
            "subcategory",
        )
        out.loc[generic, "subcategory"] = derived_sub.to_numpy()

    return out


# ==========================================================================
# Adidas
# ==========================================================================
SEGMENT_MAP = {
    "Men's Street Footwear": ("men", "footwear", "street"),
    "Men's Athletic Footwear": ("men", "footwear", "athletic"),
    "Women's Street Footwear": ("women", "footwear", "street"),
    "Women's Athletic Footwear": ("women", "footwear", "athletic"),
    "Men's Apparel": ("men", "apparel", "apparel"),
    "Women's Apparel": ("women", "apparel", "apparel"),
}

# (line name, product type suffix, subcategory)
ADIDAS_ATHLETIC_FOOTWEAR = [
    ("Ultraboost 22", "Running Shoes", "running"),
    ("Ultraboost Light", "Running Shoes", "running"),
    ("Adizero Adios Pro 3", "Racing Shoes", "running"),
    ("Adizero Boston 11", "Running Shoes", "running"),
    ("Adizero SL", "Running Shoes", "running"),
    ("Supernova 2", "Running Shoes", "running"),
    ("Solarglide 6", "Running Shoes", "running"),
    ("Solarboost 4", "Running Shoes", "running"),
    ("Duramo SL", "Running Shoes", "running"),
    ("Runfalcon 3.0", "Running Shoes", "running"),
    ("Terrex Agravic Trail", "Trail Running Shoes", "outdoor"),
    ("Terrex Free Hiker GORE-TEX", "Hiking Shoes", "outdoor"),
    ("Terrex Swift R3 GORE-TEX", "Hiking Shoes", "outdoor"),
    ("Dropset Trainer", "Training Shoes", "training-and-gym"),
    ("Powerlift 5", "Weightlifting Shoes", "training-and-gym"),
    ("Adipower Weightlifting III", "Weightlifting Shoes", "training-and-gym"),
    ("Predator Accuracy", "Firm Ground Boots", "football"),
    ("Copa Pure", "Firm Ground Boots", "football"),
    ("X Crazyfast", "Firm Ground Boots", "football"),
    ("Harden Vol. 7", "Basketball Shoes", "basketball"),
    ("Dame 8", "Basketball Shoes", "basketball"),
    ("Trae Young 3", "Basketball Shoes", "basketball"),
    ("D.O.N. Issue 5", "Basketball Shoes", "basketball"),
    ("Barricade", "Tennis Shoes", "tennis"),
    ("Adizero Ubersonic 4", "Tennis Shoes", "tennis"),
    ("Codechaos", "Golf Shoes", "training-and-gym"),
    ("Adilette Aqua", "Slides", "swimming"),
]
ADIDAS_STREET_FOOTWEAR = [
    ("Stan Smith", "Shoes", "lifestyle"),
    ("Samba OG", "Shoes", "lifestyle"),
    ("Gazelle", "Shoes", "lifestyle"),
    ("Superstar", "Shoes", "lifestyle"),
    ("Campus 00s", "Shoes", "lifestyle"),
    ("Forum Low", "Shoes", "lifestyle"),
    ("NMD_R1", "Shoes", "lifestyle"),
    ("Ozweego", "Shoes", "lifestyle"),
    ("ZX 22 Boost", "Shoes", "lifestyle"),
    ("Continental 80", "Shoes", "lifestyle"),
    ("Grand Court 2.0", "Shoes", "lifestyle"),
]
ADIDAS_APPAREL = [
    ("Tiro 23", "Track Pants", "trousers"),
    ("Own the Run", "Running Jacket", "jackets"),
    ("Own the Run", "Running Tee", "tops-and-t-shirts"),
    ("Aeroready Designed 2 Move", "Training Tee", "tops-and-t-shirts"),
    ("Z.N.E.", "Full-Zip Hoodie", "hoodies-and-sweatshirts"),
    ("Essentials 3-Stripes", "Fleece Hoodie", "hoodies-and-sweatshirts"),
    ("Essentials Fleece", "Joggers", "trousers"),
    ("Techfit", "Training Tights", "leggings"),
    ("Climacool", "Training Tank", "tops-and-t-shirts"),
    ("Terrex Multi", "RAIN.RDY Jacket", "jackets"),
    ("Adizero", "Running Singlet", "tops-and-t-shirts"),
    ("Yoga Studio Luxe", "Training Tights", "leggings"),
    ("Squadra 21", "Football Jersey", "kits-and-jerseys"),
    ("Entrada 22", "Football Shorts", "shorts"),
    ("Primeblue", "Tennis Polo", "tops-and-t-shirts"),
    ("Ultimate365", "Golf Trousers", "trousers"),
    ("Team Issue", "Full-Zip Hoodie", "hoodies-and-sweatshirts"),
    ("Fast Running", "Shorts", "shorts"),
    ("Cold.RDY Training", "Jacket", "jackets"),
    ("Aeroready Train Essentials", "Sports Bra", "sports-bras"),
]
ADIDAS_ACCESSORIES = [
    ("Tiro", "Backpack", "bags-and-backpacks"),
    ("Power VII", "Backpack", "bags-and-backpacks"),
    ("Essentials", "Cap", "hats"),
    ("Cushioned Crew", "Socks 3-Pack", "socks"),
    ("Al Rihla", "Football", "football"),
    ("Training", "Gym Bag", "bags-and-backpacks"),
]

ADIDAS_COLORWAYS = [
    ("Core Black", "black"), ("Cloud White", "white"), ("Grey Three", "grey"),
    ("Legend Ink", "navy"), ("Team Navy Blue", "navy"), ("Blue Rush", "blue"),
    ("Solar Red", "red"), ("Scarlet", "red"), ("Bliss Pink", "pink"),
    ("Green Oxide", "green"), ("Pulse Lime", "green"), ("Solar Orange", "orange"),
    ("Beam Yellow", "yellow"), ("Shadow Violet", "purple"), ("Wonder Beige", "beige"),
    ("Preloved Brown", "brown"), ("Multicolor", "multi"), ("Carbon", "grey"),
]

FOOTWEAR_SIZES_MEN = ["US {:g}".format(s) for s in np.arange(7.0, 14.0, 0.5)]
FOOTWEAR_SIZES_WOMEN = ["US {:g}".format(s) for s in np.arange(5.0, 12.0, 0.5)]


def load_adidas(rng: np.random.Generator, prov: Provenance) -> pd.DataFrame:
    path = find_raw("Adidas*.xlsx", "adidas*.xlsx", "*adidas*sales*.xlsx", "Adidas*.csv")
    section("ADIDAS -- {}".format(path.name))

    if path.suffix == ".xlsx":
        sales = pd.read_excel(path, header=4)
    else:
        sales = pd.read_csv(path, header=4)
    sales = sales.loc[:, ~sales.columns.astype(str).str.startswith("Unnamed")]
    sales = sales.dropna(subset=["Product", "Price per Unit"])
    log("loaded {:,} sales transactions across {} product segments".format(
        len(sales), sales["Product"].nunique()))
    log("NOTE: this source is transactional, not a catalogue. Real signal used: "
        "per-segment price distribution, gender/category split and units-sold "
        "volume. Product identities are synthesised from an Adidas line vocabulary.")

    seg_units = sales.groupby("Product")["Units Sold"].sum()
    seg_prices = {seg: grp["Price per Unit"].to_numpy(dtype=float)
                  for seg, grp in sales.groupby("Product")}
    log("")
    log("units sold per segment (drives how many products each segment gets):")
    for seg, units in seg_units.sort_values(ascending=False).items():
        log("  {:<28}{:>12,.0f} units   median price ${:.2f}".format(
            seg, units, float(np.median(seg_prices[seg]))))

    segments = [s for s in seg_units.index if s in SEGMENT_MAP]
    weights = seg_units.loc[segments].to_numpy(dtype=float)
    picks = weighted_choice(rng, segments, weights, ADIDAS_TARGET)

    rows = []
    for i, seg in enumerate(picks):
        gender, category, kind = SEGMENT_MAP[seg]

        if kind == "athletic":
            name, suffix, sub = ADIDAS_ATHLETIC_FOOTWEAR[rng.integers(len(ADIDAS_ATHLETIC_FOOTWEAR))]
        elif kind == "street":
            name, suffix, sub = ADIDAS_STREET_FOOTWEAR[rng.integers(len(ADIDAS_STREET_FOOTWEAR))]
        elif rng.random() < 0.12:
            category = "accessory"
            name, suffix, sub = ADIDAS_ACCESSORIES[rng.integers(len(ADIDAS_ACCESSORIES))]
        else:
            name, suffix, sub = ADIDAS_APPAREL[rng.integers(len(ADIDAS_APPAREL))]

        colorway, base_color = ADIDAS_COLORWAYS[rng.integers(len(ADIDAS_COLORWAYS))]

        # Price: resample the segment's real per-unit prices, with a modest
        # multiplier so the catalogue is not six discrete price points.
        base_price = float(rng.choice(seg_prices[seg]))
        price = round(base_price * float(rng.uniform(0.75, 1.9)) * (0.55 if category == "accessory" else 1.0), 2)
        price = max(price, 9.99)

        if category == "footwear":
            pool = FOOTWEAR_SIZES_MEN if gender == "men" else FOOTWEAR_SIZES_WOMEN
            size = pool[rng.integers(len(pool))]
        elif category == "accessory":
            size = "one-size"
        else:
            size = APPAREL_SIZES[rng.integers(len(APPAREL_SIZES))]

        gender_out = "unisex" if (category == "accessory" and rng.random() < 0.8) else gender
        title = "{} {}".format(
            {"men": "Men's", "women": "Women's", "unisex": "Unisex"}[gender_out], suffix
        )

        rows.append({
            "product_id": "AD{:05d}".format(i + 1),
            "brand": "Adidas",
            "category": category,
            "subcategory": sub,
            "gender_target": gender_out,
            "price": price,
            "size": size,
            "color": base_color,
            "_blob": "adidas {} {} {}".format(name, suffix, sub),
            "_name": "adidas " + name,
            "_title": title,
            "_colorway": colorway,
            "_segment": seg,
            "_seg_units": float(seg_units[seg]),
        })

    df = pd.DataFrame(rows)

    # Discount and stock depth are absent from a transaction log.
    on_sale = rng.random(len(df)) < 0.35
    df["discount"] = np.where(on_sale, np.round(rng.uniform(0.05, 0.45, len(df)), 3), 0.0)
    df["stock_status"] = weighted_choice(
        rng, ["in_stock", "low_stock", "out_of_stock"], [0.80, 0.13, 0.07], len(df)
    )

    prov.note("brand", "source", len(df))
    prov.note("category", "source", len(df))
    prov.note("subcategory", "derived", len(df))
    prov.note("gender_target", "source", len(df))
    prov.note("price", "derived", len(df))
    prov.note("size", "synthetic", len(df))
    prov.note("color", "synthetic", len(df))
    prov.note("discount", "synthetic", len(df))
    prov.note("stock_status", "synthetic", len(df))

    df["_bestseller"] = df["_seg_units"] >= seg_units.max() * 0.9
    log("")
    log("synthesised {:,} Adidas products".format(len(df)))
    return df


# ==========================================================================
# Shared derived attributes
# ==========================================================================
def derive_attributes(df: pd.DataFrame, rng: np.random.Generator, prov: Provenance) -> pd.DataFrame:
    n = len(df)
    blob = df["_blob"].astype(str)
    footwear = df["category"].eq("footwear").to_numpy()

    df["sport_type"] = apply_rules(
        blob, SPORT_RULES,
        lambda k: weighted_choice(
            rng, ["lifestyle", "running", "training"], [0.5, 0.3, 0.2], k
        ),
        prov, "sport_type",
    )

    df["material"] = apply_rules(
        blob, MATERIAL_RULES,
        lambda k: weighted_choice(
            rng, ["synthetic", "polyester", "cotton"], [0.4, 0.35, 0.25], k
        ),
        prov, "material",
    )

    # Cushioning and arch support only mean anything for footwear. Apparel and
    # accessories take a neutral constant by category rule: cushioning 1 = "no
    # midsole / unpadded", arch_support "medium" = "not applicable".
    shoe_blob = blob[footwear]
    cushion = pd.Series(np.ones(n, dtype=int), index=df.index)
    cushion[footwear] = apply_rules(
        shoe_blob, CUSHION_RULES,
        lambda k: rng.integers(2, 5, k),
        prov, "cushioning_level",
    ).astype(int)
    df["cushioning_level"] = cushion.to_numpy()
    prov.note("cushioning_level", "derived", int((~footwear).sum()))

    arch = pd.Series(np.full(n, "medium", dtype=object), index=df.index)
    arch[footwear] = apply_rules(
        shoe_blob, ARCH_RULES,
        lambda k: weighted_choice(rng, ["low", "medium", "high"], [0.2, 0.55, 0.25], k),
        prov, "arch_support",
    )
    df["arch_support"] = arch.to_numpy()
    prov.note("arch_support", "derived", int((~footwear).sum()))

    df["breathability"] = apply_rules(
        blob, BREATHABILITY_RULES,
        lambda k: rng.integers(2, 5, k),
        prov, "breathability",
    ).astype(int)

    df["waterproof"] = apply_rules(
        blob, WATERPROOF_RULES,
        lambda k: rng.random(k) < 0.08,
        prov, "waterproof",
    ).astype(bool)

    df["seasonality"] = apply_rules(
        blob, SEASON_RULES,
        lambda k: weighted_choice(rng, ["all-season", "summer", "winter"], [0.75, 0.15, 0.10], k),
        prov, "seasonality",
    )

    # Weight in grams: a category/subcategory base with seeded jitter, nudged
    # by keywords that really do imply a lighter or heavier build.
    base = np.select(
        [df["category"].eq("footwear").to_numpy(),
         df["category"].eq("apparel").to_numpy()],
        [290.0, 260.0], default=420.0,
    )
    light = blob.str.contains(LIGHT_RE, case=False, regex=True, na=False).to_numpy()
    heavy = blob.str.contains(HEAVY_RE, case=False, regex=True, na=False).to_numpy()
    tiny = blob.str.contains(TINY_RE, case=False, regex=True, na=False).to_numpy()
    base = np.where(light & ~heavy, base * 0.55, base)
    base = np.where(heavy, base * 1.55, base)
    base = np.where(tiny, base * 0.18, base)
    df["weight"] = np.round(base * rng.normal(1.0, 0.12, n)).clip(30, 2500).astype(int)
    keyword_weight = light | heavy | tiny
    prov.note("weight", "derived", int(keyword_weight.sum()))
    prov.note("weight", "synthetic", int((~keyword_weight).sum()))

    # Ratings and review counts do not exist in either source. Bestsellers get
    # a lift on both, which is what makes the popularity tail believable.
    best = df["_bestseller"].to_numpy(dtype=bool)
    rating = 5.0 - rng.gamma(2.2, 0.30, n)
    rating = np.where(best, rating + 0.25, rating)
    df["avg_rating"] = np.round(np.clip(rating, 1.0, 5.0), 2)

    reviews = rng.lognormal(mean=3.1, sigma=1.35, size=n)
    reviews = np.where(best, reviews * 4.5, reviews)
    df["review_count"] = np.round(reviews).clip(0, 20000).astype(int)
    prov.note("avg_rating", "synthetic", n)
    prov.note("review_count", "synthetic", n)

    # sales_rank: dense rank over a popularity score. 1 = best selling.
    score = (
        np.log1p(df["review_count"].to_numpy()) * 1.0
        + df["avg_rating"].to_numpy() * 0.6
        + best * 2.0
        + df["discount"].to_numpy() * 1.2
        - np.log1p(df["price"].to_numpy()) * 0.15
    )
    df["sales_rank"] = pd.Series(-score).rank(method="first").astype(int).to_numpy()
    prov.note("sales_rank", "derived", n)

    # Description: assembled from the real title/name plus the resolved
    # attributes, so it is usable as content-based text later.
    feature_bits = []
    for cushioning, breath, water, material, season, cat in zip(
        df["cushioning_level"], df["breathability"], df["waterproof"],
        df["material"], df["seasonality"], df["category"],
    ):
        bits = []
        if cat == "footwear":
            bits.append({1: "flat, unpadded", 2: "low-profile", 3: "balanced",
                         4: "well-cushioned", 5: "max-cushioned"}[int(cushioning)] + " midsole")
        bits.append({1: "warm and insulating", 2: "weather-shielding", 3: "moderately breathable",
                     4: "breathable", 5: "highly breathable"}[int(breath)] + " {} build".format(material))
        if water:
            bits.append("waterproof")
        bits.append("{} wear".format(season))
        feature_bits.append(", ".join(bits))

    gender_word = {"men": "Men's", "women": "Women's", "unisex": "Unisex"}
    df["product_description"] = [
        "{} {} - {}. {} {} in {}. {}.".format(
            brand, name.replace(brand + " ", "").replace(brand.lower() + " ", ""),
            title, gender_word[gender], sub.replace("-", " "), color, feats.capitalize(),
        )
        for brand, name, title, gender, sub, color, feats in zip(
            df["brand"], df["_name"], df["_title"], df["gender_target"],
            df["subcategory"], df["color"], feature_bits,
        )
    ]
    prov.note("product_description", "derived", n)
    return df


def main() -> None:
    rng = np.random.default_rng(SEED)
    prov = Provenance("products.csv")

    section("BUILD PRODUCTS  (seed={}, target={:,} rows)".format(SEED, NIKE_TARGET + ADIDAS_TARGET))

    nike = load_nike(rng, prov)
    adidas = load_adidas(rng, prov)

    df = pd.concat([nike, adidas], ignore_index=True)
    df = derive_attributes(df, rng, prov)

    assert df["product_id"].is_unique, "product_id collision"
    df = df[SCHEMA]

    write_csv(df, PRODUCTS_CSV)

    summarize(
        df,
        "products.csv",
        dist_cols=[
            "brand", "category", "sport_type", "gender_target", "stock_status", "arch_support", "seasonality", "material", "color",
            "cushioning_level", "breathability", "waterproof", "subcategory", "size",
        ],
        numeric_cols=["price", "discount", "weight", "cushioning_level",
                      "breathability", "avg_rating", "review_count", "sales_rank"],
    )
    prov.report()

    section("DONE -- products")
    log("{:,} products written to {}".format(len(df), PRODUCTS_CSV))


if __name__ == "__main__":
    main()
