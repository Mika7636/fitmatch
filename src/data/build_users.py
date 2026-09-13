"""Build data/processed/users.csv from the Fitness Tracker dataset.

Run with:  python -m src.data.build_users

Source reality check
--------------------
The Fitness Tracker (gym members exercise tracking) dataset holds 1,800 rows of
real-shaped body and training data: age, gender, height, weight, workout type,
weekly frequency and experience level. It is dirty -- roughly 1-4% nulls in
every column and stray escape characters in Workout_Type -- and its own BMI
column disagrees with height/weight, so BMI is recomputed here.

1,800 rows is short of the 5,000 users we want, so the cleaned base is
bootstrap-resampled with small jitter on age/height/weight. That preserves the
joint distribution of the real data (tall people stay heavy, older members
train less) instead of drawing each column independently.

Everything a fitness tracker cannot know -- sizes, budgets, brand and style
preference, location -- is derived from body metrics where a real relationship
exists (shoe and apparel size from height/weight/gender) and synthesised
otherwise. See the provenance table printed at the end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.common import (
    APPAREL_SIZES,
    BRANDS,
    COLORS,
    FITNESS_LEVELS,
    FOOT_ARCH_TYPES,
    LOCATIONS,
    MATERIALS,
    SEED,
    STYLE_PREFERENCES,
    USERS_CSV,
    Provenance,
    find_raw,
    log,
    section,
    summarize,
    write_csv,
)

TARGET_USERS = 5000

SCHEMA = [
    "user_id",
    "age",
    "gender",
    "height_cm",
    "weight_kg",
    "bmi",
    "shoe_size",
    "apparel_size",
    "foot_arch_type",
    "primary_sport",
    "fitness_level",
    "workouts_per_week",
    "indoor_or_outdoor",
    "budget_min",
    "budget_max",
    "preferred_brands",
    "style_preference",
    "color_preference",
    "material_preference",
    "location",
    "climate",
]

# Workout_Type -> distribution over the sports the catalogue actually sells.
SPORT_BY_WORKOUT = {
    "Strength": {"training": 0.52, "basketball": 0.10, "football": 0.09,
                 "lifestyle": 0.10, "running": 0.11, "swimming": 0.04,
                 "skateboarding": 0.04},
    "Cardio": {"running": 0.50, "training": 0.14, "football": 0.09,
               "outdoor": 0.11, "tennis": 0.08, "swimming": 0.06,
               "lifestyle": 0.02},
    "Yoga": {"yoga": 0.58, "training": 0.17, "lifestyle": 0.12,
             "running": 0.08, "outdoor": 0.05},
    "HIIT": {"training": 0.46, "running": 0.24, "basketball": 0.13,
             "football": 0.09, "tennis": 0.05, "outdoor": 0.03},
}

# primary_sport -> where that sport is practised.
VENUE_BY_SPORT = {
    "running": {"outdoor": 0.70, "indoor": 0.08, "both": 0.22},
    "training": {"indoor": 0.68, "outdoor": 0.10, "both": 0.22},
    "yoga": {"indoor": 0.80, "outdoor": 0.05, "both": 0.15},
    "basketball": {"indoor": 0.55, "outdoor": 0.20, "both": 0.25},
    "football": {"outdoor": 0.75, "indoor": 0.10, "both": 0.15},
    "tennis": {"outdoor": 0.60, "indoor": 0.20, "both": 0.20},
    "outdoor": {"outdoor": 0.90, "indoor": 0.02, "both": 0.08},
    "swimming": {"indoor": 0.70, "outdoor": 0.12, "both": 0.18},
    "skateboarding": {"outdoor": 0.80, "indoor": 0.08, "both": 0.12},
    "lifestyle": {"both": 0.50, "indoor": 0.25, "outdoor": 0.25},
}

# primary_sport -> style preference.
STYLE_BY_SPORT = {
    "running": {"performance": 0.72, "athleisure": 0.18, "casual": 0.07, "retro": 0.03},
    "training": {"performance": 0.65, "athleisure": 0.24, "casual": 0.08, "retro": 0.03},
    "yoga": {"athleisure": 0.60, "performance": 0.25, "casual": 0.12, "retro": 0.03},
    "basketball": {"performance": 0.50, "casual": 0.25, "retro": 0.18, "athleisure": 0.07},
    "football": {"performance": 0.68, "casual": 0.18, "retro": 0.09, "athleisure": 0.05},
    "tennis": {"performance": 0.62, "retro": 0.16, "casual": 0.15, "athleisure": 0.07},
    "outdoor": {"performance": 0.70, "casual": 0.18, "athleisure": 0.08, "retro": 0.04},
    "swimming": {"performance": 0.75, "athleisure": 0.12, "casual": 0.10, "retro": 0.03},
    "skateboarding": {"casual": 0.48, "retro": 0.30, "athleisure": 0.14, "performance": 0.08},
    "lifestyle": {"casual": 0.45, "athleisure": 0.28, "retro": 0.22, "performance": 0.05},
}

# fitness_level -> typical spend per item (USD). Advanced athletes replace kit
# more often and buy further up the range.
BUDGET_CENTRE = {"beginner": 65.0, "intermediate": 105.0, "advanced": 165.0}

# The source file draws height, weight and gender independently: men and women
# both average 1.74 m, and 22% of rows land at a BMI that is not survivable
# (min 10.0, max 57.5). Sizing tables built on that produce women wearing a
# US 10 shoe, so height and BMI are recalibrated onto these gender-specific
# marginals (US adult, gym-going population). The recalibration is
# rank-preserving -- the tallest person in the source stays the tallest -- so
# every within-gender ordering in the raw data survives.
HEIGHT_NORM = {"Male": (177.8, 7.2), "Female": (164.2, 6.8)}
BMI_NORM = {"Male": (25.2, 3.6), "Female": (24.0, 4.2)}
BMI_FLOOR, BMI_CEILING = 16.0, 42.0


def rank_map(values: np.ndarray, mean: float, sd: float, rng: np.random.Generator) -> np.ndarray:
    """Map values onto a normal sample of the same size, preserving rank order.

    Avoids needing scipy for an inverse normal CDF, and breaks ties between
    bootstrap duplicates as a side effect.
    """
    draws = np.sort(rng.normal(mean, sd, len(values)))
    return draws[np.argsort(np.argsort(values, kind="stable"), kind="stable")]


def sample_map(rng: np.random.Generator, keys, table: dict) -> np.ndarray:
    """Draw one value per row from a per-key categorical distribution."""
    out = np.empty(len(keys), dtype=object)
    keys = np.asarray(keys)
    for key, dist in table.items():
        mask = keys == key
        count = int(mask.sum())
        if count:
            out[mask] = rng.choice(list(dist), size=count, p=list(dist.values()))
    return out


def weighted_choice(rng: np.random.Generator, options, weights, size):
    weights = np.asarray(weights, dtype=float)
    return rng.choice(options, size=size, p=weights / weights.sum())


# ==========================================================================
# Load and clean the fitness tracker base
# ==========================================================================
def load_base(prov: Provenance) -> pd.DataFrame:
    path = find_raw(
        "gym_members*.csv", "*fitness*tracker*.csv", "*fitness*.csv", "*gym*.csv"
    )
    section("FITNESS TRACKER BASE -- {}".format(path.name))

    raw = pd.read_csv(path)
    log("loaded {:,} rows, {} columns".format(len(raw), len(raw.columns)))
    log("")
    log("nulls in source:")
    for col, count in raw.isna().sum().items():
        if count:
            log("  {:<32}{:>6,}  ({:.1f}%)".format(col, count, 100.0 * count / len(raw)))

    df = pd.DataFrame({
        "age": pd.to_numeric(raw["Age"], errors="coerce"),
        "gender": raw["Gender"].astype("string").str.strip(),
        "weight_kg": pd.to_numeric(raw["Weight (kg)"], errors="coerce"),
        "height_m": pd.to_numeric(raw["Height (m)"], errors="coerce"),
        "workout_type": raw["Workout_Type"].astype("string"),
        "workouts_per_week": pd.to_numeric(raw["Workout_Frequency (days/week)"], errors="coerce"),
        "experience": pd.to_numeric(raw["Experience_Level"], errors="coerce"),
        "source_bmi": pd.to_numeric(raw["BMI"], errors="coerce"),
    })

    # Workout_Type carries literal "\n"/"\t" escape sequences on some rows.
    dirty = df["workout_type"].str.contains(r"\\[nt]|^\s|\s$", regex=True, na=False)
    df["workout_type"] = (
        df["workout_type"].str.replace(r"\\[nt]", "", regex=True).str.strip()
    )
    log("")
    log("cleaned {:,} Workout_Type values containing stray escape characters".format(int(dirty.sum())))

    # Core body fields cannot be imputed sensibly -- drop those rows.
    core = ["age", "gender", "weight_kg", "height_m"]
    before = len(df)
    df = df.dropna(subset=core)
    df = df[df["gender"].isin(["Male", "Female"])]
    log("dropped {:,} rows missing age/gender/height/weight; {:,} remain".format(
        before - len(df), len(df)))

    # The non-core fields are imputed from their own observed distribution.
    rng_impute = np.random.default_rng(SEED)
    for col in ["workout_type", "workouts_per_week", "experience"]:
        missing = df[col].isna()
        if missing.any():
            observed = df.loc[~missing, col].to_numpy()
            df.loc[missing, col] = rng_impute.choice(observed, size=int(missing.sum()))
            log("imputed {:,} missing {} values by resampling the observed distribution".format(
                int(missing.sum()), col))

    df["height_cm"] = df["height_m"] * 100.0
    computed_bmi = df["weight_kg"] / df["height_m"] ** 2
    df["computed_bmi"] = computed_bmi
    disagree = (computed_bmi - df["source_bmi"]).abs() > 1.0
    log("")
    log("source BMI disagrees with weight/height^2 on {:,} of {:,} rows "
        "(mean source {:.1f} vs computed {:.1f}); BMI is recomputed here".format(
            int(disagree.sum()), int(df["source_bmi"].notna().sum()),
            float(df["source_bmi"].mean()), float(computed_bmi.mean())))

    implausible = (computed_bmi < BMI_FLOOR) | (computed_bmi > BMI_CEILING)
    heights = df.groupby("gender")["height_cm"].mean()
    log("")
    log("DATA QUALITY: the source draws height, weight and gender independently.")
    log("  mean height is {:.1f} cm for men and {:.1f} cm for women (real gap: ~13 cm)".format(
        heights.get("Male", float("nan")), heights.get("Female", float("nan"))))
    log("  {:,} of {:,} rows ({:.1f}%) have a BMI outside {:.0f}-{:.0f}".format(
        int(implausible.sum()), len(df), 100.0 * implausible.mean(), BMI_FLOOR, BMI_CEILING))
    log("  -> height and BMI are recalibrated per gender in the bootstrap step;")
    log("     see HEIGHT_NORM / BMI_NORM. Age, gender, workout type, frequency")
    log("     and experience level are used exactly as they come.")

    log("")
    log("cleaned base: {:,} rows".format(len(df)))
    log("  age        {:.0f}-{:.0f} (mean {:.1f})".format(
        df["age"].min(), df["age"].max(), df["age"].mean()))
    log("  height_cm  {:.0f}-{:.0f} (mean {:.1f})".format(
        df["height_cm"].min(), df["height_cm"].max(), df["height_cm"].mean()))
    log("  weight_kg  {:.0f}-{:.0f} (mean {:.1f})".format(
        df["weight_kg"].min(), df["weight_kg"].max(), df["weight_kg"].mean()))
    log("  workout types: {}".format(df["workout_type"].value_counts().to_dict()))
    return df.reset_index(drop=True)


def bootstrap(base: pd.DataFrame, rng: np.random.Generator, prov: Provenance) -> pd.DataFrame:
    """Resample the base to TARGET_USERS rows with small jitter."""
    section("BOOTSTRAP TO {:,} USERS".format(TARGET_USERS))
    idx = rng.integers(0, len(base), TARGET_USERS)
    df = base.iloc[idx].reset_index(drop=True)
    log("resampled {:,} base rows -> {:,} users ({:,} distinct base rows used)".format(
        len(base), TARGET_USERS, len(set(idx.tolist()))))

    df["age"] = np.round(df["age"] + rng.normal(0, 2.0, TARGET_USERS)).clip(18, 70).astype(int)
    log("age: source value + N(0, 2.0 yr) jitter, clipped to 18-70")

    # Rank-preserving recalibration of height and BMI, per gender. Weight then
    # follows from the two, so all three stay mutually consistent.
    height = np.empty(TARGET_USERS)
    bmi = np.empty(TARGET_USERS)
    for gender, group in df.groupby("gender").groups.items():
        rows = np.asarray(group)
        h_mean, h_sd = HEIGHT_NORM[gender]
        b_mean, b_sd = BMI_NORM[gender]
        height[rows] = rank_map(df["height_cm"].to_numpy()[rows], h_mean, h_sd, rng)
        bmi[rows] = rank_map(df["computed_bmi"].to_numpy()[rows], b_mean, b_sd, rng)
        log("{:<7} n={:,}  height -> N({:.1f}, {:.1f})  BMI -> N({:.1f}, {:.1f})".format(
            gender, len(rows), h_mean, h_sd, b_mean, b_sd))

    bmi = np.clip(bmi, BMI_FLOOR, BMI_CEILING)
    df["height_cm"] = np.round(height, 1)
    df["bmi"] = np.round(bmi, 1)
    df["weight_kg"] = np.round(df["bmi"] * (df["height_cm"] / 100.0) ** 2, 1)
    log("weight_kg recomputed as bmi * height_m^2, so the three always agree")

    prov.note("age", "source", TARGET_USERS)
    prov.note("gender", "source", TARGET_USERS)
    prov.note("workouts_per_week", "source", TARGET_USERS)
    prov.note("height_cm", "derived", TARGET_USERS)
    prov.note("bmi", "derived", TARGET_USERS)
    prov.note("weight_kg", "derived", TARGET_USERS)
    return df


# ==========================================================================
# Sizing
# ==========================================================================
def shoe_size_us(height_cm: np.ndarray, bmi: np.ndarray, is_male: np.ndarray,
                 rng: np.random.Generator) -> np.ndarray:
    """US shoe size from height, using the standard height/size regression.

    Anchors: men 175 cm -> US 10, 185 cm -> US 11.5; women 165 cm -> US 8.5,
    175 cm -> US 10.5. Heavier feet run slightly wider and longer, so a high
    BMI nudges the size up half a step.
    """
    size = np.where(is_male, 0.190 * height_cm - 23.7, 0.166 * height_cm - 18.6)
    size = size + np.where(bmi > 30, 0.5, np.where(bmi > 26, 0.25, 0.0))
    size = size + rng.normal(0, 0.35, len(size))  # feet vary at a given height
    size = np.round(size * 2) / 2
    return np.where(is_male, np.clip(size, 7.0, 14.0), np.clip(size, 5.0, 12.0))


# BMI upper bounds per apparel size, by gender.
APPAREL_BANDS_MEN = [(19.5, "XS"), (22.5, "S"), (25.5, "M"), (28.5, "L"), (32.5, "XL")]
APPAREL_BANDS_WOMEN = [(18.5, "XS"), (21.5, "S"), (24.5, "M"), (27.5, "L"), (31.5, "XL")]


def apparel_size(bmi: np.ndarray, height_cm: np.ndarray, is_male: np.ndarray) -> np.ndarray:
    out = np.full(len(bmi), "XXL", dtype=object)
    for i, (value, male) in enumerate(zip(bmi, is_male)):
        bands = APPAREL_BANDS_MEN if male else APPAREL_BANDS_WOMEN
        for limit, letter in bands:
            if value < limit:
                out[i] = letter
                break
    # Very tall people size up a step for length even at the same BMI.
    tall = height_cm > np.where(is_male, 190, 178)
    order = {s: i for i, s in enumerate(APPAREL_SIZES)}
    out[tall] = [APPAREL_SIZES[min(order[s] + 1, len(APPAREL_SIZES) - 1)] for s in out[tall]]
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    prov = Provenance("users.csv")

    section("BUILD USERS  (seed={}, target={:,} rows)".format(SEED, TARGET_USERS))

    base = load_base(prov)
    df = bootstrap(base, rng, prov)
    n = len(df)

    is_male = df["gender"].eq("Male").to_numpy()
    bmi = df["bmi"].to_numpy()
    height = df["height_cm"].to_numpy()

    out = pd.DataFrame(index=df.index)
    out["user_id"] = ["U{:05d}".format(i + 1) for i in range(n)]
    out["age"] = df["age"].to_numpy()
    out["gender"] = np.where(is_male, "male", "female")
    out["height_cm"] = height
    out["weight_kg"] = df["weight_kg"].to_numpy()
    out["bmi"] = bmi

    out["shoe_size"] = shoe_size_us(height, bmi, is_male, rng)
    out["apparel_size"] = apparel_size(bmi, height, is_male)
    prov.note("shoe_size", "derived", n)
    prov.note("apparel_size", "derived", n)

    # Flat feet correlate with a higher BMI; high arches are a bit rarer.
    p_low = np.clip(0.14 + 0.012 * (bmi - 22.0), 0.08, 0.42)
    draw = rng.random(n)
    arch = np.where(draw < p_low, "low", np.where(draw < p_low + 0.62, "normal", "high"))
    out["foot_arch_type"] = arch
    prov.note("foot_arch_type", "derived", n)

    workout = df["workout_type"].astype(str).to_numpy()
    out["primary_sport"] = sample_map(rng, workout, SPORT_BY_WORKOUT)
    prov.note("primary_sport", "derived", n)

    level = pd.Series(df["experience"].round().clip(1, 3).astype(int).to_numpy())
    out["fitness_level"] = level.map(dict(zip([1, 2, 3], FITNESS_LEVELS))).to_numpy()
    prov.note("fitness_level", "source", n)

    out["workouts_per_week"] = df["workouts_per_week"].round().clip(0, 7).astype(int).to_numpy()

    out["indoor_or_outdoor"] = sample_map(rng, out["primary_sport"].to_numpy(), VENUE_BY_SPORT)
    prov.note("indoor_or_outdoor", "derived", n)

    # Budget: centred on fitness level, widened by a lognormal so the range
    # covers the catalogue's whole price span.
    centre = pd.Series(out["fitness_level"]).map(BUDGET_CENTRE).to_numpy()
    budget_max = centre * rng.lognormal(0.0, 0.42, n)
    # Under-25s and over-55s skew lower.
    age = out["age"].to_numpy()
    budget_max = budget_max * np.where((age < 25) | (age > 55), 0.82, 1.0)
    budget_max = np.round(np.clip(budget_max, 35, 450)).astype(int)
    budget_min = np.round(budget_max * rng.uniform(0.15, 0.45, n)).astype(int)
    out["budget_min"] = np.clip(budget_min, 10, None)
    out["budget_max"] = budget_max
    prov.note("budget_min", "synthetic", n)
    prov.note("budget_max", "synthetic", n)

    # 1-3 brands, order preserved, pipe-separated.
    brand_weights = np.array([0.50, 0.35, 0.15])
    counts = rng.choice([1, 2, 3], size=n, p=[0.45, 0.40, 0.15])
    preferred = []
    for count in counts:
        picked = rng.choice(BRANDS, size=int(count), replace=False, p=brand_weights)
        preferred.append("|".join(picked))
    out["preferred_brands"] = preferred
    prov.note("preferred_brands", "synthetic", n)

    out["style_preference"] = sample_map(rng, out["primary_sport"].to_numpy(), STYLE_BY_SPORT)
    prov.note("style_preference", "derived", n)

    out["color_preference"] = weighted_choice(
        rng, COLORS,
        [0.22, 0.12, 0.10, 0.10, 0.07, 0.07, 0.05, 0.06, 0.03, 0.03, 0.03, 0.03, 0.03, 0.06],
        n,
    )
    out["material_preference"] = weighted_choice(
        rng, MATERIALS,
        [0.18, 0.10, 0.10, 0.14, 0.14, 0.14, 0.07, 0.05, 0.05, 0.03],
        n,
    )
    prov.note("color_preference", "synthetic", n)
    prov.note("material_preference", "synthetic", n)

    cities = list(LOCATIONS)
    picked_cities = rng.choice(cities, size=n)
    out["location"] = ["{}, {}".format(c, LOCATIONS[c][0]) for c in picked_cities]
    out["climate"] = [LOCATIONS[c][1] for c in picked_cities]
    prov.note("location", "synthetic", n)
    prov.note("climate", "derived", n)

    assert out["user_id"].is_unique, "user_id collision"
    assert (out["budget_min"] < out["budget_max"]).all(), "budget_min >= budget_max"
    out = out[SCHEMA]

    write_csv(out, USERS_CSV)

    summarize(
        out,
        "users.csv",
        dist_cols=["gender", "primary_sport", "fitness_level", "apparel_size",
                   "shoe_size", "foot_arch_type", "indoor_or_outdoor",
                   "style_preference", "preferred_brands", "color_preference",
                   "material_preference", "climate", "workouts_per_week"],
        numeric_cols=["age", "height_cm", "weight_kg", "bmi", "shoe_size",
                      "workouts_per_week", "budget_min", "budget_max"],
    )
    prov.report()

    section("DONE -- users")
    log("{:,} users written to {}".format(len(out), USERS_CSV))


if __name__ == "__main__":
    main()
