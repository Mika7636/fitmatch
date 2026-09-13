# Circularity between the interaction generator and the recommender

**Short version.** `data/processed/interactions.csv` is synthetic. It was
produced by `src/data/build_interactions.py`, which scored every
(user, product) pair on a hand-written fit function and then sampled from that
score. The session-2 recommender scores (user, product) pairs on a *different*
hand-written function that reads **many of the same fields**. Evaluating the
recommender against those interactions in session 5 therefore measures, in
part, how closely the recommender's rules agree with the generator's rules —
not how good the recommendations are.

This note exists so that overlap is stated up front rather than discovered by
a marker. It is not tuned around: the recommender was written to be a sensible
recommender, and the overlap is reported as-is.

---

## 1. What the generator used

`build_interactions.py` builds a log-affinity score per (user, product) and
draws from it with Gumbel top-k (weighted sampling without replacement). The
score is a Zipf popularity prior plus weighted fit terms:

| generator term | source field(s) | weight |
| --- | --- | ---: |
| Zipf popularity prior | `products.sales_rank` | `-0.85 * log(rank)` |
| sport match | `sport_type` vs `primary_sport` | `+2.40` |
| lifestyle bonus | `sport_type == "lifestyle"` | `+0.25` |
| gender match / mismatch | `gender_target` vs `gender` | `+0.90` / `-2.20` |
| inside budget | `price` vs `budget_min/max` | `+0.85` |
| budget overshoot decay | `price` vs `budget_max` | `-1.4 * log(overshoot)` |
| preferred brand | `brand` vs `preferred_brands` | `+0.75` |
| colour | `color` vs `color_preference` | `+0.35` |
| material | `material` vs `material_preference` | `+0.30` |
| arch support | `arch_support` vs `foot_arch_type` | `+0.30` |
| size in stock | `size` vs `shoe_size` / `apparel_size` | `+0.55` |
| seasonality vs climate | `seasonality` vs `climate` | `+0.25` |
| waterproof for outdoor users | `waterproof` vs `indoor_or_outdoor` | `+0.30` |
| out of stock penalty | `stock_status` | `-0.60` |

Twelve product fields in total: `sales_rank`, `sport_type`, `gender_target`,
`price`, `brand`, `color`, `material`, `arch_support`, `size`, `seasonality`,
`waterproof`, `stock_status`.

## 2. What the recommender uses

### Technique 1 — `knowledge_based.py`, constraint filter (Ch7)

| constraint | field | in the generator? |
| --- | --- | --- |
| budget (**hard**) | `price` | **yes** — `+0.85` in-budget, log overshoot decay |
| gender (**hard**) | `gender_target` | **yes** — `+0.90` / `-2.20` |
| stock (**hard**) | `stock_status` | **yes** — `-0.60` out of stock |
| sport, footwear only (**hard**) | `sport_type` | **yes** — `+2.40`, plus lifestyle `+0.25` |
| size (soft) | `size` | **yes** — `+0.55` |
| preferred brands (soft) | `brand` | **yes** — `+0.75` |
| colour (soft) | `color` | **yes** — `+0.35` |
| material (soft) | `material` | **yes** — `+0.30` |
| seasonality vs climate (soft) | `seasonality` | **yes** — `+0.25` |
| arch support (soft) | `arch_support` | **yes** — `+0.30` |

**All ten constraints overlap the generator.** The knowledge-based module is
almost a re-derivation of the generator's fit function, expressed as a filter
instead of a score. This is the strongest circularity in the project, and it is
mostly unavoidable: these are also the fields a real constraint-based
sportswear recommender would use.

The one structural difference is worth noting: the generator treats every term
as a *soft* bonus, so it happily emits an interaction with an out-of-stock,
over-budget, wrong-gender product — just less often. The recommender treats
four of them as *hard*. That difference is measurable, and it cuts against
recall rather than inflating it (see §4).

### Technique 2 — `content_based.py`, TF-IDF + cosine (Ch3)

| user-vector field | product side | in the generator? |
| --- | --- | --- |
| `primary_sport` | `sport_*` token + sport word | **yes** |
| `preferred_brands` | `brand_*` token + brand word | **yes** |
| `color_preference` | `color_*` token + colour word | **yes** |
| `material_preference` | `material_*` token + material word | **yes** |
| `climate` | `season_*`, `breathability_*` tokens | **partly** — seasonality yes, breathability no |
| `indoor_or_outdoor` | `waterproof`, `outdoor`, `trail`, `gym` | **partly** — waterproof yes, the rest no |
| `style_preference` | description words (`og`, `heritage`, `everyday`, …) | **no** |
| `fitness_level` | `cushion_*`, `archsupport_*` tokens | **partly** — arch yes, cushioning no |

**Five of eight user fields overlap** (sport, brand, colour, material, and the
seasonality half of climate).

### What does *not* overlap

These carry real signal in the recommender and had **no** role in generating
the interactions:

* `subcategory` — a `sub_*` token per product; the generator never read it.
* `product_description` — the free-text half of every TF-IDF document, and
  empirically the largest contributor to most similarity scores (see the
  `[Ch3]` explanation lines from the CLI, where description words routinely
  out-contribute the structured tokens).
* `breathability` — 5-level token on every product.
* `cushioning_level` — footwear-only token.
* **`sales_rank` / popularity.** This is the important one. The Zipf
  popularity prior was arguably the generator's *dominant* term — Spearman
  correlation between `sales_rank` and realised interaction count is
  **-0.557**, and the top 1% of the catalogue took 26.7% of all interactions.
  The recommender ignores `sales_rank` completely. Nothing in either technique
  reads it.

## 3. What this means for session 5

Three concrete consequences to put in the write-up:

1. **Precision/recall against `interactions.csv` is inflated by construction**
   on the overlapping fields, and deflated by the popularity term the
   recommender refuses to use. The two biases point in opposite directions,
   which makes the net number hard to interpret in either direction — so do
   not present a single headline accuracy figure as evidence that the
   recommender is good.

2. **A popularity baseline is the honest comparator.** Ranking every feasible
   product by `sales_rank` should score *very well* against this log, because
   popularity is what generated it. If the content-based ranker only matches a
   popularity baseline, that is the expected result, not a failure — and if it
   beats one, check whether the overlapping fields are doing the work before
   claiming a win.

3. **Report the non-overlapping evidence separately.** Coverage, catalogue
   diversity, cold-start reach (all 1,010 in-stock zero-interaction products
   are reachable — see `feasibility_report.md` §5), and constraint-satisfaction
   rate are all measurable *without* the interaction log, and none of them is
   circular. They are the parts of the evaluation that mean what they appear
   to mean.

## 4. The other direction: hard filtering *removes* ground truth

Separate from circularity, and pushing the opposite way. The hard constraints
are never relaxed, so they exclude items the user demonstrably interacted with:

| hard constraint | interactions it excludes | share of 100,003 |
| --- | ---: | ---: |
| budget | 23,900 | 23.9% |
| stock | 16,209 | 16.2% |
| gender | 6,405 | 6.4% |
| sport (footwear only) | 4,246 | 4.2% |

**57.2% of all interactions, and 60.3% of the 7,939 explicit ratings, survive
all four.** So recall@k measured against the raw log has a ceiling near 57%
before the ranker is judged at all.

This is correct behaviour, not a bug. An out-of-stock shoe over the user's
budget is not a recommendation, whatever the historical log says — the log is
generous about these because the generator scored them as soft penalties.
Session 5 should quote recall twice: against all held-out items, and against
held-out items that satisfy the hard constraints.

---

*Numbers here are produced by `python -m src.recommender.audit`, which writes
`reports/feasibility_report.md`. Generator weights are read from
`src/data/build_interactions.py`.*
