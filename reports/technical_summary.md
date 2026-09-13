# FitMatch — Technical Summary

DS&RS 541 / CSX 4207 term project. A sportswear recommender built from two
techniques on the course list: Content-Based Filtering with TF-IDF and cosine
similarity (Chapter 3), and Knowledge-Based Constraint-Based Recommendation
(Chapter 7). This document states what the system does, why it is built the way
it is, what the evaluation found, and what those findings are not evidence of.

## The problem

Buying sportswear is a constraint-satisfaction problem before it is a taste
problem. A running shoe two sizes too small is not a worse recommendation than
one in the wrong colour; it is not a recommendation at all. The same is true of
an item out of stock, priced past what the shopper will spend, or made for a
different body. Most recommender work optimises a ranking over items assumed to
be broadly acceptable, and that assumption fails here: a catalogue of ten
thousand products typically holds fewer than two thousand a given shopper could
actually buy, and the constraint eliminating the rest is usually price.

FitMatch therefore treats admissibility and desirability as different questions
deserving different machinery. A shopper fills in a profile — body
measurements, sport, training habits, budget, a handful of stated preferences
— and receives ten products, each explained, together with a plain statement
of which preferences the system could not honour. That last part is not a
courtesy: session 2 measured that the median user here has *zero* products
satisfying every preference they stated, so a system that silently returns ten
items is concealing something the shopper needs to know.

## Why these two techniques

The course list offered collaborative filtering, content-based filtering,
knowledge-based and constraint-based methods, and hybrids. The choice was
settled by one measurement rather than by preference: across five thousand
users there are only 7,939 ratings, a mean of 1.29 liked items each, and 4,258
users — 85.2% — have fewer than the three ratings the content model needs
before it will blend past behaviour into a recommendation. Collaborative
filtering has a precondition that this data cannot meet. On the training half
of the evaluation split, 1,820 of 5,000 users have no rating at all, so a
user-based filter has no row to find neighbours for and a factorisation model
has no observations from which to place them in latent space; widen the
requirement only as far as "shares a rated product with at least one other
user" and 2,462 users, 49.2%, still fail it. The item side is worse: only 2,542
of 10,000 products carry any rating, so an item-based filter can never
recommend the remaining 74.6% of the catalogue whatever its quality, and only
771 products have the two or more raters an item-item similarity needs. Matrix
density is 0.011%. Collaborative filtering would not have produced a weaker
ranking here; for a third of users it would have produced no ranking.

Content-based filtering has no such precondition: a product's document is its
own text and a user's document is the form they just filled in. All ten
thousand products are rankable, including the 1,284 nobody has interacted with
— session 2 verified that all 1,010 of those in stock appear in at least one
feasible set. Knowledge-based constraint filtering has no precondition either,
and it is the only one of the four that can express the domain facts that
matter most: a similarity score cannot represent "this shoe does not fit", a
constraint can. The two complement each other exactly — the filter knows what
is admissible and has no opinion about order, the ranker has opinions about
order and no notion of admissibility — so running them in sequence needs no
learned weighting, because neither is asked to overrule the other.

## How the cascade works

A request arrives as a profile dictionary — a row of `users.csv`, or a
partially filled web form. One normaliser decides what every missing field
means, so the two techniques can never disagree about it; an absent field
switches off the constraint that reads it rather than being guessed at, because
a form that skips the colour question must not become a user who wants black.

The Chapter 7 filter runs first over all ten thousand products. Four hard
constraints apply and are never relaxed: price inside the stated budget, a
gender target matching the shopper or marked unisex, stock status of in-stock,
and — for footwear only — a sport type matching the shopper's primary sport or
the sport-agnostic lifestyle tier. Apparel and accessories are treated as
sport-flexible, because a training tee is a reasonable suggestion for a yoga
user even though the catalogue labelled it otherwise; without that concession a
yoga shopper would be choosing from 150 products. Six soft constraints then
apply on top: colour, material, arch support, seasonality against the local
climate, preferred brands, and size. If what remains falls below fifty
products, they are dropped one at a time in a fixed order held in a single
named constant so it can be inspected and tested — colour first because it is
cosmetic, size last because a shoe that does not fit is not a recommendation.
Each drop is recorded with the count before and after, and that log is what the
interface renders. Colour is almost always the casualty: 98.7% of sampled users
lose it, because the catalogue has fourteen colours and the shopper named one.

The Chapter 3 ranker then orders whatever survived. Every product was turned at
fit time into a document — brand, category, subcategory, sport, material,
colour and the free-text description, plus engineered tokens for breathability,
waterproofing, seasonality and, for footwear, cushioning and arch support — and
vectorised into a 1,132-term TF-IDF space. The shopper's profile is turned into
a document in that same vocabulary and scored by cosine similarity. Where the
shopper has at least three ratings of four or better, their vector is blended
half and half with the L2-normalised centroid of those liked items; otherwise
the profile stands alone, which is the path 85.2% of users take. Ties break on
product identifier, so the ordering is total and stable, and neither technique
draws a random number: the same profile yields the same ten products on every
machine and run. Both modules then explain themselves — the filter by listing
the constraints each product satisfies, the ranker by naming the terms that
contributed most to the similarity — and those two streams are what the
results page shows.

## The dataset and its provenance

Three processed files are built from four Kaggle downloads by scripts that
report, per column, how many values came from the source, how many were derived
by rule, and how many were synthesised. `products.csv` holds ten thousand rows: six
thousand Nike and Jordan colourways whose names, prices, sizes and availability
are real, and four thousand Adidas rows. The Adidas source is not a catalogue
at all — it is 9,648 sales transactions across six segments with no product
names — so what is genuine in that half is the per-segment price distribution,
the gender and category split and the sales volume feeding popularity, while
product identities come from a curated line vocabulary chosen so the keyword
rules applied to the real Nike text have something to work on. The build script
says so on stdout every run. `users.csv` holds five thousand rows bootstrapped
from 1,673 cleaned fitness-tracker rows; age, gender, workout type, weekly
frequency and experience level arrive unchanged, while height and BMI are
recalibrated onto gender-specific marginals — rank-preservingly, with weight
recomputed so all three agree — because the source draws them independently
and produces women wearing a US 10 shoe.

`interactions.csv` holds 100,003 events — 62.4% views, then add-to-cart,
purchase and 7,939 ratings — and it is entirely synthetic. It was produced by
scoring every user-product pair on a hand-written fit function and sampling
from it with a Zipf popularity prior. The result is properly non-uniform: the
top 1% of the catalogue takes 26.7% of interactions, the top 10% takes 55.6%,
the Gini coefficient of product popularity is 0.682, and 8,716 of 10,000
products are touched at all. It is also plausibly aligned: 83.5% of
interactions match the user's sport or the lifestyle tier, 93.6% are compatible
on gender and 76.1% fall inside the user's budget. Conversion is deliberately
generous — realistic storefront rates would have left roughly 1,500 ratings,
too few to evaluate against — and the 1,284 products nobody touches are
deliberate cold-start cases.

## What the evaluation found

The split is leave-latest-out by timestamp: each user's most recent 20% of
events is held out, so nothing from a user's future ranks their present, and
the content model is fitted on the training half alone. A held-out item is
relevant when its rating is four or five, leaving 1,742 evaluable users holding
1,988 relevant items. Five configurations were scored at five, ten and twenty:
random, popularity, the constraint filter alone with its feasible set in
catalogue order, the content ranker alone over the unfiltered catalogue, and
the full cascade.

The cascade beats both of its own halves and beats random by a wide margin. At
K=10 it reaches a precision of 0.0024 against 0.0012 for the constraint filter
alone, 0.0009 for the content ranker alone and 0.0002 for random, and its
recall of 0.0202 is roughly double either component's. That ordering is the
result worth having, because the architecture predicts it: filtering and
ranking contribute separately and their combination is worth more than either.
Coverage says the same from the other side — 44.1% of the catalogue at K=10
against the filter's 11.5%, the ranker spreading demand across the feasible set
rather than returning its head.

The popularity baseline beats all of them, at a precision of 0.0187 and a
recall of 0.1702, and this is the single most important number in the report to
read correctly. It is not evidence that popularity is a good recommender; it is
evidence that popularity generated the ground truth. The Zipf prior on
`sales_rank` was the largest term in the interaction generator, the hundred
best-selling products average 241 interactions each against 2.2 for the hundred
worst, and neither FitMatch technique reads `sales_rank` at any point. The
popularity row measures agreement with the generator — which it has by
construction — while reaching 0.1% of the catalogue, the same ten products for
all 1,742 users. `circularity_note.md` predicted this before the evaluation was
written, and it is reported as measured rather than tuned away.

Segmenting by rating count shows the cold-start path degrading gracefully
rather than failing. Grouped by scoring mode, the profile-only path serves
1,536 of the 1,742 evaluated users and reaches 0.95 times the warm path's
recall, 0.75 times its NDCG and 0.65 times its precision. The precision gap is
real and in the expected direction — three ratings do add signal — but read
it against a warm segment of only 206 users, eighteen of whom sit in the
six-or-more bucket where one extra hit moves the average visibly.

## Limitations

The first and largest is circularity, and it undercuts every accuracy number
above. The interaction log was synthesised by a fit function that reads twelve
product fields, and the recommender reads many of the same ones: all ten of the
Chapter 7 constraints correspond to a term in the generator, and five of the
eight Chapter 3 user fields do. Precision and recall measured against that log
therefore quantify how closely two hand-written rule sets agree with one
another, not how well the system would serve a real shopper. The overlap is
enumerated field by field in `circularity_note.md`. Two consequences follow.
Nothing here should be presented as a headline accuracy claim; and the parts of
the evaluation that are *not* circular — catalogue coverage, cold-start reach,
the constraint-satisfaction rate, and the fact that every in-stock
zero-interaction product remains reachable — are the parts that mean what they
appear to mean.

The second is the recall ceiling, which pushes in the opposite direction. The
hard constraints are never relaxed, so they exclude items the user demonstrably
interacted with and rated highly. Across the whole log only 57.2% of
interactions and 60.3% of ratings survive all four; on this evaluation's own
held-out set the measured figure is 61.3%. Recall@K therefore cannot reach 1.0
no matter how good the ranking is, which is why it is reported raw, divided by
that ceiling, and again over only the reachable items. Sharper still: for 625
of the 1,742 evaluable users, *every* relevant held-out item is blocked by a
hard constraint, so their maximum achievable recall is zero for any
configuration that respects those constraints. This is correct behaviour rather
than a defect — an out-of-stock shoe over the shopper's budget is not a
recommendation, whatever the log says — but it means the raw recall numbers
understate the ranker by a large and quantified margin.

Third, the methodology itself had to be adjusted for an artifact of the
generator, which is worth stating rather than hiding. The usual convention of
excluding items a user has already seen is wrong for this log: the generator
emits a funnel — view, add to cart, purchase, rating — over a single product,
so the earlier events land in training while the rating lands in the held-out
half. Applying the convention removes 69.4% of all relevant held-out items and
drives every configuration to a precision of exactly zero. Training items are
therefore not excluded; `--exclude-seen` reproduces the other convention.

Fourth, the labels are thin and lopsided. Most evaluable users have exactly one
relevant held-out item, so recall at K is nearly a hit rate and every average
is sensitive to single hits; relevance is binary, so NDCG does not distinguish
a five-star item from a four-star one; and the cold-start share among evaluable
users is 88.2%, above the population's 85.2%, precisely because
leave-latest-out removes the recent ratings that would have qualified them.

Finally, three weaknesses sit in the system rather than in its measurement.
`style_preference` has no catalogue column and is mapped onto description
keywords — a map that once made a running profile return nineteen training
shoes out of twenty because style tokens outvoted sport; no keyword may now
reuse a `sport_type` value, but it remains the weakest field in the user
vector. Product size is one value per row rather than a stocked run, which is
why size is a tolerant soft constraint rather than a hard one. And the cascade
is one-directional: the filter cannot reconsider once the ranker has seen the
result, so a shopper whose budget admits fifty products gets a ranking over
fifty products, with no mechanism for suggesting that a slightly higher budget
would open the catalogue considerably.

---

*Every number in this document is reproduced by `python -m src.evaluation.evaluate`,
`python -m src.evaluation.cold_start` and `python -m src.recommender.audit`.
See also `reports/circularity_note.md` and `reports/feasibility_report.md`.*
