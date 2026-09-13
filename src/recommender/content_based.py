"""Technique 2 of 2: Content-Based Filtering, TF-IDF + Cosine Similarity (Ch3).

Every product is turned into a bag of words -- its brand, category,
subcategory, sport, material, colour, marketing description, and a handful of
engineered tokens for the numeric attributes.  ``TfidfVectorizer`` weights
those words by how discriminating they are across the 10,000-product
catalogue.  The user is turned into a bag of words in the *same* vocabulary,
and the recommendation score is the cosine of the angle between the two
vectors.  No neighbours, no matrix factorisation, no ratings from other users:
content-based filtering compares a user to item *descriptions*.

This module answers **how admissible products should be ordered**.  It has no
opinion about which products are admissible -- that is Chapter 7's job
(:mod:`src.recommender.knowledge_based`).  The two modules do not import each
other, so either can be tested alone.

Two modes
---------
``profile_only``
    The user vector is built from the stated profile only.  This is the
    cold-start mode, and it is the mode almost everybody gets: only 7,939
    ratings exist across 5,000 users, so **85.2% of users have fewer than
    three ratings of 4 or better** and fall back to it.
    :meth:`ContentBasedRecommender.fallback_rate` re-measures that.

``profile_plus_centroid``
    For users who do have >= ``min_ratings_for_centroid`` ratings of >= 4, the
    profile vector is blended with the L2-normalised centroid of those liked
    items -- the classic content-based user profile of Chapter 3 -- at
    ``centroid_weight``.

Engineered tokens, and where they stop
--------------------------------------
``breathability``, ``waterproof`` and ``seasonality`` produce tokens on every
product.  ``cushioning_level`` and ``arch_support`` produce tokens **only on
footwear**: 74.7% of the catalogue carries ``cushioning_level == 1`` and 80.1%
carries ``arch_support == "medium"`` purely because apparel and accessories
are given a neutral value, so emitting them catalogue-wide would flood the
vocabulary with a token that means "not applicable" (spec (c)).

Circularity warning
-------------------
``build_interactions.py`` scored user-product fit on sport, gender, budget,
brand, colour, material, arch support, size, climate and stock before
sampling.  The user vector below is built from primary_sport, preferred
brands, colour, material, climate and fitness level -- **five of those fields
overlap the generator**.  Measuring this ranker against those interactions in
session 5 therefore partly measures self-consistency with the generator.  What
does *not* overlap: subcategory, the free-text description, breathability,
cushioning level, and product popularity (``sales_rank``), which drove a large
share of the generator's sampling and which this module ignores entirely.
Field-by-field breakdown in ``reports/circularity_note.md``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data.common import INTERACTIONS_CSV, PRODUCTS_CSV
from src.recommender.profiles import normalize_profile

# A user needs this many ratings of >= LIKED_RATING before the centroid mode
# is worth using.  Below it the centroid is one or two items of noise.
MIN_RATINGS_FOR_CENTROID = 3
LIKED_RATING = 4

# How much of the blended user vector comes from the liked-item centroid.
CENTROID_WEIGHT = 0.5

# Repetition counts applied to the user document.  TF-IDF has no other way to
# say "the sport matters more than the climate", and sublinear_tf keeps the
# growth gentle (log(1+n)) rather than linear.
USER_FIELD_WEIGHTS: dict[str, int] = {
    "primary_sport": 3,
    "preferred_brands": 2,
    "style_preference": 2,
    "color_preference": 2,
    "material_preference": 2,
    "indoor_or_outdoor": 1,
    "climate": 1,
    "fitness_level": 1,
}

# `style_preference` has no column of its own in products.csv, so it is mapped
# onto vocabulary that genuinely occurs in the catalogue text -- "og" (467
# descriptions), "everyday" (95), "heritage" (78), "support" (82), "lightweight"
# (41).  Coverage of this map is reported by `python -m src.recommender.audit`
# so the weakness stays visible rather than silently scoring zero.
#
# None of these maps may use a word that is also a `sport_type` value.  An
# earlier version mapped "performance" to ("training", "gym", "court", ...),
# and a *running* profile came back 19/20 training shoes: the style tokens
# outvoted primary_sport, which carries three times the weight but only one
# term.  Style describes how a product looks, not which sport it is for.
STYLE_TOKENS: dict[str, tuple[str, ...]] = {
    "performance": ("performance", "support", "lightweight", "breathability_high"),
    "casual": ("sub_lifestyle", "everyday"),
    "athleisure": ("sub_lifestyle", "everyday", "sub_leggings",
                   "sub_hoodies_and_sweatshirts"),
    "retro": ("retro", "classic", "heritage", "og"),
}

# indoor_or_outdoor -> tokens.  "waterproof" is an engineered product token;
# the rest are description words.
CONTEXT_TOKENS: dict[str, tuple[str, ...]] = {
    "indoor": ("gym", "indoor"),
    "outdoor": ("outdoor", "trail", "waterproof", "sub_outdoor"),
    "both": ("gym", "outdoor", "trail"),
}

# climate -> the seasonality token that suits it, plus a breathability hint.
CLIMATE_TOKENS: dict[str, tuple[str, ...]] = {
    "hot": ("season_summer", "breathability_high", "breathable"),
    "tropical": ("season_summer", "breathability_high", "waterproof"),
    "temperate": ("season_allseason",),
    "continental": ("season_winter", "season_allseason"),
    "cold": ("season_winter", "breathability_low"),
}

# fitness_level -> footwear cushioning and support expectations.  A beginner
# is better served by a cushioned, supportive shoe; an advanced athlete tends
# to want a firmer, lighter one.
FITNESS_TOKENS: dict[str, tuple[str, ...]] = {
    "beginner": ("cushion_high", "archsupport_high", "support"),
    "intermediate": ("cushion_mid", "archsupport_medium"),
    "advanced": ("cushion_low", "lightweight", "performance"),
}

# Prefixes used by the engineered tokens, and how to say them out loud in an
# explanation.
_TOKEN_LABELS: dict[str, str] = {
    "brand": "brand",
    "cat": "category",
    "sub": "subcategory",
    "sport": "sport",
    "color": "colour",
    "material": "material",
    "season": "season",
    "breathability": "breathability",
    "cushion": "cushioning",
    "archsupport": "arch support",
}

_PREFIX_RE = re.compile(r"^([a-z]+)_(.+)$")


def _slug(value: Any) -> str:
    """`"bags-and-backpacks"` -> `"bags_and_backpacks"`.

    TfidfVectorizer's default token pattern splits on hyphens but not on
    underscores, so every engineered token is built with underscores to
    survive tokenisation as one term.
    """
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _band(value: float, low: float, high: float, names: Sequence[str]) -> str:
    """Bucket a 1-5 attribute into low / mid / high."""
    if value <= low:
        return names[0]
    if value >= high:
        return names[2]
    return names[1]


def product_tokens(row: Mapping[str, Any]) -> list[str]:
    """The engineered half of one product's document.

    Structured fields become prefixed tokens (``sport_running``) so that an
    exact field match is a distinct term from the same word appearing in prose.
    The plain word is emitted alongside it so the description text can match
    too.

    ``cushioning_level`` and ``arch_support`` are emitted **only for
    footwear**, because on apparel and accessories they carry a neutral
    placeholder rather than a measurement (spec (c)).
    """
    tokens: list[str] = []
    brand = _slug(row.get("brand"))
    category = _slug(row.get("category"))
    subcategory = _slug(row.get("subcategory"))
    sport = _slug(row.get("sport_type"))
    material = _slug(row.get("material"))
    color = _slug(row.get("color"))

    tokens += ["brand_" + brand, brand]
    tokens += ["cat_" + category, category]
    tokens += ["sub_" + subcategory] + subcategory.split("_")
    tokens += ["sport_" + sport, sport]
    tokens += ["material_" + material, material]
    tokens += ["color_" + color, color]

    breathability = float(row.get("breathability") or 3)
    tokens.append(
        "breathability_"
        + _band(breathability, 2, 4, ("low", "mid", "high"))
    )
    if breathability >= 4:
        tokens.append("breathable")

    if bool(row.get("waterproof")):
        tokens.append("waterproof")

    season = _slug(row.get("seasonality"))          # all-season -> all_season
    tokens.append("season_" + ("allseason" if season == "all_season" else season))

    # Footwear-only tokens.
    if str(row.get("category")).strip().lower() == "footwear":
        cushioning = float(row.get("cushioning_level") or 1)
        tokens.append("cushion_" + _band(cushioning, 2, 4, ("low", "mid", "high")))
        tokens.append("cushion_lvl" + str(int(cushioning)))
        tokens.append("archsupport_" + _slug(row.get("arch_support")))
    return tokens


def build_product_documents(products: pd.DataFrame) -> list[str]:
    """One text document per product, in catalogue row order.

    Content = brand + category + subcategory + sport_type + material + color
    + product_description + the engineered tokens from :func:`product_tokens`.
    """
    descriptions = products["product_description"].fillna("").astype(str).str.lower()
    documents: list[str] = []
    for row, description in zip(products.to_dict("records"), descriptions):
        documents.append(" ".join(product_tokens(row)) + " " + description)
    return documents


@dataclass
class UserVector:
    """A user's TF-IDF vector plus how it was built, so the pipeline can
    report the cold-start fallback instead of hiding it."""

    vector: sp.csr_matrix
    mode: str                 # "profile_only" | "profile_plus_centroid"
    document: str
    n_liked: int
    in_vocabulary: int
    out_of_vocabulary: int

    @property
    def used_history(self) -> bool:
        return self.mode == "profile_plus_centroid"


class ContentBasedRecommender:
    """Content-Based Filtering with TF-IDF + Cosine Similarity (Chapter 3).

    Parameters
    ----------
    products:
        The catalogue.  Documents are built in row order and that order is the
        model's index, so the frame must not be reordered after fitting.
    interactions:
        Optional interaction log.  Only ``event_type == "rating"`` rows with
        ``rating >= LIKED_RATING`` are read, and only to build the per-user
        centroid of mode (b).  Pass ``None`` to force ``profile_only``
        everywhere -- which is what a session-3 web form will do.
    min_ratings_for_centroid:
        Below this many liked items, the centroid is noise and the model falls
        back to ``profile_only``.
    centroid_weight:
        Share of the blended vector taken from the liked-item centroid.

    Notes
    -----
    Fitting is deterministic: ``TfidfVectorizer`` has no random component, so
    the same catalogue always yields the same vocabulary, the same IDF weights
    and the same scores.
    """

    def __init__(
        self,
        products: pd.DataFrame,
        interactions: pd.DataFrame | None = None,
        min_ratings_for_centroid: int = MIN_RATINGS_FOR_CENTROID,
        centroid_weight: float = CENTROID_WEIGHT,
    ) -> None:
        if not 0.0 <= centroid_weight <= 1.0:
            raise ValueError("centroid_weight must be between 0 and 1")
        self.products = products.reset_index(drop=True)
        self.min_ratings_for_centroid = int(min_ratings_for_centroid)
        self.centroid_weight = float(centroid_weight)

        self.product_ids: np.ndarray = self.products["product_id"].to_numpy()
        self._row_of: dict[str, int] = {
            pid: row for row, pid in enumerate(self.product_ids)
        }
        self._liked: dict[str, list[int]] = (
            self._index_liked(interactions) if interactions is not None else {}
        )

        self.vectorizer: TfidfVectorizer | None = None
        self.matrix: sp.csr_matrix | None = None
        self._vocabulary: dict[str, int] = {}

    @classmethod
    def from_csv(
        cls,
        products_path=PRODUCTS_CSV,
        interactions_path=INTERACTIONS_CSV,
        **kwargs: Any,
    ) -> "ContentBasedRecommender":
        """Load catalogue and interactions from disk, then fit."""
        interactions = None
        if interactions_path is not None:
            interactions = pd.read_csv(interactions_path)
        model = cls(pd.read_csv(products_path), interactions, **kwargs)
        return model.fit()

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self) -> "ContentBasedRecommender":
        """Fit the TF-IDF vectorizer over the whole catalogue.

        ``sublinear_tf`` because a description repeating "breathable" three
        times is not three times as breathable; ``min_df=2`` to drop the
        one-off tokens a single product name contributes.
        """
        self.vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            min_df=2,
            stop_words="english",
            dtype=np.float64,
        )
        self.matrix = self.vectorizer.fit_transform(build_product_documents(self.products))
        self._vocabulary = self.vectorizer.vocabulary_
        return self

    @property
    def is_fitted(self) -> bool:
        return self.matrix is not None

    def export(self) -> dict[str, Any]:
        """The fitted state worth caching: the vectorizer and the document
        matrix.  Everything else is cheap to rebuild from the catalogue."""
        self._require_fit()
        return {
            "vectorizer": self.vectorizer,
            "matrix": self.matrix,
            "product_ids": self.product_ids,
        }

    def attach(self, state: Mapping[str, Any]) -> "ContentBasedRecommender":
        """Adopt a previously fitted vectorizer and matrix instead of
        refitting.  Used by the joblib cache in :mod:`src.recommender.pipeline`
        so the session-3 API never rebuilds TF-IDF per request.
        """
        cached_ids = np.asarray(state["product_ids"])
        if len(cached_ids) != len(self.product_ids) or not np.array_equal(
            cached_ids, self.product_ids
        ):
            raise ValueError("cached TF-IDF model was fitted on a different catalogue")
        self.vectorizer = state["vectorizer"]
        self.matrix = state["matrix"]
        self._vocabulary = self.vectorizer.vocabulary_
        return self

    def _require_fit(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("call fit() before scoring")

    @property
    def feature_names(self) -> np.ndarray:
        self._require_fit()
        return self.vectorizer.get_feature_names_out()

    def _index_liked(self, interactions: pd.DataFrame) -> dict[str, list[int]]:
        """user_id -> catalogue row numbers of the items they rated >= 4."""
        ratings = interactions[interactions["event_type"] == "rating"]
        ratings = ratings[pd.to_numeric(ratings["rating"], errors="coerce") >= LIKED_RATING]
        liked: dict[str, list[int]] = {}
        for user_id, product_id in zip(ratings["user_id"], ratings["product_id"]):
            row = self._row_of.get(product_id)
            if row is not None:
                liked.setdefault(user_id, []).append(row)
        return liked

    # ------------------------------------------------------------------
    # The user vector
    # ------------------------------------------------------------------
    def build_user_document(self, profile: Mapping[str, Any]) -> str:
        """Turn a profile into a document in the product vocabulary.

        Every token here is one a product could also emit -- that is the whole
        point of content-based filtering, and it is why ``style_preference``
        has to be mapped onto catalogue words rather than used raw.
        """
        profile = normalize_profile(profile)
        tokens: list[str] = []

        def add(field: str, values: Iterable[str]) -> None:
            weight = USER_FIELD_WEIGHTS.get(field, 1)
            for value in values:
                tokens.extend([value] * weight)

        sport = profile["primary_sport"]
        if sport:
            add("primary_sport", ("sport_" + _slug(sport), _slug(sport)))

        for brand in profile["preferred_brands"]:
            add("preferred_brands", ("brand_" + _slug(brand), _slug(brand)))

        style = profile["style_preference"]
        if style in STYLE_TOKENS:
            add("style_preference", STYLE_TOKENS[style])

        color = profile["color_preference"]
        if color:
            add("color_preference", ("color_" + _slug(color), _slug(color)))

        material = profile["material_preference"]
        if material:
            add("material_preference", ("material_" + _slug(material), _slug(material)))

        context = profile["indoor_or_outdoor"]
        if context in CONTEXT_TOKENS:
            add("indoor_or_outdoor", CONTEXT_TOKENS[context])

        climate = profile["climate"]
        if climate in CLIMATE_TOKENS:
            add("climate", CLIMATE_TOKENS[climate])

        fitness = profile["fitness_level"]
        if fitness in FITNESS_TOKENS:
            add("fitness_level", FITNESS_TOKENS[fitness])

        return " ".join(tokens)

    def user_vector(
        self, profile: Mapping[str, Any], allow_history: bool = True
    ) -> UserVector:
        """Build the user's TF-IDF vector and say which mode produced it.

        Mode (b) -- profile + centroid of items rated >= 4 -- is used when the
        profile carries a ``user_id`` with at least
        ``min_ratings_for_centroid`` such ratings.  Otherwise mode (a),
        profile only.  Pass ``allow_history=False`` to force mode (a), which
        is what a cold-start test or a web-form profile wants.
        """
        self._require_fit()
        profile = normalize_profile(profile)
        document = self.build_user_document(profile)

        tokens = document.split()
        in_vocab = sum(1 for token in set(tokens) if token in self._vocabulary)
        out_vocab = len(set(tokens)) - in_vocab

        profile_vec = self.vectorizer.transform([document])
        profile_vec = _l2_normalize(profile_vec)

        liked_rows = (
            self._liked.get(profile.get("user_id"), []) if allow_history else []
        )
        n_liked = len(liked_rows)
        if n_liked < self.min_ratings_for_centroid:
            return UserVector(profile_vec, "profile_only", document, n_liked,
                              in_vocab, out_vocab)

        centroid = _l2_normalize(
            sp.csr_matrix(self.matrix[liked_rows].mean(axis=0))
        )
        blended = _l2_normalize(
            (1.0 - self.centroid_weight) * profile_vec + self.centroid_weight * centroid
        )
        return UserVector(blended, "profile_plus_centroid", document, n_liked,
                          in_vocab, out_vocab)

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------
    def score(
        self,
        profile: Mapping[str, Any],
        candidates: pd.DataFrame | Sequence[str] | None = None,
        allow_history: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, UserVector]:
        """Cosine similarity of the user vector against the candidates.

        Returns ``(product_ids, scores, user_vector)`` in candidate order.
        ``candidates`` may be a frame with a ``product_id`` column (typically
        the feasible set from Chapter 7), a sequence of product ids, or
        ``None`` for the whole catalogue.
        """
        self._require_fit()
        rows, product_ids = self._candidate_rows(candidates)
        vector = self.user_vector(profile, allow_history=allow_history)
        if len(rows) == 0:
            return product_ids, np.zeros(0, dtype=float), vector
        scores = cosine_similarity(vector.vector, self.matrix[rows]).ravel()
        return product_ids, scores, vector

    def rank(
        self,
        profile: Mapping[str, Any],
        candidates: pd.DataFrame | Sequence[str] | None = None,
        top_n: int = 10,
        allow_history: bool = True,
    ) -> pd.DataFrame:
        """Rank candidates for one user, best first.

        Ties are broken by ``product_id`` so the ordering is total and stable:
        two products with identical documents must not swap places between
        runs.  Columns: ``product_id``, ``score``, ``rank``, ``mode``.
        """
        product_ids, scores, vector = self.score(
            profile, candidates, allow_history=allow_history
        )
        if len(product_ids) == 0:
            return pd.DataFrame(
                {"product_id": [], "score": [], "rank": [], "mode": []}
            )
        # lexsort is last-key-major: sort by product_id, then by -score.
        order = np.lexsort((product_ids, -scores))[: max(int(top_n), 0)]
        return pd.DataFrame(
            {
                "product_id": product_ids[order],
                "score": scores[order],
                "rank": np.arange(1, len(order) + 1),
                "mode": vector.mode,
            }
        )

    def _candidate_rows(
        self, candidates: pd.DataFrame | Sequence[str] | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Candidate product ids -> catalogue row numbers, dropping unknowns."""
        if candidates is None:
            return np.arange(len(self.products)), self.product_ids
        if isinstance(candidates, pd.DataFrame):
            ids = candidates["product_id"].to_numpy()
        else:
            ids = np.asarray(list(candidates))
        rows = np.array([self._row_of[pid] for pid in ids if pid in self._row_of],
                        dtype=int)
        kept = np.array([pid for pid in ids if pid in self._row_of], dtype=object)
        return rows, kept

    # ------------------------------------------------------------------
    # Explanation
    # ------------------------------------------------------------------
    def explain(
        self,
        user: Mapping[str, Any],
        product: Mapping[str, Any] | pd.Series | str,
        top_k: int = 5,
        allow_history: bool = True,
    ) -> list[str]:
        """The TF-IDF terms that contributed most to this pairing, in words.

        Both vectors are L2-normalised, so the elementwise product sums to the
        cosine similarity: each line below is a named slice of the score.
        """
        self._require_fit()
        product_id = (
            product if isinstance(product, str) else str(product["product_id"])
        )
        row = self._row_of.get(product_id)
        if row is None:
            return []

        vector = self.user_vector(user, allow_history=allow_history)
        contribution = vector.vector.multiply(self.matrix[row]).tocoo()
        if contribution.nnz == 0:
            return ["no shared vocabulary between this profile and the product"]

        names = self.feature_names
        pairs = sorted(
            zip(contribution.col, contribution.data), key=lambda p: -p[1]
        )[: max(int(top_k), 0)]
        return [
            "{} (contributes {:.3f} of the {:.3f} similarity)".format(
                _readable_term(names[col]), value, contribution.data.sum()
            )
            for col, value in pairs
        ]

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def fallback_rate(self, user_ids: Iterable[str]) -> dict[str, Any]:
        """What fraction of these users cannot use mode (b)?

        Session-1 data gives 7,939 ratings across 5,000 users, so this is
        expected to be high; the point is that the number is reported rather
        than assumed.
        """
        ids = list(user_ids)
        counts = np.array([len(self._liked.get(uid, [])) for uid in ids])
        eligible = int((counts >= self.min_ratings_for_centroid).sum())
        total = max(len(ids), 1)
        return {
            "n_users": len(ids),
            "n_profile_plus_centroid": eligible,
            "n_profile_only": len(ids) - eligible,
            "fallback_rate": (len(ids) - eligible) / total,
            "min_ratings_for_centroid": self.min_ratings_for_centroid,
            "mean_liked_items": float(counts.mean()) if len(ids) else 0.0,
        }


def _l2_normalize(vector: sp.spmatrix) -> sp.csr_matrix:
    """Scale a sparse row vector to unit length; leave an all-zero row alone."""
    vector = sp.csr_matrix(vector)
    norm = np.sqrt(vector.multiply(vector).sum())
    if norm > 0:
        vector = vector / norm
    return sp.csr_matrix(vector)


def _readable_term(term: str) -> str:
    """`"sport_running"` -> `"sport: running"`; a bare word stays a bare word."""
    match = _PREFIX_RE.match(term)
    if match:
        prefix, rest = match.groups()
        label = _TOKEN_LABELS.get(prefix)
        if label:
            return "{}: {}".format(label, rest.replace("_", " "))
    return "description mentions '{}'".format(term)
