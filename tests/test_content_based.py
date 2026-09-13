"""Technique 2 in isolation: Content-Based Filtering, TF-IDF + Cosine.

Nothing in this file imports the knowledge-based module. The ranker is fed
the raw catalogue, or an arbitrary candidate list, never a feasible set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.recommender.content_based import (
    CONTEXT_TOKENS,
    LIKED_RATING,
    STYLE_TOKENS,
    ContentBasedRecommender,
    build_product_documents,
    product_tokens,
)


# ----------------------------------------------------------------------
# Document construction, including the spec (c) rules
# ----------------------------------------------------------------------
def test_cushioning_and_arch_tokens_are_footwear_only(products):
    """Spec (c): 74.7% of products carry cushioning_level 1 and 80.1% carry
    arch_support "medium" only because non-footwear gets a neutral value.
    Emitting those tokens catalogue-wide would flood the vocabulary."""
    footwear = products[products["category"] == "footwear"].iloc[0].to_dict()
    apparel = products[products["category"] == "apparel"].iloc[0].to_dict()
    accessory = products[products["category"] == "accessory"].iloc[0].to_dict()

    footwear_tokens = product_tokens(footwear)
    assert any(token.startswith("cushion_") for token in footwear_tokens)
    assert any(token.startswith("archsupport_") for token in footwear_tokens)

    for row in (apparel, accessory):
        tokens = product_tokens(row)
        assert not any(token.startswith("cushion_") for token in tokens)
        assert not any(token.startswith("archsupport_") for token in tokens)


def test_cushioning_tokens_appear_on_roughly_the_footwear_share(content, products):
    """A sanity check on the above at catalogue scale: the cushioning tokens
    must not be attached to more rows than there is footwear."""
    documents = build_product_documents(products)
    with_cushion = sum(1 for document in documents if "cushion_" in document)
    assert with_cushion == int((products["category"] == "footwear").sum())


def test_waterproof_token_is_emitted_only_when_true(products):
    waterproof = products[products["waterproof"]].iloc[0].to_dict()
    dry = products[~products["waterproof"]].iloc[0].to_dict()
    assert "waterproof" in product_tokens(waterproof)
    assert "waterproof" not in product_tokens(dry)


def test_every_required_field_reaches_the_document(products):
    row = products.iloc[0].to_dict()
    tokens = set(product_tokens(row))
    assert "brand_" + row["brand"].lower() in tokens
    assert "cat_" + row["category"].lower() in tokens
    assert "sport_" + row["sport_type"].lower() in tokens
    assert "material_" + row["material"].lower() in tokens
    assert "color_" + row["color"].lower() in tokens
    assert any(token.startswith("sub_") for token in tokens)
    assert any(token.startswith("breathability_") for token in tokens)
    assert any(token.startswith("season_") for token in tokens)


def test_engineered_tokens_survive_tokenisation(content):
    """Underscored tokens must come back as single terms -- TfidfVectorizer's
    default pattern splits on hyphens, which is why nothing here uses one."""
    vocabulary = set(content.feature_names)
    for token in ("sport_running", "cat_footwear", "color_black",
                  "material_mesh", "season_summer", "cushion_high",
                  "archsupport_low", "breathability_high"):
        assert token in vocabulary, token


# ----------------------------------------------------------------------
# The user vector
# ----------------------------------------------------------------------
def test_user_document_lands_in_the_product_vocabulary(content, users):
    """A user token that no product can emit scores nothing, so coverage is
    the thing to test."""
    for profile in users.head(50).to_dict("records"):
        vector = content.user_vector(profile, allow_history=False)
        assert vector.vector.nnz > 0
        assert vector.in_vocabulary >= 5


def test_style_tokens_all_exist_in_the_catalogue(content):
    """products.csv has no style column, so style_preference is mapped onto
    catalogue words. If that map rots, the field silently scores zero."""
    vocabulary = set(content.feature_names)
    for style, tokens in STYLE_TOKENS.items():
        missing = [token for token in tokens if token not in vocabulary]
        assert not missing, "{}: {} not in vocabulary".format(style, missing)


def test_context_tokens_all_exist_in_the_catalogue(content):
    vocabulary = set(content.feature_names)
    for context, tokens in CONTEXT_TOKENS.items():
        missing = [token for token in tokens if token not in vocabulary]
        assert not missing, "{}: {} not in vocabulary".format(context, missing)


def test_missing_fields_do_not_invent_preferences(content):
    """A near-empty web-form profile must still produce a usable vector,
    without pretending the user said things they did not."""
    sparse = content.user_vector({"primary_sport": "running"}, allow_history=False)
    assert sparse.vector.nnz > 0
    assert "sport_running" in sparse.document
    assert "color_" not in sparse.document
    assert "brand_" not in sparse.document


def test_user_vectors_are_unit_length(content, users):
    for profile in users.head(20).to_dict("records"):
        vector = content.user_vector(profile, allow_history=False).vector
        assert np.isclose(np.sqrt(vector.multiply(vector).sum()), 1.0)


# ----------------------------------------------------------------------
# The two modes
# ----------------------------------------------------------------------
def test_mode_a_is_used_when_there_is_too_little_history(content, users, interactions):
    """Under min_ratings_for_centroid liked items -> profile_only."""
    ratings = interactions[
        (interactions["event_type"] == "rating")
        & (pd.to_numeric(interactions["rating"], errors="coerce") >= LIKED_RATING)
    ]
    counts = ratings.groupby("user_id").size()
    thin = counts[counts < content.min_ratings_for_centroid].index[0]
    profile = users[users["user_id"] == thin].iloc[0].to_dict()
    assert content.user_vector(profile).mode == "profile_only"


def test_mode_b_is_used_when_there_is_enough_history(content, users, interactions):
    ratings = interactions[
        (interactions["event_type"] == "rating")
        & (pd.to_numeric(interactions["rating"], errors="coerce") >= LIKED_RATING)
    ]
    counts = ratings.groupby("user_id").size()
    rich = counts[counts >= content.min_ratings_for_centroid].index[0]
    profile = users[users["user_id"] == rich].iloc[0].to_dict()

    vector = content.user_vector(profile)
    assert vector.mode == "profile_plus_centroid"
    assert vector.n_liked >= content.min_ratings_for_centroid
    assert vector.used_history

    # And it must be suppressible, for cold-start testing and for web forms.
    assert content.user_vector(profile, allow_history=False).mode == "profile_only"


def test_a_model_built_without_interactions_is_always_cold(products, users):
    """This is the session-3 configuration: a web form has no history."""
    model = ContentBasedRecommender(products, interactions=None).fit()
    for profile in users.head(25).to_dict("records"):
        assert model.user_vector(profile).mode == "profile_only"


def test_fallback_rate_matches_the_data(content, users):
    """Session 2 spec asks for this fraction to be reported, not assumed."""
    stats = content.fallback_rate(users["user_id"])
    assert stats["n_users"] == len(users)
    assert stats["n_profile_only"] + stats["n_profile_plus_centroid"] == len(users)
    # 7,939 ratings across 5,000 users: most people cannot reach mode (b).
    assert 0.75 < stats["fallback_rate"] < 0.95


# ----------------------------------------------------------------------
# Ranking
# ----------------------------------------------------------------------
def test_ranking_is_sorted_and_scores_are_cosines(content, users):
    ranked = content.rank(users.iloc[41].to_dict(), top_n=20)
    assert len(ranked) == 20
    assert list(ranked["rank"]) == list(range(1, 21))
    assert ranked["score"].is_monotonic_decreasing
    assert ((ranked["score"] >= -1e-9) & (ranked["score"] <= 1 + 1e-9)).all()


def test_ranking_respects_the_candidate_list(content, products, users):
    candidates = products.sample(200, random_state=42)
    ranked = content.rank(users.iloc[0].to_dict(), candidates, top_n=10)
    assert set(ranked["product_id"]) <= set(candidates["product_id"])


def test_ranking_an_empty_candidate_set_is_empty_not_an_error(content, products, users):
    ranked = content.rank(users.iloc[0].to_dict(), products.head(0), top_n=10)
    assert len(ranked) == 0
    assert list(ranked.columns) == ["product_id", "score", "rank", "mode"]


def test_unknown_product_ids_are_dropped_not_fatal(content, users):
    ranked = content.rank(users.iloc[0].to_dict(), ["NOPE-1", "NOPE-2"], top_n=5)
    assert len(ranked) == 0


def test_the_ranker_prefers_the_right_sport(content, products):
    """The technique has to actually work: a running profile against footwear
    only should surface running or lifestyle shoes, not football boots."""
    footwear = products[products["category"] == "footwear"]
    ranked = content.rank(
        {"primary_sport": "running", "style_preference": "performance"},
        footwear,
        top_n=20,
        allow_history=False,
    )
    sports = products.set_index("product_id").loc[ranked["product_id"], "sport_type"]
    assert (sports == "running").mean() >= 0.5


def test_ties_are_broken_by_product_id(content, products):
    """Two identical documents must not swap places between runs."""
    ranked = content.rank({"primary_sport": "yoga"}, products, top_n=500)
    for score, group in ranked.groupby("score"):
        if len(group) > 1:
            assert list(group["product_id"]) == sorted(group["product_id"])


# ----------------------------------------------------------------------
# Explanations
# ----------------------------------------------------------------------
def test_explain_returns_readable_terms(content, users):
    profile = users.iloc[41].to_dict()
    ranked = content.rank(profile, top_n=1)
    reasons = content.explain(profile, ranked["product_id"].iloc[0])
    assert 1 <= len(reasons) <= 5
    assert all(isinstance(reason, str) and reason for reason in reasons)
    assert not any("_" in reason.split(" (")[0] for reason in reasons), (
        "engineered tokens must be prettified, not shown raw"
    )


def test_explain_on_an_unknown_product_is_empty(content, users):
    assert content.explain(users.iloc[0].to_dict(), "NOT-A-PRODUCT") == []


# ----------------------------------------------------------------------
# Fitting contract
# ----------------------------------------------------------------------
def test_scoring_before_fitting_is_an_error(products):
    model = ContentBasedRecommender(products)
    assert not model.is_fitted
    with pytest.raises(RuntimeError, match="fit"):
        model.rank({"primary_sport": "running"})


def test_rejects_a_nonsense_centroid_weight(products):
    with pytest.raises(ValueError):
        ContentBasedRecommender(products, centroid_weight=1.5)


def test_export_and_attach_round_trip(content, products):
    """The joblib cache in pipeline.py depends on this."""
    fresh = ContentBasedRecommender(products)
    fresh.attach(content.export())
    assert fresh.is_fitted
    profile = {"primary_sport": "tennis", "color_preference": "white"}
    pd.testing.assert_frame_equal(
        fresh.rank(profile, top_n=10), content.rank(profile, top_n=10)
    )


def test_attach_rejects_a_different_catalogue(content, products):
    fresh = ContentBasedRecommender(products.head(100))
    with pytest.raises(ValueError, match="different catalogue"):
        fresh.attach(content.export())
