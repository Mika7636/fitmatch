"""The cascade end to end, and the six guarantees session 2 is specified on.

These are the tests that matter for the report:

1. no result ever exceeds budget_max, mismatches gender, or is out of stock;
2. every one of the ten primary_sport values returns >= 10 results for a
   median profile;
3. yoga and swimming users -- the thin catalogue segments -- still return 10;
4. the 1,284 zero-interaction products remain reachable;
5. a brand-new profile with no history returns valid results;
6. determinism: the same input gives the same output at seed 42.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data.common import SPORTS
from src.recommender.pipeline import (
    FitMatchRecommender,
    Recommendation,
    _canonical_user_id,
    parse_profile_argument,
    build_parser,
    main,
)
from src.recommender.profiles import GENDER_TO_TARGET

TOP_N = 10


def _assert_hard_guarantees(items: pd.DataFrame, profile: dict) -> None:
    """Guarantee 1, factored out because nearly every test re-checks it."""
    assert (items["price"] <= profile["budget_max"] + 1e-9).all()
    assert (items["price"] >= profile["budget_min"] - 1e-9).all()
    assert (items["stock_status"] == "in_stock").all()
    if profile.get("gender") in GENDER_TO_TARGET:
        allowed = {GENDER_TO_TARGET[profile["gender"]], "unisex"}
        assert set(items["gender_target"]) <= allowed


# ----------------------------------------------------------------------
# 1. Hard guarantees on the final output
# ----------------------------------------------------------------------
@pytest.mark.parametrize("index", [0, 41, 137, 999, 2500, 4321, 4999])
def test_results_never_break_a_hard_constraint(engine, users, products, index):
    profile = users.iloc[index].to_dict()
    result = engine.recommend(profile["user_id"], top_n=TOP_N)

    assert len(result.items) == TOP_N
    _assert_hard_guarantees(result.items, profile)

    stock = products.set_index("product_id").loc[result.product_ids, "stock_status"]
    assert (stock == "in_stock").all()


def test_hard_guarantees_hold_across_a_broad_sweep(engine, users, products):
    """The parametrised cases above are spot checks; this is the sweep."""
    stock = products.set_index("product_id")["stock_status"]
    for profile in users.iloc[::250].to_dict("records"):
        result = engine.recommend(profile["user_id"], top_n=TOP_N, explain=False)
        assert len(result.items) == TOP_N, profile["user_id"]
        _assert_hard_guarantees(result.items, profile)
        assert (stock.loc[result.product_ids] == "in_stock").all()


def test_no_duplicate_products_in_one_result(engine, users):
    for profile in users.head(30).to_dict("records"):
        ids = engine.recommend(profile["user_id"], top_n=TOP_N, explain=False).product_ids
        assert len(ids) == len(set(ids))


# ----------------------------------------------------------------------
# 2. Every sport returns at least ten results
# ----------------------------------------------------------------------
@pytest.mark.parametrize("sport", SPORTS)
def test_every_sport_returns_ten_for_a_median_profile(engine, median_profiles, sport):
    profile = median_profiles[sport]
    result = engine.recommend(profile, top_n=TOP_N)
    assert len(result.items) == TOP_N, "{}: only {} results".format(
        sport, len(result.items)
    )
    _assert_hard_guarantees(result.items, profile)
    assert result.n_feasible >= TOP_N


# ----------------------------------------------------------------------
# 3. The thin catalogue segments
# ----------------------------------------------------------------------
@pytest.mark.parametrize("sport", ["yoga", "swimming"])
def test_thin_segments_still_return_ten(engine, users, sport):
    """744 yoga users face 150 yoga products; 125 swimming users face 109.
    Treating lifestyle as the sport-agnostic tier and apparel as
    sport-flexible is what makes this pass -- spec (b)."""
    segment = users[users["primary_sport"] == sport]
    assert len(segment) > 0
    for profile in segment.head(50).to_dict("records"):
        result = engine.recommend(profile["user_id"], top_n=TOP_N, explain=False)
        assert len(result.items) == TOP_N, "{} user {} got {}".format(
            sport, profile["user_id"], len(result.items)
        )
        _assert_hard_guarantees(result.items, profile)


def test_every_real_user_of_a_thin_sport_is_served(engine, users):
    """Not a sample: every single skateboarding user, the smallest segment."""
    segment = users[users["primary_sport"] == "skateboarding"]
    for profile in segment.to_dict("records"):
        result = engine.recommend(profile["user_id"], top_n=TOP_N, explain=False)
        assert len(result.items) == TOP_N


# ----------------------------------------------------------------------
# 4. Cold products stay reachable through the whole cascade
# ----------------------------------------------------------------------
def test_zero_interaction_products_remain_reachable(engine, products, interactions):
    """Reachable = survives the cascade's constraint stage for somebody. The
    ranker never sees sales_rank, so a cold product competes on its text."""
    touched = set(interactions["product_id"].unique())
    cold = products[~products["product_id"].isin(touched)]
    assert len(cold) == 1284

    cold_in_stock = set(cold.loc[cold["stock_status"] == "in_stock", "product_id"])
    reachable: set[str] = set()
    for sport in SPORTS:
        for gender in ("male", "female"):
            for low, high in ((0, 60), (60, 150), (150, 2000)):
                feasible, _ = engine.knowledge.filter(
                    {"primary_sport": sport, "gender": gender,
                     "budget_min": low, "budget_max": high}
                )
                reachable.update(feasible["product_id"])

    assert cold_in_stock <= reachable, "{} cold products unreachable".format(
        len(cold_in_stock - reachable)
    )


def test_a_cold_product_can_actually_be_recommended(engine, products, interactions):
    """Stronger than reachability: a zero-interaction product must be able to
    win a top-10 slot, not merely survive the filter."""
    touched = set(interactions["product_id"].unique())
    cold = products[
        (~products["product_id"].isin(touched))
        & (products["stock_status"] == "in_stock")
    ]
    target = cold.sort_values("product_id").iloc[0]

    profile = {
        "primary_sport": target["sport_type"],
        "gender": "male" if target["gender_target"] == "men" else "female",
        "budget_min": float(target["price"]) - 1,
        "budget_max": float(target["price"]) + 1,
        "color_preference": target["color"],
        "material_preference": target["material"],
        "preferred_brands": target["brand"],
    }
    ids = engine.recommend(profile, top_n=TOP_N, explain=False).product_ids
    assert target["product_id"] in ids


def test_recommendations_are_not_all_from_the_popular_head(engine, users, interactions):
    """The content ranker ignores sales_rank entirely, so results should not
    collapse onto the interaction head."""
    popularity = interactions["product_id"].value_counts()
    head = set(popularity.head(1000).index)
    recommended: set[str] = set()
    for profile in users.iloc[::200].to_dict("records"):
        recommended.update(
            engine.recommend(profile["user_id"], top_n=TOP_N, explain=False).product_ids
        )
    assert len(recommended - head) > len(recommended) * 0.3


# ----------------------------------------------------------------------
# 5. A brand-new profile with no history
# ----------------------------------------------------------------------
def test_a_brand_new_profile_gets_valid_results(engine):
    """The session-3 web-form case: never seen before, no user_id, no history."""
    profile = {
        "gender": "female",
        "primary_sport": "yoga",
        "shoe_size": 7.5,
        "apparel_size": "S",
        "foot_arch_type": "normal",
        "fitness_level": "beginner",
        "indoor_or_outdoor": "indoor",
        "budget_min": 25,
        "budget_max": 120,
        "preferred_brands": ["Nike", "Adidas"],
        "style_preference": "athleisure",
        "color_preference": "black",
        "material_preference": "cotton",
        "climate": "temperate",
    }
    result = engine.recommend(profile, top_n=TOP_N)

    assert len(result.items) == TOP_N
    assert result.user_id is None
    assert result.content_mode == "profile_only"
    _assert_hard_guarantees(result.items, profile)
    assert all(result.explanations[pid]["constraints"] for pid in result.product_ids)


def test_a_nearly_empty_profile_still_works(engine):
    """Everything optional actually is optional."""
    result = engine.recommend({"primary_sport": "running"}, top_n=TOP_N)
    assert len(result.items) == TOP_N
    assert result.content_mode == "profile_only"


def test_an_empty_profile_still_works(engine):
    result = engine.recommend({}, top_n=TOP_N)
    assert len(result.items) == TOP_N


def test_an_unstated_gender_is_not_silently_a_mens_store(engine):
    result = engine.recommend({"primary_sport": "lifestyle"}, top_n=200, explain=False)
    assert len(set(result.items["gender_target"])) > 1


# ----------------------------------------------------------------------
# 6. Determinism
# ----------------------------------------------------------------------
def test_same_input_same_output(engine, users, seed):
    assert seed == 42
    for profile in users.head(15).to_dict("records"):
        first = engine.recommend(profile["user_id"], top_n=TOP_N, explain=False)
        second = engine.recommend(profile["user_id"], top_n=TOP_N, explain=False)
        pd.testing.assert_frame_equal(first.items, second.items)


def test_deterministic_across_freshly_built_models(products, users, interactions):
    """Two independently constructed engines must agree exactly -- no hidden
    state, no dict-ordering dependence, no random seeding anywhere."""
    left = FitMatchRecommender(products, users, interactions)
    right = FitMatchRecommender(products, users, interactions)
    for user_id in ("U00042", "U01000", "U04999"):
        pd.testing.assert_frame_equal(
            left.recommend(user_id, top_n=TOP_N, explain=False).items,
            right.recommend(user_id, top_n=TOP_N, explain=False).items,
        )


def test_a_shuffled_catalogue_gives_the_same_recommendations(
    products, users, interactions
):
    """Row order in products.csv must not change the answer."""
    baseline = FitMatchRecommender(products, users, interactions)
    shuffled = FitMatchRecommender(
        products.sample(frac=1.0, random_state=42), users, interactions
    )
    for user_id in ("U00042", "U02500"):
        left = baseline.recommend(user_id, top_n=TOP_N, explain=False).items
        right = shuffled.recommend(user_id, top_n=TOP_N, explain=False).items
        assert list(left["product_id"]) == list(right["product_id"])


def test_explanations_are_deterministic(engine, users):
    profile = users.iloc[41].to_dict()
    first = engine.recommend(profile["user_id"], top_n=5)
    second = engine.recommend(profile["user_id"], top_n=5)
    assert first.explanations == second.explanations


# ----------------------------------------------------------------------
# The cascade wiring itself
# ----------------------------------------------------------------------
def test_the_cascade_ranks_only_what_the_filter_admitted(engine, users):
    """Technique 2 must never reach outside technique 1's feasible set."""
    profile = users.iloc[41].to_dict()
    feasible, _ = engine.knowledge.filter(profile)
    result = engine.recommend(profile["user_id"], top_n=TOP_N)
    assert set(result.product_ids) <= set(feasible["product_id"])


def test_both_techniques_explain_every_result(engine, users):
    """The grader has to be able to see both techniques at work per item."""
    result = engine.recommend(users.iloc[41]["user_id"], top_n=5)
    for product_id in result.product_ids:
        why = result.explanations[product_id]
        assert why["constraints"], "Ch7 explanation missing for " + product_id
        assert why["content"], "Ch3 explanation missing for " + product_id


def test_result_carries_the_relaxation_log(engine, users):
    result = engine.recommend(users.iloc[41]["user_id"], top_n=TOP_N)
    assert len(result.relaxation_log) >= 2
    assert result.relaxation_log.n_feasible == result.n_feasible
    assert set(result.relaxation_log.hard_excluded) == {
        "budget", "gender", "stock", "sport"
    }


def test_top_n_is_honoured(engine, users):
    for top_n in (1, 5, 10, 25, 50):
        result = engine.recommend(users.iloc[41]["user_id"], top_n=top_n, explain=False)
        assert len(result.items) == top_n
        assert list(result.items["rank"]) == list(range(1, top_n + 1))


def test_scores_are_descending(engine, users):
    result = engine.recommend(users.iloc[100]["user_id"], top_n=TOP_N, explain=False)
    assert result.items["score"].is_monotonic_decreasing


# ----------------------------------------------------------------------
# Profile resolution and the CLI
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "value,expected",
    [(42, "U00042"), ("42", "U00042"), ("U00042", "U00042"), ("u42", "U00042")],
)
def test_user_ids_canonicalise(value, expected):
    assert _canonical_user_id(value) == expected


def test_all_the_user_id_spellings_agree(engine):
    reference = engine.recommend("U00042", top_n=TOP_N, explain=False).items
    for spelling in (42, "42", "u42"):
        pd.testing.assert_frame_equal(
            engine.recommend(spelling, top_n=TOP_N, explain=False).items, reference
        )


def test_an_unknown_user_id_is_a_clear_error(engine):
    with pytest.raises(KeyError, match="U99999"):
        engine.recommend("U99999")


def test_to_dict_is_json_serialisable(engine, users):
    result = engine.recommend(users.iloc[41]["user_id"], top_n=3)
    payload = json.dumps(result.to_dict(), default=str)
    restored = json.loads(payload)
    assert len(restored["items"]) == 3
    assert restored["relaxation_log"]
    assert restored["hard_excluded"]
    assert restored["content_mode"] in ("profile_only", "profile_plus_centroid")


def test_str_renders_both_techniques(engine, users):
    text = str(engine.recommend(users.iloc[41]["user_id"], top_n=2))
    assert "[Ch7]" in text and "[Ch3]" in text


def test_cli_parses_the_documented_invocation():
    args = build_parser().parse_args(["--user-id", "42", "--top-n", "10"])
    assert args.user_id == "42" and args.top_n == 10


def test_cli_requires_exactly_one_source():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--user-id", "42", "--profile", "{}"])


def test_cli_runs_end_to_end(capsys):
    """`python -m src.recommender.pipeline --user-id 42 --top-n 10`."""
    assert main(["--user-id", "42", "--top-n", "10"]) == 0
    out = capsys.readouterr().out
    assert "U00042" in out and "[Ch7]" in out


def test_cli_json_mode_emits_valid_json(capsys):
    assert main(["--profile", '{"primary_sport": "tennis"}', "--top-n", "3",
                 "--json", "--no-explain"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["items"]) == 3


def test_cli_reports_an_unknown_user(capsys):
    assert main(["--user-id", "99999"]) == 1
    assert "U99999" in capsys.readouterr().err


# ----------------------------------------------------------------------
# Profile input forms
# ----------------------------------------------------------------------
def test_profile_argument_accepts_json():
    assert parse_profile_argument('{"primary_sport": "yoga"}') == {
        "primary_sport": "yoga"
    }


@pytest.mark.parametrize(
    "text,expected",
    [
        ("primary_sport=yoga", {"primary_sport": "yoga"}),
        (
            "primary_sport=yoga,budget_max=90",
            {"primary_sport": "yoga", "budget_max": "90"},
        ),
        (
            " gender = female ; apparel_size = M ",
            {"gender": "female", "apparel_size": "M"},
        ),
        (
            "preferred_brands=Nike|Adidas",
            {"preferred_brands": "Nike|Adidas"},
        ),
    ],
)
def test_profile_argument_accepts_key_value_pairs(text, expected):
    """The key=value form exists because Windows PowerShell 5.1 strips the
    quotes out of a JSON argument before the process ever sees it."""
    assert parse_profile_argument(text) == expected


def test_key_value_profile_survives_normalisation(engine):
    """The pairs arrive as strings; normalize_profile does the coercion."""
    profile = parse_profile_argument(
        "primary_sport=yoga,gender=female,budget_min=20,budget_max=90,"
        "preferred_brands=Nike|Adidas"
    )
    result = engine.recommend(profile, top_n=TOP_N, explain=False)
    assert len(result.items) == TOP_N
    assert (result.items["price"] <= 90).all()
    assert (result.items["price"] >= 20).all()
    assert set(result.items["gender_target"]) <= {"women", "unisex"}


def test_malformed_json_names_the_shell_problem():
    with pytest.raises(ValueError, match="PowerShell"):
        parse_profile_argument('{primary_sport: yoga}')


def test_a_json_array_is_rejected():
    with pytest.raises(ValueError, match="object"):
        parse_profile_argument('["yoga"]')


def test_a_fragment_that_is_neither_form_is_rejected():
    with pytest.raises(ValueError, match="neither JSON nor key=value"):
        parse_profile_argument("just some words")


def test_cli_accepts_key_value_pairs(capsys):
    assert main(["--profile", "primary_sport=yoga,gender=female",
                 "--top-n", "3", "--no-explain"]) == 0
    assert "feasible set" in capsys.readouterr().out


def test_cli_accepts_a_profile_file(tmp_path, capsys):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"primary_sport": "tennis"}), encoding="utf-8")
    assert main(["--profile-file", str(path), "--top-n", "3", "--no-explain",
                 "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)["items"]) == 3


def test_cli_reads_a_bom_prefixed_profile_file(tmp_path, capsys):
    """PowerShell's Out-File writes a UTF-8 BOM, so the reader uses utf-8-sig."""
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"primary_sport": "tennis"}), encoding="utf-8-sig")
    assert main(["--profile-file", str(path), "--top-n", "2", "--no-explain"]) == 0
    assert "feasible set" in capsys.readouterr().out


def test_cli_rejects_a_bad_profile_without_a_traceback(capsys):
    assert main(["--profile", "{nope}", "--top-n", "3"]) == 2
    assert "PowerShell" in capsys.readouterr().err


def test_cli_rejects_a_missing_profile_file(tmp_path, capsys):
    assert main(["--profile-file", str(tmp_path / "nope.json")]) == 2
    assert capsys.readouterr().err


def test_the_three_profile_sources_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--profile", "a=b", "--profile-file", "x.json"])
