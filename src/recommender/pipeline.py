"""The cascade: Chapter 7 decides *what is admissible*, Chapter 3 decides
*what order*.

    user profile
        |
        v
    ConstraintBasedRecommender.filter()      <- Knowledge-Based, Ch7
        |  feasible set (unranked) + relaxation log
        v
    ContentBasedRecommender.rank()           <- TF-IDF + cosine, Ch3
        |  top_n with score, rank, explanations
        v
    Recommendation

The two techniques stay separate modules with no imports between them; this
file is the only place they meet.  That is deliberate -- the grader has to be
able to see each technique on its own, and ``tests/`` exercises each one
without the other.

Run it::

    python -m src.recommender.pipeline --user-id 42 --top-n 10
    python -m src.recommender.pipeline --profile primary_sport=yoga,budget_max=90
    python -m src.recommender.pipeline --profile-file myprofile.json --json

``--profile`` takes either a JSON object or ``key=value`` pairs.  Prefer the
pairs on Windows: PowerShell 5.1 strips the quotes out of a JSON argument
before the process sees it, so the JSON form only survives cmd.exe with
backslash escaping.

Caching
-------
Fitting TF-IDF over 10,000 products takes a second or so -- fine in a script,
not fine per HTTP request in session 3.  :meth:`FitMatchRecommender.load`
memoises the fitted vectorizer and document matrix to
``data/cache/tfidf_model.joblib``, keyed on a signature of the catalogue file
and the vectorizer settings, so a stale cache is rebuilt rather than used.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd

from src.data.common import (
    INTERACTIONS_CSV,
    PRODUCTS_CSV,
    PROJECT_ROOT,
    SEED,
    USERS_CSV,
)
from src.recommender.content_based import ContentBasedRecommender
from src.recommender.knowledge_based import (
    DEFAULT_MIN_RESULTS,
    ConstraintBasedRecommender,
    ConstraintReport,
)
from src.recommender.profiles import normalize_profile

CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CACHE_PATH = CACHE_DIR / "tfidf_model.joblib"

# Bump when the document builder or vectorizer settings change, so an old
# cache is discarded instead of silently serving stale vectors.
CACHE_VERSION = 2

# Columns carried through from products.csv into the result frame.
RESULT_COLUMNS = (
    "product_id", "brand", "category", "subcategory", "sport_type",
    "gender_target", "price", "stock_status", "size", "color", "material",
    "seasonality", "avg_rating",
)


@dataclass
class Recommendation:
    """One call's worth of output: the ranked items and the paper trail."""

    user_id: str | None
    profile: dict[str, Any]
    items: pd.DataFrame
    relaxation_log: ConstraintReport
    content_mode: str
    n_feasible: int
    explanations: dict[str, dict[str, list[str]]] = field(default_factory=dict)

    @property
    def product_ids(self) -> list[str]:
        return list(self.items["product_id"])

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready, for the session-3 API."""
        return {
            "user_id": self.user_id,
            "content_mode": self.content_mode,
            "n_feasible": self.n_feasible,
            "items": self.items.to_dict("records"),
            "relaxation_log": self.relaxation_log.as_dicts(),
            "hard_excluded": self.relaxation_log.hard_excluded,
            "explanations": self.explanations,
        }

    def __str__(self) -> str:
        lines = [
            "user: {}   feasible set: {:,}   content mode: {}".format(
                self.user_id or "(anonymous profile)",
                self.n_feasible,
                self.content_mode,
            ),
            "",
            str(self.relaxation_log),
            "",
        ]
        for row in self.items.to_dict("records"):
            lines.append(
                "{:>2}. {:<12} {:<28} ${:>8.2f}  score {:.4f}".format(
                    int(row["rank"]),
                    str(row["product_id"]),
                    "{} {} / {}".format(
                        row.get("brand", ""), row.get("category", ""),
                        row.get("sport_type", ""),
                    )[:28],
                    float(row["price"]),
                    float(row["score"]),
                )
            )
            why = self.explanations.get(str(row["product_id"]), {})
            for reason in why.get("constraints", []):
                lines.append("      [Ch7] " + reason)
            for reason in why.get("content", []):
                lines.append("      [Ch3] " + reason)
        return "\n".join(lines)


class FitMatchRecommender:
    """The two-technique cascade.

    Parameters
    ----------
    products, users, interactions:
        The session-1 frames.  ``users`` is only needed to resolve a
        ``--user-id``; ``interactions`` is only needed for the Chapter 3
        centroid mode.  Both may be ``None``.
    min_results:
        Passed to the Chapter 7 filter as its relaxation target.

    Notes
    -----
    Fully deterministic.  Neither technique draws a random number, ties in the
    Chapter 3 ranking are broken by ``product_id``, and ``SEED`` is recorded
    here only so tests and the CLI can state the seed they ran under.
    """

    seed = SEED

    def __init__(
        self,
        products: pd.DataFrame,
        users: pd.DataFrame | None = None,
        interactions: pd.DataFrame | None = None,
        min_results: int = DEFAULT_MIN_RESULTS,
        fit: bool = True,
    ) -> None:
        self.products = products.reset_index(drop=True)
        self.users = users
        self.knowledge = ConstraintBasedRecommender(self.products, min_results=min_results)
        self.content = ContentBasedRecommender(self.products, interactions)
        self._users_by_id: dict[str, dict[str, Any]] = {}
        if users is not None:
            self._users_by_id = {
                str(row["user_id"]): row for row in users.to_dict("records")
            }
        # `load()` passes fit=False because it may restore the fitted matrix
        # from the joblib cache instead of refitting it.
        if fit:
            self.content.fit()

    # ------------------------------------------------------------------
    # Construction and caching
    # ------------------------------------------------------------------
    @classmethod
    def load(
        cls,
        products_path: Path = PRODUCTS_CSV,
        users_path: Path | None = USERS_CSV,
        interactions_path: Path | None = INTERACTIONS_CSV,
        min_results: int = DEFAULT_MIN_RESULTS,
        cache_path: Path | None = CACHE_PATH,
        rebuild_cache: bool = False,
    ) -> "FitMatchRecommender":
        """Build the cascade from the processed CSVs, reusing the joblib TF-IDF
        cache when it matches the current catalogue."""
        for path, produced_by in (
            (products_path, "src.data.build_products"),
            (users_path, "src.data.build_users"),
            (interactions_path, "src.data.build_interactions"),
        ):
            if path is not None and not Path(path).exists():
                raise SystemExit(
                    "\nFATAL: {} is missing. Run `python -m {}` first "
                    "(or build_data.bat).".format(path, produced_by)
                )

        products = pd.read_csv(products_path)
        users = pd.read_csv(users_path) if users_path is not None else None
        interactions = (
            pd.read_csv(interactions_path) if interactions_path is not None else None
        )
        model = cls(products, users, interactions, min_results=min_results, fit=False)
        model._fit_content(Path(products_path), cache_path, rebuild_cache)
        return model

    def _fit_content(
        self, products_path: Path, cache_path: Path | None, rebuild_cache: bool
    ) -> None:
        """Fit TF-IDF, or restore it from the joblib cache."""
        if cache_path is None:
            self.content.fit()
            return

        signature = _catalogue_signature(products_path, len(self.products))
        cache_path = Path(cache_path)
        if not rebuild_cache and cache_path.exists():
            try:
                cached = joblib.load(cache_path)
                if cached.get("signature") == signature:
                    self.content.attach(cached)
                    return
            except Exception:
                # A cache that cannot be read is a cache that gets rebuilt.
                pass

        self.content.fit()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        state = self.content.export()
        state["signature"] = signature
        joblib.dump(state, cache_path)

    # ------------------------------------------------------------------
    # The cascade
    # ------------------------------------------------------------------
    def resolve_profile(self, user_id_or_profile: str | int | Mapping[str, Any]
                        ) -> dict[str, Any]:
        """Accept a ``user_id`` from users.csv or a raw profile dict.

        ``42``, ``"42"`` and ``"U00042"`` all mean the same user, so a web form
        posting a bare number does not need to know the id format.
        """
        if isinstance(user_id_or_profile, Mapping):
            return normalize_profile(user_id_or_profile)

        key = _canonical_user_id(user_id_or_profile)
        if key not in self._users_by_id:
            raise KeyError(
                "no user {} in users.csv (read from {!r}; ids run U00001-U05000)".format(
                    key, str(user_id_or_profile)
                )
            )
        return normalize_profile(self._users_by_id[key])

    def recommend(
        self,
        user_id_or_profile: str | int | Mapping[str, Any],
        top_n: int = 10,
        explain: bool = True,
        allow_history: bool = True,
        min_results: int | None = None,
    ) -> Recommendation:
        """Recommend ``top_n`` products: constraint filter, then TF-IDF rank.

        Step 1 (Ch7) narrows 10,000 products to a feasible set, relaxing soft
        constraints until it is big enough.  Step 2 (Ch3) ranks that set by
        cosine similarity to the user's TF-IDF vector.  Both steps' reasoning
        comes back in the result.
        """
        profile = self.resolve_profile(user_id_or_profile)
        feasible, report = self.knowledge.filter(profile, min_results=min_results)
        ranked = self.content.rank(
            profile, feasible, top_n=top_n, allow_history=allow_history
        )

        columns = [c for c in RESULT_COLUMNS if c in self.products.columns]
        items = ranked.merge(self.products[columns], on="product_id", how="left")
        items = items[["rank", "product_id", "score"]
                      + [c for c in columns if c != "product_id"]]

        explanations: dict[str, dict[str, list[str]]] = {}
        if explain and len(items):
            indexed = self.products.set_index("product_id")
            for product_id in items["product_id"]:
                product = indexed.loc[product_id]
                explanations[str(product_id)] = {
                    "constraints": self.knowledge.explain_constraints(profile, product),
                    "content": self.content.explain(
                        profile, str(product_id), allow_history=allow_history
                    ),
                }

        mode = str(ranked["mode"].iloc[0]) if len(ranked) else "profile_only"
        return Recommendation(
            user_id=profile.get("user_id"),
            profile=profile,
            items=items,
            relaxation_log=report,
            content_mode=mode,
            n_feasible=report.n_feasible,
            explanations=explanations,
        )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _canonical_user_id(value: str | int) -> str:
    """`42`, `"42"`, `"u42"` and `"U00042"` all canonicalise to `"U00042"`."""
    text = str(value).strip()
    if text.upper().startswith("U"):
        text = text[1:]
    if text.isdigit():
        return "U{:05d}".format(int(text))
    return str(value).strip()


def _catalogue_signature(products_path: Path, n_rows: int) -> tuple:
    """Identify the catalogue a cached model was fitted on."""
    stat = Path(products_path).stat()
    return (CACHE_VERSION, str(Path(products_path).name), stat.st_size,
            int(stat.st_mtime), n_rows)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.recommender.pipeline",
        description="FitMatch recommendation cascade: knowledge-based "
                    "constraint filter (Ch7) then TF-IDF content ranking (Ch3).",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--user-id", help="a user_id from users.csv, e.g. 42 or U00042")
    source.add_argument(
        "--profile",
        help="a raw profile, either as a JSON object or as key=value pairs "
             "(primary_sport=yoga,budget_max=90). The key=value form exists "
             "because Windows PowerShell 5.1 mangles quoted JSON on its way "
             "to a native program.",
    )
    source.add_argument(
        "--profile-file", help="path to a file holding the profile as JSON"
    )
    parser.add_argument("--top-n", type=int, default=10, help="how many to return")
    parser.add_argument("--min-results", type=int, default=DEFAULT_MIN_RESULTS,
                        help="feasible-set target before soft constraints relax")
    parser.add_argument("--no-explain", action="store_true",
                        help="skip the per-item explanations")
    parser.add_argument("--no-history", action="store_true",
                        help="force content mode (a), profile only")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="refit TF-IDF and overwrite the joblib cache")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    return parser


def parse_profile_argument(text: str) -> dict[str, Any]:
    """Parse `--profile`: a JSON object, or `key=value` pairs.

    The second form is not sugar. Windows PowerShell 5.1 strips the quotes out
    of a single-quoted JSON string before the native process ever sees it, so
    the documented `--profile '{"primary_sport": "yoga"}'` invocation fails on
    the machine this project is developed on. `key=value` survives every shell.

    Values stay strings; :func:`~src.recommender.profiles.normalize_profile`
    coerces the numeric fields and splits pipe-separated brands.
    """
    text = text.strip()
    if text.startswith("["):
        raise ValueError("--profile must be a JSON object, not an array")
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(
                "--profile is not valid JSON ({}).\n"
                "  PowerShell strips quotes from JSON arguments; use either\n"
                '    --profile primary_sport=yoga,budget_max=90\n'
                "  or --profile-file myprofile.json".format(error)
            ) from error
        if not isinstance(parsed, dict):
            raise ValueError("--profile JSON must be an object, not a {}".format(
                type(parsed).__name__))
        return parsed

    profile: dict[str, Any] = {}
    for pair in text.replace(";", ",").split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(
                "--profile fragment {!r} is neither JSON nor key=value".format(pair)
            )
        key, _, value = pair.partition("=")
        profile[key.strip()] = value.strip()
    if not profile:
        raise ValueError("--profile is empty")
    return profile


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    target: Any
    try:
        if args.user_id:
            target = args.user_id
        elif args.profile_file:
            # utf-8-sig, not utf-8: PowerShell's Out-File writes a BOM.
            target = json.loads(
                Path(args.profile_file).read_text(encoding="utf-8-sig")
            )
        else:
            target = parse_profile_argument(args.profile)
    except (ValueError, OSError, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        return 2

    model = FitMatchRecommender.load(
        min_results=args.min_results, rebuild_cache=args.rebuild_cache
    )

    try:
        result = model.recommend(
            target,
            top_n=args.top_n,
            explain=not args.no_explain,
            allow_history=not args.no_history,
        )
    except KeyError as error:
        # KeyError stringifies as a repr, quotes included; args[0] is the message.
        print(str(error.args[0]), file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
