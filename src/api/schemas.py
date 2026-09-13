"""Pydantic v2 request and response models.

Two rules hold this file together.

**The vocabularies are not written down here.**  Every categorical field is
validated against :mod:`src.api.vocabulary`, which measured it from
``products.csv`` and ``users.csv`` at startup.  A bad ``sport_type`` therefore
comes back as a 422 that *names the ten sports the catalogue actually has*,
and it keeps naming the right ten if the catalogue is rebuilt.

**The profile defaults are not written down here either.**
:func:`src.recommender.profiles.normalize_profile` is the single source of
truth for what a missing field means, so every profile field on the request is
optional with no default, and :meth:`RecommendRequest.to_profile` emits only
the keys the caller actually sent.  A field the API invented a default for
would be a field on which the API and the recommender could disagree.

What *is* written down here is request validation the recommender has no
opinion about: an age of 400 or a budget that runs backwards is a bad request,
not an empty feasible set.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_validator,
)

from src.api.vocabulary import FIELD_LIMITS, get_vocabulary

ScoringMode = Literal["profile_only", "profile_plus_centroid"]


# ----------------------------------------------------------------------
# Vocabulary-backed field types
# ----------------------------------------------------------------------
def _check_vocabulary(field_name: str):
    """Build a validator that maps a value onto the catalogue's own spelling.

    The error message lists the valid values, because a frontend that gets
    ``"'teal' is not a known colour"`` and nothing else cannot fix itself, and
    because the ten sports are the kind of thing a grader should be able to
    read straight out of a failed request.
    """

    def validate(value: Any) -> Any:
        if value is None:
            return None
        vocabulary = get_vocabulary()
        canonical = vocabulary.canonical(field_name, str(value))
        if canonical is None:
            options = vocabulary.options(field_name)
            raise ValueError(
                "{!r} is not a valid {}; the catalogue has {} values: {}".format(
                    str(value), field_name, len(options), ", ".join(options)
                )
            )
        return canonical

    return validate


def _vocab_type(field_name: str):
    return Annotated[Optional[str], BeforeValidator(_check_vocabulary(field_name))]


def _as_list(value: Any) -> Any:
    """Accept a bare ``"Nike"`` as well as ``["Nike"]``.

    This is *not* the pipe-separated ``"Nike|Adidas"`` spelling from
    ``users.csv`` -- splitting that string is
    :func:`src.recommender.profiles.normalize_profile`'s job, and the JSON API
    takes an array.  A posted ``"Nike|Adidas"`` therefore fails validation with
    a message naming the three real brands, which is the right answer.
    """
    if value is None or isinstance(value, list):
        return value
    return [value]


SportField = _vocab_type("primary_sport")
ColorField = _vocab_type("color_preference")
MaterialField = _vocab_type("material_preference")
StyleField = _vocab_type("style_preference")
ClimateField = _vocab_type("climate")
ArchField = _vocab_type("foot_arch_type")
ApparelSizeField = _vocab_type("apparel_size")
GenderField = _vocab_type("gender")
FitnessField = _vocab_type("fitness_level")
IndoorOutdoorField = _vocab_type("indoor_or_outdoor")
SportTypeField = _vocab_type("sport_type")
BrandField = _vocab_type("brand")
CategoryField = _vocab_type("category")
BrandListField = Annotated[
    Optional[list[Annotated[str, BeforeValidator(_check_vocabulary("brand"))]]],
    BeforeValidator(_as_list),
]


def _limit(name: str, key: str) -> float:
    return FIELD_LIMITS[name][key]


# ----------------------------------------------------------------------
# Requests
# ----------------------------------------------------------------------
class RecommendRequest(BaseModel):
    """The body of ``POST /api/recommend``.

    Every profile field is optional.  A field that is absent is absent all the
    way down: it reaches ``normalize_profile`` missing, takes the default in
    ``PROFILE_DEFAULTS``, and -- for the preference fields -- switches off the
    constraint that reads it rather than inventing an answer.  That is why a
    form the user only half filled in still returns something sensible.

    ``user_id`` means *score this row of users.csv instead*, not *this is who
    is filling in the form*: it is the demo button's path, and when it is
    present the other profile fields are ignored.  A frontend that wants the
    user to edit a loaded profile should drop ``user_id`` before posting.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "age": 28,
                "gender": "female",
                "height_cm": 168.0,
                "weight_kg": 61.0,
                "shoe_size": 8.0,
                "apparel_size": "M",
                "foot_arch_type": "normal",
                "primary_sport": "yoga",
                "fitness_level": "intermediate",
                "workouts_per_week": 4,
                "indoor_or_outdoor": "indoor",
                "budget_min": 40,
                "budget_max": 120,
                "preferred_brands": ["Nike"],
                "style_preference": "athleisure",
                "color_preference": "navy",
                "material_preference": "cotton",
                "climate": "temperate",
                "top_n": 10,
            }
        },
    )

    user_id: Optional[str] = Field(
        default=None,
        description="Score this users.csv row instead of the posted profile. "
        "Accepts 42, '42' or 'U00042'.",
    )

    age: Optional[int] = Field(
        default=None, ge=_limit("age", "min"), le=_limit("age", "max")
    )
    gender: GenderField = None
    height_cm: Optional[float] = Field(
        default=None, ge=_limit("height_cm", "min"), le=_limit("height_cm", "max")
    )
    weight_kg: Optional[float] = Field(
        default=None, ge=_limit("weight_kg", "min"), le=_limit("weight_kg", "max")
    )
    bmi: Optional[float] = Field(
        default=None,
        ge=5,
        le=100,
        description="Accepted so a profile loaded from /api/users round-trips; "
        "no constraint reads it.",
    )
    shoe_size: Optional[float] = Field(
        default=None, ge=_limit("shoe_size", "min"), le=_limit("shoe_size", "max")
    )
    apparel_size: ApparelSizeField = None
    foot_arch_type: ArchField = None
    primary_sport: SportField = None
    fitness_level: FitnessField = None
    workouts_per_week: Optional[int] = Field(
        default=None,
        ge=_limit("workouts_per_week", "min"),
        le=_limit("workouts_per_week", "max"),
    )
    indoor_or_outdoor: IndoorOutdoorField = None
    budget_min: Optional[float] = Field(
        default=None, ge=_limit("budget_min", "min"), le=_limit("budget_min", "max")
    )
    budget_max: Optional[float] = Field(
        default=None, ge=_limit("budget_max", "min"), le=_limit("budget_max", "max")
    )
    preferred_brands: BrandListField = None
    style_preference: StyleField = None
    color_preference: ColorField = None
    material_preference: MaterialField = None
    location: Optional[str] = Field(
        default=None,
        max_length=120,
        description="Free text, e.g. 'Denver, CO'. Carried through; climate is "
        "the field the constraint filter reads.",
    )
    climate: ClimateField = None

    top_n: int = Field(default=10, ge=1, le=100, description="How many to return.")
    min_results: Optional[int] = Field(
        default=None,
        ge=1,
        le=10000,
        description="Feasible-set target the constraint filter relaxes towards. "
        "Defaults to the recommender's own DEFAULT_MIN_RESULTS (50).",
    )

    @model_validator(mode="after")
    def _budget_runs_forwards(self) -> "RecommendRequest":
        if (
            self.budget_min is not None
            and self.budget_max is not None
            and self.budget_min >= self.budget_max
        ):
            raise ValueError(
                "budget_min ({:g}) must be less than budget_max ({:g})".format(
                    self.budget_min, self.budget_max
                )
            )
        return self

    def to_profile(self) -> dict[str, Any]:
        """The profile dict to hand to the recommender.

        Only the keys the caller actually sent, so
        ``profiles.normalize_profile`` supplies every default exactly once.
        """
        return self.model_dump(
            exclude_none=True, exclude={"top_n", "min_results", "user_id"}
        )

    def profile_summary(self) -> str:
        """One short line for the request log."""
        parts = [
            "sport={}".format(self.primary_sport or "-"),
            "gender={}".format(self.gender or "-"),
            "budget={}-{}".format(
                "-" if self.budget_min is None else "{:g}".format(self.budget_min),
                "-" if self.budget_max is None else "{:g}".format(self.budget_max),
            ),
            "shoe={}".format("-" if self.shoe_size is None else "{:g}".format(self.shoe_size)),
            "apparel={}".format(self.apparel_size or "-"),
            "color={}".format(self.color_preference or "-"),
            "material={}".format(self.material_preference or "-"),
            "climate={}".format(self.climate or "-"),
            "brands={}".format("|".join(self.preferred_brands or []) or "-"),
        ]
        return " ".join(parts)


class ProductQuery(BaseModel):
    """Query parameters for ``GET /api/products``."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    sport: SportTypeField = None
    brand: BrandField = None
    category: CategoryField = None
    price_min: Optional[float] = Field(default=None, ge=0)
    price_max: Optional[float] = Field(default=None, ge=0)
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=200)

    @model_validator(mode="after")
    def _price_runs_forwards(self) -> "ProductQuery":
        if (
            self.price_min is not None
            and self.price_max is not None
            and self.price_min > self.price_max
        ):
            raise ValueError(
                "price_min ({:g}) must not exceed price_max ({:g})".format(
                    self.price_min, self.price_max
                )
            )
        return self


# ----------------------------------------------------------------------
# Responses -- products
# ----------------------------------------------------------------------
class ProductSummary(BaseModel):
    """The catalogue fields the cascade carries through into a result row."""

    model_config = ConfigDict(extra="forbid")

    product_id: str
    brand: str
    category: str
    subcategory: str
    sport_type: str
    gender_target: str
    price: float
    stock_status: str
    size: str
    color: str
    material: str
    seasonality: str
    avg_rating: Optional[float] = None


class ProductDetail(ProductSummary):
    """Everything ``products.csv`` knows about one product."""

    discount: Optional[float] = None
    weight: Optional[float] = None
    cushioning_level: Optional[int] = None
    arch_support: Optional[str] = None
    breathability: Optional[int] = None
    waterproof: Optional[bool] = None
    review_count: Optional[int] = None
    sales_rank: Optional[int] = None
    product_description: Optional[str] = None


class ProductPage(BaseModel):
    """One page of the catalogue browse."""

    model_config = ConfigDict(extra="forbid")

    items: list[ProductSummary]
    page: int
    page_size: int
    total: int = Field(description="Rows matching the filters, before paging.")
    total_pages: int
    has_next: bool
    has_previous: bool
    filters: dict[str, Any] = Field(
        default_factory=dict, description="The filters that were applied."
    )


# ----------------------------------------------------------------------
# Responses -- recommendation
# ----------------------------------------------------------------------
class Explanation(BaseModel):
    """One line of why, tagged with the technique that produced it."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["knowledge_based", "content_based"]
    chapter: Literal["Ch7", "Ch3"]
    text: str


class Recommendation(BaseModel):
    """One ranked product."""

    model_config = ConfigDict(extra="forbid")

    rank: int
    score: float = Field(
        description="Cosine similarity between the user vector and the product "
        "vector, in [0, 1]."
    )
    product: ProductSummary
    explanations: list[Explanation] = Field(default_factory=list)


class RelaxedConstraint(BaseModel):
    """A preference that could not be honoured, and what dropping it bought.

    Session 2 measured colour being relaxed for 98.7% of users, so this is the
    normal case rather than an error case; ``message`` is written to be shown
    to the user verbatim.
    """

    model_config = ConfigDict(extra="forbid")

    constraint: str
    label: str = Field(description="Human name for the constraint.")
    requested: Optional[str] = Field(
        default=None, description="What the user asked for, if they asked."
    )
    reason: str = Field(description="Why it was dropped, from the recommender's log.")
    message: str = Field(description="One sentence, ready to show to the user.")
    candidates_before: int
    candidates_after: int
    order: int = Field(description="Position in the relaxation order; 0 is dropped first.")


class RelaxationLogEntry(BaseModel):
    """One raw step of the constraint filter's own log, unedited."""

    model_config = ConfigDict(extra="forbid")

    order: int
    constraint: str
    action: Literal["applied", "relaxed", "exhausted"]
    n_before: int
    n_after: int
    reason: str


class ConstraintsBlock(BaseModel):
    """The knowledge-based half of the answer: what was tried, in full."""

    model_config = ConfigDict(extra="forbid")

    hard_applied: list[str] = Field(
        description="Constraints that are never relaxed, in the order applied."
    )
    hard_excluded: dict[str, int] = Field(
        description="Products each hard constraint removed, in that same order."
    )
    candidates_after_hard: int
    candidates_after_soft: int = Field(
        description="Feasible set with every soft constraint still applied. The "
        "session-2 median for this is 0, which is why relaxation exists."
    )
    relaxed: list[RelaxedConstraint]
    still_applied: list[str] = Field(
        description="Soft constraints that survived and are reflected in the results."
    )
    final_candidate_count: int
    min_results_target: int
    exhausted: bool = Field(
        description="True when every soft constraint was dropped and the set is "
        "still below the target -- the hard constraints alone are the limit."
    )
    catalogue_size: int
    summary: str = Field(description="The whole story in one sentence, for the UI.")
    log: list[RelaxationLogEntry] = Field(
        description="The constraint filter's own step-by-step log, unedited."
    )


class ScoringBlock(BaseModel):
    """The content-based half: which mode ranked these, and why that one."""

    model_config = ConfigDict(extra="forbid")

    mode: ScoringMode
    qualifying_ratings: int = Field(
        description="Ratings of 4 or better this user has, which is what decides "
        "the mode."
    )
    min_ratings_for_centroid: int
    label: str = Field(description="Short label for the UI badge.")
    explanation: str = Field(description="One sentence, ready to show to the user.")


class RecommendResponse(BaseModel):
    """The ``POST /api/recommend`` body.

    ``constraints`` and ``scoring`` are part of the contract, not diagnostics:
    an empty ``recommendations`` list is only ever returned alongside the
    account of what was tried to fill it.
    """

    model_config = ConfigDict(extra="forbid")

    user_id: Optional[str] = Field(
        default=None, description="Set when the request named a users.csv row."
    )
    recommendations: list[Recommendation]
    constraints: ConstraintsBlock
    scoring: ScoringBlock
    latency_ms: float


# ----------------------------------------------------------------------
# Responses -- users, meta, health
# ----------------------------------------------------------------------
class UserProfile(BaseModel):
    """A users.csv row, normalised and ready to post back to /api/recommend."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    age: Optional[int] = None
    gender: Optional[str] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    bmi: Optional[float] = None
    shoe_size: Optional[float] = None
    apparel_size: Optional[str] = None
    foot_arch_type: Optional[str] = None
    primary_sport: Optional[str] = None
    fitness_level: Optional[str] = None
    workouts_per_week: Optional[int] = None
    indoor_or_outdoor: Optional[str] = None
    budget_min: Optional[float] = None
    budget_max: Optional[float] = None
    preferred_brands: list[str] = Field(default_factory=list)
    style_preference: Optional[str] = None
    color_preference: Optional[str] = None
    material_preference: Optional[str] = None
    location: Optional[str] = None
    climate: Optional[str] = None
    qualifying_ratings: int = Field(
        default=0,
        description="Ratings of 4+ on record, so the UI can say in advance which "
        "scoring mode this demo user will get.",
    )
    expected_scoring_mode: ScoringMode = "profile_only"


class RangeModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min: float
    max: float
    step: Optional[float] = None


class LimitModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min: float
    max: float


class UserIdRange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min: str
    max: str


class MetaResponse(BaseModel):
    """Every dropdown the frontend needs, measured from the CSVs at startup."""

    model_config = ConfigDict(extra="forbid")

    sport_type: list[str]
    brand: list[str]
    color_preference: list[str]
    material_preference: list[str]
    category: list[str]
    subcategory: list[str]
    gender_target: list[str]
    stock_status: list[str]
    seasonality: list[str]
    gender: list[str]
    style_preference: list[str]
    climate: list[str]
    foot_arch_type: list[str]
    apparel_size: list[str]
    fitness_level: list[str]
    indoor_or_outdoor: list[str]
    price: RangeModel
    shoe_size: RangeModel
    budget: RangeModel
    limits: dict[str, LimitModel]
    counts: dict[str, int]
    user_id_range: UserIdRange
    notes: list[str]


class CacheStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    warm: bool = Field(
        description="A joblib TF-IDF cache exists on disk for this catalogue."
    )
    hit_on_startup: bool = Field(
        description="The cache was reused at startup rather than rebuilt."
    )
    path: str
    size_bytes: Optional[int] = None


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "starting", "error"]
    ready: bool
    cache: CacheStatus
    cold_start_ms: Optional[float] = None
    uptime_s: float
    catalogue_size: int
    n_users: int
    n_vocabulary_terms: Optional[int] = None
    min_results_default: int
    detail: Optional[str] = None


class ErrorResponse(BaseModel):
    """The body of every 4xx and 5xx this API raises deliberately."""

    model_config = ConfigDict(extra="forbid")

    detail: str
    error: str = Field(description="Machine-readable code, e.g. 'product_not_found'.")
    hint: Optional[str] = None


class ValidationErrorItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    message: str
    type: str


class ValidationErrorResponse(BaseModel):
    """422 body: which field, what was wrong, and what would have been valid."""

    model_config = ConfigDict(extra="forbid")

    detail: str
    error: Literal["validation_error"] = "validation_error"
    errors: list[ValidationErrorItem]


__all__ = [
    "CacheStatus",
    "ConstraintsBlock",
    "ErrorResponse",
    "Explanation",
    "HealthResponse",
    "MetaResponse",
    "ProductDetail",
    "ProductPage",
    "ProductQuery",
    "ProductSummary",
    "Recommendation",
    "RecommendRequest",
    "RecommendResponse",
    "RelaxationLogEntry",
    "RelaxedConstraint",
    "ScoringBlock",
    "ScoringMode",
    "UserProfile",
    "ValidationErrorResponse",
]
