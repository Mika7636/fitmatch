/**
 * TypeScript mirrors of `src/api/schemas.py`.
 *
 * Hand-written rather than generated, but field-for-field: every interface
 * here corresponds to one Pydantic model, in the same order, and the optional
 * markers match the Python `Optional[...] = None` exactly. The Python models
 * are `extra="forbid"`, so an interface with a field the API does not send is
 * a bug on this side, not a tolerance.
 *
 * The vocabularies are deliberately *not* union types. They are measured from
 * products.csv and users.csv at API startup and served by `/api/meta`; baking
 * them into the type system here would be the hardcoding the backend exists to
 * prevent. They are `string`, and the option lists come from `MetaResponse`.
 */

// ---------------------------------------------------------------------------
// Recommendation request
// ---------------------------------------------------------------------------

/**
 * The profile half of `POST /api/recommend`, matching `RecommendRequest`.
 *
 * Every field is optional. The API forwards only what is sent to
 * `profiles.normalize_profile`, which owns the defaults; a field omitted here
 * switches off the constraint that reads it rather than being guessed at.
 */
export interface ProfileFields {
  age?: number
  gender?: string
  height_cm?: number
  weight_kg?: number
  bmi?: number
  shoe_size?: number
  apparel_size?: string
  foot_arch_type?: string
  primary_sport?: string
  fitness_level?: string
  workouts_per_week?: number
  indoor_or_outdoor?: string
  budget_min?: number
  budget_max?: number
  preferred_brands?: string[]
  style_preference?: string
  color_preference?: string
  material_preference?: string
  location?: string
  climate?: string
}

/**
 * The full request body.
 *
 * `user_id` is not "who is asking" -- it means *score this stored users.csv
 * row and ignore every other field in the body*. Sending it alongside an
 * edited profile silently scores the original row instead of the edits, so
 * `buildRecommendRequest` in `context/ProfileContext.tsx` is the only place
 * allowed to set it.
 */
export interface RecommendRequest extends ProfileFields {
  user_id?: string
  top_n?: number
  min_results?: number
}

// ---------------------------------------------------------------------------
// Products
// ---------------------------------------------------------------------------
export interface ProductSummary {
  product_id: string
  brand: string
  category: string
  subcategory: string
  sport_type: string
  gender_target: string
  price: number
  stock_status: string
  size: string
  color: string
  material: string
  seasonality: string
  avg_rating: number | null
}

export interface ProductDetail extends ProductSummary {
  discount: number | null
  weight: number | null
  cushioning_level: number | null
  arch_support: string | null
  breathability: number | null
  waterproof: boolean | null
  review_count: number | null
  sales_rank: number | null
  product_description: string | null
}

export interface ProductPage {
  items: ProductSummary[]
  page: number
  page_size: number
  total: number
  total_pages: number
  has_next: boolean
  has_previous: boolean
  filters: Record<string, unknown>
}

export interface ProductQuery {
  sport?: string
  brand?: string
  category?: string
  price_min?: number
  price_max?: number
  page?: number
  page_size?: number
}

// ---------------------------------------------------------------------------
// Recommendation response
// ---------------------------------------------------------------------------

/** One line of why, tagged with the technique that produced it. */
export interface Explanation {
  /** `knowledge_based` is the Ch7 constraint filter, `content_based` is Ch3. */
  source: 'knowledge_based' | 'content_based'
  chapter: 'Ch7' | 'Ch3'
  text: string
}

export interface Recommendation {
  rank: number
  /** Cosine similarity in [0, 1]. */
  score: number
  product: ProductSummary
  explanations: Explanation[]
}

/**
 * A preference the constraint filter could not honour.
 *
 * `message` is written by the API to be displayed verbatim -- session 2
 * measured colour being relaxed for 98.7% of users, so this is the system
 * working as designed, and the UI must not restate it as a warning.
 */
export interface RelaxedConstraint {
  constraint: string
  label: string
  requested: string | null
  reason: string
  message: string
  candidates_before: number
  candidates_after: number
  order: number
}

export interface RelaxationLogEntry {
  order: number
  constraint: string
  action: 'applied' | 'relaxed' | 'exhausted'
  n_before: number
  n_after: number
  reason: string
}

export interface ConstraintsBlock {
  hard_applied: string[]
  /** Products each hard constraint removed, applied in `hard_applied` order. */
  hard_excluded: Record<string, number>
  candidates_after_hard: number
  /** With every soft constraint still applied. The session-2 median is 0. */
  candidates_after_soft: number
  relaxed: RelaxedConstraint[]
  still_applied: string[]
  final_candidate_count: number
  min_results_target: number
  exhausted: boolean
  catalogue_size: number
  /** One sentence covering the whole filter. Display verbatim. */
  summary: string
  log: RelaxationLogEntry[]
}

export type ScoringMode = 'profile_only' | 'profile_plus_centroid'

export interface ScoringBlock {
  mode: ScoringMode
  qualifying_ratings: number
  min_ratings_for_centroid: number
  label: string
  explanation: string
}

export interface RecommendResponse {
  /** Set only when the request named a users.csv row. */
  user_id: string | null
  recommendations: Recommendation[]
  constraints: ConstraintsBlock
  scoring: ScoringBlock
  latency_ms: number
}

// ---------------------------------------------------------------------------
// Users
// ---------------------------------------------------------------------------
export interface UserProfile extends ProfileFields {
  user_id: string
  preferred_brands: string[]
  qualifying_ratings: number
  expected_scoring_mode: ScoringMode
}

// ---------------------------------------------------------------------------
// Meta
// ---------------------------------------------------------------------------
export interface NumericRange {
  min: number
  max: number
  step: number | null
}

export interface NumericLimit {
  min: number
  max: number
}

export interface MetaResponse {
  sport_type: string[]
  brand: string[]
  color_preference: string[]
  material_preference: string[]
  category: string[]
  subcategory: string[]
  gender_target: string[]
  stock_status: string[]
  seasonality: string[]
  gender: string[]
  style_preference: string[]
  climate: string[]
  foot_arch_type: string[]
  apparel_size: string[]
  fitness_level: string[]
  indoor_or_outdoor: string[]
  price: NumericRange
  shoe_size: NumericRange
  budget: NumericRange
  /** Numeric bounds the API enforces; the form mirrors these client-side. */
  limits: Record<string, NumericLimit>
  counts: Record<string, number>
  user_id_range: { min: string; max: string }
  notes: string[]
}

// ---------------------------------------------------------------------------
// Health and errors
// ---------------------------------------------------------------------------
export interface HealthResponse {
  status: 'ok' | 'starting' | 'error'
  ready: boolean
  cache: {
    warm: boolean
    hit_on_startup: boolean
    path: string
    size_bytes: number | null
  }
  cold_start_ms: number | null
  uptime_s: number
  catalogue_size: number
  n_users: number
  n_vocabulary_terms: number | null
  min_results_default: number
  detail: string | null
}

/** Body of every deliberate 4xx/5xx except 422. */
export interface ErrorBody {
  detail: string
  error: string
  hint: string | null
}

export interface ValidationErrorItem {
  field: string
  message: string
  type: string
}

/** Body of a 422: which field, what was wrong, what would have been valid. */
export interface ValidationErrorBody {
  detail: string
  error: 'validation_error'
  errors: ValidationErrorItem[]
}
