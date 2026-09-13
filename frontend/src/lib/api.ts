/**
 * The only place in the app that calls `fetch`.
 *
 * Components ask for data through these functions and never build a URL or
 * read a status code themselves. That is what keeps the contract in one
 * reviewable file: if `src/api/schemas.py` changes, this file and
 * `lib/types.ts` are the two that have to change with it.
 *
 * Errors arrive as `ApiError`, which keeps the API's own message. The backend
 * writes genuinely useful ones -- a 422 on `primary_sport` names all ten valid
 * sports, a 404 on a user names the id range -- so the UI surfaces
 * `error.detail` rather than inventing "Something went wrong".
 */

import type {
  HealthResponse,
  MetaResponse,
  ProductDetail,
  ProductPage,
  ProductQuery,
  RecommendRequest,
  RecommendResponse,
  UserProfile,
  ValidationErrorBody,
  ValidationErrorItem,
} from './types'

/**
 * Backend origin. `VITE_API_BASE_URL` overrides it; the default matches
 * `run_api.bat`, which serves on 127.0.0.1:8000.
 */
export const API_BASE_URL: string = (
  import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'
).replace(/\/+$/, '')

/** How long any single request may take before we give the user a way out. */
const REQUEST_TIMEOUT_MS = 20_000

/**
 * A failed request, carrying whatever the API said about it.
 *
 * `status` is 0 when the request never reached the server at all -- which in
 * this project almost always means the backend is not running, so that case
 * gets its own message naming `run_api.bat`.
 */
export class ApiError extends Error {
  readonly status: number
  readonly code: string
  readonly hint: string | null
  /** Per-field messages from a 422; empty for every other status. */
  readonly fieldErrors: ValidationErrorItem[]

  constructor(
    message: string,
    options: {
      status: number
      code?: string
      hint?: string | null
      fieldErrors?: ValidationErrorItem[]
    },
  ) {
    super(message)
    this.name = 'ApiError'
    this.status = options.status
    this.code = options.code ?? 'error'
    this.hint = options.hint ?? null
    this.fieldErrors = options.fieldErrors ?? []
  }

  /** True when the API rejected the request body field by field. */
  get isValidation(): boolean {
    return this.status === 422
  }

  /** True when the backend could not be reached at all. */
  get isOffline(): boolean {
    return this.status === 0
  }

  /** Field name -> message, for wiring a 422 back onto the form inputs. */
  byField(): Record<string, string> {
    const out: Record<string, string> = {}
    for (const item of this.fieldErrors) out[item.field] = item.message
    return out
  }
}

function isValidationBody(body: unknown): body is ValidationErrorBody {
  return (
    typeof body === 'object' &&
    body !== null &&
    (body as { error?: unknown }).error === 'validation_error' &&
    Array.isArray((body as { errors?: unknown }).errors)
  )
}

async function toApiError(response: Response): Promise<ApiError> {
  let body: unknown = null
  try {
    body = await response.json()
  } catch {
    // A non-JSON error body (a proxy page, say) leaves `body` null and we
    // fall through to the status text below.
  }

  if (isValidationBody(body)) {
    return new ApiError(body.detail, {
      status: response.status,
      code: body.error,
      fieldErrors: body.errors,
    })
  }

  if (typeof body === 'object' && body !== null && 'detail' in body) {
    const shaped = body as { detail: unknown; error?: unknown; hint?: unknown }
    return new ApiError(String(shaped.detail), {
      status: response.status,
      code: typeof shaped.error === 'string' ? shaped.error : 'error',
      hint: typeof shaped.hint === 'string' ? shaped.hint : null,
    })
  }

  return new ApiError(
    `${response.status} ${response.statusText || 'request failed'}`,
    { status: response.status },
  )
}

async function request<T>(
  path: string,
  init: RequestInit & { signal?: AbortSignal } = {},
): Promise<T> {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)

  // A caller's own signal (a component unmounting) also aborts this request.
  const caller = init.signal
  if (caller) {
    if (caller.aborted) controller.abort()
    else caller.addEventListener('abort', () => controller.abort(), { once: true })
  }

  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      signal: controller.signal,
      headers: { Accept: 'application/json', ...init.headers },
    })
  } catch (error) {
    // An abort the caller asked for is not an error worth rewriting; let it
    // propagate so `useAsync` can ignore it.
    if (caller?.aborted) throw error
    if (controller.signal.aborted) {
      throw new ApiError(
        `The request to ${API_BASE_URL} timed out after ${REQUEST_TIMEOUT_MS / 1000}s.`,
        { status: 0, code: 'timeout' },
      )
    }
    throw new ApiError(`Could not reach the FitMatch API at ${API_BASE_URL}.`, {
      status: 0,
      code: 'network_error',
      hint: 'Start the backend with run_api.bat, then try again.',
    })
  } finally {
    clearTimeout(timeout)
  }

  if (!response.ok) throw await toApiError(response)
  return (await response.json()) as T
}

/** Drop undefined/null/empty entries, then encode. */
function queryString(params: Record<string, unknown>): string {
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === '') continue
    search.set(key, String(value))
  }
  const encoded = search.toString()
  return encoded ? `?${encoded}` : ''
}

// ---------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------

/** `GET /api/health` -- readiness and whether the joblib cache is warm. */
export function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return request<HealthResponse>('/api/health', { signal })
}

/**
 * `GET /api/meta` -- every dropdown, measured from the CSVs at startup.
 *
 * Fetched once and held in `MetaContext`; nothing in the UI may hardcode an
 * option list, because the API rejects any value outside these with a 422.
 */
export function getMeta(signal?: AbortSignal): Promise<MetaResponse> {
  return request<MetaResponse>('/api/meta', { signal })
}

/** `GET /api/products` -- one filtered, paginated page of the catalogue. */
export function getProducts(
  query: ProductQuery = {},
  signal?: AbortSignal,
): Promise<ProductPage> {
  return request<ProductPage>(`/api/products${queryString({ ...query })}`, {
    signal,
  })
}

/** `GET /api/products/{id}` -- every column, for the detail modal. */
export function getProduct(
  productId: string,
  signal?: AbortSignal,
): Promise<ProductDetail> {
  return request<ProductDetail>(
    `/api/products/${encodeURIComponent(productId)}`,
    { signal },
  )
}

/**
 * `GET /api/users/{id}` -- a stored profile for the "load a sample" button.
 *
 * Accepts `42`, `"42"` or `"U00042"`. The response carries
 * `expected_scoring_mode`, so the form can say which ranking mode this user
 * will get before anything is submitted.
 */
export function getUser(
  userId: string | number,
  signal?: AbortSignal,
): Promise<UserProfile> {
  return request<UserProfile>(`/api/users/${encodeURIComponent(String(userId))}`, {
    signal,
  })
}

/**
 * `POST /api/recommend` -- run the two-technique cascade.
 *
 * The body is built by `ProfileContext.buildRecommendRequest()`, which is the
 * only code that decides whether `user_id` belongs in it. Passing a body with
 * both `user_id` and profile fields would score the stored row and discard the
 * profile, so that decision is deliberately not made here.
 */
export function postRecommend(
  body: RecommendRequest,
  signal?: AbortSignal,
): Promise<RecommendResponse> {
  return request<RecommendResponse>('/api/recommend', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
}
