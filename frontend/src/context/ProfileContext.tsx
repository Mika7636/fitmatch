/**
 * The multi-step form's state, and the one rule that makes it correct.
 *
 * ## The `user_id` rule
 *
 * `POST /api/recommend` treats `user_id` as *"score this stored users.csv row
 * and ignore the rest of the body"*. So a body carrying both `user_id` and an
 * edited profile scores the original row and silently discards every edit --
 * the user would see results for somebody else's profile with no indication
 * anything was wrong.
 *
 * This module is therefore the only place `user_id` is ever set on a request.
 * `loadSample()` records the id; **any** call to `setField` clears it. What
 * that buys, concretely:
 *
 *   load sample U00042, submit untouched -> { user_id: "U00042", top_n }
 *   load sample U00042, change one field -> { age, gender, ... } and no user_id
 *
 * `source` exposes which of those is about to happen so the UI can say so out
 * loud rather than leaving it to trust.
 *
 * ## Why strings
 *
 * Every draft field is a string, including the numeric ones. A number-typed
 * field cannot represent "the user has cleared this input and is mid-typing",
 * which is how forms end up unable to delete a digit. Conversion to the typed
 * `ProfileFields` the API expects happens once, in `toProfileFields`, and an
 * empty string means *absent* -- which is exactly what the backend wants,
 * since an omitted field switches its constraint off instead of guessing.
 *
 * State is in memory only. No localStorage: a refresh starts clean, which is
 * the behaviour asked for.
 */

import {
  createContext,
  use,
  useCallback,
  useMemo,
  useReducer,
  useState,
  type ReactNode,
} from 'react'

import { ApiError, postRecommend } from '../lib/api'
import type {
  ProfileFields,
  RecommendRequest,
  RecommendResponse,
  UserProfile,
} from '../lib/types'

// ---------------------------------------------------------------------------
// The draft
// ---------------------------------------------------------------------------
export interface ProfileDraft {
  // Step 1 -- the body
  age: string
  gender: string
  height_cm: string
  weight_kg: string
  shoe_size: string
  apparel_size: string
  foot_arch_type: string
  // Step 2 -- the activity
  primary_sport: string
  fitness_level: string
  workouts_per_week: string
  indoor_or_outdoor: string
  climate: string
  // Step 3 -- the preferences
  budget_min: string
  budget_max: string
  preferred_brands: string[]
  style_preference: string
  color_preference: string
  material_preference: string
}

export type DraftField = keyof ProfileDraft

export const EMPTY_DRAFT: ProfileDraft = {
  age: '',
  gender: '',
  height_cm: '',
  weight_kg: '',
  shoe_size: '',
  apparel_size: '',
  foot_arch_type: '',
  primary_sport: '',
  fitness_level: '',
  workouts_per_week: '',
  indoor_or_outdoor: '',
  climate: '',
  budget_min: '',
  budget_max: '',
  preferred_brands: [],
  style_preference: '',
  color_preference: '',
  material_preference: '',
}

/** Which fields belong to which step, for per-step validation. */
export const STEP_FIELDS: readonly (readonly DraftField[])[] = [
  ['age', 'gender', 'height_cm', 'weight_kg', 'shoe_size', 'apparel_size', 'foot_arch_type'],
  ['primary_sport', 'fitness_level', 'workouts_per_week', 'indoor_or_outdoor', 'climate'],
  [
    'budget_min',
    'budget_max',
    'preferred_brands',
    'style_preference',
    'color_preference',
    'material_preference',
  ],
] as const

export const STEP_TITLES = ['The athlete', 'The training', 'The preferences'] as const

/** Where the pending request's profile comes from. */
export type ProfileSource =
  /** Typed by hand; posts the profile fields. */
  | 'manual'
  /** Loaded from users.csv and untouched; posts `user_id`. */
  | 'sample'
  /** Loaded from users.csv and then edited; posts the edited fields. */
  | 'edited-sample'

export type DraftErrors = Partial<Record<DraftField, string>>

// ---------------------------------------------------------------------------
// Conversion and validation
// ---------------------------------------------------------------------------
function numberOrUndefined(value: string): number | undefined {
  const trimmed = value.trim()
  if (!trimmed) return undefined
  const parsed = Number(trimmed)
  return Number.isFinite(parsed) ? parsed : undefined
}

/**
 * Draft -> the typed body the API takes.
 *
 * Empty strings are dropped rather than sent as `null`: the request model is
 * `extra="forbid"` with every field optional, and an omitted field is what
 * tells `normalize_profile` to leave that constraint switched off.
 */
export function toProfileFields(draft: ProfileDraft): ProfileFields {
  const fields: ProfileFields = {}

  const age = numberOrUndefined(draft.age)
  if (age !== undefined) fields.age = Math.round(age)
  const workouts = numberOrUndefined(draft.workouts_per_week)
  if (workouts !== undefined) fields.workouts_per_week = Math.round(workouts)

  const height = numberOrUndefined(draft.height_cm)
  if (height !== undefined) fields.height_cm = height
  const weight = numberOrUndefined(draft.weight_kg)
  if (weight !== undefined) fields.weight_kg = weight
  const shoe = numberOrUndefined(draft.shoe_size)
  if (shoe !== undefined) fields.shoe_size = shoe
  const budgetMin = numberOrUndefined(draft.budget_min)
  if (budgetMin !== undefined) fields.budget_min = budgetMin
  const budgetMax = numberOrUndefined(draft.budget_max)
  if (budgetMax !== undefined) fields.budget_max = budgetMax

  if (draft.gender) fields.gender = draft.gender
  if (draft.apparel_size) fields.apparel_size = draft.apparel_size
  if (draft.foot_arch_type) fields.foot_arch_type = draft.foot_arch_type
  if (draft.primary_sport) fields.primary_sport = draft.primary_sport
  if (draft.fitness_level) fields.fitness_level = draft.fitness_level
  if (draft.indoor_or_outdoor) fields.indoor_or_outdoor = draft.indoor_or_outdoor
  if (draft.climate) fields.climate = draft.climate
  if (draft.style_preference) fields.style_preference = draft.style_preference
  if (draft.color_preference) fields.color_preference = draft.color_preference
  if (draft.material_preference) {
    fields.material_preference = draft.material_preference
  }
  if (draft.preferred_brands.length) {
    fields.preferred_brands = [...draft.preferred_brands]
  }

  return fields
}

/** A users.csv row -> the draft, for the "load a sample profile" button. */
export function draftFromUser(user: UserProfile): ProfileDraft {
  const text = (value: number | string | undefined | null): string =>
    value === undefined || value === null ? '' : String(value)

  return {
    age: text(user.age),
    gender: text(user.gender),
    height_cm: text(user.height_cm),
    weight_kg: text(user.weight_kg),
    shoe_size: text(user.shoe_size),
    apparel_size: text(user.apparel_size),
    foot_arch_type: text(user.foot_arch_type),
    primary_sport: text(user.primary_sport),
    fitness_level: text(user.fitness_level),
    workouts_per_week: text(user.workouts_per_week),
    indoor_or_outdoor: text(user.indoor_or_outdoor),
    climate: text(user.climate),
    budget_min: text(user.budget_min),
    budget_max: text(user.budget_max),
    preferred_brands: [...(user.preferred_brands ?? [])],
    style_preference: text(user.style_preference),
    color_preference: text(user.color_preference),
    material_preference: text(user.material_preference),
  }
}

export interface FieldLimits {
  age: { min: number; max: number }
  height_cm: { min: number; max: number }
  weight_kg: { min: number; max: number }
  shoe_size: { min: number; max: number }
  workouts_per_week: { min: number; max: number }
}

/**
 * Client-side validation mirroring the Pydantic rules in
 * `src/api/schemas.py`.
 *
 * A mirror, not a replacement: the API still validates, and a 422 it returns
 * is surfaced on the offending field. This exists so the common mistakes are
 * caught without a round trip, not so the backend can be trusted less.
 */
export function validateDraft(
  draft: ProfileDraft,
  limits: FieldLimits,
): DraftErrors {
  const errors: DraftErrors = {}

  const checkRange = (
    field: 'age' | 'height_cm' | 'weight_kg' | 'shoe_size' | 'workouts_per_week',
    label: string,
    unit = '',
  ) => {
    const raw = draft[field].trim()
    if (!raw) return
    const value = Number(raw)
    if (!Number.isFinite(value)) {
      errors[field] = `${label} must be a number.`
      return
    }
    const bound = limits[field]
    if (value < bound.min || value > bound.max) {
      errors[field] = `${label} must be between ${bound.min}${unit} and ${bound.max}${unit}.`
    }
  }

  checkRange('age', 'Age')
  checkRange('height_cm', 'Height', 'cm')
  checkRange('weight_kg', 'Weight', 'kg')
  checkRange('shoe_size', 'Shoe size')
  checkRange('workouts_per_week', 'Workouts per week')

  const min = numberOrUndefined(draft.budget_min)
  const max = numberOrUndefined(draft.budget_max)
  if (draft.budget_min.trim() && min === undefined) {
    errors.budget_min = 'Minimum budget must be a number.'
  }
  if (draft.budget_max.trim() && max === undefined) {
    errors.budget_max = 'Maximum budget must be a number.'
  }
  if (min !== undefined && min < 0) {
    errors.budget_min = 'Minimum budget cannot be negative.'
  }
  // The API's own rule is strict: budget_min must be *less than* budget_max,
  // so equal values are rejected too.
  if (min !== undefined && max !== undefined && min >= max) {
    errors.budget_max = `Maximum budget must be more than the minimum ($${min}).`
  }

  return errors
}

// ---------------------------------------------------------------------------
// The form state machine
// ---------------------------------------------------------------------------

/**
 * Everything the `user_id` rule depends on, in one plain object.
 *
 * Kept as a reducer rather than three `useState` calls so the rule itself is a
 * pure function of (state, action) and can be verified without a browser --
 * which `scripts/verify.tsx` does. A rule this easy to get wrong, and this
 * silent when wrong, should not be spread across three setters.
 */
export interface ProfileFormState {
  draft: ProfileDraft
  /** The loaded row while it is still untouched. Null once edited. */
  sampleUserId: string | null
  /** The row that was loaded, kept after editing for display only. */
  loadedFrom: UserProfile | null
}

export const INITIAL_FORM_STATE: ProfileFormState = {
  draft: EMPTY_DRAFT,
  sampleUserId: null,
  loadedFrom: null,
}

export type ProfileAction =
  | { type: 'set-field'; field: DraftField; value: string | string[] }
  | { type: 'load-sample'; user: UserProfile }
  | { type: 'reset' }

export function profileFormReducer(
  state: ProfileFormState,
  action: ProfileAction,
): ProfileFormState {
  switch (action.type) {
    case 'load-sample':
      return {
        draft: draftFromUser(action.user),
        sampleUserId: action.user.user_id,
        loadedFrom: action.user,
      }

    case 'set-field': {
      if (state.draft[action.field] === action.value) return state
      return {
        ...state,
        draft: { ...state.draft, [action.field]: action.value },
        // The rule. An edited profile is no longer the stored row, so the
        // request must stop carrying its id -- otherwise the API scores the
        // row and silently discards every edit.
        sampleUserId: null,
      }
    }

    case 'reset':
      return INITIAL_FORM_STATE

    default:
      return state
  }
}

/**
 * The exact body to POST.
 *
 * Two shapes, never a hybrid: either the stored row by id, or the profile
 * fields. A body carrying both would be scored as the former with the latter
 * thrown away.
 */
export function buildRecommendRequest(
  state: ProfileFormState,
  topN: number,
): RecommendRequest {
  if (state.sampleUserId) return { user_id: state.sampleUserId, top_n: topN }
  return { ...toProfileFields(state.draft), top_n: topN }
}

/** Where the pending request's profile comes from, for the UI to say so. */
export function profileSource(state: ProfileFormState): ProfileSource {
  if (state.sampleUserId) return 'sample'
  return state.loadedFrom ? 'edited-sample' : 'manual'
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------
interface ProfileState {
  draft: ProfileDraft
  setField: <K extends DraftField>(field: K, value: ProfileDraft[K]) => void
  reset: () => void

  /** Populate from a users.csv row and remember the id. */
  loadSample: (user: UserProfile) => void
  sampleUserId: string | null
  loadedFrom: UserProfile | null
  source: ProfileSource

  step: number
  setStep: (step: number) => void
  errors: DraftErrors
  setErrors: (errors: DraftErrors) => void

  topN: number
  setTopN: (topN: number) => void

  /** Exactly what would be posted. Rendered in the UI, and used by submit. */
  buildRecommendRequest: () => RecommendRequest
  submit: () => Promise<RecommendResponse>
  result: RecommendResponse | null
  submitting: boolean
  submitError: unknown
  clearSubmitError: () => void
}

const ProfileContext = createContext<ProfileState | null>(null)

export function ProfileProvider({ children }: { children: ReactNode }) {
  const [form, dispatch] = useReducer(profileFormReducer, INITIAL_FORM_STATE)
  const [step, setStep] = useState(0)
  const [errors, setErrors] = useState<DraftErrors>({})
  const [topN, setTopN] = useState(10)
  const [result, setResult] = useState<RecommendResponse | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState<unknown>(null)

  const setField = useCallback<ProfileState['setField']>((field, value) => {
    dispatch({ type: 'set-field', field, value })
    setErrors((current) => {
      if (!(field in current)) return current
      const next = { ...current }
      delete next[field]
      return next
    })
  }, [])

  const loadSample = useCallback((user: UserProfile) => {
    dispatch({ type: 'load-sample', user })
    setErrors({})
  }, [])

  const reset = useCallback(() => {
    dispatch({ type: 'reset' })
    setErrors({})
    setStep(0)
    setResult(null)
    setSubmitError(null)
  }, [])

  const buildRequest = useCallback(
    () => buildRecommendRequest(form, topN),
    [form, topN],
  )

  const submit = useCallback(async (): Promise<RecommendResponse> => {
    setSubmitting(true)
    setSubmitError(null)
    try {
      const response = await postRecommend(buildRecommendRequest(form, topN))
      setResult(response)
      return response
    } catch (caught) {
      setSubmitError(caught)
      // A 422 the client-side mirror missed still belongs on the fields.
      if (caught instanceof ApiError && caught.isValidation) {
        const mapped: DraftErrors = {}
        for (const item of caught.fieldErrors) {
          const field = item.field.split('.')[0] as DraftField
          if (field in EMPTY_DRAFT) mapped[field] = item.message
        }
        if (Object.keys(mapped).length) {
          setErrors((current) => ({ ...current, ...mapped }))
        }
      }
      throw caught
    } finally {
      setSubmitting(false)
    }
  }, [form, topN])

  const value = useMemo<ProfileState>(
    () => ({
      draft: form.draft,
      setField,
      reset,
      loadSample,
      sampleUserId: form.sampleUserId,
      loadedFrom: form.loadedFrom,
      source: profileSource(form),
      step,
      setStep,
      errors,
      setErrors,
      topN,
      setTopN,
      buildRecommendRequest: buildRequest,
      submit,
      result,
      submitting,
      submitError,
      clearSubmitError: () => setSubmitError(null),
    }),
    [
      form,
      setField,
      reset,
      loadSample,
      step,
      errors,
      topN,
      buildRequest,
      submit,
      result,
      submitting,
      submitError,
    ],
  )

  return <ProfileContext value={value}>{children}</ProfileContext>
}

export function useProfile(): ProfileState {
  const value = use(ProfileContext)
  if (!value) throw new Error('useProfile must be used inside <ProfileProvider>')
  return value
}
