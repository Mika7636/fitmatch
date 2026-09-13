/**
 * The three-step form.
 *
 * Step state and draft state both live in `ProfileContext`, so moving between
 * steps -- or leaving for the catalogue and coming back -- keeps everything
 * typed so far. Nothing is persisted beyond the tab's lifetime.
 */

import { useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { SampleLoader } from '../components/profile/SampleLoader'
import {
  StepAthlete,
  StepPreferences,
  StepTraining,
} from '../components/profile/steps'
import {
  Button,
  ErrorState,
  Eyebrow,
  Skeleton,
  cx,
} from '../components/ui/primitives'
import { limitFor, useMeta } from '../context/MetaContext'
import {
  STEP_FIELDS,
  STEP_TITLES,
  useProfile,
  validateDraft,
  type DraftErrors,
} from '../context/ProfileContext'

function StepIndicator({
  step,
  onSelect,
}: {
  step: number
  onSelect: (index: number) => void
}) {
  return (
    <ol className="flex flex-wrap gap-2" aria-label="Progress">
      {STEP_TITLES.map((title, index) => {
        const state =
          index === step ? 'current' : index < step ? 'done' : 'upcoming'
        return (
          <li key={title} className="flex-1 basis-40">
            <button
              type="button"
              onClick={() => onSelect(index)}
              aria-current={state === 'current' ? 'step' : undefined}
              className={cx(
                'group flex w-full flex-col gap-2 text-left',
                state === 'upcoming' && 'opacity-60 hover:opacity-100',
              )}
            >
              <span
                aria-hidden="true"
                className={cx(
                  'block h-1.5 w-full transition-colors',
                  state === 'current'
                    ? 'bg-crimson-600'
                    : state === 'done'
                      ? 'bg-crimson-800'
                      : 'bg-ink-600 group-hover:bg-ink-600/80',
                )}
              />
              <span
                className={cx(
                  'font-display text-[11px] font-bold uppercase tracking-[0.14em]',
                  state === 'current' ? 'text-ash' : 'text-ash-muted',
                )}
              >
                <span className="text-crimson-500">{index + 1}</span> {title}
              </span>
            </button>
          </li>
        )
      })}
    </ol>
  )
}

function FormSkeleton() {
  return (
    <div className="space-y-6">
      <Skeleton className="h-8 w-48" />
      <div className="grid gap-6 sm:grid-cols-2">
        {Array.from({ length: 4 }).map((_, index) => (
          <div key={index} className="space-y-2">
            <Skeleton className="h-3 w-24" />
            <Skeleton className="h-11 w-full" />
          </div>
        ))}
      </div>
      <Skeleton className="h-3 w-20" />
      <Skeleton className="h-11 w-full max-w-sm" />
    </div>
  )
}

export function ProfileBuilder() {
  const navigate = useNavigate()
  const { meta, loading, error, reload } = useMeta()
  const {
    draft,
    step,
    setStep,
    setErrors,
    submit,
    submitting,
    submitError,
    buildRecommendRequest,
    topN,
    setTopN,
  } = useProfile()
  const [showRequest, setShowRequest] = useState(false)

  const limits = {
    age: limitFor(meta, 'age'),
    height_cm: limitFor(meta, 'height_cm'),
    weight_kg: limitFor(meta, 'weight_kg'),
    shoe_size: limitFor(meta, 'shoe_size'),
    workouts_per_week: limitFor(meta, 'workouts_per_week'),
  }

  /** Validate only the fields on `index`, so Next does not flag later steps. */
  const validateStep = (index: number): boolean => {
    const all = validateDraft(draft, limits)
    const scoped: DraftErrors = {}
    for (const field of STEP_FIELDS[index]) {
      if (all[field]) scoped[field] = all[field]
    }
    setErrors(scoped)
    return Object.keys(scoped).length === 0
  }

  const goNext = () => {
    if (!validateStep(step)) return
    setStep(Math.min(step + 1, STEP_TITLES.length - 1))
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const goBack = () => {
    setStep(Math.max(step - 1, 0))
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const onSubmit = async () => {
    // The whole draft this time, not just the visible step.
    const all = validateDraft(draft, limits)
    if (Object.keys(all).length > 0) {
      setErrors(all)
      const firstBadStep = STEP_FIELDS.findIndex((fields) =>
        fields.some((field) => all[field]),
      )
      if (firstBadStep >= 0) setStep(firstBadStep)
      window.scrollTo({ top: 0, behavior: 'smooth' })
      return
    }
    try {
      await submit()
      navigate('/results')
    } catch {
      // `submitError` is rendered below; a 422 is also mapped onto the fields
      // by the context, so there is nothing to do here.
    }
  }

  const request = buildRecommendRequest()

  return (
    <div className="mx-auto max-w-4xl px-4 py-10 sm:px-6 sm:py-14">
      <Eyebrow>Profile builder</Eyebrow>
      <h1 className="mt-2 text-4xl text-ash sm:text-5xl">Build your profile</h1>
      <p className="mt-3 max-w-2xl text-sm leading-relaxed text-ash-dim">
        Every field is optional. Anything you leave blank switches off the
        constraint that reads it rather than being guessed at — so a half-filled
        form still returns sensible results, just a wider set.
      </p>

      <div className="mt-8">
        <StepIndicator step={step} onSelect={setStep} />
      </div>

      <div className="mt-8">
        <SampleLoader
          idRange={meta?.user_id_range ?? { min: 'U00001', max: 'U05000' }}
        />
      </div>

      {error != null && (
        <ErrorState
          error={error}
          onRetry={reload}
          className="mt-8"
        />
      )}

      <form
        className="mt-8"
        onSubmit={(event) => {
          event.preventDefault()
          if (step === STEP_TITLES.length - 1) void onSubmit()
          else goNext()
        }}
      >
        {loading && !meta ? (
          <FormSkeleton />
        ) : (
          <>
            {step === 0 && <StepAthlete />}
            {step === 1 && <StepTraining />}
            {step === 2 && <StepPreferences />}
          </>
        )}

        {step === STEP_TITLES.length - 1 && (
          <div className="mt-8 border border-ink-600 bg-ink-800 p-5">
            <label
              htmlFor="top_n"
              className="eyebrow block text-ash-dim"
            >
              How many results
            </label>
            <input
              id="top_n"
              type="range"
              min={1}
              max={50}
              step={1}
              value={topN}
              onChange={(event) => setTopN(Number(event.target.value))}
              className="range-thumb mt-3 h-6 w-full bg-ink-600"
              style={{ background: 'transparent' }}
              aria-valuetext={`${topN} results`}
            />
            <p className="mt-1 text-sm text-ash">
              Top <strong className="font-display text-lg">{topN}</strong>
            </p>
          </div>
        )}

        {submitError != null && (
          <ErrorState error={submitError} className="mt-8" />
        )}

        <div className="mt-8 flex flex-wrap items-center gap-3 border-t border-ink-700 pt-6">
          {step > 0 && (
            <Button variant="secondary" type="button" onClick={goBack}>
              Back
            </Button>
          )}

          <Button type="submit" disabled={submitting} className="min-w-44">
            {step === STEP_TITLES.length - 1
              ? submitting
                ? 'Matching…'
                : 'Find my gear'
              : 'Next'}
          </Button>

          <button
            type="button"
            onClick={() => setShowRequest((open) => !open)}
            aria-expanded={showRequest}
            className="ml-auto font-display text-[11px] font-bold uppercase tracking-[0.14em] text-ash-muted underline decoration-dotted underline-offset-4 hover:text-ash"
          >
            {showRequest ? 'Hide' : 'Show'} request body
          </button>
        </div>
      </form>

      {/*
        The exact JSON that will be posted. It is here because the single
        easiest mistake in this app is shipping `user_id` alongside an edited
        profile, and the cheapest way to prove it is not happening is to show
        the body.
      */}
      {showRequest && (
        <div className="mt-4 border border-ink-600 bg-ink-950 p-4">
          <p className="eyebrow text-ash-muted">POST /api/recommend</p>
          <pre className="mt-2 overflow-x-auto text-xs leading-relaxed text-ash-dim">
            {JSON.stringify(request, null, 2)}
          </pre>
          <p className="mt-2 text-[11px] text-ash-muted">
            {'user_id' in request
              ? 'Scoring the stored row by id — the other fields are deliberately absent, because the API would ignore them.'
              : 'No user_id: the API scores exactly these fields.'}
          </p>
        </div>
      )}
    </div>
  )
}
