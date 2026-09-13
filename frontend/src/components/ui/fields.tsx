/**
 * Form controls: labelled, error-aware, and keyboard-navigable.
 *
 * Every control here takes a real `id` and renders a real `<label for>`, with
 * errors wired through `aria-describedby` and `aria-invalid`. That is the
 * accessibility floor for a form this size, and doing it in the primitives
 * means no individual field can forget.
 *
 * Option lists are always passed in from `/api/meta`. None of these components
 * knows a single catalogue value.
 */

import { useId, type ReactNode } from 'react'

import { titleCase } from '../../lib/format'
import { cx } from './primitives'

// ---------------------------------------------------------------------------
// Field wrapper
// ---------------------------------------------------------------------------
interface FieldProps {
  id: string
  label: string
  error?: string
  hint?: ReactNode
  optional?: boolean
  children: (aria: {
    id: string
    'aria-invalid': boolean | undefined
    'aria-describedby': string | undefined
  }) => ReactNode
}

export function Field({
  id,
  label,
  error,
  hint,
  optional,
  children,
}: FieldProps) {
  const errorId = `${id}-error`
  const hintId = `${id}-hint`
  const describedBy =
    [error ? errorId : null, hint ? hintId : null].filter(Boolean).join(' ') ||
    undefined

  return (
    <div className="flex flex-col gap-1.5">
      <label
        htmlFor={id}
        className="eyebrow flex items-baseline gap-2 text-ash-dim"
      >
        {label}
        {optional && (
          <span className="text-[10px] font-medium normal-case tracking-normal text-ash-muted">
            optional
          </span>
        )}
      </label>

      {children({
        id,
        'aria-invalid': error ? true : undefined,
        'aria-describedby': describedBy,
      })}

      {hint && (
        <p id={hintId} className="text-xs leading-relaxed text-ash-muted">
          {hint}
        </p>
      )}
      {error && (
        <p id={errorId} role="alert" className="text-xs font-medium text-crimson-500">
          {error}
        </p>
      )}
    </div>
  )
}

const CONTROL_BASE =
  'w-full bg-ink-900 border-2 px-3 py-2.5 text-sm text-ash placeholder:text-ash-muted/70 transition-colors'
const CONTROL_OK = 'border-ink-600 hover:border-ash-muted focus:border-crimson-500'
const CONTROL_BAD = 'border-crimson-500'

// ---------------------------------------------------------------------------
// Text / number input
// ---------------------------------------------------------------------------
export function NumberField({
  id,
  label,
  value,
  onChange,
  min,
  max,
  step,
  unit,
  error,
  hint,
  placeholder,
  optional,
}: {
  id: string
  label: string
  value: string
  onChange: (value: string) => void
  min?: number
  max?: number
  step?: number
  unit?: string
  error?: string
  hint?: ReactNode
  placeholder?: string
  optional?: boolean
}) {
  return (
    <Field id={id} label={label} error={error} hint={hint} optional={optional}>
      {(aria) => (
        <div className="relative">
          <input
            {...aria}
            type="number"
            inputMode="decimal"
            value={value}
            min={min}
            max={max}
            step={step}
            placeholder={placeholder}
            onChange={(event) => onChange(event.target.value)}
            className={cx(
              CONTROL_BASE,
              error ? CONTROL_BAD : CONTROL_OK,
              unit && 'pr-12',
            )}
          />
          {unit && (
            <span
              aria-hidden="true"
              className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-xs text-ash-muted"
            >
              {unit}
            </span>
          )}
        </div>
      )}
    </Field>
  )
}

// ---------------------------------------------------------------------------
// Select
// ---------------------------------------------------------------------------
export function SelectField({
  id,
  label,
  value,
  onChange,
  options,
  placeholder = 'No preference',
  error,
  hint,
  optional,
  loading,
}: {
  id: string
  label: string
  value: string
  onChange: (value: string) => void
  options: string[]
  placeholder?: string
  error?: string
  hint?: ReactNode
  optional?: boolean
  loading?: boolean
}) {
  return (
    <Field id={id} label={label} error={error} hint={hint} optional={optional}>
      {(aria) => (
        <select
          {...aria}
          value={value}
          disabled={loading}
          onChange={(event) => onChange(event.target.value)}
          className={cx(
            CONTROL_BASE,
            error ? CONTROL_BAD : CONTROL_OK,
            'appearance-none bg-[length:14px] bg-[right_0.75rem_center] bg-no-repeat pr-9 disabled:opacity-50',
          )}
          style={{
            backgroundImage:
              "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 12 8' fill='%239a9a9a'%3E%3Cpath d='M1 1l5 5 5-5'/%3E%3C/svg%3E\")",
          }}
        >
          <option value="">{loading ? 'Loading…' : placeholder}</option>
          {options.map((option) => (
            <option key={option} value={option}>
              {titleCase(option)}
            </option>
          ))}
        </select>
      )}
    </Field>
  )
}

// ---------------------------------------------------------------------------
// Segmented choice -- a radio group that looks like a button bar
// ---------------------------------------------------------------------------
export function ChoiceField({
  label,
  value,
  onChange,
  options,
  error,
  hint,
  optional,
  allowClear = true,
}: {
  label: string
  value: string
  onChange: (value: string) => void
  options: string[]
  error?: string
  hint?: ReactNode
  optional?: boolean
  allowClear?: boolean
}) {
  const groupId = useId()
  const errorId = `${groupId}-error`
  const hintId = `${groupId}-hint`

  return (
    <fieldset
      aria-describedby={
        [error ? errorId : null, hint ? hintId : null].filter(Boolean).join(' ') ||
        undefined
      }
    >
      <legend className="eyebrow mb-1.5 flex items-baseline gap-2 text-ash-dim">
        {label}
        {optional && (
          <span className="text-[10px] font-medium normal-case tracking-normal text-ash-muted">
            optional
          </span>
        )}
      </legend>

      <div className="flex flex-wrap gap-2">
        {options.map((option) => {
          const selected = value === option
          return (
            <button
              key={option}
              type="button"
              aria-pressed={selected}
              onClick={() => onChange(selected && allowClear ? '' : option)}
              className={cx(
                'border-2 px-3.5 py-2 font-display text-xs font-bold uppercase tracking-[0.1em] transition-colors',
                selected
                  ? 'border-crimson-600 bg-crimson-700 text-white'
                  : 'border-ink-600 bg-ink-900 text-ash-dim hover:border-ash-muted hover:text-ash',
              )}
            >
              {titleCase(option)}
            </button>
          )
        })}
      </div>

      {hint && (
        <p id={hintId} className="mt-1.5 text-xs leading-relaxed text-ash-muted">
          {hint}
        </p>
      )}
      {error && (
        <p id={errorId} role="alert" className="mt-1.5 text-xs font-medium text-crimson-500">
          {error}
        </p>
      )}
    </fieldset>
  )
}

// ---------------------------------------------------------------------------
// Multi-select -- preferred brands
// ---------------------------------------------------------------------------
export function MultiSelectField({
  label,
  values,
  onChange,
  options,
  hint,
  optional,
}: {
  label: string
  values: string[]
  onChange: (values: string[]) => void
  options: string[]
  hint?: ReactNode
  optional?: boolean
}) {
  const groupId = useId()
  const hintId = `${groupId}-hint`

  const toggle = (option: string) => {
    onChange(
      values.includes(option)
        ? values.filter((value) => value !== option)
        : [...values, option],
    )
  }

  return (
    <fieldset aria-describedby={hint ? hintId : undefined}>
      <legend className="eyebrow mb-1.5 flex items-baseline gap-2 text-ash-dim">
        {label}
        {optional && (
          <span className="text-[10px] font-medium normal-case tracking-normal text-ash-muted">
            optional
          </span>
        )}
      </legend>

      <div className="flex flex-wrap gap-2">
        {options.map((option) => {
          const selected = values.includes(option)
          return (
            <button
              key={option}
              type="button"
              role="checkbox"
              aria-checked={selected}
              onClick={() => toggle(option)}
              className={cx(
                'flex items-center gap-2 border-2 px-3.5 py-2 font-display text-xs font-bold uppercase tracking-[0.1em] transition-colors',
                selected
                  ? 'border-crimson-600 bg-crimson-700 text-white'
                  : 'border-ink-600 bg-ink-900 text-ash-dim hover:border-ash-muted hover:text-ash',
              )}
            >
              <span
                aria-hidden="true"
                className={cx(
                  'flex h-3.5 w-3.5 items-center justify-center border',
                  selected ? 'border-white bg-white/20' : 'border-ash-muted',
                )}
              >
                {selected && (
                  <svg viewBox="0 0 12 12" className="h-2.5 w-2.5" fill="none">
                    <path
                      d="M2 6.5l2.5 2.5L10 3"
                      stroke="currentColor"
                      strokeWidth="2.5"
                    />
                  </svg>
                )}
              </span>
              {option}
            </button>
          )
        })}
      </div>

      {hint && (
        <p id={hintId} className="mt-1.5 text-xs leading-relaxed text-ash-muted">
          {hint}
        </p>
      )}
    </fieldset>
  )
}
