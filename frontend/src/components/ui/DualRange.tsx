/**
 * A two-handle range slider, built from two stacked native `<input
 * type="range">`.
 *
 * Native inputs rather than pointer maths: they are keyboard-operable for free
 * (arrows, Home, End, Page Up/Down), they announce themselves correctly to a
 * screen reader, and they respect the OS's own touch targets. The CSS in
 * `index.css` makes each input's track transparent so only the thumbs show,
 * and the visible track is drawn by this component.
 *
 * Both handles stay usable when they meet: the two inputs are stacked with
 * `pointer-events: none` on the track and `auto` on the thumbs, so whichever
 * thumb is under the cursor takes the drag.
 */

import { useId } from 'react'

import { cx } from './primitives'

interface DualRangeProps {
  label: string
  min: number
  max: number
  step: number
  low: number
  high: number
  onChange: (low: number, high: number) => void
  format: (value: number) => string
  error?: string
  hint?: string
  /**
   * True when no range has been chosen yet.
   *
   * The thumbs still sit at the extremes, but the readout says so instead of
   * naming a range that is not being applied -- an untouched budget sends no
   * budget at all, and a slider reading "$10 - $450" would be claiming a
   * constraint that is not in the request.
   */
  unset?: boolean
  unsetLabel?: string
}

export function DualRange({
  label,
  min,
  max,
  step,
  low,
  high,
  onChange,
  format,
  error,
  hint,
  unset = false,
  unsetLabel = 'Any',
}: DualRangeProps) {
  const baseId = useId()
  const lowId = `${baseId}-low`
  const highId = `${baseId}-high`
  const errorId = `${baseId}-error`
  const hintId = `${baseId}-hint`

  const span = Math.max(1, max - min)
  const lowPercent = ((Math.min(low, high) - min) / span) * 100
  const highPercent = ((Math.max(low, high) - min) / span) * 100

  // Each handle is clamped against the other so the pair can never cross --
  // the API rejects budget_min >= budget_max, and a slider that can produce
  // one is a slider that produces 422s.
  const handleLow = (value: number) => onChange(Math.min(value, high - step), high)
  const handleHigh = (value: number) => onChange(low, Math.max(value, low + step))

  const describedBy =
    [error ? errorId : null, hint ? hintId : null].filter(Boolean).join(' ') ||
    undefined

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
        <span className="eyebrow text-ash-dim">{label}</span>
        <span className="font-display text-lg font-bold text-ash">
          {unset ? (
            <span className="text-ash-muted">{unsetLabel}</span>
          ) : (
            <>
              {format(low)} <span className="text-ash-muted">—</span>{' '}
              {format(high)}
            </>
          )}
        </span>
      </div>

      <div className="relative h-6">
        {/* Track */}
        <div className="absolute top-1/2 h-1.5 w-full -translate-y-1/2 bg-ink-600" />
        {/* Selected span */}
        <div
          className={cx(
            'absolute top-1/2 h-1.5 -translate-y-1/2',
            unset ? 'bg-ink-600' : 'bg-crimson-600',
          )}
          style={{
            left: `${lowPercent}%`,
            width: `${Math.max(0, highPercent - lowPercent)}%`,
          }}
        />

        <label htmlFor={lowId} className="sr-only">
          {label} — minimum
        </label>
        <input
          id={lowId}
          type="range"
          min={min}
          max={max}
          step={step}
          value={low}
          aria-describedby={describedBy}
          aria-valuetext={format(low)}
          onChange={(event) => handleLow(Number(event.target.value))}
          className="range-thumb absolute inset-0 h-6 w-full"
        />

        <label htmlFor={highId} className="sr-only">
          {label} — maximum
        </label>
        <input
          id={highId}
          type="range"
          min={min}
          max={max}
          step={step}
          value={high}
          aria-describedby={describedBy}
          aria-valuetext={format(high)}
          onChange={(event) => handleHigh(Number(event.target.value))}
          className="range-thumb absolute inset-0 h-6 w-full"
        />
      </div>

      <div className="mt-1 flex justify-between text-[11px] text-ash-muted">
        <span>{format(min)}</span>
        <span>{format(max)}</span>
      </div>

      {hint && (
        <p id={hintId} className={cx('mt-2 text-xs leading-relaxed text-ash-muted')}>
          {hint}
        </p>
      )}
      {error && (
        <p id={errorId} role="alert" className="mt-2 text-xs font-medium text-crimson-500">
          {error}
        </p>
      )}
    </div>
  )
}
