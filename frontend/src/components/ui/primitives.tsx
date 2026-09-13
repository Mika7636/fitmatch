/**
 * The small shared pieces: buttons, badges, chips, panels, skeletons, errors.
 *
 * Kept in one file because each is a handful of lines and they are always
 * imported together. Anything with real behaviour (the dual slider, the
 * multi-select, the modal) gets its own file.
 */

import type { ButtonHTMLAttributes, ReactNode } from 'react'

import { ApiError } from '../../lib/api'

export function cx(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(' ')
}

// ---------------------------------------------------------------------------
// Button
// ---------------------------------------------------------------------------
type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'onLight'

const BUTTON_STYLES: Record<ButtonVariant, string> = {
  primary:
    'bg-crimson-700 text-white hover:bg-crimson-600 active:bg-crimson-800 border-2 border-crimson-700 hover:border-crimson-600',
  secondary:
    'bg-transparent text-ash border-2 border-ash/40 hover:border-ash hover:bg-ash/10',
  ghost: 'bg-transparent text-ash-dim hover:text-ash border-2 border-transparent',
  onLight:
    'bg-ink-900 text-ash border-2 border-ink-900 hover:bg-crimson-700 hover:border-crimson-700',
}

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant
  size?: 'sm' | 'md' | 'lg'
}

export function Button({
  variant = 'primary',
  size = 'md',
  className,
  children,
  ...rest
}: ButtonProps) {
  const sizes = {
    sm: 'px-4 py-2 text-xs',
    md: 'px-6 py-3 text-sm',
    lg: 'px-8 py-4 text-base',
  }
  return (
    <button
      className={cx(
        'font-display uppercase tracking-[0.12em] font-bold transition-colors duration-150',
        'disabled:opacity-40 disabled:cursor-not-allowed',
        sizes[size],
        BUTTON_STYLES[variant],
        className,
      )}
      {...rest}
    >
      {children}
    </button>
  )
}

// ---------------------------------------------------------------------------
// Labels and panels
// ---------------------------------------------------------------------------
export function Eyebrow({
  children,
  className,
}: {
  children: ReactNode
  className?: string
}) {
  return <p className={cx('eyebrow text-crimson-500', className)}>{children}</p>
}

/** A dark panel with the diagonal crimson edge the deck uses. */
export function Panel({
  children,
  className,
}: {
  children: ReactNode
  className?: string
}) {
  return (
    <div
      className={cx(
        'relative border border-ink-600 bg-ink-800',
        'before:absolute before:left-0 before:top-0 before:h-full before:w-1 before:bg-crimson-700',
        className,
      )}
    >
      {children}
    </div>
  )
}

/**
 * A neutral informational chip.
 *
 * `tone` never includes a warning colour on purpose: the two things this app
 * has most to say -- that constraints were relaxed, and that the cold-start
 * ranker was used -- are normal outcomes, not problems, and colouring them
 * amber would misreport the system.
 */
export function Badge({
  children,
  tone = 'neutral',
  className,
  title,
}: {
  children: ReactNode
  tone?: 'neutral' | 'crimson' | 'outline'
  className?: string
  title?: string
}) {
  const tones = {
    neutral: 'bg-ink-700 text-ash-dim border border-ink-600',
    crimson: 'bg-crimson-700 text-white border border-crimson-600',
    outline: 'bg-transparent text-ash-dim border border-ash/30',
  }
  return (
    <span
      title={title}
      className={cx(
        'inline-flex items-center gap-1.5 px-2.5 py-1 text-[11px] font-medium leading-tight',
        tones[tone],
        className,
      )}
    >
      {children}
    </span>
  )
}

// ---------------------------------------------------------------------------
// Skeletons
// ---------------------------------------------------------------------------
export function Skeleton({ className }: { className?: string }) {
  return <div className={cx('skeleton', className)} aria-hidden="true" />
}

/** A whole loading screen, announced to assistive tech. */
export function LoadingBlock({
  label,
  children,
}: {
  label: string
  children: ReactNode
}) {
  return (
    <div role="status" aria-live="polite" aria-busy="true">
      <span className="sr-only">{label}</span>
      {children}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------
/**
 * The one error surface.
 *
 * It prints what the API actually said. The backend's messages are written to
 * be read -- a 422 names every valid value for the field, a 404 names the id
 * range -- so replacing them with "Something went wrong" would throw away the
 * most useful part of the response.
 */
export function ErrorState({
  error,
  onRetry,
  className,
}: {
  error: unknown
  onRetry?: () => void
  className?: string
}) {
  const apiError = error instanceof ApiError ? error : null
  const message =
    apiError?.message ??
    (error instanceof Error ? error.message : 'The request failed.')

  return (
    <div
      role="alert"
      className={cx(
        'border-2 border-crimson-600 bg-crimson-900/20 p-5 sm:p-6',
        className,
      )}
    >
      <p className="eyebrow text-crimson-500">
        {apiError?.isOffline
          ? 'Backend unreachable'
          : `Request failed${apiError ? ` — ${apiError.status}` : ''}`}
      </p>
      <p className="mt-2 text-sm text-ash">{message}</p>

      {apiError?.hint && (
        <p className="mt-2 text-sm text-ash-muted">{apiError.hint}</p>
      )}

      {apiError?.isValidation && apiError.fieldErrors.length > 0 && (
        <ul className="mt-3 space-y-1 text-sm text-ash-dim">
          {apiError.fieldErrors.map((item) => (
            <li key={`${item.field}-${item.type}`}>
              <span className="font-mono text-xs text-crimson-500">
                {item.field}
              </span>{' '}
              {item.message}
            </li>
          ))}
        </ul>
      )}

      {onRetry && (
        <Button variant="secondary" size="sm" className="mt-4" onClick={onRetry}>
          Try again
        </Button>
      )}
    </div>
  )
}

/** An empty result that is not an error -- no products, no catalogue matches. */
export function EmptyState({
  title,
  children,
}: {
  title: string
  children?: ReactNode
}) {
  return (
    <div className="border border-dashed border-ink-600 bg-ink-800/60 p-8 text-center">
      <h3 className="text-lg text-ash">{title}</h3>
      {children && <div className="mt-2 text-sm text-ash-muted">{children}</div>}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Rating
// ---------------------------------------------------------------------------
export function Stars({
  rating,
  className,
}: {
  rating: number | null
  className?: string
}) {
  if (rating === null) {
    return <span className={cx('text-xs text-ash-muted', className)}>No rating</span>
  }
  const rounded = Math.round(rating * 2) / 2
  return (
    <span
      className={cx('inline-flex items-center gap-1', className)}
      aria-label={`Rated ${rating.toFixed(1)} out of 5`}
    >
      <span aria-hidden="true" className="inline-flex">
        {[1, 2, 3, 4, 5].map((position) => {
          const filled = rounded >= position
          const half = !filled && rounded >= position - 0.5
          return (
            <svg
              key={position}
              viewBox="0 0 20 20"
              className="h-3.5 w-3.5"
              fill={filled || half ? 'currentColor' : 'none'}
              stroke="currentColor"
              strokeWidth="1.5"
              style={{ color: filled || half ? '#C41010' : '#5a5a5a' }}
            >
              {half && (
                <defs>
                  <linearGradient id={`half-${position}`}>
                    <stop offset="50%" stopColor="#C41010" />
                    <stop offset="50%" stopColor="transparent" />
                  </linearGradient>
                </defs>
              )}
              <path
                d="M10 1.5l2.6 5.3 5.9.9-4.2 4.1 1 5.8-5.3-2.8-5.3 2.8 1-5.8L1.5 7.7l5.9-.9z"
                fill={half ? `url(#half-${position})` : undefined}
              />
            </svg>
          )
        })}
      </span>
      <span className="text-xs text-ash-muted">{rating.toFixed(1)}</span>
    </span>
  )
}
