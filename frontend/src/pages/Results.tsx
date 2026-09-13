/**
 * The results page.
 *
 * Reads the response held in `ProfileContext` -- there is no result fetched
 * here, because the submit that produced it happened on the profile form and
 * re-posting on mount would double every request.
 *
 * The zero-result branch is the one worth reading. The hard constraints are
 * never relaxed, so they can legitimately leave nothing; when they do, the API
 * returns a `constraints.summary` naming the constraint that did it and what
 * to change. That sentence is rendered as-is, with a button back to the step
 * that owns the offending field.
 */

import { useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'

import { ProductModal } from '../components/ProductModal'
import { HowChosen, ScoringModeBadge } from '../components/results/HowChosen'
import { ResultCard } from '../components/results/ResultCard'
import { Badge, Button, Eyebrow, cx } from '../components/ui/primitives'
import { useProfile } from '../context/ProfileContext'
import { formatCount } from '../lib/format'
import type { ConstraintsBlock, Recommendation } from '../lib/types'

type SortKey = 'score' | 'price-asc' | 'price-desc' | 'rating'

const SORTS: { key: SortKey; label: string }[] = [
  { key: 'score', label: 'Match score' },
  { key: 'price-asc', label: 'Price ↑' },
  { key: 'price-desc', label: 'Price ↓' },
  { key: 'rating', label: 'Rating' },
]

function sortResults(items: Recommendation[], key: SortKey): Recommendation[] {
  const sorted = [...items]
  switch (key) {
    case 'price-asc':
      return sorted.sort((a, b) => a.product.price - b.product.price)
    case 'price-desc':
      return sorted.sort((a, b) => b.product.price - a.product.price)
    case 'rating':
      return sorted.sort(
        (a, b) => (b.product.avg_rating ?? -1) - (a.product.avg_rating ?? -1),
      )
    default:
      // The API already returns rank order; restoring it is just rank ascending.
      return sorted.sort((a, b) => a.rank - b.rank)
  }
}

/**
 * Which form step owns the constraint that emptied the result set.
 *
 * Budget lives on step 3, gender on step 1, sport on step 2. Stock is nobody's
 * field -- there is nothing the user can change about it -- so it sends them
 * to the preferences step, where widening the budget is the realistic move.
 */
function stepForHardConstraint(constraints: ConstraintsBlock): number {
  const entries = Object.entries(constraints.hard_excluded)
  if (entries.length === 0) return 2
  const [worst] = entries.sort((a, b) => b[1] - a[1])[0]
  if (worst === 'gender') return 0
  if (worst === 'sport') return 1
  return 2
}

function NoResults({ constraints }: { constraints: ConstraintsBlock }) {
  const navigate = useNavigate()
  const { setStep } = useProfile()
  const targetStep = stepForHardConstraint(constraints)
  const stepLabels = ['the athlete', 'the training', 'the preferences']

  return (
    <div className="border-2 border-crimson-700 bg-ink-800 p-6 sm:p-8">
      <Eyebrow>No products qualified</Eyebrow>
      <h2 className="mt-3 text-3xl text-ash">Nothing survived the hard filter</h2>

      {/* The API's own account of what happened. */}
      <p className="mt-4 max-w-3xl text-sm leading-relaxed text-ash-dim">
        {constraints.summary}
      </p>

      <ul className="mt-6 space-y-px">
        {constraints.hard_applied.map((name) => (
          <li
            key={name}
            className="flex items-center justify-between gap-4 bg-ink-900 px-3 py-2.5"
          >
            <span className="text-sm text-ash">{name}</span>
            <span className="font-mono text-xs text-ash-muted">
              −{formatCount(constraints.hard_excluded[name] ?? 0)} products
            </span>
          </li>
        ))}
      </ul>

      <div className="mt-6 flex flex-wrap gap-3">
        <Button
          type="button"
          onClick={() => {
            setStep(targetStep)
            navigate('/profile')
          }}
        >
          Adjust {stepLabels[targetStep]}
        </Button>
        <Link
          to="/catalog"
          className="inline-flex items-center border-2 border-ash/40 px-6 py-3 font-display text-sm font-bold uppercase tracking-[0.12em] text-ash transition-colors hover:border-ash hover:bg-ash/10"
        >
          Browse the catalogue instead
        </Link>
      </div>
    </div>
  )
}

export function Results() {
  const { result, submitting, submit, sampleUserId, loadedFrom } = useProfile()
  const [sort, setSort] = useState<SortKey>('score')
  const [openProductId, setOpenProductId] = useState<string | null>(null)
  const navigate = useNavigate()

  const sorted = useMemo(
    () => (result ? sortResults(result.recommendations, sort) : []),
    [result, sort],
  )
  const bestScore = useMemo(
    () =>
      result?.recommendations.reduce(
        (best, item) => Math.max(best, item.score),
        0,
      ) ?? 0,
    [result],
  )

  // Arriving here directly -- a refresh, or a bookmarked URL. State is in
  // memory only by design, so there is nothing to restore.
  if (!result) {
    return (
      <div className="mx-auto max-w-3xl px-4 py-20 text-center sm:px-6">
        <Eyebrow>No results yet</Eyebrow>
        <h1 className="mt-3 text-4xl text-ash">Build a profile first</h1>
        <p className="mt-4 text-sm leading-relaxed text-ash-dim">
          Results are held in memory for this tab only — nothing is saved to the
          browser — so a refresh clears them. Fill in the three-step form and
          your matches appear here.
        </p>
        <Link
          to="/profile"
          className="mt-8 inline-flex items-center gap-2 border-2 border-crimson-700 bg-crimson-700 px-8 py-4 font-display text-base font-bold uppercase tracking-[0.12em] text-white transition-colors hover:bg-crimson-600"
        >
          Build a profile
          <span aria-hidden="true">→</span>
        </Link>
      </div>
    )
  }

  const { constraints, scoring, recommendations, latency_ms } = result
  const scoredStoredRow = Boolean(result.user_id)

  return (
    <div className="mx-auto max-w-5xl px-4 py-10 sm:px-6 sm:py-14">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <Eyebrow>Your matches</Eyebrow>
          <h1 className="mt-2 text-4xl text-ash sm:text-5xl">
            {recommendations.length > 0
              ? `Top ${recommendations.length}`
              : 'No matches'}
          </h1>
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <ScoringModeBadge scoring={scoring} />
            {scoredStoredRow ? (
              <Badge tone="outline">
                Scored stored profile {result.user_id}
              </Badge>
            ) : (
              <Badge tone="outline">
                Scored the profile you entered
                {loadedFrom && !sampleUserId
                  ? ` (edited from ${loadedFrom.user_id})`
                  : ''}
              </Badge>
            )}
            <Badge tone="outline">
              {formatCount(constraints.final_candidate_count)} candidates ranked
            </Badge>
          </div>
        </div>

        <div className="flex gap-2">
          <Button
            variant="secondary"
            size="sm"
            type="button"
            onClick={() => navigate('/profile')}
          >
            Edit profile
          </Button>
          <Button
            variant="secondary"
            size="sm"
            type="button"
            disabled={submitting}
            onClick={() => void submit()}
          >
            {submitting ? 'Re-running…' : 'Re-run'}
          </Button>
        </div>
      </div>

      <div className="mt-8">
        <HowChosen
          constraints={constraints}
          scoring={scoring}
          latencyMs={latency_ms}
        />
      </div>

      {recommendations.length === 0 ? (
        <div className="mt-8">
          <NoResults constraints={constraints} />
        </div>
      ) : (
        <>
          <div className="mt-10 flex flex-wrap items-center justify-between gap-3 border-b border-ink-700 pb-3">
            <h2 className="text-xl text-ash">Ranked results</h2>
            <div className="flex flex-wrap items-center gap-2">
              <span className="eyebrow text-ash-muted" id="sort-label">
                Sort
              </span>
              <div
                role="group"
                aria-labelledby="sort-label"
                className="flex flex-wrap gap-1"
              >
                {SORTS.map((option) => (
                  <button
                    key={option.key}
                    type="button"
                    aria-pressed={sort === option.key}
                    onClick={() => setSort(option.key)}
                    className={cx(
                      'border px-3 py-1.5 font-display text-[11px] font-bold uppercase tracking-[0.1em] transition-colors',
                      sort === option.key
                        ? 'border-crimson-600 bg-crimson-700 text-white'
                        : 'border-ink-600 text-ash-dim hover:border-ash-muted hover:text-ash',
                    )}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {sort !== 'score' && (
            <p className="mt-3 text-xs text-ash-muted">
              Re-ordered for display. The rank badge keeps the recommender's own
              ordering by match score.
            </p>
          )}

          <div className="mt-5 space-y-4">
            {sorted.map((item) => (
              <ResultCard
                key={item.product.product_id}
                recommendation={item}
                bestScore={bestScore}
                onOpen={setOpenProductId}
              />
            ))}
          </div>
        </>
      )}

      <ProductModal
        productId={openProductId}
        onClose={() => setOpenProductId(null)}
      />
    </div>
  )
}
