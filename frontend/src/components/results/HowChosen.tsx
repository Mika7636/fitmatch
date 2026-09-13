/**
 * "How these were chosen" -- the explainability panel, open by default.
 *
 * Order matches the order the backend actually applied things: hard
 * constraints and what each removed, then the soft constraints that were
 * relaxed, then the final candidate count, then the scoring mode.
 *
 * Each step names the technique behind it in lower-case small print --
 * "constraint-based filter", "TF-IDF cosine similarity" -- rather than the
 * "Chapter 7 ·" / "Chapter 3 ·" prefixes it used to carry. The mapping is
 * still on screen; it is just no longer the first thing read.
 *
 * Two rules, both from the session-3 contract:
 *
 *  1. The relaxation sentences are `constraints.relaxed[].message`, rendered
 *     verbatim. The backend writes them to be displayed; paraphrasing them
 *     here would put a second, drifting copy of the explanation in the app.
 *  2. Nothing here is styled as a warning. Relaxation is the normal path --
 *     the median profile matches zero products with every preference applied
 *     -- and the cold-start ranker serves 85.2% of requests. Amber borders on
 *     either would misreport a working system as a degraded one.
 */

import { useState } from 'react'

import { formatCount, titleCase } from '../../lib/format'
import type { ConstraintsBlock, ScoringBlock } from '../../lib/types'
import { Badge, cx } from '../ui/primitives'

/** What each hard constraint means, in words. Labels only -- no logic. */
const HARD_CONSTRAINT_COPY: Record<string, string> = {
  budget: 'Price inside your budget',
  gender: 'Made for your gender, or unisex',
  stock: 'In stock right now',
  sport: 'Footwear built for your sport, or lifestyle',
}

export function ScoringModeBadge({ scoring }: { scoring: ScoringBlock }) {
  return (
    <Badge tone="neutral" title={scoring.explanation}>
      <span
        aria-hidden="true"
        className="h-1.5 w-1.5 rounded-full bg-ash-muted"
      />
      {scoring.label}
    </Badge>
  )
}

function Stat({
  value,
  label,
  emphasis,
}: {
  value: string
  label: string
  emphasis?: boolean
}) {
  return (
    <div className="border border-ink-600 bg-ink-900 p-3">
      <p
        className={cx(
          'font-display text-2xl font-black',
          emphasis ? 'text-crimson-500' : 'text-ash',
        )}
      >
        {value}
      </p>
      <p className="mt-0.5 text-[11px] leading-tight text-ash-muted">{label}</p>
    </div>
  )
}

export function HowChosen({
  constraints,
  scoring,
  latencyMs,
}: {
  constraints: ConstraintsBlock
  scoring: ScoringBlock
  latencyMs: number
}) {
  const [open, setOpen] = useState(true)

  return (
    <section
      aria-labelledby="how-chosen-heading"
      className="border border-ink-600 bg-ink-800"
    >
      <div className="flex items-start justify-between gap-4 border-b border-ink-700 p-5">
        <div>
          <h2 id="how-chosen-heading" className="text-xl text-ash">
            How these were chosen
          </h2>
          <p className="mt-1.5 max-w-2xl text-sm leading-relaxed text-ash-dim">
            {constraints.summary}
          </p>
        </div>
        <button
          type="button"
          onClick={() => setOpen((value) => !value)}
          aria-expanded={open}
          aria-controls="how-chosen-body"
          className="shrink-0 border border-ink-600 px-3 py-1.5 font-display text-[11px] font-bold uppercase tracking-[0.12em] text-ash-dim hover:border-ash-muted hover:text-ash"
        >
          {open ? 'Hide' : 'Show'}
        </button>
      </div>

      <div id="how-chosen-body" hidden={!open} className="space-y-7 p-5">
        {/* 1 -- the hard filter */}
        <div>
          <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
            <h3 className="text-base text-ash">1 — Hard constraints</h3>
            <span className="text-[11px] lowercase text-ash-muted">
              constraint-based filter · never relaxed
            </span>
          </div>

          <ul className="mt-3 space-y-px">
            {constraints.hard_applied.map((name) => {
              const removed = constraints.hard_excluded[name] ?? 0
              return (
                <li
                  key={name}
                  className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1 bg-ink-900 px-3 py-2.5"
                >
                  <span className="text-sm text-ash">
                    {HARD_CONSTRAINT_COPY[name] ?? titleCase(name)}
                  </span>
                  <span className="font-mono text-xs text-ash-muted">
                    −{formatCount(removed)} products
                  </span>
                </li>
              )
            })}
          </ul>

          <p className="mt-3 text-sm text-ash-dim">
            {formatCount(constraints.catalogue_size)} products →{' '}
            <strong className="font-display text-base text-ash">
              {formatCount(constraints.candidates_after_hard)}
            </strong>{' '}
            survived all four.
          </p>
        </div>

        {/* 2 -- the soft constraints */}
        <div>
          <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
            <h3 className="text-base text-ash">2 — Your preferences</h3>
            <span className="text-[11px] text-ash-muted">
              soft · relaxed cheapest-first until enough candidates remain
            </span>
          </div>

          <p className="mt-3 text-sm text-ash-dim">
            With every preference applied at once:{' '}
            <strong className="font-display text-base text-ash">
              {formatCount(constraints.candidates_after_soft)}
            </strong>{' '}
            candidate{constraints.candidates_after_soft === 1 ? '' : 's'}. The
            ranker targets {formatCount(constraints.min_results_target)}.
          </p>

          {constraints.relaxed.length > 0 ? (
            <ol className="mt-4 space-y-2">
              {constraints.relaxed.map((item) => (
                <li
                  key={item.constraint}
                  className="border-l-2 border-ink-600 bg-ink-900 p-3.5"
                >
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge tone="outline">Set aside</Badge>
                    <span className="font-display text-xs font-bold uppercase tracking-[0.12em] text-ash">
                      {item.label}
                    </span>
                    {item.requested && (
                      <span className="text-xs text-ash-muted">
                        you asked for {item.requested}
                      </span>
                    )}
                  </div>

                  {/* The API's own sentence, verbatim. */}
                  <p className="mt-2 text-sm leading-relaxed text-ash-dim">
                    {item.message}
                  </p>

                  <p className="mt-2 font-mono text-[11px] text-ash-muted">
                    {formatCount(item.candidates_before)} →{' '}
                    {formatCount(item.candidates_after)} candidates
                  </p>
                </li>
              ))}
            </ol>
          ) : (
            <p className="mt-4 border-l-2 border-crimson-700 bg-ink-900 p-3.5 text-sm text-ash-dim">
              Every preference you stated was honoured — nothing had to be set
              aside.
            </p>
          )}

          {constraints.still_applied.length > 0 && (
            <p className="mt-3 text-xs text-ash-muted">
              Still applied:{' '}
              {constraints.still_applied.map((name) => titleCase(name)).join(', ')}.
            </p>
          )}

          {constraints.exhausted && (
            <p className="mt-3 text-xs text-ash-muted">
              Every soft constraint was given up and the set is still under
              target — the hard constraints alone are the limit here.
            </p>
          )}
        </div>

        {/* 3 -- the numbers */}
        <div>
          <h3 className="text-base text-ash">3 — What the ranker saw</h3>
          <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
            <Stat
              value={formatCount(constraints.catalogue_size)}
              label="Catalogue"
            />
            <Stat
              value={formatCount(constraints.candidates_after_hard)}
              label="After hard constraints"
            />
            <Stat
              value={formatCount(constraints.candidates_after_soft)}
              label="With all preferences"
            />
            <Stat
              value={formatCount(constraints.final_candidate_count)}
              label="Final candidates ranked"
              emphasis
            />
          </div>
        </div>

        {/* 4 -- the scoring mode */}
        <div>
          <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
            <h3 className="text-base text-ash">4 — Scoring mode</h3>
            <span className="text-[11px] lowercase text-ash-muted">
              TF-IDF cosine similarity
            </span>
          </div>

          <div className="mt-3 flex flex-wrap items-center gap-2">
            <ScoringModeBadge scoring={scoring} />
            <Badge tone="outline">
              {scoring.qualifying_ratings} qualifying rating
              {scoring.qualifying_ratings === 1 ? '' : 's'} · {scoring.min_ratings_for_centroid}{' '}
              needed for history
            </Badge>
          </div>

          <p className="mt-2.5 text-sm leading-relaxed text-ash-dim">
            {scoring.explanation}
          </p>
        </div>

        <p className="border-t border-ink-700 pt-4 font-mono text-[11px] text-ash-muted">
          Answered in {latencyMs.toFixed(0)} ms.
        </p>
      </div>
    </section>
  )
}
