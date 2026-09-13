/**
 * One ranked product.
 *
 * The explanation chips are the point of the card, not decoration: the
 * assignment's explainability requirement is that a user can see *why* an item
 * is here, and the API returns those reasons already written, tagged with the
 * technique that produced them.
 *
 * Which technique wrote which group is named under each heading, in lower-case
 * small print. It used to be a pair of "CH7"/"CH3" badge chips, which shouted
 * the coursework's structure over the actual reason the product is on screen.
 * The mapping stays discoverable -- the landing page's explainer spells both
 * techniques out in full -- it just is not the loudest thing on the card.
 */

import { Fragment } from 'react'

import {
  formatPrice,
  formatScore,
  scoreBarWidth,
  sentenceCase,
  titleCase,
} from '../../lib/format'
import type { Explanation, Recommendation } from '../../lib/types'
import { ProductPhoto } from '../ui/ProductPhoto'
import { Badge, Stars, cx } from '../ui/primitives'

function ExplanationChips({ explanations }: { explanations: Explanation[] }) {
  const knowledge = explanations.filter((item) => item.source === 'knowledge_based')
  const content = explanations.filter((item) => item.source === 'content_based')

  const groups = [
    {
      key: 'knowledge',
      title: 'Constraints it satisfies',
      technique: 'constraint-based filter',
      items: knowledge,
      chip: 'border-crimson-700/60 bg-crimson-900/25 text-ash-dim',
    },
    {
      key: 'content',
      title: 'Why it ranked here',
      technique: 'TF-IDF cosine similarity',
      items: content,
      chip: 'border-ink-600 bg-ink-900 text-ash-dim',
    },
  ].filter((group) => group.items.length > 0)

  if (groups.length === 0) return null

  return (
    <div className="mt-4 space-y-3">
      {groups.map((group) => (
        <div key={group.key}>
          <p className="eyebrow text-ash-muted">{group.title}</p>
          <p className="mt-0.5 text-[10px] lowercase italic text-ash-muted/70">
            {group.technique}
          </p>
          <ul className="mt-2 flex flex-wrap gap-1.5">
            {group.items.map((item, index) => (
              <li
                key={`${group.key}-${index}`}
                className={cx('border px-2.5 py-1 text-[11px] leading-snug', group.chip)}
              >
                {item.text}
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  )
}

export function ResultCard({
  recommendation,
  bestScore,
  onOpen,
}: {
  recommendation: Recommendation
  bestScore: number
  onOpen: (productId: string) => void
}) {
  const { product, score, rank, explanations } = recommendation
  const name = `${product.brand} ${titleCase(product.subcategory)}`

  return (
    <article className="group relative border border-ink-600 bg-ink-800 transition-colors hover:border-crimson-700">
      <div className="flex flex-col sm:flex-row">
        <div className="relative shrink-0 sm:w-48">
          <ProductPhoto
            product_id={product.product_id}
            brand={product.brand}
            category={product.category}
            color={product.color}
            sport_type={product.sport_type}
            subcategory={product.subcategory}
            size="card"
            className="w-full"
          />
          <span className="absolute left-0 top-0 bg-crimson-700 px-2.5 py-1 font-display text-sm font-black text-white">
            {rank}
          </span>
        </div>

        <div className="flex-1 p-4 sm:p-5">
          <div className="flex flex-wrap items-start justify-between gap-x-4 gap-y-2">
            <div className="min-w-0">
              <h3 className="text-lg leading-tight text-ash">
                <button
                  type="button"
                  onClick={() => onOpen(product.product_id)}
                  className="text-left hover:text-crimson-500 focus-visible:text-crimson-500"
                >
                  {name}
                  <span className="sr-only"> — open product details</span>
                </button>
              </h3>
              <p className="mt-1 font-mono text-[11px] text-ash-muted">
                {product.product_id} · {titleCase(product.sport_type)} ·{' '}
                {titleCase(product.color)} · {titleCase(product.material)} · size{' '}
                {product.size}
              </p>
            </div>

            <div className="text-right">
              <p className="font-display text-2xl font-black text-ash">
                {formatPrice(product.price)}
              </p>
              <Stars rating={product.avg_rating} className="mt-1 justify-end" />
            </div>
          </div>

          {/* Match score */}
          <div className="mt-4">
            <div className="flex items-baseline justify-between gap-2">
              <span className="eyebrow text-ash-muted">Match score</span>
              <span className="font-display text-sm font-bold text-crimson-500">
                {formatScore(score)}
              </span>
            </div>
            <div
              className="mt-1.5 h-2 w-full bg-ink-900"
              role="img"
              aria-label={`Match score ${formatScore(score)}, rank ${rank}`}
            >
              <div
                className="h-full bg-crimson-600"
                style={{ width: scoreBarWidth(score, bestScore) }}
              />
            </div>
            <p className="mt-1 text-[10px] text-ash-muted">
              Cosine similarity, shown relative to the best match on this page.
            </p>
          </div>

          <div className="mt-3 flex flex-wrap gap-1.5">
            <Badge tone="outline">{sentenceCase(product.stock_status)}</Badge>
            <Badge tone="outline">{titleCase(product.gender_target)}</Badge>
            <Badge tone="outline">{titleCase(product.seasonality)}</Badge>
          </div>

          <ExplanationChips explanations={explanations} />

          <button
            type="button"
            onClick={() => onOpen(product.product_id)}
            className="mt-4 font-display text-[11px] font-bold uppercase tracking-[0.14em] text-ash-muted underline decoration-dotted underline-offset-4 transition-colors hover:text-crimson-500"
          >
            Full details
          </button>
        </div>
      </div>
    </article>
  )
}

export function ResultCardSkeleton() {
  return (
    <div className="border border-ink-600 bg-ink-800">
      <div className="flex flex-col sm:flex-row">
        <div className="skeleton h-40 w-full sm:h-auto sm:w-48" />
        <div className="flex-1 space-y-3 p-5">
          <div className="skeleton h-5 w-2/3" />
          <div className="skeleton h-3 w-1/2" />
          <div className="skeleton h-2 w-full" />
          <div className="flex gap-2">
            {[0, 1, 2].map((index) => (
              <Fragment key={index}>
                <div className="skeleton h-6 w-24" />
              </Fragment>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
