/**
 * Product detail, fetched on open from `GET /api/products/{id}`.
 *
 * Shared by the results list and the catalogue grid, which is why it takes a
 * product id rather than a product: both screens hold only the summary fields,
 * and the description, cushioning, breathability and the rest come from the
 * detail endpoint.
 */

import { useAsync } from '../hooks/useAsync'
import { getProduct } from '../lib/api'
import { formatPrice, sentenceCase, titleCase } from '../lib/format'
import { ProductPhoto } from './ui/ProductPhoto'
import { Modal } from './ui/Modal'
import { Badge, Button, ErrorState, Skeleton, Stars } from './ui/primitives'

function Spec({ label, value }: { label: string; value: string }) {
  return (
    <div className="border border-ink-600 bg-ink-900 p-3">
      <dt className="eyebrow text-ash-muted">{label}</dt>
      <dd className="mt-1 text-sm text-ash">{value}</dd>
    </div>
  )
}

/** 1–5 integer levels rendered as a bar, for breathability and cushioning. */
function Level({ label, value }: { label: string; value: number }) {
  return (
    <div className="border border-ink-600 bg-ink-900 p-3">
      <dt className="eyebrow text-ash-muted">{label}</dt>
      <dd className="mt-1.5 flex items-center gap-2">
        <span className="flex gap-0.5" aria-hidden="true">
          {[1, 2, 3, 4, 5].map((step) => (
            <span
              key={step}
              className={
                step <= value ? 'h-3 w-4 bg-crimson-600' : 'h-3 w-4 bg-ink-600'
              }
            />
          ))}
        </span>
        <span className="text-sm text-ash">{value}/5</span>
      </dd>
    </div>
  )
}

export function ProductModal({
  productId,
  onClose,
}: {
  productId: string | null
  onClose: () => void
}) {
  const { data, loading, error, reload } = useAsync(
    (signal) =>
      productId
        ? getProduct(productId, signal)
        : Promise.resolve(null),
    [productId],
  )

  return (
    <Modal
      open={productId !== null}
      onClose={onClose}
      title={data ? `${data.brand} ${titleCase(data.subcategory)}` : 'Product details'}
    >
      <div className="flex items-start justify-between gap-4 border-b border-ink-700 p-4 sm:p-5">
        <div className="min-w-0">
          <p className="eyebrow text-crimson-500">Product detail</p>
          <h2 className="mt-1 truncate text-2xl text-ash">
            {loading || !data
              ? 'Loading…'
              : `${data.brand} ${titleCase(data.subcategory)}`}
          </h2>
          {data && (
            <p className="mt-1 font-mono text-[11px] text-ash-muted">
              {data.product_id}
            </p>
          )}
        </div>
        <Button variant="ghost" size="sm" onClick={onClose} type="button">
          Close ✕
        </Button>
      </div>

      <div className="max-h-[70vh] overflow-y-auto p-4 sm:p-5">
        {loading && (
          <div role="status" aria-live="polite" className="space-y-4">
            <span className="sr-only">Loading product details</span>
            <Skeleton className="h-44 w-full" />
            <Skeleton className="h-5 w-1/3" />
            <Skeleton className="h-16 w-full" />
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
              {Array.from({ length: 6 }).map((_, index) => (
                <Skeleton key={index} className="h-16 w-full" />
              ))}
            </div>
          </div>
        )}

        {error != null && <ErrorState error={error} onRetry={reload} />}

        {data && !loading && (
          <div className="space-y-5">
            <ProductPhoto
              product_id={data.product_id}
              brand={data.brand}
              category={data.category}
              color={data.color}
              sport_type={data.sport_type}
              subcategory={data.subcategory}
              size="modal"
              priority
              className="w-full"
            />
            <p className="-mt-3 text-[11px] text-ash-muted">
              Representative stock photography, matched to this product's
              category and sport — no source dataset carries product images.
            </p>

            <div className="flex flex-wrap items-center justify-between gap-3">
              <p className="font-display text-3xl font-black text-ash">
                {formatPrice(data.price)}
                {data.discount !== null && data.discount > 0 && (
                  <span className="ml-2 align-middle text-sm font-bold text-crimson-500">
                    −{Math.round(data.discount * 100)}%
                  </span>
                )}
              </p>
              <Stars rating={data.avg_rating} />
            </div>

            <div className="flex flex-wrap gap-1.5">
              <Badge tone="crimson">{titleCase(data.sport_type)}</Badge>
              <Badge tone="outline">{titleCase(data.category)}</Badge>
              <Badge tone="outline">{titleCase(data.gender_target)}</Badge>
              <Badge tone="outline">{sentenceCase(data.stock_status)}</Badge>
              {data.waterproof && <Badge tone="outline">Waterproof</Badge>}
            </div>

            {data.product_description && (
              <p className="text-sm leading-relaxed text-ash-dim">
                {data.product_description}
              </p>
            )}

            <dl className="grid grid-cols-2 gap-2 sm:grid-cols-3">
              <Spec label="Brand" value={data.brand} />
              <Spec label="Size" value={data.size} />
              <Spec label="Colour" value={titleCase(data.color)} />
              <Spec label="Material" value={titleCase(data.material)} />
              <Spec label="Seasonality" value={titleCase(data.seasonality)} />
              <Spec label="Subcategory" value={titleCase(data.subcategory)} />
              {data.arch_support && (
                <Spec label="Arch support" value={titleCase(data.arch_support)} />
              )}
              {data.weight !== null && (
                <Spec label="Weight" value={`${data.weight} g`} />
              )}
              {data.review_count !== null && (
                <Spec label="Reviews" value={String(data.review_count)} />
              )}
              {data.breathability !== null && (
                <Level label="Breathability" value={data.breathability} />
              )}
              {data.cushioning_level !== null && (
                <Level label="Cushioning" value={data.cushioning_level} />
              )}
            </dl>

            {/*
              sales_rank is shown as catalogue metadata and nothing more.
              Neither technique reads popularity -- see
              reports/circularity_note.md -- so it must not look like it
              influenced the ranking.
            */}
            {data.sales_rank !== null && (
              <p className="border-t border-ink-700 pt-3 text-[11px] text-ash-muted">
                Catalogue sales rank #{data.sales_rank}. Recorded in the data
                but unused by either recommender technique — neither the
                constraint filter nor the content ranker reads popularity.
              </p>
            )}
          </div>
        )}
      </div>
    </Modal>
  )
}
