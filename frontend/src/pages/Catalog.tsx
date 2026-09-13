/**
 * Catalogue browse: filter sidebar plus a paginated grid.
 *
 * Paging is the API's, not the client's: `page` and `page_size` go to
 * `GET /api/products` and `total`, `total_pages`, `has_next` and `has_previous`
 * come back from it. Nothing is fetched wholesale and sliced here -- the
 * catalogue is 10,000 rows.
 *
 * Filter state lives in the URL query string so a filtered view can be shared
 * and the browser's back button steps through filter changes.
 */

import { useSearchParams } from 'react-router-dom'

import { ProductModal } from '../components/ProductModal'
import { ProductPhoto } from '../components/ui/ProductPhoto'
import { SelectField } from '../components/ui/fields'
import {
  Badge,
  Button,
  EmptyState,
  ErrorState,
  Eyebrow,
  Skeleton,
  Stars,
  cx,
} from '../components/ui/primitives'
import { useMeta } from '../context/MetaContext'
import { useAsync } from '../hooks/useAsync'
import { getProducts } from '../lib/api'
import { formatCount, formatPrice, titleCase } from '../lib/format'
import type { ProductSummary } from '../lib/types'

const PAGE_SIZE = 24

function ProductTile({
  product,
  onOpen,
}: {
  product: ProductSummary
  onOpen: (productId: string) => void
}) {
  return (
    <li>
      <button
        type="button"
        onClick={() => onOpen(product.product_id)}
        className="group flex h-full w-full flex-col border border-ink-600 bg-ink-800 text-left transition-colors hover:border-crimson-700"
      >
        <ProductPhoto
          product_id={product.product_id}
          brand={product.brand}
          category={product.category}
          color={product.color}
          sport_type={product.sport_type}
          subcategory={product.subcategory}
          size="tile"
          className="w-full"
        />
        <div className="flex flex-1 flex-col p-3.5">
          <h3 className="text-sm leading-snug text-ash group-hover:text-white">
            {titleCase(product.subcategory)}
          </h3>
          <p className="mt-1 font-mono text-[10px] text-ash-muted">
            {titleCase(product.sport_type)} · size {product.size}
          </p>
          <div className="mt-auto flex items-end justify-between gap-2 pt-3">
            <span className="font-display text-lg font-black text-ash">
              {formatPrice(product.price)}
            </span>
            <Stars rating={product.avg_rating} />
          </div>
        </div>
      </button>
    </li>
  )
}

export function Catalog() {
  const { meta } = useMeta()
  const [params, setParams] = useSearchParams()

  const sport = params.get('sport') ?? ''
  const brand = params.get('brand') ?? ''
  const category = params.get('category') ?? ''
  const priceMin = params.get('price_min') ?? ''
  const priceMax = params.get('price_max') ?? ''
  const page = Math.max(1, Number(params.get('page') ?? '1') || 1)
  const openProductId = params.get('product')

  const setParam = (key: string, value: string) => {
    const next = new URLSearchParams(params)
    if (value) next.set(key, value)
    else next.delete(key)
    // Any filter change invalidates the page number; page 7 of the old result
    // set is rarely page 7 of the new one.
    if (key !== 'page' && key !== 'product') next.delete('page')
    setParams(next, { replace: key === 'product' })
  }

  const { data, loading, error, reload } = useAsync(
    (signal) =>
      getProducts(
        {
          sport: sport || undefined,
          brand: brand || undefined,
          category: category || undefined,
          price_min: priceMin ? Number(priceMin) : undefined,
          price_max: priceMax ? Number(priceMax) : undefined,
          page,
          page_size: PAGE_SIZE,
        },
        signal,
      ),
    [sport, brand, category, priceMin, priceMax, page],
  )

  const hasFilters = Boolean(sport || brand || category || priceMin || priceMax)

  return (
    <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6 sm:py-14">
      <Eyebrow>Catalogue</Eyebrow>
      <h1 className="mt-2 text-4xl text-ash sm:text-5xl">Browse everything</h1>
      <p className="mt-3 max-w-2xl text-sm leading-relaxed text-ash-dim">
        The full catalogue, ordered by product id. Deliberately not sorted by
        popularity — neither recommender technique reads sales rank, so a
        browse view that ranked by it would show an ordering the recommender
        does not believe in.
      </p>

      <div className="mt-8 grid gap-8 lg:grid-cols-[260px_1fr]">
        {/* Filters */}
        <aside aria-labelledby="filters-heading" className="lg:sticky lg:top-24 lg:self-start">
          <div className="border border-ink-600 bg-ink-800 p-4">
            <div className="flex items-center justify-between gap-2">
              <h2 id="filters-heading" className="text-base text-ash">
                Filters
              </h2>
              {hasFilters && (
                <button
                  type="button"
                  onClick={() => setParams(new URLSearchParams())}
                  className="font-display text-[10px] font-bold uppercase tracking-[0.12em] text-crimson-500 hover:text-crimson-600"
                >
                  Clear all
                </button>
              )}
            </div>

            <div className="mt-4 space-y-4">
              <SelectField
                id="filter-sport"
                label="Sport"
                value={sport}
                onChange={(value) => setParam('sport', value)}
                options={meta?.sport_type ?? []}
                placeholder="All sports"
                loading={!meta}
              />
              <SelectField
                id="filter-brand"
                label="Brand"
                value={brand}
                onChange={(value) => setParam('brand', value)}
                options={meta?.brand ?? []}
                placeholder="All brands"
                loading={!meta}
              />
              <SelectField
                id="filter-category"
                label="Category"
                value={category}
                onChange={(value) => setParam('category', value)}
                options={meta?.category ?? []}
                placeholder="All categories"
                loading={!meta}
              />

              <fieldset>
                <legend className="eyebrow mb-1.5 text-ash-dim">
                  Price range
                </legend>
                <div className="flex items-center gap-2">
                  <label htmlFor="filter-price-min" className="sr-only">
                    Minimum price
                  </label>
                  <input
                    id="filter-price-min"
                    type="number"
                    min={0}
                    inputMode="decimal"
                    placeholder={meta ? String(Math.floor(meta.price.min)) : 'Min'}
                    value={priceMin}
                    onChange={(event) => setParam('price_min', event.target.value)}
                    className="w-full border-2 border-ink-600 bg-ink-900 px-2.5 py-2 text-sm text-ash hover:border-ash-muted focus:border-crimson-500"
                  />
                  <span aria-hidden="true" className="text-ash-muted">
                    –
                  </span>
                  <label htmlFor="filter-price-max" className="sr-only">
                    Maximum price
                  </label>
                  <input
                    id="filter-price-max"
                    type="number"
                    min={0}
                    inputMode="decimal"
                    placeholder={meta ? String(Math.ceil(meta.price.max)) : 'Max'}
                    value={priceMax}
                    onChange={(event) => setParam('price_max', event.target.value)}
                    className="w-full border-2 border-ink-600 bg-ink-900 px-2.5 py-2 text-sm text-ash hover:border-ash-muted focus:border-crimson-500"
                  />
                </div>
              </fieldset>
            </div>
          </div>
        </aside>

        {/* Grid */}
        <section aria-label="Products">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-ink-700 pb-3">
            <p className="text-sm text-ash-dim" role="status" aria-live="polite">
              {loading && !data
                ? 'Loading products…'
                : data
                  ? `${formatCount(data.total)} product${data.total === 1 ? '' : 's'}${
                      hasFilters ? ' match these filters' : ''
                    }`
                  : ''}
            </p>
            {data && data.total > 0 && (
              <Badge tone="outline">
                Page {data.page} of {formatCount(data.total_pages)}
              </Badge>
            )}
          </div>

          {error != null && <ErrorState error={error} onRetry={reload} className="mt-6" />}

          {loading && (
            <ul
              className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-3 xl:grid-cols-4"
              role="status"
              aria-busy="true"
            >
              <span className="sr-only">Loading products</span>
              {Array.from({ length: 8 }).map((_, index) => (
                <li key={index} className="border border-ink-600 bg-ink-800">
                  <Skeleton className="h-36 w-full" />
                  <div className="space-y-2 p-3.5">
                    <Skeleton className="h-3 w-16" />
                    <Skeleton className="h-4 w-3/4" />
                    <Skeleton className="h-3 w-1/2" />
                  </div>
                </li>
              ))}
            </ul>
          )}

          {!loading && data && data.items.length === 0 && (
            <EmptyState title="Nothing matches those filters">
              Widen the price range or clear a filter to see more of the{' '}
              {formatCount(meta?.counts.products ?? 0)}-product catalogue.
            </EmptyState>
          )}

          {!loading && data && data.items.length > 0 && (
            <>
              <ul className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-3 xl:grid-cols-4">
                {data.items.map((product) => (
                  <ProductTile
                    key={product.product_id}
                    product={product}
                    onOpen={(id) => setParam('product', id)}
                  />
                ))}
              </ul>

              <nav
                aria-label="Pagination"
                className="mt-8 flex flex-wrap items-center justify-between gap-3 border-t border-ink-700 pt-5"
              >
                <Button
                  variant="secondary"
                  size="sm"
                  type="button"
                  disabled={!data.has_previous}
                  onClick={() => setParam('page', String(page - 1))}
                >
                  ← Previous
                </Button>

                <span className={cx('text-xs text-ash-muted')}>
                  Showing {formatCount((data.page - 1) * data.page_size + 1)}–
                  {formatCount(
                    Math.min(data.page * data.page_size, data.total),
                  )}{' '}
                  of {formatCount(data.total)}
                </span>

                <Button
                  variant="secondary"
                  size="sm"
                  type="button"
                  disabled={!data.has_next}
                  onClick={() => setParam('page', String(page + 1))}
                >
                  Next →
                </Button>
              </nav>
            </>
          )}
        </section>
      </div>

      <ProductModal
        productId={openProductId}
        onClose={() => setParam('product', '')}
      />
    </div>
  )
}
