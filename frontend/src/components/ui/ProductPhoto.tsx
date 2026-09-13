/**
 * A product's photograph, with the generated tile underneath it.
 *
 * Three layers, bottom to top:
 *
 *   1. `ProductArtwork` — the SVG tile derived from the product's colour,
 *      category and id. It is the *loading* state, so a card never flashes
 *      empty, and it is the *error* state, so a missing or corrupt file leaves
 *      a deliberate-looking tile rather than a broken-image icon. It is not
 *      dead code kept around: it is what makes the photo layer safe to fail.
 *   2. the `<img>`, `object-fit: cover` inside a fixed aspect ratio, faded in
 *      on load so the swap from tile to photo is not a flicker.
 *   3. a crimson-to-transparent scrim and the brand/category label. The scrim
 *      is what lets white text sit on a photograph of unknown brightness — a
 *      white trainer on a white background would otherwise swallow it.
 *
 * The label lives here rather than in `ProductArtwork` (which is told
 * `showLabel={false}`) because it has to be above the photo, not under it.
 */

import { useState } from 'react'

import { titleCase } from '../../lib/format'
import { resolveProductImage, type PhotoSubject } from '../../lib/productImage'
import { ProductArtwork, type ArtworkSize } from './ProductArtwork'
import { cx } from './primitives'

const TEXT_SCALE: Record<ArtworkSize, { brand: string; meta: string; pad: string }> = {
  tile: { brand: 'text-sm', meta: 'text-[9px]', pad: 'p-2.5' },
  card: { brand: 'text-base', meta: 'text-[10px]', pad: 'p-3' },
  modal: { brand: 'text-2xl', meta: 'text-[11px]', pad: 'p-4' },
}

/**
 * Fixed ratios so a grid of cards stays on its rows whatever shape the
 * underlying photo is — the library runs from 225x225 thumbnails to
 * 4000x6000 originals.
 *
 * The card is the exception: in the result list it sits in a fixed-width
 * column and stretches to the card's own height on desktop, so it takes a
 * height rather than a ratio there.
 */
const SHAPE: Record<ArtworkSize, string> = {
  tile: 'aspect-[4/3]',
  card: 'aspect-[4/3] sm:aspect-auto sm:h-full',
  modal: 'aspect-[16/10]',
}

export interface ProductPhotoProps extends PhotoSubject {
  color: string
  size?: ArtworkSize
  /**
   * Load immediately instead of lazily.
   *
   * Only the product modal sets this: it opens on demand and is the one place
   * the image is unquestionably in view. Everything in a list or a grid stays
   * lazy, so scrolling the 10,000-product catalogue fetches what is on screen.
   */
  priority?: boolean
  className?: string
}

export function ProductPhoto({
  product_id,
  brand,
  category,
  color,
  sport_type,
  subcategory,
  size = 'card',
  priority = false,
  className,
}: ProductPhotoProps) {
  const [loaded, setLoaded] = useState(false)
  const [failed, setFailed] = useState(false)

  // `brand` goes in as well as onto the label: the resolver will not hand back
  // a photo carrying a rival's logo, which is what stops the label below
  // contradicting the photograph above it.
  const photo = resolveProductImage({
    product_id,
    brand,
    category,
    sport_type,
    subcategory,
  })
  const text = TEXT_SCALE[size]

  return (
    <div
      className={cx('relative overflow-hidden bg-ink-950', SHAPE[size], className)}
    >
      {/* Layer 1 — always rendered, so there is never a blank frame. */}
      <ProductArtwork
        productId={product_id}
        brand={brand}
        category={category}
        color={color}
        size={size}
        showLabel={false}
        className="absolute inset-0 h-full w-full"
      />

      {/* Layer 2 — the photograph. Dropped entirely once it has failed, so a
          404 settles on the tile permanently instead of retrying on rerender. */}
      {!failed && (
        <img
          src={photo.src}
          alt={photo.alt}
          loading={priority ? 'eager' : 'lazy'}
          decoding="async"
          draggable={false}
          onLoad={() => setLoaded(true)}
          onError={() => setFailed(true)}
          className={cx(
            'absolute inset-0 h-full w-full object-cover transition-opacity duration-300',
            loaded ? 'opacity-100' : 'opacity-0',
          )}
        />
      )}

      {/* Layer 3a — the scrim. Crimson into transparent, bottom up, so the
          label below reads on any photograph without dimming the whole image. */}
      <div
        aria-hidden="true"
        className="absolute inset-0 bg-gradient-to-t from-crimson-900 via-ink-950/55 to-transparent"
        style={{
          backgroundImage:
            'linear-gradient(to top, rgba(107,0,0,0.92) 0%, rgba(10,10,10,0.62) 38%, rgba(10,10,10,0.12) 62%, rgba(10,10,10,0) 100%)',
        }}
      />

      {/* The deck's crimson corner wedge, kept from the tile so a photographed
          card and a fallback card still read as the same design. */}
      <div
        aria-hidden="true"
        className="absolute right-0 top-0 h-10 w-10 bg-crimson-700"
        style={{ clipPath: 'polygon(100% 0, 100% 100%, 0 0)' }}
      />

      {/* Layer 3b — the label. */}
      <div
        className={cx(
          'absolute inset-x-0 bottom-0 flex items-end justify-between gap-2',
          text.pad,
        )}
      >
        <div className="min-w-0">
          <p
            className={cx(
              'truncate font-display font-black uppercase leading-none tracking-tight text-white',
              text.brand,
            )}
            style={{ textShadow: '0 1px 6px rgba(0,0,0,0.75)' }}
          >
            {brand}
          </p>
          <p
            className={cx(
              'mt-1 font-display font-bold uppercase leading-none tracking-[0.16em] text-ash',
              text.meta,
            )}
            style={{ textShadow: '0 1px 6px rgba(0,0,0,0.85)' }}
          >
            {titleCase(category)}
          </p>
        </div>

        {size !== 'tile' && (
          <span
            className={cx(
              'shrink-0 font-display font-bold uppercase leading-none tracking-[0.14em] text-ash-dim',
              text.meta,
            )}
            style={{ textShadow: '0 1px 6px rgba(0,0,0,0.85)' }}
          >
            {titleCase(color)}
          </span>
        )}
      </div>
    </div>
  )
}
