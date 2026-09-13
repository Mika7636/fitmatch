/**
 * The product tile: generated artwork, not a photograph.
 *
 * None of the four raw sources carries a product image — the Nike export's
 * only URL is a product *page*, the Amazon file is reviews, and the Adidas
 * workbook is aggregate sales rows with no product identities at all. On top
 * of that, 4,000 of the 10,000 catalogue rows are synthesised, so no photo of
 * them exists to find. Rather than ship broken `<img>` frames or reach out to
 * a placeholder service at render time, every product gets a tile drawn from
 * its own attributes.
 *
 * It is derived, not decorative:
 *
 *   colour   the product's `color`, mapped to a real hex value, as the band
 *            and the glyph — a navy jacket tile is navy
 *   glyph    a silhouette per `category`: footwear, apparel, accessory
 *   pattern  diagonal hatching whose angle, spacing, band position and wedge
 *            size are all seeded by a hash of `product_id`, so two products
 *            sharing a colour and a category still do not look the same
 *
 * Everything is inline SVG plus CSS. No network requests, no image files, no
 * external services — which also means the tiles render identically offline
 * and in the print/PDF version of the report.
 *
 * The brand and category are HTML overlaid on the SVG rather than `<text>`
 * inside it, so their size is set in CSS and stays legible at every scale the
 * three call sites use.
 */

import { useId } from 'react'

import { titleCase } from '../../lib/format'
import { hashId, hashSlice } from '../../lib/hash'
import { cx } from './primitives'

/**
 * `products.color` -> a real hex value.
 *
 * The fourteen names are the catalogue's own vocabulary (`GET /api/meta`
 * `color_preference`). Values are picked to be recognisable as the colour
 * named while still sitting on a near-black tile: `black` is lifted to a
 * graphite so it is visible at all, `white` and `beige` are pulled slightly
 * off pure so they do not glare.
 */
const COLOR_HEX: Record<string, string> = {
  beige: '#D6C6A8',
  black: '#2E2E2E',
  blue: '#2563C9',
  brown: '#6B4A2F',
  green: '#2E8B57',
  grey: '#8A8A8A',
  navy: '#22305F',
  orange: '#E2701E',
  pink: '#E5709B',
  purple: '#7B4BA8',
  red: '#C41010',
  white: '#F2F2F2',
  yellow: '#E8C21C',
  // `multi` has no single value, so it gets a gradient built from three of
  // the others rather than an arbitrary fourth colour.
  multi: '#C41010',
}

const MULTI_STOPS = ['#C41010', '#2563C9', '#E8C21C']

/** Anything the catalogue grows later still gets a tile, in deck crimson. */
const FALLBACK_HEX = '#8B0000'

function hexFor(color: string): string {
  return COLOR_HEX[color.toLowerCase()] ?? FALLBACK_HEX
}

/**
 * Relative luminance, for deciding what reads on top of the colour.
 *
 * A yellow or white band needs dark ink over it; a navy one needs light.
 */
function isLight(hex: string): boolean {
  const value = hex.replace('#', '')
  const r = parseInt(value.slice(0, 2), 16) / 255
  const g = parseInt(value.slice(2, 4), 16) / 255
  const b = parseInt(value.slice(4, 6), 16) / 255
  const channel = (c: number) =>
    c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4
  const luminance =
    0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)
  return luminance > 0.42
}

// ---------------------------------------------------------------------------
// Category silhouettes, drawn in a 0 0 120 70 space
// ---------------------------------------------------------------------------
const GLYPHS: Record<string, string> = {
  // Side-profile trainer: toe box, laced upper, heel, sole.
  footwear:
    'M6 54 L6 44 C6 39 9 37 15 36 L34 32 C42 30 48 26 54 21 L66 11 C70 8 75 8 78 12 L84 20 C88 25 94 28 102 30 L110 32 C115 33 117 36 117 41 L117 51 C117 54 115 56 112 56 L10 56 C7 56 6 55 6 54 Z M6 49 L117 49',
  // T-shirt: shoulders, sleeves, body.
  apparel:
    'M44 8 L52 6 C57 13 65 13 70 6 L78 8 L100 20 L90 36 L80 30 L80 62 C68 66 54 66 42 62 L42 30 L32 36 L22 20 Z',
  // Backpack: haul loop, body, front pocket.
  accessory:
    'M46 14 C46 6 54 3 61 3 C68 3 76 6 76 14 L82 17 C88 20 90 25 90 32 L90 60 C90 64 88 66 84 66 L38 66 C34 66 32 64 32 60 L32 32 C32 25 34 20 40 17 Z M46 40 L76 40 L76 58 L46 58 Z',
}

export type ArtworkSize = 'tile' | 'card' | 'modal'

const TEXT_SCALE: Record<ArtworkSize, { brand: string; meta: string; pad: string }> = {
  tile: { brand: 'text-sm', meta: 'text-[9px]', pad: 'p-2.5' },
  card: { brand: 'text-base', meta: 'text-[10px]', pad: 'p-3' },
  modal: { brand: 'text-2xl', meta: 'text-[11px]', pad: 'p-4' },
}

export interface ProductArtworkProps {
  productId: string
  brand: string
  category: string
  color: string
  size?: ArtworkSize
  className?: string
  /**
   * Draw the brand/category/colour strip.
   *
   * `ProductPhoto` sets this false and draws its own, because when a
   * photograph covers the artwork the label has to sit above the photo, not
   * underneath it.
   */
  showLabel?: boolean
}

/**
 * One product's generated tile.
 *
 * Used at all three call sites — result cards, the catalogue grid and the
 * product modal — so a product looks the same wherever it appears, which is
 * what makes the tile function as recognition rather than decoration.
 */
export function ProductArtwork({
  productId,
  brand,
  category,
  color,
  size = 'card',
  className,
  showLabel = true,
}: ProductArtworkProps) {
  // SVG ids must be unique per instance: several tiles share a page, and
  // duplicate pattern ids would make them all render the first one's fill.
  const uid = useId().replace(/:/g, '')
  const hash = hashId(productId)

  const key = color.toLowerCase()
  const hex = hexFor(key)
  const isMulti = key === 'multi'
  const light = isLight(hex)

  // Five independent seeded parameters, each from its own byte of the hash.
  const hatchAngle = 20 + hashSlice(hash, 0, 5) * 14 // 20..76 degrees
  const hatchGap = 6 + hashSlice(hash, 8, 5) * 2 // 6..14 px
  const bandShift = -30 + hashSlice(hash, 16, 12) * 9 // where the colour band sits
  const wedge = 26 + hashSlice(hash, 24, 8) * 5 // crimson corner wedge size
  const glyphNudge = -6 + hashSlice(hash, 4, 13) // glyph drift, so it is not pinned

  const glyph = GLYPHS[category.toLowerCase()] ?? GLYPHS.accessory
  const text = TEXT_SCALE[size]

  return (
    <div className={cx('relative overflow-hidden bg-ink-950', className)}>
      <svg
        viewBox="0 0 320 200"
        preserveAspectRatio="xMidYMid slice"
        className="absolute inset-0 h-full w-full"
        role="img"
        aria-label={`${titleCase(color)} ${category}, ${brand}`}
      >
        <defs>
          <linearGradient id={`base-${uid}`} x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="#151515" />
            <stop offset="60%" stopColor="#0C0C0C" />
            <stop offset="100%" stopColor="#050505" />
          </linearGradient>

          {isMulti ? (
            <linearGradient id={`band-${uid}`} x1="0" y1="0" x2="1" y2="0">
              {MULTI_STOPS.map((stop, index) => (
                <stop
                  key={stop}
                  offset={`${(index / (MULTI_STOPS.length - 1)) * 100}%`}
                  stopColor={stop}
                />
              ))}
            </linearGradient>
          ) : (
            <linearGradient id={`band-${uid}`} x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor={hex} stopOpacity="0.95" />
              <stop offset="100%" stopColor={hex} stopOpacity="0.55" />
            </linearGradient>
          )}

          {/* Diagonal hatching, angle and spacing seeded by the id. */}
          <pattern
            id={`hatch-${uid}`}
            width={hatchGap}
            height={hatchGap}
            patternTransform={`rotate(${hatchAngle})`}
            patternUnits="userSpaceOnUse"
          >
            <line
              x1="0"
              y1="0"
              x2="0"
              y2={hatchGap}
              stroke="#FFFFFF"
              strokeOpacity="0.07"
              strokeWidth="1.5"
            />
          </pattern>

          {/* The band is the only place the product colour reaches full
              strength, so the glyph over it is clipped to stay readable. */}
          <clipPath id={`frame-${uid}`}>
            <rect x="0" y="0" width="320" height="200" />
          </clipPath>
        </defs>

        <g clipPath={`url(#frame-${uid})`}>
          <rect width="320" height="200" fill={`url(#base-${uid})`} />

          {/* The colour band: a wide diagonal slab, positioned by the hash. */}
          <polygon
            points={`${bandShift},200 ${bandShift + 120},0 ${bandShift + 250},0 ${bandShift + 130},200`}
            fill={`url(#band-${uid})`}
            opacity="0.85"
          />

          <rect width="320" height="200" fill={`url(#hatch-${uid})`} />

          {/* Deck signature: the hard crimson corner wedge. */}
          <polygon
            points={`${320 - wedge},0 320,0 320,${wedge}`}
            fill="#A80000"
          />

          {/* Category silhouette. Dark ink over a light band, light over a
              dark one, so it never disappears into the colour underneath. */}
          <g
            transform={`translate(${96 + glyphNudge}, 62) scale(0.95)`}
            fill="none"
            stroke={light ? '#0A0A0A' : '#E8E8E8'}
            strokeOpacity={light ? 0.55 : 0.5}
            strokeWidth="3"
            strokeLinejoin="round"
            strokeLinecap="round"
          >
            <path d={glyph} />
          </g>

          {/* Scrim under the overlaid text, so brand stays legible whatever
              the band is doing behind it. */}
          <linearGradient id={`scrim-${uid}`} x1="0" y1="1" x2="0" y2="0">
            <stop offset="0%" stopColor="#050505" stopOpacity="0.92" />
            <stop offset="55%" stopColor="#050505" stopOpacity="0.45" />
            <stop offset="100%" stopColor="#050505" stopOpacity="0" />
          </linearGradient>
          <rect
            x="0"
            y="96"
            width="320"
            height="104"
            fill={`url(#scrim-${uid})`}
          />
        </g>
      </svg>

      {/* Brand and category in HTML, so CSS controls the size and they stay
          readable at card scale. */}
      {showLabel && (
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
          >
            {brand}
          </p>
          <p
            className={cx(
              'mt-1 font-display font-bold uppercase leading-none tracking-[0.16em] text-ash-dim',
              text.meta,
            )}
          >
            {titleCase(category)}
          </p>
        </div>

        {/* A swatch, so the mapped colour is named as well as shown. */}
        <div className="flex shrink-0 items-center gap-1.5">
          <span
            aria-hidden="true"
            className="block h-3 w-3 border border-white/40"
            style={
              isMulti
                ? {
                    backgroundImage: `linear-gradient(135deg, ${MULTI_STOPS.join(', ')})`,
                  }
                : { backgroundColor: hex }
            }
          />
          <span
            className={cx(
              'font-display font-bold uppercase leading-none tracking-[0.14em] text-ash-muted',
              text.meta,
            )}
          >
            {titleCase(color)}
          </span>
        </div>
      </div>
      )}
    </div>
  )
}
