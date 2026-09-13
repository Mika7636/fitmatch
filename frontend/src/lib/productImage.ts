/**
 * Which photograph to show for a product.
 *
 * `products.csv` has 10,000 rows and no image column, and no source dataset
 * carries product photography — so per-product photos are not merely missing,
 * they cannot exist (4,000 of the rows are synthesised from a line vocabulary,
 * so there is no real product to photograph). What exists instead is 110
 * pieces of representative stock imagery, and this module is the deterministic
 * mapping from a product's attributes onto them.
 *
 * Three properties matter, in this order:
 *
 *  1. **The photo does not contradict the label.** A photographed swoosh under
 *     the word ADIDAS is worse than no photograph at all, so a photo carrying
 *     a visible catalogue-brand mark is only ever shown for that brand. See
 *     `PHOTO_BRAND` for the audit and the exact rule.
 *  2. **The photo makes sense for the product.** A basketball shoe shows
 *     basketball shoes; a hoodie shows a hoodie. This is why the tier tables
 *     below are hand-written per catalogue subcategory rather than being some
 *     clever generic scheme — the catalogue's own vocabulary is irregular
 *     (`matching-sets` are crew-neck sweatshirts; `accessory/clothing` is
 *     sleeves and sport bands) and only a lookup gets those right.
 *  3. **The same product always gets the same photo.** The choice within a
 *     bucket is `hashId(product_id)`, never `Math.random` and never array
 *     order, so a product looks the same across renders, sessions and
 *     machines — which is what makes the results page reproducible for a
 *     screenshot or a marker.
 *
 * (1) outranks (2): where a bucket holds no photo a brand may use, resolution
 * falls through to the next tier rather than showing a rival's logo, so those
 * products get a less specific but honest photo. `scripts/check-images.mjs`
 * prints exactly which (bucket, brand) pairs that happens for.
 *
 * Resolution order, as specified:
 *
 *     category + sport_type + subcategory     most specific
 *     category + sport_type
 *     category + subcategory
 *     category                                fallback, always defined
 *
 * The resolver is **total by construction**: every category has a fallback
 * pool, there is a final default, and if even that holds nothing the brand may
 * use, the filter is dropped for that one product rather than returning
 * nothing. (No catalogue product reaches that last step today — the build
 * check asserts it.)
 * The only way it could point at a missing *file* is if `PHOTO_POOLS` drifted
 * from what is on disk, which is exactly what `scripts/check-images.mjs`
 * asserts at build time.
 */

import { hashId } from './hash'

/** Served straight out of `public/`, so the path is the URL. */
export const PHOTO_DIR = '/products'

/**
 * The photo library: pool name -> how many files it has.
 *
 * Files are `<pool>-<n>.jpg`, n from 1. Counts rather than 110 literal
 * filenames because the naming is regular and the build check verifies both
 * directions against the real directory — a file added, removed or misnamed
 * fails the build rather than 404ing in the demo.
 */
export const PHOTO_POOLS = {
  'footwear-basketball': 3,
  'footwear-football': 3,
  'footwear-lifestyle': 6,
  'footwear-outdoor': 3,
  'footwear-running': 3,
  'footwear-tennis': 3,
  'footwear-training': 3,
  'apparel-hoodie': 3,
  'apparel-jacket': 5,
  'apparel-joggers': 4,
  'apparel-leggings': 5,
  'apparel-running-top': 3,
  'apparel-shorts': 5,
  'apparel-sweatshirt': 4,
  'apparel-swimwear': 7,
  'apparel-tank': 3,
  'accessory-bag': 10,
  'accessory-bottle': 14,
  'accessory-cap': 8,
  'accessory-headband': 11,
  'accessory-socks': 4,
} as const

export type PhotoPool = keyof typeof PHOTO_POOLS

/** Human description per pool, for alt text. */
const POOL_SUBJECT: Record<PhotoPool, string> = {
  'footwear-basketball': 'basketball shoes',
  'footwear-football': 'football boots',
  'footwear-lifestyle': 'lifestyle sneakers',
  'footwear-outdoor': 'hiking boots',
  'footwear-running': 'running shoes',
  'footwear-tennis': 'tennis shoes',
  'footwear-training': 'training shoes',
  'apparel-hoodie': 'a hoodie',
  'apparel-jacket': 'a sports jacket',
  'apparel-joggers': 'joggers',
  'apparel-leggings': 'leggings',
  'apparel-running-top': 'a running top',
  'apparel-shorts': 'athletic shorts',
  'apparel-sweatshirt': 'a sweatshirt',
  'apparel-swimwear': 'swimwear',
  'apparel-tank': 'a training vest',
  'accessory-bag': 'a gym bag',
  'accessory-bottle': 'a sports bottle',
  'accessory-cap': 'a cap',
  'accessory-headband': 'a sports headband',
  'accessory-socks': 'sports socks',
}

// ---------------------------------------------------------------------------
// Brand compatibility
// ---------------------------------------------------------------------------
/**
 * The catalogue's entire brand vocabulary, lower-cased.
 *
 * Measured, not assumed: products.csv holds 5,219 Nike, 4,000 Adidas and 781
 * Jordan rows and nothing else. This list is what `PHOTO_BRAND` is allowed to
 * name, and what the compatibility rule is defined over.
 */
export const CATALOGUE_BRANDS = ['nike', 'adidas', 'jordan'] as const

export type CatalogueBrand = (typeof CATALOGUE_BRANDS)[number]

/**
 * Which photographs carry a visible catalogue-brand mark, and whose.
 *
 * Hand-built by looking at all 110 files, not by reading their names — the
 * filenames say what the subject is (`accessory-cap-4`), never whose logo is
 * printed on it. A file is listed here when a Nike swoosh / wordmark, an
 * Adidas trefoil, three-stripe or badge-of-sport, or a Jordan Jumpman is
 * legible in the frame at card size. An Air Jordan 1 is listed as `jordan`
 * even though a swoosh is also visible: the shoe is a Jordan-brand product.
 *
 * **The rule this drives:** a file listed here may only be shown for products
 * of that brand; a file absent from here may be shown for any brand. So a
 * photographed swoosh can never sit under the word ADIDAS.
 *
 * **What is deliberately *not* listed.** Some photos carry a mark belonging to
 * a brand the catalogue does not sell — The North Face (`footwear-outdoor-3`),
 * Puma (`footwear-training-3`), Under Armour (`apparel-jacket-3`), Gymshark
 * (`apparel-hoodie-3`), Guess (`apparel-hoodie-1`), owayo
 * (`apparel-running-top-1..3`), ASICS shoes in frame (`apparel-leggings-5`),
 * New Era / NY Yankees (`accessory-cap-1`, `-2`), and assorted no-name
 * wordmarks (FORZA, GOR-7, Bonkers, BEYOND). Those are a separate realism
 * problem: they are equally wrong under all three labels, so restricting them
 * would remove photos without removing a contradiction — and it would empty
 * `apparel-running-top` for every brand at once. They stay shared, and the
 * build check lists them so the decision is visible rather than silent.
 */
export const PHOTO_BRAND: Readonly<Record<string, CatalogueBrand>> = {
  // --- Nike ---------------------------------------------------------------
  'footwear-basketball-1.jpg': 'nike', // swoosh on the shoe, and on the joggers
  'footwear-basketball-2.jpg': 'nike', // Air Force 1: swoosh + AIR on the midsole
  'footwear-basketball-3.jpg': 'nike', // LeBron: swoosh, NIKE REACT on the outsole
  'footwear-football-1.jpg': 'nike', // Phantom: swoosh on both boots
  'footwear-lifestyle-3.jpg': 'nike', // Roshe: large swoosh, both shoes
  'footwear-running-1.jpg': 'nike', // tonal swoosh on the pink upper
  'footwear-tennis-3.jpg': 'nike', // swoosh on both shoes
  'accessory-bag-8.jpg': 'nike',
  'accessory-bag-9.jpg': 'nike',
  'accessory-bag-10.jpg': 'nike',
  'accessory-bottle-8.jpg': 'nike',
  'accessory-bottle-9.jpg': 'nike',
  'accessory-bottle-10.jpg': 'nike',
  'accessory-bottle-11.jpg': 'nike',
  'accessory-bottle-12.jpg': 'nike',
  'accessory-bottle-13.jpg': 'nike', // NIKE wordmark + swoosh
  'accessory-bottle-14.jpg': 'nike',
  'accessory-cap-6.jpg': 'nike',
  'accessory-cap-7.jpg': 'nike', // swoosh on the side, college mark on the front
  'accessory-cap-8.jpg': 'nike', // swoosh on the side, college mark on the front
  'accessory-headband-7.jpg': 'nike',
  'accessory-headband-8.jpg': 'nike',
  'accessory-headband-9.jpg': 'nike', // NIKE wordmark + swoosh
  'accessory-headband-10.jpg': 'nike',
  'accessory-headband-11.jpg': 'nike',
  'accessory-socks-3.jpg': 'nike',
  'accessory-socks-4.jpg': 'nike', // NIKE wordmark on all three pairs

  // --- Jordan -------------------------------------------------------------
  'footwear-lifestyle-2.jpg': 'jordan', // Air Jordan 1 Low "Black Toe"
  'apparel-jacket-4.jpg': 'jordan', // Jumpman on the chest

  // --- Adidas -------------------------------------------------------------
  'apparel-shorts-4.jpg': 'adidas', // badge-of-sport, all-over print
  'apparel-swimwear-6.jpg': 'adidas', // three stripes + badge-of-sport
  'apparel-swimwear-7.jpg': 'adidas', // three stripes + trefoil
  'accessory-bag-1.jpg': 'adidas',
  'accessory-bag-2.jpg': 'adidas', // trefoil
  'accessory-bag-3.jpg': 'adidas', // badge-of-sport + adidas wordmark
  'accessory-bag-4.jpg': 'adidas',
  'accessory-bag-5.jpg': 'adidas',
  'accessory-bag-6.jpg': 'adidas',
  'accessory-bag-7.jpg': 'adidas',
  'accessory-bottle-1.jpg': 'adidas',
  'accessory-bottle-3.jpg': 'adidas',
  'accessory-bottle-4.jpg': 'adidas',
  'accessory-bottle-5.jpg': 'adidas', // adidas wordmark + three stripes
  'accessory-bottle-6.jpg': 'adidas',
  'accessory-bottle-7.jpg': 'adidas',
  'accessory-cap-3.jpg': 'adidas', // trefoil set inside the varsity "A"
  'accessory-cap-4.jpg': 'adidas', // adidas wordmark + trefoil
  'accessory-cap-5.jpg': 'adidas', // trefoil
  'accessory-headband-1.jpg': 'adidas',
  'accessory-headband-2.jpg': 'adidas',
  'accessory-headband-3.jpg': 'adidas',
  'accessory-headband-4.jpg': 'adidas',
  'accessory-headband-5.jpg': 'adidas',
  'accessory-headband-6.jpg': 'adidas', // adidas wordmark
  'accessory-socks-1.jpg': 'adidas',
  'accessory-socks-2.jpg': 'adidas',
}

/**
 * A catalogue brand name as `PHOTO_BRAND` spells it, or `null`.
 *
 * `null` means "not one of the three brands the catalogue sells", which the
 * compatibility rule treats as *no brand-marked photo is safe* — an unknown
 * label over a swoosh is the same contradiction as ADIDAS over one. It is not
 * reachable from products.csv today; it exists so that adding a fourth brand
 * degrades to plain photography instead of quietly borrowing Nike's.
 */
export function catalogueBrand(brand: string | null | undefined): CatalogueBrand | null {
  const value = (brand ?? '').trim().toLowerCase()
  return (CATALOGUE_BRANDS as readonly string[]).includes(value)
    ? (value as CatalogueBrand)
    : null
}

/** Whether `file` may be shown for a product of `brand`. */
export function isPhotoCompatible(file: string, brand: CatalogueBrand | null): boolean {
  const mark = PHOTO_BRAND[file]
  return mark === undefined || mark === brand
}

// ---------------------------------------------------------------------------
// Tier 2 — category + sport_type
// ---------------------------------------------------------------------------
/**
 * Footwear only, and deliberately so.
 *
 * For a shoe the sport *is* the product type: a running shoe and a football
 * boot are different objects. For apparel and accessories the sport is a label
 * on an otherwise ordinary garment — a yoga jacket is a jacket — so there are
 * no apparel or accessory entries here, and those fall through to tier 3 where
 * `subcategory` decides. Putting `apparel|yoga -> leggings` here would show
 * leggings for a yoga *jacket*, because tier 2 outranks tier 3.
 */
const BY_CATEGORY_SPORT: Record<string, PhotoPool[]> = {
  'footwear|basketball': ['footwear-basketball'],
  'footwear|football': ['footwear-football'],
  'footwear|outdoor': ['footwear-outdoor'],
  'footwear|running': ['footwear-running'],
  'footwear|tennis': ['footwear-tennis'],
  'footwear|training': ['footwear-training'],
  'footwear|lifestyle': ['footwear-lifestyle'],
  // Skate shoes are sneakers; the lifestyle pool is the honest match.
  'footwear|skateboarding': ['footwear-lifestyle'],
  // No swim-footwear photo exists in the library, so these get generic
  // sneakers rather than something actively wrong.
  'footwear|swimming': ['footwear-lifestyle'],
  // The catalogue has no yoga footwear at all; defined so the table is total
  // against the sport vocabulary rather than because it is ever hit.
  'footwear|yoga': ['footwear-training'],
}

// ---------------------------------------------------------------------------
// Tier 3 — category + subcategory
// ---------------------------------------------------------------------------
/**
 * The workhorse for apparel and accessories.
 *
 * Several mappings are counter-intuitive until you read the data:
 * `matching-sets` is overwhelmingly oversized crew-neck sweatshirts,
 * `accessory/clothing` is compression sleeves and sport bands, and
 * `accessory/football` is actual footballs — for which the library has no
 * photo, so it takes the generic sports-kit pool rather than pretending.
 */
const BY_CATEGORY_SUBCATEGORY: Record<string, PhotoPool[]> = {
  // Apparel
  'apparel|tops-and-t-shirts': ['apparel-running-top', 'apparel-tank'],
  'apparel|hoodies-and-sweatshirts': ['apparel-hoodie', 'apparel-sweatshirt'],
  'apparel|matching-sets': ['apparel-sweatshirt', 'apparel-hoodie'],
  'apparel|shorts': ['apparel-shorts'],
  'apparel|jackets': ['apparel-jacket'],
  'apparel|trousers-and-tights': ['apparel-leggings'],
  'apparel|trousers': ['apparel-joggers'],
  'apparel|leggings': ['apparel-leggings'],
  'apparel|kits-and-jerseys': ['apparel-running-top', 'apparel-tank'],
  'apparel|sports-bras': ['apparel-tank'],
  'apparel|hats': ['accessory-cap'],
  'apparel|skirts-and-dresses': ['apparel-shorts'],
  'apparel|tracksuits': ['apparel-joggers', 'apparel-sweatshirt'],
  'apparel|clothing': ['apparel-running-top', 'apparel-tank'],
  'apparel|sport-clothing': ['apparel-running-top', 'apparel-tank'],
  'apparel|swimwear': ['apparel-swimwear'],
  'apparel|bags-and-backpacks': ['accessory-bag'],

  // Accessory
  'accessory|socks': ['accessory-socks'],
  'accessory|bags-and-backpacks': ['accessory-bag'],
  'accessory|clothing': ['accessory-headband', 'accessory-cap'],
  'accessory|hats': ['accessory-cap'],
  'accessory|football': ['accessory-bottle', 'accessory-bag'],
  'accessory|jackets': ['apparel-jacket'],

  // Footwear — tier 2 covers every sport, so these are a safety net for rows
  // whose sport is missing or new.
  'footwear|shoes': ['footwear-lifestyle'],
  'footwear|hats': ['accessory-cap'],
}

// ---------------------------------------------------------------------------
// Tier 1 — category + sport_type + subcategory
// ---------------------------------------------------------------------------
/**
 * Refinements, built from a small table rather than written out.
 *
 * These exist for the subcategories that say nothing about the garment
 * (`clothing`, `matching-sets`, `kits-and-jerseys`, …). For those, and only
 * those, the sport is the better signal: a swimming "kit" is swimwear, a yoga
 * "set" is leggings. Anywhere the subcategory is already specific, tier 1 stays
 * empty so tier 3 decides.
 */
const VAGUE_APPAREL_SUBCATEGORIES = [
  'clothing',
  'sport-clothing',
  'matching-sets',
  'kits-and-jerseys',
  'tracksuits',
] as const

const SPORT_REFINEMENT: Record<string, PhotoPool[]> = {
  swimming: ['apparel-swimwear'],
  yoga: ['apparel-leggings'],
  running: ['apparel-running-top'],
}

const BY_CATEGORY_SPORT_SUBCATEGORY: Record<string, PhotoPool[]> = (() => {
  const table: Record<string, PhotoPool[]> = {}
  for (const [sport, pools] of Object.entries(SPORT_REFINEMENT)) {
    for (const subcategory of VAGUE_APPAREL_SUBCATEGORIES) {
      table[`apparel|${sport}|${subcategory}`] = pools
    }
  }
  return table
})()

// ---------------------------------------------------------------------------
// Tier 4 — category alone, and the final default
// ---------------------------------------------------------------------------
const BY_CATEGORY: Record<string, PhotoPool[]> = {
  footwear: ['footwear-lifestyle'],
  apparel: ['apparel-running-top', 'apparel-tank', 'apparel-hoodie'],
  accessory: [
    'accessory-bottle',
    'accessory-bag',
    'accessory-cap',
    'accessory-headband',
  ],
}

/** Reached only if the catalogue grows a category none of the above knows. */
const DEFAULT_POOLS: PhotoPool[] = ['footwear-lifestyle']

// ---------------------------------------------------------------------------
// Resolution
// ---------------------------------------------------------------------------
export type PhotoTier =
  | 'category+sport+subcategory'
  | 'category+sport'
  | 'category+subcategory'
  | 'category'
  | 'default'

export interface ResolvedPhoto {
  /** URL under `public/`, ready for an `<img src>`. */
  src: string
  pool: PhotoPool
  /** Which rule matched — surfaced in the build check and the verify run. */
  tier: PhotoTier
  /**
   * True when brand compatibility pushed this product past a bucket it would
   * otherwise have matched — i.e. the photo is less specific than the tier
   * tables alone would give. The build check counts these.
   */
  brand_restricted: boolean
  /** Describes the photo *and* says it is representative, not the product. */
  alt: string
}

/** The fields the resolver reads. Any product shape with these will do. */
export interface PhotoSubject {
  product_id: string
  /** Drives brand compatibility; anything outside `CATALOGUE_BRANDS` is
   *  treated as "no brand", which allows only unmarked photography. */
  brand: string
  category: string
  sport_type: string
  subcategory: string
}

function normalise(value: string | null | undefined): string {
  return (value ?? '').trim().toLowerCase()
}

/** Flatten a bucket of pools into the concrete files it offers. */
function filesFor(pools: PhotoPool[]): { pool: PhotoPool; file: string }[] {
  const files: { pool: PhotoPool; file: string }[] = []
  for (const pool of pools) {
    const count = PHOTO_POOLS[pool]
    for (let index = 1; index <= count; index += 1) {
      files.push({ pool, file: `${pool}-${index}.jpg` })
    }
  }
  return files
}

interface Bucket {
  pools: PhotoPool[]
  tier: PhotoTier
}

/**
 * Every bucket this product matches, most specific first.
 *
 * This used to return the first match and stop. It returns the whole chain now
 * because brand compatibility can empty the most specific bucket — there is no
 * Adidas basketball photograph in the library — and the honest answer is then
 * the next bucket down, not a Nike shoe under an Adidas label.
 */
function candidates(product: PhotoSubject): Bucket[] {
  const category = normalise(product.category)
  const sport = normalise(product.sport_type)
  const subcategory = normalise(product.subcategory)

  const chain: Bucket[] = []
  const push = (pools: PhotoPool[] | undefined, tier: PhotoTier) => {
    if (pools) chain.push({ pools, tier })
  }

  push(
    BY_CATEGORY_SPORT_SUBCATEGORY[`${category}|${sport}|${subcategory}`],
    'category+sport+subcategory',
  )
  push(BY_CATEGORY_SPORT[`${category}|${sport}`], 'category+sport')
  push(BY_CATEGORY_SUBCATEGORY[`${category}|${subcategory}`], 'category+subcategory')
  push(BY_CATEGORY[category], 'category')
  chain.push({ pools: DEFAULT_POOLS, tier: 'default' })
  return chain
}

/**
 * The photo for one product.
 *
 * Pure and total: same input, same output, and it always returns a file that
 * `PHOTO_POOLS` declares — which the build check has proven to exist.
 */
export function resolveProductImage(product: PhotoSubject): ResolvedPhoto {
  const brand = catalogueBrand(product.brand)
  const chain = candidates(product)

  // The first bucket in the chain that offers this brand anything at all.
  let index = 0
  let files: { pool: PhotoPool; file: string }[] = []
  for (; index < chain.length; index += 1) {
    files = filesFor(chain[index].pools).filter((entry) =>
      isPhotoCompatible(entry.file, brand),
    )
    if (files.length > 0) break
  }

  // Last resort, so the function stays total: if the brand can use nothing in
  // any bucket — which no catalogue product hits, and the build check proves
  // it — take the most specific bucket unfiltered. A wrong logo beats a
  // broken image, but only here, and only if the library gets much narrower.
  if (files.length === 0) {
    index = 0
    files = filesFor(chain[0].pools)
  }

  const bucket = chain[index]
  // Spread across the whole bucket, not just across pools, so a bucket of
  // 2 pools x 5 photos uses all ten rather than alternating between two.
  const chosen = files[hashId(product.product_id) % files.length]

  return {
    src: `${PHOTO_DIR}/${chosen.file}`,
    pool: chosen.pool,
    tier: bucket.tier,
    brand_restricted: index > 0,
    alt: `Representative photograph of ${POOL_SUBJECT[chosen.pool]}, standing in for this ${normalise(product.category)} product`,
  }
}

/** Every file the library declares — used by the build-time check. */
export function allDeclaredFiles(): string[] {
  const files: string[] = []
  for (const [pool, count] of Object.entries(PHOTO_POOLS)) {
    for (let index = 1; index <= count; index += 1) {
      files.push(`${pool}-${index}.jpg`)
    }
  }
  return files.sort()
}

/**
 * How many photos each bucket offers each brand, after the compatibility rule.
 *
 * A zero here is not a bug -- it is the finding the rule exists to surface: it
 * says that brand's products in that bucket now fall through to a less
 * specific photo. `scripts/check-images.mjs` prints every zero, with the
 * product count behind it, so thinning the library is never silent.
 */
export interface BucketCoverage {
  /** The tier-table key, e.g. `footwear|basketball`. */
  key: string
  tier: PhotoTier
  pools: PhotoPool[]
  total: number
  compatible: Record<CatalogueBrand, number>
}

export function brandCoverage(): BucketCoverage[] {
  const tables: [PhotoTier, Record<string, PhotoPool[]>][] = [
    ['category+sport+subcategory', BY_CATEGORY_SPORT_SUBCATEGORY],
    ['category+sport', BY_CATEGORY_SPORT],
    ['category+subcategory', BY_CATEGORY_SUBCATEGORY],
    ['category', BY_CATEGORY],
    ['default', { default: DEFAULT_POOLS }],
  ]

  const rows: BucketCoverage[] = []
  for (const [tier, table] of tables) {
    for (const [key, pools] of Object.entries(table)) {
      const files = filesFor(pools)
      const compatible = {} as Record<CatalogueBrand, number>
      for (const brand of CATALOGUE_BRANDS) {
        compatible[brand] = files.filter((entry) =>
          isPhotoCompatible(entry.file, brand),
        ).length
      }
      rows.push({ key, tier, pools, total: files.length, compatible })
    }
  }
  return rows
}

/** Every pool any tier can select — used by the build-time check. */
export function allReferencedPools(): string[] {
  const pools = new Set<string>()
  for (const table of [
    BY_CATEGORY_SPORT_SUBCATEGORY,
    BY_CATEGORY_SPORT,
    BY_CATEGORY_SUBCATEGORY,
    BY_CATEGORY,
  ]) {
    for (const list of Object.values(table)) for (const pool of list) pools.add(pool)
  }
  for (const pool of DEFAULT_POOLS) pools.add(pool)
  return [...pools].sort()
}
