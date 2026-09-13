/**
 * Build-time guarantee that no product can resolve to a missing photograph.
 *
 * Runs as part of `npm run build`, before Vite. It fails the build rather than
 * letting a renamed or deleted file turn into a 404 in the middle of a demo.
 *
 * Four passes, cheapest first:
 *
 *   1. every file PHOTO_POOLS declares exists on disk
 *   2. every file on disk is declared (catches an added photo nothing uses)
 *   3. every pool any tier table references is declared in PHOTO_POOLS
 *   4. every (category, sport, subcategory) the catalogue actually contains
 *      resolves to a file that exists, and no product resolves to a photo
 *      carrying a rival brand's logo
 *   5. brand compatibility leaves every *fallback* bucket usable by every
 *      brand, and reports the specific buckets a brand now falls through
 *
 * Pass 4 reads data/processed/products.csv when it is there. That file is
 * gitignored and rebuilt from the raw downloads, so on a fresh checkout it is
 * absent -- in which case the pass is skipped with a note rather than failing,
 * because passes 1-3 already make the resolver total. When the CSV is present
 * this checks all 10,000 rows individually.
 */
import { createServer } from 'vite'
import { readdirSync, readFileSync, existsSync } from 'node:fs'
import { join } from 'node:path'

const PHOTO_DIR = join(process.cwd(), 'public', 'products')
const PRODUCTS_CSV = join(process.cwd(), '..', 'data', 'processed', 'products.csv')

const problems = []
const note = (message) => console.log(`   ${message}`)

const server = await createServer({
  configFile: false,
  root: process.cwd(),
  logLevel: 'error',
  server: { middlewareMode: true, hmr: false },
  optimizeDeps: { noDiscovery: true },
})

try {
  const mod = await server.ssrLoadModule('/src/lib/productImage.ts')
  const {
    resolveProductImage,
    allDeclaredFiles,
    allReferencedPools,
    brandCoverage,
    CATALOGUE_BRANDS,
    PHOTO_BRAND,
    PHOTO_POOLS,
  } = mod

  console.log('Checking product photography...\n')

  // --- 1 & 2: the manifest and the directory agree ------------------------
  if (!existsSync(PHOTO_DIR)) {
    problems.push(`photo directory is missing: ${PHOTO_DIR}`)
  } else {
    const onDisk = new Set(readdirSync(PHOTO_DIR).filter((n) => n.endsWith('.jpg')))
    const declared = allDeclaredFiles()

    const missing = declared.filter((f) => !onDisk.has(f))
    for (const file of missing) problems.push(`declared but not on disk: ${file}`)

    const undeclared = [...onDisk].filter((f) => !declared.includes(f)).sort()
    for (const file of undeclared) problems.push(`on disk but not declared: ${file}`)

    note(`${declared.length} declared, ${onDisk.size} on disk`)
  }

  // --- 3: every referenced pool is a real pool ----------------------------
  const known = new Set(Object.keys(PHOTO_POOLS))
  for (const pool of allReferencedPools()) {
    if (!known.has(pool)) problems.push(`tier table references unknown pool: ${pool}`)
  }
  note(`${allReferencedPools().length} pools referenced by the tier tables`)

  // --- 4: every real product resolves to a real file ----------------------
  if (!existsSync(PRODUCTS_CSV)) {
    note('products.csv absent (gitignored) - skipping the per-product pass')
  } else {
    const text = readFileSync(PRODUCTS_CSV, 'utf8')
    const lines = text.split(/\r?\n/).filter(Boolean)
    const header = lines[0].split(',')
    const column = (name) => header.indexOf(name)
    const idAt = column('product_id')
    const brandAt = column('brand')
    const catAt = column('category')
    const sportAt = column('sport_type')
    const subAt = column('subcategory')

    const declared = new Set(allDeclaredFiles())
    const tierCounts = {}
    const poolCounts = {}
    const fellThrough = {}
    let checked = 0

    for (const line of lines.slice(1)) {
      // Only the first columns are needed and none of them contains a comma,
      // so a split is enough -- the quoted description is the last column.
      const cells = line.split(',')
      const product = {
        product_id: cells[idAt],
        brand: cells[brandAt],
        category: cells[catAt],
        sport_type: cells[sportAt],
        subcategory: cells[subAt],
      }
      if (!product.product_id) continue
      const resolved = resolveProductImage(product)
      const file = resolved.src.replace('/products/', '')
      if (!declared.has(file)) {
        problems.push(
          `${product.product_id} (${product.category}/${product.sport_type}/${product.subcategory}) -> missing ${file}`,
        )
      }
      tierCounts[resolved.tier] = (tierCounts[resolved.tier] ?? 0) + 1
      poolCounts[resolved.pool] = (poolCounts[resolved.pool] ?? 0) + 1
      if (resolved.brand_restricted) {
        const key = `${product.brand} ${product.category}/${product.sport_type}/${product.subcategory}`
        fellThrough[key] = (fellThrough[key] ?? 0) + 1
      }
      // The contradiction this whole rule exists to prevent.
      const mark = PHOTO_BRAND[file]
      if (mark !== undefined && mark !== product.brand.trim().toLowerCase()) {
        problems.push(
          `${product.product_id} is ${product.brand} but resolves to ${file}, which shows ${mark} branding`,
        )
      }
      checked += 1
    }

    note(`${checked.toLocaleString()} products resolved`)
    note('by tier:')
    for (const [tier, count] of Object.entries(tierCounts).sort((a, b) => b[1] - a[1])) {
      note(`   ${String(count).padStart(6)}  ${tier}`)
    }
    const unused = Object.keys(PHOTO_POOLS).filter((p) => !poolCounts[p])
    note(
      unused.length
        ? `pools never selected: ${unused.join(', ')}`
        : 'every pool is selected by at least one product',
    )

    const restricted = Object.entries(fellThrough).sort((a, b) => b[1] - a[1])
    const restrictedTotal = restricted.reduce((sum, [, n]) => sum + n, 0)
    if (restricted.length) {
      note('')
      note(
        `${restrictedTotal.toLocaleString()} products fall through a bucket because it holds no photo of their brand:`,
      )
      for (const [key, count] of restricted) note(`   ${String(count).padStart(6)}  ${key}`)
    } else {
      note('no product is pushed off its bucket by brand compatibility')
    }
  }

  // --- 5: brand compatibility leaves every bucket usable ------------------
  // A bucket with nothing for a brand is allowed -- resolution falls through
  // to the next tier -- but it must never be silent, and the *fallback*
  // buckets have to hold something for everyone or the resolver has nowhere
  // left to go. That last part is an error; the rest is a report.
  console.log()
  note('brand compatibility:')
  const marked = Object.keys(PHOTO_BRAND).length
  note(`${marked} of ${allDeclaredFiles().length} photos carry a catalogue-brand mark`)

  const empty = []
  for (const bucket of brandCoverage()) {
    for (const brand of CATALOGUE_BRANDS) {
      if (bucket.compatible[brand] === 0) empty.push({ bucket, brand })
    }
  }
  for (const { bucket, brand } of empty) {
    const line = `${bucket.key} (${bucket.pools.join(' + ')}) has no ${brand} photo`
    // The last-resort tiers must stay usable, or the resolver has to drop the
    // filter to answer at all.
    if (bucket.tier === 'category' || bucket.tier === 'default') {
      problems.push(`fallback bucket ${line}`)
    } else {
      note(`   ${line} -> ${brand} products here fall through`)
    }
  }
  if (!empty.length) note('every bucket holds at least one photo for every brand')
} finally {
  await server.close()
}

console.log()
if (problems.length) {
  console.error(`IMAGE CHECK FAILED - ${problems.length} problem(s):`)
  for (const problem of problems.slice(0, 40)) console.error(`  - ${problem}`)
  if (problems.length > 40) console.error(`  ... and ${problems.length - 40} more`)
  process.exit(1)
}
console.log('Image check passed: every product resolves to a file that exists.')
