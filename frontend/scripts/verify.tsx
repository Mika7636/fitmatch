/**
 * End-to-end verification against a live API, without a browser.
 *
 *   run_api.bat            # terminal 1
 *   npm run verify         # terminal 2
 *
 * It does two things a type-checker cannot:
 *
 *  1. **Replays the real flows** against `http://127.0.0.1:8000` through the
 *     app's own `src/lib/api.ts` client -- sample load, edit, submit, heavy
 *     relaxation, zero results, catalogue paging, product detail -- and
 *     asserts the contract each one depends on.
 *
 *  2. **Renders the real components** with those real payloads via
 *     `react-dom/server`, then asserts on the HTML. That is how the two rules
 *     that are easy to violate silently get checked: that every relaxation
 *     sentence reaches the page *verbatim*, and that neither relaxation nor
 *     the cold-start mode is styled as a warning.
 *
 * `renderToStaticMarkup` runs no effects, so components that fetch their own
 * data are exercised through their loading state only; everything that takes
 * its data as props is exercised fully. The flows those effects would drive
 * are covered directly by the API calls in part 1.
 */

import { existsSync, readdirSync } from 'node:fs'
import { join } from 'node:path'

import { renderToStaticMarkup } from 'react-dom/server'
import { MemoryRouter } from 'react-router-dom'

import { HowChosen } from '../src/components/results/HowChosen'
import { ResultCard } from '../src/components/results/ResultCard'
import { ProductArtwork } from '../src/components/ui/ProductArtwork'
import { ProductPhoto } from '../src/components/ui/ProductPhoto'
import { Landing } from '../src/pages/Landing'
import { Layout } from '../src/components/layout/Layout'
import { MetaProvider } from '../src/context/MetaContext'
import {
  INITIAL_FORM_STATE,
  ProfileProvider,
  buildRecommendRequest,
  profileFormReducer,
  profileSource,
  toProfileFields,
  validateDraft,
  type ProfileFormState,
} from '../src/context/ProfileContext'
import {
  PHOTO_BRAND,
  allDeclaredFiles,
  resolveProductImage,
} from '../src/lib/productImage'
import {
  ApiError,
  getMeta,
  getProduct,
  getProducts,
  getUser,
  postRecommend,
} from '../src/lib/api'
import type { RecommendRequest, RecommendResponse } from '../src/lib/types'

// ---------------------------------------------------------------------------
// Tiny assertion harness
// ---------------------------------------------------------------------------
let passed = 0
const failures: string[] = []

function check(label: string, condition: boolean, detail = ''): void {
  if (condition) {
    passed += 1
    console.log(`  ok   ${label}`)
  } else {
    failures.push(`${label}${detail ? ` — ${detail}` : ''}`)
    console.log(`  FAIL ${label}${detail ? ` — ${detail}` : ''}`)
  }
}

function section(title: string): void {
  console.log(`\n${title}\n${'-'.repeat(title.length)}`)
}

const LIMITS = {
  age: { min: 13, max: 100 },
  height_cm: { min: 130, max: 230 },
  weight_kg: { min: 30, max: 250 },
  shoe_size: { min: 3, max: 20 },
  workouts_per_week: { min: 0, max: 21 },
}

// ---------------------------------------------------------------------------
// 1. Meta
// ---------------------------------------------------------------------------
async function verifyMeta() {
  section('GET /api/meta — the dropdowns')
  const meta = await getMeta()

  check('10 sports', meta.sport_type.length === 10, String(meta.sport_type.length))
  check('3 brands', meta.brand.length === 3)
  check('14 colours', meta.color_preference.length === 14)
  check('10 materials', meta.material_preference.length === 10)
  check('4 style preferences', meta.style_preference.length === 4)
  check('5 climates', meta.climate.length === 5)
  check('3 categories', meta.category.length === 3)
  check('3 foot arch types', meta.foot_arch_type.length === 3)
  check('6 apparel sizes', meta.apparel_size.length === 6)
  check(
    'apparel sizes in ladder order, not alphabetical',
    meta.apparel_size.join() === 'XS,S,M,L,XL,XXL',
    meta.apparel_size.join(),
  )
  check('shoe size range present', meta.shoe_size.min > 0 && meta.shoe_size.max > meta.shoe_size.min)
  check('price range present', meta.price.max > meta.price.min)
  check('numeric limits served', meta.limits.age.min === 13 && meta.limits.age.max === 100)
  return meta
}

// ---------------------------------------------------------------------------
// 2. The user_id rule -- the contract detail that fails silently
// ---------------------------------------------------------------------------
async function verifyUserIdRule() {
  section('The user_id rule — load a sample, edit it, submit')

  const user = await getUser(42)
  check('GET /api/users/42 resolves the padded id', user.user_id === 'U00042', user.user_id)
  check('profile has a sport to populate the form', Boolean(user.primary_sport))

  // Pristine sample.
  let state: ProfileFormState = profileFormReducer(INITIAL_FORM_STATE, {
    type: 'load-sample',
    user,
  })
  check('loading a sample records the id', state.sampleUserId === 'U00042')
  check('loading a sample fills the draft', state.draft.primary_sport === user.primary_sport)
  check('source is "sample" while untouched', profileSource(state) === 'sample')

  const pristineBody = buildRecommendRequest(state, 5)
  check('untouched sample posts user_id', pristineBody.user_id === 'U00042')
  check(
    'untouched sample posts NOTHING else but top_n',
    Object.keys(pristineBody).sort().join() === 'top_n,user_id',
    Object.keys(pristineBody).join(),
  )

  // Now edit one field.
  state = profileFormReducer(state, {
    type: 'set-field',
    field: 'primary_sport',
    value: 'yoga',
  })
  check('editing clears the sample id', state.sampleUserId === null)
  check('source becomes "edited-sample"', profileSource(state) === 'edited-sample')
  check('the loaded row is still remembered for display', state.loadedFrom?.user_id === 'U00042')

  const editedBody = buildRecommendRequest(state, 5)
  check(
    'edited profile posts NO user_id',
    !('user_id' in editedBody),
    JSON.stringify(Object.keys(editedBody)),
  )
  check('edited profile posts the edit', editedBody.primary_sport === 'yoga')
  check('edited profile carries the rest of the draft', editedBody.age === user.age)

  // And prove it end to end: the API echoes user_id only when it scored a row.
  const pristineResponse = await postRecommend(pristineBody)
  const editedResponse = await postRecommend(editedBody)
  check(
    'API confirms it scored the stored row',
    pristineResponse.user_id === 'U00042',
    String(pristineResponse.user_id),
  )
  check(
    'API confirms it scored the posted profile, not the row',
    editedResponse.user_id === null,
    String(editedResponse.user_id),
  )
  check(
    'the edit actually changed the results',
    JSON.stringify(pristineResponse.recommendations.map((r) => r.product.product_id)) !==
      JSON.stringify(editedResponse.recommendations.map((r) => r.product.product_id)),
  )

  // A reset must not leave the id behind.
  const afterReset = profileFormReducer(state, { type: 'reset' })
  check('reset clears everything', afterReset.sampleUserId === null && afterReset.loadedFrom === null)

  return { user, pristineResponse }
}

// ---------------------------------------------------------------------------
// 3. Heavy relaxation, rendered
// ---------------------------------------------------------------------------
async function verifyRelaxation(): Promise<RecommendResponse> {
  section('Heavy relaxation — narrow budget + yoga + a specific colour')

  const body: RecommendRequest = {
    age: 31,
    gender: 'female',
    height_cm: 166,
    weight_kg: 58,
    shoe_size: 7.5,
    apparel_size: 'S',
    foot_arch_type: 'high',
    primary_sport: 'yoga',
    fitness_level: 'intermediate',
    workouts_per_week: 5,
    indoor_or_outdoor: 'indoor',
    budget_min: 40,
    budget_max: 65,
    preferred_brands: ['Jordan'],
    style_preference: 'athleisure',
    color_preference: 'purple',
    material_preference: 'gore-tex',
    climate: 'tropical',
    top_n: 5,
  }

  const response = await postRecommend(body)
  const { constraints, scoring } = response

  check('results came back', response.recommendations.length > 0)
  check('soft constraints emptied the set', constraints.candidates_after_soft < constraints.candidates_after_hard)
  check('relaxation happened', constraints.relaxed.length > 0, `${constraints.relaxed.length} relaxed`)
  check(
    'colour was relaxed, as session 2 predicted',
    constraints.relaxed.some((item) => item.constraint === 'color_preference'),
  )
  check('a relaxed constraint is not also still applied',
    constraints.relaxed.every((item) => !constraints.still_applied.includes(item.constraint)))

  // Render the panel with the real payload.
  const html = renderToStaticMarkup(
    <HowChosen
      constraints={constraints}
      scoring={scoring}
      latencyMs={response.latency_ms}
    />,
  )

  check('panel renders the API summary verbatim', html.includes(escapeHtml(constraints.summary)))
  for (const item of constraints.relaxed) {
    check(
      `panel renders the API sentence for "${item.constraint}" verbatim`,
      html.includes(escapeHtml(item.message)),
      item.message.slice(0, 60),
    )
  }
  check('panel renders the scoring explanation verbatim', html.includes(escapeHtml(scoring.explanation)))
  check('panel names the scoring label', html.includes(escapeHtml(scoring.label)))
  check('panel is open by default', !html.includes('hidden=""'))
  check('panel names the constraint technique', html.includes('constraint-based filter'))
  check('panel names the ranking technique', html.includes('TF-IDF cosine similarity'))
  check('panel drops the chapter prefixes', !html.includes('Chapter 7') && !html.includes('Chapter 3'))

  for (const name of constraints.hard_applied) {
    check(
      `panel reports what "${name}" removed`,
      html.includes(String(constraints.hard_excluded[name]).replace(/\B(?=(\d{3})+(?!\d))/g, ',')),
    )
  }

  // Relaxation must not be dressed as a failure.
  const WARNING_WORDS = ['warning', 'error', 'failed', 'problem', 'sorry', 'unfortunately']
  const lower = html.toLowerCase()
  for (const word of WARNING_WORDS) {
    check(`panel avoids alarm word "${word}"`, !lower.includes(word))
  }
  check('panel uses no amber/yellow warning styling', !/\b(amber|yellow)-\d{3}\b/.test(html))
  check('panel raises no alert role', !html.includes('role="alert"'))

  // A card, with its explanation chips.
  const best = Math.max(...response.recommendations.map((item) => item.score))
  const cardHtml = renderToStaticMarkup(
    <ResultCard
      recommendation={response.recommendations[0]}
      bestScore={best}
      onOpen={() => {}}
    />,
  )
  const explanations = response.recommendations[0].explanations
  check('card has explanations to show', explanations.length > 0)
  for (const line of explanations) {
    check(
      `card renders explanation verbatim: ${line.source}`,
      cardHtml.includes(escapeHtml(line.text)),
      line.text.slice(0, 50),
    )
  }
  // The technique mapping stays on the card, but as lower-case small print
  // under each heading rather than a shouted "CH7"/"CH3" badge.
  check('card groups the constraint reasons', cardHtml.includes('Constraints it satisfies'))
  check('card groups the ranking reasons', cardHtml.includes('Why it ranked here'))
  check('card names the constraint technique', cardHtml.includes('constraint-based filter'))
  check('card names the ranking technique', cardHtml.includes('TF-IDF cosine similarity'))
  check('card no longer shouts a chapter badge', !/>\s*Ch[37]\s*</.test(cardHtml))
  check('card shows the match score as a percentage', /\d+\.\d%/.test(cardHtml))

  // The photograph, with the generated tile underneath it as the fallback.
  check('card renders a photograph', cardHtml.includes('<img'))
  check('card photo is served locally, not from a CDN', !/https?:\/\//.test(cardHtml))
  check('card photo comes from /products/', cardHtml.includes('src="/products/'))
  check('card photo is cropped to fill', cardHtml.includes('object-cover'))
  check('card photo below the fold is lazy', cardHtml.includes('loading="lazy"'))
  check(
    'card photo alt says it is representative',
    /alt="Representative photograph of [^"]+"/.test(cardHtml),
  )
  check('the generated tile is still underneath', cardHtml.includes('<svg'))
  check(
    'card shows the brand over the photo',
    cardHtml.includes(escapeHtml(response.recommendations[0].product.brand)),
  )
  check(
    'card artwork is labelled for assistive tech',
    cardHtml.includes('role="img"') && cardHtml.includes('aria-label'),
  )

  return response
}

// ---------------------------------------------------------------------------
// 4. Zero results
// ---------------------------------------------------------------------------
async function verifyZeroResults() {
  section('Zero results — a budget above every product in the catalogue')

  const response = await postRecommend({
    primary_sport: 'yoga',
    gender: 'female',
    budget_min: 1650,
    budget_max: 1699,
    top_n: 10,
  })
  const { constraints } = response

  check('no recommendations', response.recommendations.length === 0)
  check('nothing survived the hard filter', constraints.candidates_after_hard === 0)
  check('the API still explains itself', constraints.summary.length > 0)
  check('the summary names budget', constraints.summary.toLowerCase().includes('budget'))
  check('budget is reported as the culprit', constraints.hard_excluded.budget > 0)
  check('every soft constraint was exhausted', constraints.exhausted)

  const html = renderToStaticMarkup(
    <HowChosen
      constraints={constraints}
      scoring={response.scoring}
      latencyMs={response.latency_ms}
    />,
  )
  check('the empty case still renders the API summary', html.includes(escapeHtml(constraints.summary)))
  return response
}

// ---------------------------------------------------------------------------
// 5. Scoring modes
// ---------------------------------------------------------------------------
async function verifyScoringModes() {
  section('Scoring mode — cold start and the centroid path')

  const anonymous = await postRecommend({ primary_sport: 'basketball', top_n: 3 })
  check('an anonymous profile is profile_only', anonymous.scoring.mode === 'profile_only')
  check('it reports zero qualifying ratings', anonymous.scoring.qualifying_ratings === 0)
  check('it carries a UI label', anonymous.scoring.label.length > 0, anonymous.scoring.label)

  // U00351 has the most 4+ ratings in the session-1 interaction build.
  const heavyRater = await getUser('U00351')
  check(
    'a heavy rater is predicted as profile_plus_centroid',
    heavyRater.expected_scoring_mode === 'profile_plus_centroid',
    heavyRater.expected_scoring_mode,
  )
  const scored = await postRecommend({ user_id: heavyRater.user_id, top_n: 3 })
  check(
    'and the recommendation agrees',
    scored.scoring.mode === 'profile_plus_centroid',
    scored.scoring.mode,
  )
  check(
    'the /api/users prediction matches the actual rating count',
    scored.scoring.qualifying_ratings === heavyRater.qualifying_ratings,
  )
}

// ---------------------------------------------------------------------------
// 6. Validation -- the client mirror and the API's own 422
// ---------------------------------------------------------------------------
async function verifyValidation() {
  section('Validation — client mirror and API 422')

  const bad = { ...INITIAL_FORM_STATE.draft, budget_min: '200', budget_max: '50' }
  const errors = validateDraft(bad, LIMITS)
  check('client catches a backwards budget', Boolean(errors.budget_max))

  check(
    'client catches an out-of-range age',
    Boolean(validateDraft({ ...INITIAL_FORM_STATE.draft, age: '120' }, LIMITS).age),
  )
  check(
    'client accepts the boundary age',
    !validateDraft({ ...INITIAL_FORM_STATE.draft, age: '100' }, LIMITS).age,
  )
  check(
    'client catches an out-of-range shoe size',
    Boolean(validateDraft({ ...INITIAL_FORM_STATE.draft, shoe_size: '21' }, LIMITS).shoe_size),
  )
  check(
    'client rejects equal budgets, as the API does',
    Boolean(
      validateDraft(
        { ...INITIAL_FORM_STATE.draft, budget_min: '100', budget_max: '100' },
        LIMITS,
      ).budget_max,
    ),
  )

  // And the API's own 422 arrives with the valid values named.
  try {
    await postRecommend({ primary_sport: 'quidditch' } as RecommendRequest)
    check('an invalid sport is rejected', false, 'no error thrown')
  } catch (caught) {
    const error = caught as ApiError
    check('an invalid sport is a 422', error.status === 422)
    check('the 422 names the field', error.fieldErrors[0]?.field === 'primary_sport')
    check('the 422 names every valid sport', error.message.includes('yoga') && error.message.includes('skateboarding'))
    check('the client can map it back onto a field', 'primary_sport' in error.byField())
  }

  // An empty profile is valid: every field is optional.
  const emptyBody = toProfileFields(INITIAL_FORM_STATE.draft)
  check('an empty draft posts an empty body', Object.keys(emptyBody).length === 0)
  const emptyResponse = await postRecommend({ ...emptyBody, top_n: 3 })
  check('and the API still returns results', emptyResponse.recommendations.length === 3)
}

// ---------------------------------------------------------------------------
// 7. Catalogue paging and product detail
// ---------------------------------------------------------------------------
async function verifyCatalogue() {
  section('Catalogue — paging, filters, product detail')

  const first = await getProducts({ page: 1, page_size: 24 })
  const second = await getProducts({ page: 2, page_size: 24 })

  check('page 1 is full', first.items.length === 24)
  check('the total is the whole catalogue', first.total === 10000)
  check('total_pages is computed by the API', first.total_pages === Math.ceil(10000 / 24))
  check('page 1 has a next and no previous', first.has_next && !first.has_previous)
  check('page 2 has a previous', second.has_previous)
  const firstIds = new Set(first.items.map((item) => item.product_id))
  check('pages do not overlap', second.items.every((item) => !firstIds.has(item.product_id)))

  // The default order interleaves brands, so the catalogue does not open on a
  // single-brand block the way the CSV write order made it.
  const pageBrands = first.items.map((item) => item.brand)
  check(
    'page 1 is not one brand',
    new Set(pageBrands).size > 1,
    [...new Set(pageBrands)].join(', '),
  )
  check(
    'page 1 carries the catalogue mix, Nike ahead of Adidas ahead of Jordan',
    pageBrands.filter((b) => b === 'Nike').length >
      pageBrands.filter((b) => b === 'Adidas').length &&
      pageBrands.filter((b) => b === 'Adidas').length >
        pageBrands.filter((b) => b === 'Jordan').length &&
      pageBrands.includes('Jordan'),
  )
  const again = await getProducts({ page: 1, page_size: 24 })
  check(
    'the same request returns the same page',
    again.items.map((item) => item.product_id).join(',') ===
      first.items.map((item) => item.product_id).join(','),
  )

  const filtered = await getProducts({
    sport: 'yoga',
    brand: 'Nike',
    category: 'apparel',
    price_min: 30,
    price_max: 90,
    page_size: 50,
  })
  check('filters narrow the set', filtered.total > 0 && filtered.total < first.total)
  check(
    'every filtered row matches every filter',
    filtered.items.every(
      (item) =>
        item.sport_type === 'yoga' &&
        item.brand === 'Nike' &&
        item.category === 'apparel' &&
        item.price >= 30 &&
        item.price <= 90,
    ),
  )
  check('the API echoes the filters back', filtered.filters.sport === 'yoga')

  const detail = await getProduct(first.items[0].product_id)
  check('product detail resolves', detail.product_id === first.items[0].product_id)
  check('detail carries the description the summary lacks', typeof detail.product_description === 'string')
  check('detail carries the extra columns', 'cushioning_level' in detail && 'sales_rank' in detail)

  try {
    await getProduct('NOPE12345')
    check('an unknown product is a 404', false, 'no error thrown')
  } catch (caught) {
    const error = caught as ApiError
    check('an unknown product is a 404', error.status === 404)
    check('with a machine-readable code', error.code === 'product_not_found')
    check('and a hint', Boolean(error.hint))
  }

  try {
    await getUser(99999)
    check('an unknown user is a 404', false, 'no error thrown')
  } catch (caught) {
    check('an unknown user is a 404', (caught as ApiError).status === 404)
  }
}

// ---------------------------------------------------------------------------
// 8. The pages render
// ---------------------------------------------------------------------------
function verifyPagesRender() {
  section('Pages render without throwing')

  const shell = (path: string) =>
    renderToStaticMarkup(
      <MemoryRouter initialEntries={[path]}>
        <MetaProvider>
          <ProfileProvider>
            <Layout />
          </ProfileProvider>
        </MetaProvider>
      </MemoryRouter>,
    )

  const landing = renderToStaticMarkup(
    <MemoryRouter>
      <MetaProvider>
        <Landing />
      </MetaProvider>
    </MemoryRouter>,
  )
  check('landing renders the tagline', landing.includes('The right gear for the right athlete'))
  // The cards and the panel no longer shout the chapter numbers, so the
  // landing explainer is where the technique-to-chapter mapping has to stay
  // discoverable. These two assertions are what keep it there.
  check('landing still maps the constraint filter to its chapter', landing.includes('Chapter 7'))
  check('landing still maps the content ranker to its chapter', landing.includes('Chapter 3'))
  check('landing explains TF-IDF cosine similarity', landing.includes('TF-IDF'))
  check('landing has the Find My Gear call to action', /find my gear/i.test(landing))

  for (const path of ['/', '/profile', '/results', '/catalog', '/nope']) {
    let html = ''
    let threw: unknown = null
    try {
      html = shell(path)
    } catch (caught) {
      threw = caught
    }
    check(`${path} renders`, threw === null, threw ? String(threw) : '')
    check(`${path} renders a skip link`, html.includes('Skip to content'))
  }
}

// ---------------------------------------------------------------------------
// 9. Generated product artwork
// ---------------------------------------------------------------------------
async function verifyArtwork() {
  section('Generated tile — the photo fallback, still deterministic')

  const page = await getProducts({ page: 1, page_size: 60 })
  const render = (product: (typeof page.items)[number]) =>
    renderToStaticMarkup(
      <ProductArtwork
        productId={product.product_id}
        brand={product.brand}
        category={product.category}
        color={product.color}
        size="card"
      />,
    )

  const first = page.items[0]
  const html = render(first)

  check('no <img> element — this layer never loads a file', !html.includes('<img'))
  check('drawn as inline SVG', html.includes('<svg'))
  check('no remote URL of any kind', !/https?:\/\//.test(html))
  check('brand is rendered', html.includes(escapeHtml(first.brand)))
  check('category is rendered', html.toLowerCase().includes(first.category.toLowerCase()))
  check('carries an accessible label', html.includes('role="img"'))

  // Deterministic: the same product always yields the same tile.
  check('same product renders identically twice', render(first) === render(first))

  // Colour-derived: the tile uses the product's own colour as a real hex.
  const byColour = new Map<string, (typeof page.items)[number]>()
  for (const item of page.items) {
    if (!byColour.has(item.color)) byColour.set(item.color, item)
  }
  const palettes = new Set<string>()
  for (const [colour, item] of byColour) {
    const markup = render(item)
    const found = markup.match(/#[0-9A-Fa-f]{6}/g) ?? []
    check(`"${colour}" tile carries a hex fill`, found.length > 0)
    palettes.add(found.join())
  }
  check(
    'different colours produce different palettes',
    palettes.size === byColour.size,
    `${palettes.size} distinct of ${byColour.size} colours`,
  )

  // Seeded by product_id: same colour + category still differ.
  const sameLook = page.items.filter(
    (item) => item.color === first.color && item.category === first.category,
  )
  if (sameLook.length >= 2) {
    check(
      'two products sharing colour and category still differ',
      render(sameLook[0]) !== render(sameLook[1]),
    )
  }

  // Every catalogue colour maps to something -- no product falls through.
  const meta = await getMeta()
  for (const colour of meta.color_preference) {
    const markup = renderToStaticMarkup(
      <ProductArtwork
        productId="NK00000001"
        brand="Nike"
        category="footwear"
        color={colour}
        size="card"
      />,
    )
    check(`colour "${colour}" renders`, markup.includes('<svg'))
  }

  // Every category glyph resolves.
  for (const category of meta.category) {
    const markup = renderToStaticMarkup(
      <ProductArtwork
        productId="AD00000001"
        brand="Adidas"
        category={category}
        color="blue"
        size="tile"
      />,
    )
    check(`category "${category}" renders a glyph`, markup.includes('<path'))
  }
}

// ---------------------------------------------------------------------------
// 10. Product photography
// ---------------------------------------------------------------------------
async function verifyPhotos() {
  section('Product photography — resolution, determinism, files on disk')

  const photoDir = join(process.cwd(), 'public', 'products')
  const onDisk = new Set(
    existsSync(photoDir) ? readdirSync(photoDir).filter((n) => n.endsWith('.jpg')) : [],
  )
  check('the photo directory exists', onDisk.size > 0, `${onDisk.size} files`)
  check(
    'every declared file is on disk',
    allDeclaredFiles().every((file) => onDisk.has(file)),
  )
  check(
    'every file on disk is declared',
    [...onDisk].every((file) => allDeclaredFiles().includes(file)),
  )

  // Resolution over a real slice of the catalogue.
  const page = await getProducts({ page: 1, page_size: 200 })
  const tiers = new Map<string, number>()
  let allExist = true
  for (const product of page.items) {
    const resolved = resolveProductImage(product)
    if (!onDisk.has(resolved.src.replace('/products/', ''))) allExist = false
    tiers.set(resolved.tier, (tiers.get(resolved.tier) ?? 0) + 1)
  }
  check(`all ${page.items.length} products on page 1 resolve to a real file`, allExist)
  check('resolution uses the specific tiers, not just the fallback', !tiers.has('default'))

  // Determinism: same product, same photo, every time.
  const sample = page.items[0]
  check(
    'the same product always resolves to the same photo',
    resolveProductImage(sample).src === resolveProductImage(sample).src,
  )

  // Brand: never a rival's logo under the label.
  const conflicts = page.items.filter((item) => {
    const mark = PHOTO_BRAND[resolveProductImage(item).src.replace('/products/', '')]
    return mark !== undefined && mark !== item.brand.trim().toLowerCase()
  })
  check(
    "no product resolves to a photo carrying another brand's logo",
    conflicts.length === 0,
    conflicts.length
      ? `${conflicts[0].brand} ${conflicts[0].product_id} -> ${resolveProductImage(conflicts[0]).src}`
      : `${page.items.length} products checked`,
  )
  // The same product must still resolve the same way whatever else changes --
  // brand is now an input, so a different brand is allowed a different photo,
  // but the same product is not.
  check(
    'brand filtering did not cost determinism',
    page.items.every(
      (item) => resolveProductImage(item).src === resolveProductImage({ ...item }).src,
    ),
  )

  // Sense: the photo has to match what the product is.
  const shoes = page.items.filter((item) => item.category === 'footwear')
  check(
    'footwear resolves to footwear photography',
    shoes.length > 0 &&
      shoes.every((item) => resolveProductImage(item).pool.startsWith('footwear-')),
  )
  const runningShoes = shoes.filter((item) => item.sport_type === 'running')
  if (runningShoes.length > 0) {
    check(
      'running shoes resolve to the running-shoe pool',
      runningShoes.every(
        (item) => resolveProductImage(item).pool === 'footwear-running',
      ),
    )
  }
  const hoodies = page.items.filter(
    (item) => item.subcategory === 'hoodies-and-sweatshirts',
  )
  if (hoodies.length > 0) {
    check(
      'hoodies resolve to hoodie or sweatshirt photography',
      hoodies.every((item) =>
        ['apparel-hoodie', 'apparel-sweatshirt'].includes(
          resolveProductImage(item).pool,
        ),
      ),
    )
  }

  // Spread: a bucket with several photos should not collapse onto one.
  const lifestyleShoes = page.items.filter(
    (item) => item.category === 'footwear' && item.sport_type === 'lifestyle',
  )
  if (lifestyleShoes.length >= 12) {
    const used = new Set(lifestyleShoes.map((item) => resolveProductImage(item).src))
    check(
      'the hash spreads across a multi-photo pool',
      used.size >= 4,
      `${used.size} of 6 lifestyle photos used across ${lifestyleShoes.length} products`,
    )
  }

  // The rendered component.
  const html = renderToStaticMarkup(
    <ProductPhoto
      product_id={sample.product_id}
      brand={sample.brand}
      category={sample.category}
      color={sample.color}
      sport_type={sample.sport_type}
      subcategory={sample.subcategory}
      size="tile"
    />,
  )
  check('component renders an <img>', html.includes('<img'))
  check('component points at the resolved file', html.includes(resolveProductImage(sample).src))
  check('component crops with object-cover', html.includes('object-cover'))
  check('component holds a fixed aspect ratio', html.includes('aspect-'))
  check('component lazy-loads by default', html.includes('loading="lazy"'))
  check('component keeps the generated tile underneath', html.includes('<svg'))
  check('component draws a gradient scrim over the photo', html.includes('linear-gradient(to top'))
  check('component labels the brand over the photo', html.includes(escapeHtml(sample.brand)))

  const modalHtml = renderToStaticMarkup(
    <ProductPhoto
      product_id={sample.product_id}
      brand={sample.brand}
      category={sample.category}
      color={sample.color}
      sport_type={sample.sport_type}
      subcategory={sample.subcategory}
      size="modal"
      priority
    />,
  )
  check('the modal loads its photo eagerly', modalHtml.includes('loading="eager"'))
  check(
    'the modal shows the same photo as the card',
    modalHtml.includes(resolveProductImage(sample).src),
  )
}

// ---------------------------------------------------------------------------
/** Escape the way React does, so "contains verbatim" compares like for like. */
function escapeHtml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#x27;')
}

async function main() {
  console.log('FitMatch frontend verification — against the live API\n')
  try {
    await verifyMeta()
    await verifyUserIdRule()
    await verifyRelaxation()
    await verifyZeroResults()
    await verifyScoringModes()
    await verifyValidation()
    await verifyCatalogue()
    await verifyArtwork()
    await verifyPhotos()
    verifyPagesRender()
  } catch (caught) {
    if (caught instanceof ApiError && caught.isOffline) {
      console.error(
        `\nThe API is not running. Start it with run_api.bat, then re-run.\n${caught.message}`,
      )
      process.exit(2)
    }
    throw caught
  }

  console.log(`\n${'='.repeat(60)}`)
  if (failures.length === 0) {
    console.log(`ALL ${passed} CHECKS PASSED`)
  } else {
    console.log(`${passed} passed, ${failures.length} FAILED:`)
    for (const failure of failures) console.log(`  - ${failure}`)
  }
  process.exit(failures.length === 0 ? 0 : 1)
}

void main()
