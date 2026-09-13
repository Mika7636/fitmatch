# FitMatch frontend

React + Vite + TypeScript + Tailwind v4 over the session-3 FastAPI backend.
The root `README.md` has the project-level picture; this file is the detail.

```bash
npm install
npm run dev        # http://localhost:5173  (or: run_frontend.bat from the project root)
```

**The backend must be running.** Start `run_api.bat` in another terminal
first — the header shows a red dot and every screen explains itself if it is
not.

| command | what it does |
| --- | --- |
| `npm run dev` | Vite dev server on port 5173 |
| `npm run build` | typecheck, then a production build into `dist/` |
| `npm run preview` | serve the production build |
| `npm run typecheck` | `tsc -b --force`, no emit |
| `npm run lint` | oxlint |
| `npm run verify` | end-to-end checks against a running API (see below) |

## Why the scripts call `node` directly

This checkout's absolute path contains a comma (`…\Uni\4th,1st\…`). cmd.exe
treats a comma as an argument separator, so npm's generated `.bin` shims
resolve to nonsense — `npx tsc` looks for `C:\Users\user\Uni\4th,1st\typescript\bin\tsc`
and fails. Every script therefore invokes `node node_modules/<pkg>/…` and
skips the shim. `npx` will not work here; `npm run <script>` will.

## Port 5173 is not negotiable

`src/api/main.py` allows CORS from `http://localhost:5173` and
`http://127.0.0.1:5173` only, so `vite.config.ts` sets `strictPort: true`. A
clash fails loudly instead of quietly moving to 5174, where the browser would
block every request with no obvious cause.

## Layout

```
src/
  lib/
    types.ts        TypeScript mirrors of src/api/schemas.py, model for model
    api.ts          the ONLY file that calls fetch
    format.ts       display helpers (title case, currency, score bars)
  context/
    MetaContext     /api/meta, fetched once, shared
    ProfileContext  the multi-step draft, and the user_id rule
  hooks/useAsync    one request + loading + error + retry, with abort
  components/
    layout/         header, nav, footer, backend status dot
    ui/             buttons, badges, fields, dual range slider, modal
    profile/        the three form steps and the sample loader
    results/        result cards, explanation chips, "How these were chosen"
  pages/            Landing, ProfileBuilder, Results, Catalog
scripts/            the headless verification harness
```

## The three contract details

### 1. `user_id` means "score the stored row, ignore the body"

`POST /api/recommend` treats `user_id` as an instruction to score that
`users.csv` row and discard every other field. So a body carrying both a
`user_id` and an edited profile silently scores the original row — the user
sees somebody else's results with nothing to indicate it.

The rule lives in `profileFormReducer` in `context/ProfileContext.tsx`:
loading a sample records the id, and **any** `set-field` clears it.
`buildRecommendRequest` then emits one of exactly two shapes, never a hybrid:

```ts
{ user_id: 'U00042', top_n: 10 }              // sample, untouched
{ age: 51, gender: 'male', …, top_n: 10 }     // anything else, no user_id
```

It is a pure reducer rather than three `useState` calls specifically so the
rule can be tested without a browser, which `npm run verify` does. The profile
form also has a "Show request body" toggle that prints the exact JSON, and the
sample loader says in words which of the two is queued up.

### 2. Relaxation is the normal path

The median profile matches **zero** products with every preference applied;
colour is relaxed for 98.7% of profiles. `constraints.relaxed[]` arrives with
a `message` the backend wrote to be displayed, and
`components/results/HowChosen.tsx` renders those sentences **verbatim** —
no paraphrasing, no second copy of the explanation in this codebase.

Nothing about it is styled as a warning: no amber, no alert role, no "failed".
`npm run verify` asserts all of that against the rendered HTML.

### 3. The scoring mode is informational

85.2% of requests use `profile_only`. `scoring.label` appears as a neutral grey
badge next to the results heading and again in the panel, with
`scoring.explanation` beneath it. Grey, not red.

## Product photography

`public/products/` holds 110 representative stock photographs, named for what
they depict: `<category>-<type>-<n>.jpg`, e.g. `footwear-running-2.jpg`,
`apparel-hoodie-1.jpg`, `accessory-cap-5.jpg`.

`lib/productImage.ts` maps a product onto one of them, resolving in order:

1. `category + sport_type + subcategory` — refinements for subcategories that
   say nothing about the garment (`clothing`, `matching-sets`, …), where the
   sport is the better signal;
2. `category + sport_type` — **footwear only**, because for a shoe the sport
   *is* the product type. Deliberately empty for apparel and accessories: a
   yoga jacket is a jacket, and an `apparel|yoga` rule here would outrank the
   subcategory and show leggings;
3. `category + subcategory` — the workhorse for apparel and accessories;
4. `category` — always defined, so the resolver is total.

Within a bucket the pick is `hashId(product_id)` (`lib/hash.ts`), so a product
shows the same photo on every render while the distribution spreads across the
whole bucket. `npm run check:images` — which `npm run build` runs first —
fails if the manifest and the directory disagree or if any of the 10,000
products resolves to a missing file.

Some catalogue vocabulary is irregular and the tables encode that:
`matching-sets` is mostly crew-neck sweatshirts, `accessory/clothing` is
compression sleeves and sport bands, and `accessory/football` is actual
footballs (no ball photo exists, so it takes the generic sports-kit pool).

The root README carries the note on why no real product photography exists,
plus credits and licence.

## `ProductPhoto` and `ProductArtwork`

`ProductPhoto` is three layers:

1. `ProductArtwork` — the generated SVG tile, kept as the **loading** state so
   a card never flashes empty and as the **error** state so a 404 settles on a
   deliberate-looking tile rather than a broken-image icon;
2. the `<img>`, `object-fit: cover` in a fixed aspect ratio, faded in on load,
   `loading="lazy"` everywhere except the modal (which opens on demand and is
   unquestionably in view);
3. a crimson-to-transparent scrim plus the brand/category label, so white text
   stays legible over a photograph of unknown brightness — a white trainer on a
   white background would otherwise swallow it.

The label lives in `ProductPhoto`, not `ProductArtwork` (which is passed
`showLabel={false}`), because it has to sit above the photo.

## Technique labels, not chapter badges

The result cards used to carry "CH7"/"CH3" badge chips. They shouted the
coursework's structure over the actual reason a product was on screen, so they
are gone. What replaced them:

- cards keep the headings "Constraints it satisfies" and "Why it ranked here",
  with `constraint-based filter` and `TF-IDF cosine similarity` in lower-case
  small print beneath;
- the "How these were chosen" panel keeps its four numbered steps, with the
  same two technique names in the subheadings and no `Chapter 7 ·` prefix.

The mapping from technique to chapter stays discoverable on the landing page's
three-step explainer, which is the one place it is the actual subject rather
than a label on something else. `npm run verify` asserts both halves: that the
technique names are on the cards and in the panel, and that the landing page
still names the chapters.

## `npm run verify`

An end-to-end check with no browser, run against a live API:

```bash
run_api.bat          # terminal 1
npm run verify       # terminal 2
```

It replays every flow through the app's own `lib/api.ts` — sample load, edit,
submit, heavy relaxation, zero results, catalogue paging, product detail, 404s
and a 422 — and then renders the real components with those real payloads via
`react-dom/server`, asserting on the HTML that each API sentence appears
verbatim and that nothing is dressed up as an error. 174 checks.

It runs through Vite's own SSR module loader (`scripts/run-verify.mjs`), so it
needs no bundler dependency of its own and transforms the same source the
browser gets. `renderToStaticMarkup` runs no effects, so components that fetch
their own data are covered through their loading state, and the flows those
effects drive are covered by the direct API calls instead.

## Configuration

`VITE_API_BASE_URL` overrides the backend origin; it defaults to
`http://127.0.0.1:8000`. Copy `.env.example` to `.env.local` to change it.

## Accessibility notes

- Every input has a real `<label for>`; errors are wired through
  `aria-invalid` and `aria-describedby`.
- One focus style, defined once in `index.css`, visible on dark red, near-black
  and light grey alike.
- The dual range slider is two native `<input type="range">` elements, so it is
  keyboard-operable and announces itself without custom ARIA.
- The product modal traps focus, closes on Escape, and returns focus to the
  trigger.
- Skip link, landmark regions, `aria-live` on the result counts, and
  `prefers-reduced-motion` honoured.

Colour contrast (WCAG AA needs 4.5:1 for body text):

| pair | ratio |
| --- | --- |
| white on `#8B0000` | ~11.6:1 |
| white on `#A80000` | ~8.6:1 |
| `#E8E8E8` on `#0A0A0A` | ~17.9:1 |
| `#0A0A0A` on `#E8E8E8` | ~17.9:1 |

Crimson is only ever a background for white text, never text on dark.

## State

Profile state is in memory only — no `localStorage`, as specified. A refresh
clears it, and `/results` says so and offers the form rather than showing a
blank page. Catalogue filters live in the URL query string instead, so a
filtered view is shareable and the back button steps through filter changes.
