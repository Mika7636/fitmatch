/**
 * The landing page.
 *
 * The three-step explainer is the part that earns its place: it names both
 * techniques, in order, with the catalogue numbers the API reports. A
 * recommender coursework demo that does not say what it is doing is missing
 * the point of the assignment.
 */

import { Link } from 'react-router-dom'

import { useMeta } from '../context/MetaContext'
import { formatCount } from '../lib/format'
import { Eyebrow } from '../components/ui/primitives'

const STEPS = [
  {
    index: '01',
    title: 'You describe the athlete',
    chapter: null,
    body: 'Body, sport, training load, climate, budget and taste. Anything you leave blank stays blank — an unstated preference switches its filter off rather than being guessed at.',
  },
  {
    index: '02',
    title: 'A constraint filter narrows the field',
    chapter: 'Chapter 7 — knowledge-based',
    body: 'Explicit domain rules, not learned weights. Budget, gender, stock and sport are hard and never bend. Colour, material, arch support, season, brand and size are soft, and get relaxed in a fixed order until enough candidates survive to rank.',
  },
  {
    index: '03',
    title: 'TF-IDF cosine similarity ranks what survives',
    chapter: 'Chapter 3 — content-based',
    body: 'Every product becomes a bag of words from its brand, sport, material, colour and description. Your profile becomes one too, in the same vocabulary. The match score is the cosine of the angle between them — no ratings from other users involved.',
  },
]

export function Landing() {
  const { meta } = useMeta()
  const productCount = meta?.counts.products
  const sportCount = meta?.counts.sport_type

  return (
    <div>
      {/* Hero */}
      <section className="relative overflow-hidden border-b border-ink-700 bg-ink-950">
        <div
          aria-hidden="true"
          className="absolute inset-y-0 right-0 hidden w-1/2 skew-x-[-12deg] translate-x-24 stripe-crimson opacity-90 lg:block"
        />
        <div
          aria-hidden="true"
          className="absolute inset-0 stripe-ink"
        />

        <div className="relative mx-auto max-w-7xl px-4 py-20 sm:px-6 sm:py-28 lg:py-36">
          <div className="max-w-2xl">
            <Eyebrow>Sportswear recommendation engine</Eyebrow>

            <h1 className="mt-4 text-6xl leading-[0.92] sm:text-7xl lg:text-8xl">
              <span className="block text-ash">Fit</span>
              <span className="block text-crimson-500">Match</span>
            </h1>

            <p className="mt-6 max-w-xl font-display text-xl font-semibold uppercase leading-tight tracking-wide text-ash sm:text-2xl">
              The right gear for the right athlete — every time.
            </p>

            <p className="mt-5 max-w-xl text-base leading-relaxed text-ash-dim">
              Two recommendation techniques in sequence: a knowledge-based
              constraint filter decides what is admissible, then TF-IDF cosine
              similarity decides the order. Every result comes back with the
              reasons behind it.
            </p>

            <div className="mt-9 flex flex-wrap gap-3">
              <Link
                to="/profile"
                className="group inline-flex items-center gap-3 border-2 border-crimson-700 bg-crimson-700 px-8 py-4 font-display text-base font-bold uppercase tracking-[0.12em] text-white transition-colors hover:border-crimson-600 hover:bg-crimson-600"
              >
                Find my gear
                <span
                  aria-hidden="true"
                  className="transition-transform group-hover:translate-x-1"
                >
                  →
                </span>
              </Link>
              <Link
                to="/catalog"
                className="inline-flex items-center border-2 border-ash/40 px-8 py-4 font-display text-base font-bold uppercase tracking-[0.12em] text-ash transition-colors hover:border-ash hover:bg-ash/10"
              >
                Browse the catalogue
              </Link>
            </div>

            {productCount !== undefined && (
              <dl className="mt-12 flex flex-wrap gap-x-10 gap-y-4">
                {[
                  { value: formatCount(productCount), label: 'Products' },
                  { value: formatCount(sportCount ?? 0), label: 'Sports' },
                  {
                    value: formatCount(meta?.counts.users ?? 0),
                    label: 'Sample profiles',
                  },
                ].map((stat) => (
                  <div key={stat.label}>
                    <dt className="eyebrow text-ash-muted">{stat.label}</dt>
                    <dd className="font-display text-3xl font-black text-ash">
                      {stat.value}
                    </dd>
                  </div>
                ))}
              </dl>
            )}
          </div>
        </div>
      </section>

      {/* Three-step explainer */}
      <section className="bg-ash py-16 sm:py-24">
        <div className="mx-auto max-w-7xl px-4 sm:px-6">
          <Eyebrow className="text-crimson-700">How it works</Eyebrow>
          <h2 className="mt-3 max-w-3xl text-4xl text-ink-900 sm:text-5xl">
            Two techniques, in sequence
          </h2>

          <ol className="mt-12 grid gap-px bg-ink-900/15 sm:grid-cols-3">
            {STEPS.map((step) => (
              <li key={step.index} className="relative bg-ash p-6 sm:p-8">
                <span
                  aria-hidden="true"
                  className="font-display text-5xl font-black text-crimson-700/25"
                >
                  {step.index}
                </span>
                <h3 className="mt-3 text-xl text-ink-900">{step.title}</h3>
                {step.chapter && (
                  <p className="mt-2 inline-block bg-crimson-700 px-2 py-1 font-display text-[10px] font-bold uppercase tracking-[0.14em] text-white">
                    {step.chapter}
                  </p>
                )}
                <p className="mt-3 text-sm leading-relaxed text-ink-700">
                  {step.body}
                </p>
              </li>
            ))}
          </ol>

          <div className="mt-12 border-l-4 border-crimson-700 bg-ink-900 p-6 sm:p-8">
            <h3 className="text-lg text-ash">
              Why you will see “relaxed” preferences
            </h3>
            <p className="mt-3 max-w-3xl text-sm leading-relaxed text-ash-dim">
              With every stated preference applied at once, the median profile
              in this catalogue matches <strong className="text-ash">zero</strong>{' '}
              products — colour alone rules out almost everything, and it is
              relaxed for 98.7% of profiles. So the filter gives up the cheapest
              preferences first, in a fixed order, until enough candidates
              survive to rank. Your results will say exactly which ones went and
              what each one cost. That is the design working, not a failure.
            </p>
            <Link
              to="/profile"
              className="mt-6 inline-flex items-center gap-2 font-display text-sm font-bold uppercase tracking-[0.12em] text-crimson-500 hover:text-crimson-600"
            >
              Build a profile
              <span aria-hidden="true">→</span>
            </Link>
          </div>
        </div>
      </section>
    </div>
  )
}
