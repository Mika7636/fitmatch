/**
 * Display helpers.
 *
 * The catalogue's vocabularies are lower-case machine values (`gore-tex`,
 * `indoor_or_outdoor`, `all-season`). They must be *displayed* in title case
 * but *sent* exactly as they came from `/api/meta`, so every function here
 * returns a new string for the screen and never mutates what gets posted.
 */

const CURRENCY = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  maximumFractionDigits: 2,
})

const INTEGER = new Intl.NumberFormat('en-US')

export function formatPrice(value: number): string {
  return CURRENCY.format(value)
}

export function formatCount(value: number): string {
  return INTEGER.format(value)
}

/** `gore-tex` -> `Gore-Tex`, `indoor_or_outdoor` -> `Indoor Or Outdoor`. */
export function titleCase(value: string): string {
  return value
    .replace(/[_]/g, ' ')
    .split(' ')
    .map((word) =>
      word
        .split('-')
        .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
        .join('-'),
    )
    .join(' ')
}

/** `in_stock` -> `In stock`. Sentence case, for status text. */
export function sentenceCase(value: string): string {
  const spaced = value.replace(/[_-]/g, ' ')
  return spaced.charAt(0).toUpperCase() + spaced.slice(1)
}

/**
 * Cosine similarity as a percentage.
 *
 * Scores are genuinely low -- a TF-IDF cosine of 0.17 is a strong match in a
 * 1,132-term vocabulary -- so this is labelled "match" in the UI and never
 * presented as a probability or a confidence.
 */
export function formatScore(score: number): string {
  return `${(score * 100).toFixed(1)}%`
}

/** Width for the score bar, scaled so typical scores are readable.
 *
 * Raw cosine values cluster below 0.3, which would make every bar a stub. The
 * bar is scaled against the best score on the page instead, so it compares
 * results to each other -- which is what a ranked list is for -- rather than
 * to an absolute 1.0 nothing ever reaches.
 */
export function scoreBarWidth(score: number, best: number): string {
  if (best <= 0) return '0%'
  const ratio = Math.max(0.06, Math.min(1, score / best))
  return `${(ratio * 100).toFixed(1)}%`
}

/** The product image placeholder: deterministic per product, not random. */
export function productHue(productId: string): number {
  let hash = 0
  for (let index = 0; index < productId.length; index += 1) {
    hash = (hash * 31 + productId.charCodeAt(index)) % 360
  }
  return hash
}

export function formatRating(rating: number | null): string {
  return rating === null ? 'No rating' : rating.toFixed(1)
}

/** `U00042` -> `42`, for showing a sample-profile id compactly. */
export function shortUserId(userId: string): string {
  const digits = userId.replace(/^U0*/i, '')
  return digits || userId
}
