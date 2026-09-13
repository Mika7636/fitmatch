/**
 * One hash, shared by everything that needs a stable per-product choice.
 *
 * Both the generated artwork and the photo resolver key off `product_id`, and
 * they must agree on what "deterministic" means: the same product has to get
 * the same tile and the same photograph on every render, in every session, on
 * every machine. Two separate implementations would be two things to keep in
 * step, so there is one.
 */

/**
 * FNV-1a over the string, then an avalanche step.
 *
 * The avalanche matters here. Catalogue ids are near-sequential
 * (`NK14072420`, `NK14072421`), and plain FNV-1a leaves neighbouring inputs
 * with neighbouring outputs — which would hand consecutive products the same
 * photo and visibly band the catalogue grid. Mixing the high bits down makes
 * every output bit depend on every input bit.
 */
export function hashId(value: string): number {
  let hash = 0x811c9dc5
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index)
    hash = Math.imul(hash, 0x01000193)
  }
  hash ^= hash >>> 16
  hash = Math.imul(hash, 0x7feb352d)
  hash ^= hash >>> 15
  return hash >>> 0
}

/** Pull an independent small value out of its own byte of a hash. */
export function hashSlice(hash: number, shift: number, modulo: number): number {
  return ((hash >>> shift) & 0xff) % modulo
}
