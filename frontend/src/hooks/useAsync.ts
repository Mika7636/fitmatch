/**
 * One request, with its loading and error states.
 *
 * Every screen in the app is "fetch something, show a skeleton, show an error
 * with a retry, or show the data". Doing that by hand in each component is how
 * blank screens happen, so it is done once here.
 */

import { useCallback, useEffect, useRef, useState } from 'react'

export interface AsyncState<T> {
  data: T | null
  loading: boolean
  error: unknown
  /** Re-run the request; used by every error state's "Try again" button. */
  reload: () => void
}

/**
 * Run `fn` on mount and whenever `deps` change.
 *
 * The request is aborted if the component unmounts or the deps change while
 * it is in flight, and a resolved-but-stale response is dropped rather than
 * written into state -- which is what stops a slow page-1 response from
 * overwriting a fast page-2 one in the catalogue.
 */
export function useAsync<T>(
  fn: (signal: AbortSignal) => Promise<T>,
  deps: readonly unknown[],
): AsyncState<T> {
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<unknown>(null)
  const [nonce, setNonce] = useState(0)

  // `fn` is typically an inline arrow, so it is a new function every render;
  // keeping it in a ref lets the effect depend on `deps` alone.
  const fnRef = useRef(fn)
  fnRef.current = fn

  useEffect(() => {
    const controller = new AbortController()
    let active = true

    setLoading(true)
    setError(null)

    fnRef
      .current(controller.signal)
      .then((result) => {
        if (!active) return
        setData(result)
        setLoading(false)
      })
      .catch((caught: unknown) => {
        if (!active || controller.signal.aborted) return
        setError(caught)
        setLoading(false)
      })

    return () => {
      active = false
      controller.abort()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce])

  const reload = useCallback(() => setNonce((value) => value + 1), [])

  return { data, loading, error, reload }
}
