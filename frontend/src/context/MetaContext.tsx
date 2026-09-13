/**
 * `/api/meta`, fetched once and shared.
 *
 * Every dropdown in the app reads from here. Nothing hardcodes an option list:
 * the backend measures these from products.csv and users.csv at startup and
 * rejects anything outside them with a 422, so a hardcoded list would break
 * the moment the catalogue is rebuilt.
 */

import { createContext, use, type ReactNode } from 'react'

import { useAsync } from '../hooks/useAsync'
import { getMeta } from '../lib/api'
import type { MetaResponse } from '../lib/types'

interface MetaState {
  meta: MetaResponse | null
  loading: boolean
  error: unknown
  reload: () => void
}

const MetaContext = createContext<MetaState | null>(null)

export function MetaProvider({ children }: { children: ReactNode }) {
  const { data, loading, error, reload } = useAsync(
    (signal) => getMeta(signal),
    [],
  )

  return (
    <MetaContext value={{ meta: data, loading, error, reload }}>
      {children}
    </MetaContext>
  )
}

export function useMeta(): MetaState {
  const value = use(MetaContext)
  if (!value) throw new Error('useMeta must be used inside <MetaProvider>')
  return value
}

/**
 * The numeric bounds the API enforces, with the values from
 * `src/api/vocabulary.py` as the fallback when meta has not loaded yet.
 *
 * The fallback exists so a form can render and validate before the network
 * settles; it is not a second source of truth, and the served values win the
 * instant they arrive.
 */
export const FALLBACK_LIMITS = {
  age: { min: 13, max: 100 },
  height_cm: { min: 130, max: 230 },
  weight_kg: { min: 30, max: 250 },
  shoe_size: { min: 3, max: 20 },
  workouts_per_week: { min: 0, max: 21 },
  budget_min: { min: 0, max: 100000 },
  budget_max: { min: 0, max: 100000 },
} as const

export function limitFor(
  meta: MetaResponse | null,
  field: keyof typeof FALLBACK_LIMITS,
): { min: number; max: number } {
  return meta?.limits?.[field] ?? FALLBACK_LIMITS[field]
}
