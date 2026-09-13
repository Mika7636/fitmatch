/**
 * The frame: skip link, header, nav, footer.
 *
 * The backend status dot in the header is deliberate. A frontend talking to a
 * local FastAPI process fails in exactly one way most of the time -- the
 * process is not running -- and a page that says so in the corner beats four
 * separate error panels saying it one screen at a time.
 */

import { NavLink, Outlet } from 'react-router-dom'

import { useAsync } from '../../hooks/useAsync'
import { API_BASE_URL, getHealth } from '../../lib/api'
import { formatCount } from '../../lib/format'
import { cx } from '../ui/primitives'

const NAV = [
  { to: '/', label: 'Home', end: true },
  { to: '/profile', label: 'Build profile', end: false },
  { to: '/results', label: 'Results', end: false },
  { to: '/catalog', label: 'Catalogue', end: false },
]

function BackendStatus() {
  const { data, loading, error } = useAsync((signal) => getHealth(signal), [])

  const tone = loading
    ? 'bg-ash-muted'
    : error || !data?.ready
      ? 'bg-crimson-500'
      : 'bg-emerald-500'

  const label = loading
    ? 'Checking the API…'
    : error
      ? `API unreachable at ${API_BASE_URL} — start it with run_api.bat`
      : data?.ready
        ? `API ready — ${formatCount(data.catalogue_size)} products, ${formatCount(data.n_users)} profiles, TF-IDF cache ${data.cache.hit_on_startup ? 'reused' : 'rebuilt'} in ${Math.round(data.cold_start_ms ?? 0)} ms`
        : (data?.detail ?? 'API not ready')

  return (
    <span
      className="flex items-center gap-2 text-[11px] text-ash-muted"
      title={label}
    >
      <span
        className={cx('h-2 w-2 shrink-0 rounded-full', tone)}
        aria-hidden="true"
      />
      <span className="sr-only">{label}</span>
      <span aria-hidden="true" className="hidden sm:inline">
        {loading ? 'API…' : error ? 'API offline' : 'API ready'}
      </span>
    </span>
  )
}

export function Layout() {
  return (
    <div className="flex min-h-screen flex-col bg-ink-900">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:bg-crimson-700 focus:px-4 focus:py-2 focus:font-display focus:text-sm focus:uppercase focus:text-white"
      >
        Skip to content
      </a>

      <header className="sticky top-0 z-30 border-b border-ink-700 bg-ink-950/95 backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-6 gap-y-3 px-4 py-3 sm:px-6">
          <NavLink
            to="/"
            className="group flex items-center gap-2.5"
            aria-label="FitMatch home"
          >
            <span
              aria-hidden="true"
              className="block h-7 w-2 skew-x-[-14deg] bg-crimson-600 transition-colors group-hover:bg-crimson-500"
            />
            <span className="font-display text-xl font-black uppercase tracking-tight text-ash">
              Fit<span className="text-crimson-500">Match</span>
            </span>
          </NavLink>

          <nav aria-label="Main" className="order-3 w-full sm:order-2 sm:w-auto">
            <ul className="flex flex-wrap items-center gap-x-1 gap-y-1">
              {NAV.map((item) => (
                <li key={item.to}>
                  <NavLink
                    to={item.to}
                    end={item.end}
                    className={({ isActive }) =>
                      cx(
                        'block px-3 py-1.5 font-display text-xs font-bold uppercase tracking-[0.12em] transition-colors',
                        isActive
                          ? 'bg-crimson-700 text-white'
                          : 'text-ash-dim hover:bg-ink-700 hover:text-ash',
                      )
                    }
                  >
                    {item.label}
                  </NavLink>
                </li>
              ))}
            </ul>
          </nav>

          <div className="order-2 ml-auto sm:order-3">
            <BackendStatus />
          </div>
        </div>
      </header>

      <main id="main" className="flex-1">
        <Outlet />
      </main>

      <footer className="border-t border-ink-700 bg-ink-950">
        <div className="mx-auto flex max-w-7xl flex-col gap-2 px-4 py-6 text-xs text-ash-muted sm:flex-row sm:items-center sm:justify-between sm:px-6">
          <p>
            FitMatch — DS&amp;RS 541 / CSX 4207 term project. Knowledge-based
            constraint filtering (Ch7) + TF-IDF content ranking (Ch3).
          </p>
          <p className="font-mono text-[11px]">{API_BASE_URL}</p>
        </div>
      </footer>
    </div>
  )
}
