/**
 * Routes and the two providers everything below them needs.
 *
 * `MetaProvider` sits above the router so `/api/meta` is fetched once for the
 * session rather than per page; `ProfileProvider` sits there too so the
 * multi-step draft survives a trip to the catalogue and back.
 */

import { Link, Route, Routes } from 'react-router-dom'

import { Layout } from './components/layout/Layout'
import { Eyebrow } from './components/ui/primitives'
import { MetaProvider } from './context/MetaContext'
import { ProfileProvider } from './context/ProfileContext'
import { Catalog } from './pages/Catalog'
import { Landing } from './pages/Landing'
import { ProfileBuilder } from './pages/ProfileBuilder'
import { Results } from './pages/Results'

function NotFound() {
  return (
    <div className="mx-auto max-w-2xl px-4 py-20 text-center sm:px-6">
      <Eyebrow>404</Eyebrow>
      <h1 className="mt-3 text-5xl text-ash">Page not found</h1>
      <p className="mt-4 text-sm text-ash-dim">
        That route does not exist in this app.
      </p>
      <Link
        to="/"
        className="mt-8 inline-flex items-center border-2 border-crimson-700 bg-crimson-700 px-8 py-4 font-display text-base font-bold uppercase tracking-[0.12em] text-white transition-colors hover:bg-crimson-600"
      >
        Back to the start
      </Link>
    </div>
  )
}

export default function App() {
  return (
    <MetaProvider>
      <ProfileProvider>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<Landing />} />
            <Route path="profile" element={<ProfileBuilder />} />
            <Route path="results" element={<Results />} />
            <Route path="catalog" element={<Catalog />} />
            <Route path="*" element={<NotFound />} />
          </Route>
        </Routes>
      </ProfileProvider>
    </MetaProvider>
  )
}
