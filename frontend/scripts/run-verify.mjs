/**
 * Run scripts/verify.tsx in Node, through Vite's own SSR module loader.
 *
 * Vite transforms the TSX and resolves the `src/` imports exactly as it does
 * for the browser build, so the verification exercises the same code the app
 * ships -- and it needs no bundler dependency of its own, because the project
 * already has Vite.
 */
import { createServer } from 'vite'

const server = await createServer({
  configFile: false,
  root: process.cwd(),
  logLevel: 'error',
  server: { middlewareMode: true, hmr: false },
  optimizeDeps: { noDiscovery: true },
})

try {
  await server.ssrLoadModule('/scripts/verify.tsx')
} finally {
  // verify.tsx calls process.exit itself; this only runs if it threw first.
  await server.close()
}
