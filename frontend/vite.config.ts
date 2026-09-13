import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Port 5173 is not incidental: the FastAPI backend allows CORS from
// http://localhost:5173 and http://127.0.0.1:5173 only (src/api/main.py,
// DEFAULT_CORS_ORIGINS). `strictPort` makes a clash fail loudly rather than
// silently moving to 5174, where every request would be blocked by the
// browser with no obvious cause.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 5173, strictPort: true },
})
