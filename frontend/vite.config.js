import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  base: '/ML-Project-CV-Analysis/',
  plugins: [react()],
  cacheDir: '.vite_cache',
  server: {
    port: 5173,
    host: true,
    strictPort: true,
  },
  test: {
    // Only run unit tests in src/ - exclude Playwright e2e tests
    include: ['src/**/*.{test,spec}.{js,jsx,ts,tsx}'],
    exclude: ['e2e/**', 'node_modules/**'],
    environment: 'jsdom',
  },
})
