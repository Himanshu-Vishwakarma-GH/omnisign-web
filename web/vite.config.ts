import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// GitHub Pages base path for the omnisign-web repo.
export default defineConfig({
  base: '/omnisign-web/',
  plugins: [react()],
  build: {
    chunkSizeWarningLimit: 2000,
  },
})
