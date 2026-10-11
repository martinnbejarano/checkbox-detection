import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// In development the API runs on its own port; in production FastAPI serves this build on one origin.
const API = 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/detect': API, '/health': API } },
})
