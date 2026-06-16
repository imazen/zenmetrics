import path from "path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

// The dashboard binary serves the built assets from `dist/` (ZEN_WEB_DIR). During `npm run dev`,
// /api/* is proxied to the local Rust server on :3000 so the SPA talks to a real backend.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 3100,
    proxy: {
      "/api": "http://localhost:3000",
    },
  },
})
