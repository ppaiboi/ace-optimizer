import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Static SPA — Vercel auto-detects Vite (framework preset "Vite"),
// build => `dist`, no extra config needed.
export default defineConfig({
  plugins: [react()],
});
