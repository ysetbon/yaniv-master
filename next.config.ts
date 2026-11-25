import type { NextConfig } from "next";

// The GitHub Pages workflow (.github/workflows/pages.yml) sets PAGES_BASE_PATH to the site's sub-path
// ("/yaniv-master"): the build is then a static export served under that path. Unset (npm run dev / build /
// start on your own machine), nothing changes.
const pagesBase = process.env.PAGES_BASE_PATH;

const nextConfig: NextConfig = {
  allowedDevOrigins: ["127.0.0.1"],
  ...(pagesBase !== undefined ? { output: "export", basePath: pagesBase, images: { unoptimized: true } } : {}),
  // basePath is not added to fetch() URLs, so the page prefixes the model files with this itself.
  env: { NEXT_PUBLIC_BASE_PATH: pagesBase ?? "" },
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
