# AGENTS.md

Guide for AI agents and contributors working on the openinfer documentation site.

## Project overview

Static documentation site for [openinfer](https://github.com/openinfer-project/openinfer), built with [Astro](https://astro.build/) and [Starlight](https://starlight.astro.build/), deployed to Cloudflare Workers as static assets.

- **Production URL:** https://open-infer.org
- **Node:** 22+
- **Package manager:** npm

## Quick start

```bash
npm install
npm run dev        # http://localhost:4321
```

If routes 404 unexpectedly after config or content changes, clear the Astro cache and restart:

```bash
rm -rf .astro && npm run dev
```

## Repository layout

```
src/
  content/docs/       # All doc pages (Markdown / MDX)
  styles/custom.css   # Site-wide theme overrides (light + dark)
  assets/             # Logo and other bundled assets
  content.config.ts   # Starlight content collection
public/               # Static files served as-is (images, favicon)
tests/
  routes.mjs          # Canonical list of public routes (shared by tests)
  static.test.mjs     # Post-build static output checks
  pages.spec.mjs      # Playwright E2E smoke tests
astro.config.mjs      # Astro + Starlight config (sidebar, SEO, code blocks)
wrangler.jsonc        # Cloudflare Workers static asset deploy
```

## Content

All pages live under `src/content/docs/`:

| Path | Purpose |
| --- | --- |
| `index.mdx` | Homepage |
| `getting-started.md` | Build and first request |
| `blog/*.md` | Blog posts |
| `models/*.md` | Per-model benchmark and usage pages |

Benchmark images go in `public/` (e.g. `public/blog/openinfer-010/qps.png`, `public/models/qwen3-4b/perf.png`) and are referenced with absolute paths in Markdown:

```markdown
![Serving sweep](/blog/openinfer-010/qps.png)
```

### Adding a new page

1. Create the file under `src/content/docs/`.
2. Add a sidebar entry in `astro.config.mjs` — use `link:` (not `slug:`) with a trailing slash, e.g. `link: '/models/qwen3-8b/'`.
3. Update `tests/routes.mjs`:
   - Add the URL to `SITE_ROUTES`.
   - Add the corresponding `dist/` HTML path to `DIST_PAGES`.
4. Run the full test suite (see below).

The site uses `trailingSlash: 'always'`. Internal links and sidebar entries must include the trailing slash.

## Styling conventions

- **Doc-first, minimal** — inspired by [uv docs](https://docs.astral.sh/uv): plain typography, flat borders, no marketing chrome.
- **Fonts:** Roboto + Roboto Mono (loaded in `astro.config.mjs` `head`).
- **Code blocks:** plain by default (no terminal frame). The `openinfer-plain-code` Expressive Code plugin sets `frame: 'none'` unless `frame=` is explicitly set in a fence meta string.
- **Theme tweaks** belong in `src/styles/custom.css`. Avoid one-off inline styles in content files.
- **Logo:** configured as a string path in `astro.config.mjs` (`logo: { src: './src/assets/logo.png', ... }`). Do not pass an imported `ImageMetadata` object.

## Build and preview

```bash
npm run build      # output → dist/
npm run preview    # serve dist/ at http://localhost:4321
```

## Testing

Tests are split into static (filesystem) and E2E (browser) layers. The default `test` script runs all of them in order.

```bash
npm test              # build + static + e2e
npm run test:static   # Node built-in test runner only (requires prior build)
npm run test:e2e      # Playwright only (starts preview server automatically)
```

### Static tests (`tests/static.test.mjs`)

Run after `npm run build`. They verify:

- Every expected HTML file exists under `dist/`.
- `sitemap-0.xml` includes all routes from `SITE_ROUTES`.
- `404.html` is the Starlight-styled page, not a bare error.

### E2E tests (`tests/pages.spec.mjs`)

Playwright starts `astro preview` on port 4321 (see `playwright.config.mjs`). They verify:

- Each route in `SITE_ROUTES` returns HTTP 200 and renders a real page (not the 404 content).
- Unknown paths return 404 with the expected message.
- Sidebar navigation works (Getting Started link).

First-time Playwright setup:

```bash
npx playwright install chromium
```

When adding or removing pages, always update `tests/routes.mjs` — both test files import from it.

## Deploy

Production deploys via Cloudflare Workers static assets:

```bash
npm run build
npx wrangler deploy
```

`wrangler.jsonc` sets `html_handling: "auto-trailing-slash"` so paths without a trailing slash still resolve in production.

For CI/CD, connect the repo in the Cloudflare dashboard with build command `npm run build`.

## Common pitfalls

| Issue | Cause | Fix |
| --- | --- | --- |
| All routes 404 in dev | Stale `.astro` cache or old dev server | `rm -rf .astro`, restart `npm run dev` |
| Sidebar build error: slug does not exist | Using `slug:` in sidebar | Use `link: '/path/'` instead |
| `getImage` / logo path error | Hero `image.file` or imported logo object | Use string path in `logo.src`; avoid hero image imports in frontmatter |
| New page passes locally but fails CI | Route not in `tests/routes.mjs` | Add URL + dist path, rerun `npm test` |
| E2E port conflict | Another process on 4321 | Stop other dev/preview servers before `npm run test:e2e` |

## What not to change without reason

- Do not switch back to `template: splash` or heavy marketing layouts on the homepage — the site is doc-first.
- Do not add interactive chart libraries for benchmark tables; use summary images plus Markdown tables.
- Do not commit secrets (`.env`, credentials).
- Do not skip git hooks or force-push to `main` unless explicitly requested.

## Related docs

- Human-oriented overview: [README.md](./README.md)
- Choosing blog titles / topics: [.claude/skills/seo-keyword-eval/SKILL.md](./.claude/skills/seo-keyword-eval/SKILL.md) — measure keyword demand and competition before writing
- Starlight docs: https://starlight.astro.build/
- Astro content collections: https://docs.astro.build/en/guides/content-collections/
