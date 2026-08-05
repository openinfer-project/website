# pegainfer website

Documentation site for [pegainfer](https://github.com/pegainfer-project/pegainfer),
built with [Astro Starlight](https://starlight.astro.build/) and deployed on
Cloudflare Workers (static assets).

## Develop

```bash
npm install
npm run dev      # http://localhost:4321
```

Requires Node 22+.

## Content

Pages live in `src/content/docs/`:

- `index.mdx` — landing page
- `getting-started.md` — build & first request
- `models/<model>.md` — one page per model line; add the new page to the
  `sidebar` in `astro.config.mjs`

## Deploy

The site deploys to Cloudflare Workers as static assets (`wrangler.jsonc`).

```bash
npm run build
npx wrangler deploy
```

For automatic deploys, connect this repo in the Cloudflare dashboard
(Workers & Pages → Create → connect to Git) with build command
`npm run build` — pushes to `main` deploy to production and PRs get
preview URLs.
