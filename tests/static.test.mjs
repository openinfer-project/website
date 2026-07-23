import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, it } from 'node:test';
import { DIST_PAGES, REDIRECTS, SITE_ROUTES } from './routes.mjs';

const distDir = join(import.meta.dirname, '..', 'dist');

describe('static build output', () => {
	for (const page of DIST_PAGES) {
		it(`dist/${page} exists`, () => {
			assert.ok(existsSync(join(distDir, page)), `missing dist/${page}`);
		});
	}

	it('sitemap lists every expected route', () => {
		const sitemap = readFileSync(join(distDir, 'sitemap-0.xml'), 'utf8');
		for (const route of SITE_ROUTES) {
			const loc = `https://open-infer.org${route === '/' ? '/' : route}`;
			assert.match(sitemap, new RegExp(`<loc>${loc}</loc>`), `sitemap missing ${loc}`);
		}
	});

	it('preserves legacy URLs as permanent redirects', () => {
		const redirects = readFileSync(join(distDir, '_redirects'), 'utf8')
			.split('\n')
			.map((line) => line.trim())
			.filter((line) => line && !line.startsWith('#'));

		assert.deepEqual(
			redirects,
			REDIRECTS.map(([source, destination, status]) => `${source} ${destination} ${status}`),
		);
	});

	it('404 page is Starlight styled, not a bare error', () => {
		const html = readFileSync(join(distDir, '404.html'), 'utf8');
		assert.match(html, /Page not found/i);
		assert.match(html, /starlight|openinfer/i);
	});
});
