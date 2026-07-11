// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import sitemap from '@astrojs/sitemap';

const SITE = 'https://open-infer.org';
// TODO(seo): replace with a dedicated 1200x630 social card at /og-card.png.
// favicon.png is square and renders small in link previews — fine as an interim.
const OG_IMAGE = `${SITE}/favicon.png`;

// https://astro.build/config
export default defineConfig({
	site: SITE,
	trailingSlash: 'always',
	integrations: [
		sitemap(),
		starlight({
			title: 'openinfer',
			logo: { src: './src/assets/logo.png', alt: 'openinfer' },
			favicon: '/favicon.png',
			customCss: ['./src/styles/custom.css'],
			expressiveCode: {
				styleOverrides: {
					borderRadius: '0.375rem',
					borderWidth: '1px',
					codeFontSize: '0.8125rem',
					codeLineHeight: '1.65',
					frames: {
						frameBoxShadowCssValue: 'none',
					},
				},
				plugins: [
					{
						name: 'openinfer-plain-code',
						hooks: {
							preprocessCode: ({ codeBlock }) => {
								// uv-style: plain code blocks unless frame= is set explicitly
								if (codeBlock.metaOptions.getString('frame') === undefined) {
									codeBlock.props.frame = 'none';
								}
							},
						},
					},
				],
			},
			// Starlight already emits <title>, description, og:title/description/url,
			// canonical, and twitter:card=summary_large_image. These fill the gaps:
			// a social preview image and SoftwareApplication structured data.
			head: [
				{
					tag: 'link',
					attrs: { rel: 'preconnect', href: 'https://fonts.googleapis.com' },
				},
				{
					tag: 'link',
					attrs: {
						rel: 'preconnect',
						href: 'https://fonts.gstatic.com',
						crossorigin: true,
					},
				},
				{
					tag: 'link',
					attrs: {
						rel: 'stylesheet',
						href: 'https://fonts.googleapis.com/css2?family=Roboto:ital,wght@0,400;0,500;0,700;1,400&family=Roboto+Mono:ital,wght@0,400;0,500;1,400&display=swap',
					},
				},
				{ tag: 'meta', attrs: { property: 'og:image', content: OG_IMAGE } },
				{ tag: 'meta', attrs: { name: 'twitter:image', content: OG_IMAGE } },
				{
					tag: 'script',
					attrs: { type: 'application/ld+json' },
					content: JSON.stringify({
						'@context': 'https://schema.org',
						'@type': 'SoftwareApplication',
						name: 'openinfer',
						description:
							'Pure Rust + CUDA LLM inference engine — no PyTorch, OpenAI-compatible, serves Qwen3 to Kimi-K2.',
						url: SITE,
						applicationCategory: 'DeveloperApplication',
						operatingSystem: 'Linux, Windows',
						programmingLanguage: ['Rust', 'CUDA'],
						codeRepository: 'https://github.com/openinfer-project/openinfer',
						license: 'https://www.apache.org/licenses/LICENSE-2.0',
						offers: { '@type': 'Offer', price: '0', priceCurrency: 'USD' },
					}),
				},
			],
			social: [
				{
					icon: 'github',
					label: 'GitHub',
					href: 'https://github.com/openinfer-project/openinfer',
				},
			],
			sidebar: [
				{ label: 'Getting Started', link: '/getting-started/' },
				{
					label: 'Blogs',
					items: [
						{ label: 'All Posts', link: '/blog/' },
						{
							label: 'See Qwen3 Decode as a CUDA Graph',
							link: '/blog/cuda-graph-export/',
						},
						{
							label: 'OpenInfer 0.1.0: Production-Grade Rust Inference',
							link: '/blog/openinfer-010/',
						},
						{
							label: 'Co-locating Prefill and Decode',
							link: '/blog/green-ctx/',
						},
					],
				},
				{
					label: 'Models',
					items: [
						{ label: 'Qwen3-4B / 8B / 32B', link: '/models/qwen3-4b/' },
						{ label: 'Qwen3.5-4B / 9B / 27B', link: '/models/qwen35-4b/' },
					],
				},
			],
		}),
	],
});
