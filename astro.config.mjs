// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// https://astro.build/config
export default defineConfig({
	site: 'https://open-infer.org',
	integrations: [
		starlight({
			title: 'openinfer',
			logo: { src: './src/assets/logo.png' },
			favicon: '/favicon.png',
			customCss: ['./src/styles/custom.css'],
			social: [
				{
					icon: 'github',
					label: 'GitHub',
					href: 'https://github.com/openinfer-project/openinfer',
				},
			],
			sidebar: [
				{ label: 'Getting Started', slug: 'getting-started' },
				{
					label: 'Blogs',
					items: [
						{ label: 'All Posts', slug: 'blog' },
						{
							label: 'OpenInfer 0.1.0: Production-Grade Rust Inference',
							slug: 'blog/openinfer-010',
						},
						{
							label: 'Green Contexts',
							slug: 'blog/green-ctx',
						},
					],
				},
				{
					label: 'Models',
					items: [{ label: 'Qwen3-4B', slug: 'models/qwen3-4b' }],
				},
			],
		}),
	],
});
