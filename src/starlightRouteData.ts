import { defineRouteMiddleware, type StarlightRouteData } from '@astrojs/starlight/route-data';

const SITE = 'https://open-infer.org';
const SOCIAL_IMAGE = `${SITE}/og-card.png`;

type Head = StarlightRouteData['head'];
type MetaKey = 'name' | 'property';

function setMeta(head: Head, key: MetaKey, value: string, content: string) {
	const existing = head.find((entry) => entry.tag === 'meta' && entry.attrs?.[key] === value);
	if (existing) {
		existing.attrs = { ...existing.attrs, content };
		return;
	}

	head.push({ tag: 'meta', attrs: { [key]: value, content } });
}

function softwareApplicationJsonLd() {
	return {
		'@context': 'https://schema.org',
		'@type': 'SoftwareApplication',
		name: 'pegainfer',
		description:
			'Pure Rust + CUDA LLM inference engine — no PyTorch, OpenAI-compatible, serves Qwen3 to Kimi-K2.',
		url: SITE,
		applicationCategory: 'DeveloperApplication',
		operatingSystem: 'Linux, Windows',
		programmingLanguage: ['Rust', 'CUDA'],
		codeRepository: 'https://github.com/pegainfer-project/pegainfer',
		license: 'https://www.apache.org/licenses/LICENSE-2.0',
		image: SOCIAL_IMAGE,
		offers: { '@type': 'Offer', price: '0', priceCurrency: 'USD' },
	};
}

export const onRequest = defineRouteMiddleware((context) => {
	const route = context.locals.starlightRoute;
	const { data } = route.entry;
	const canonical = new URL(context.url.pathname, SITE).href;
	const isHomepage = context.url.pathname === '/';
	const isBlogIndex = context.url.pathname === '/blog/';
	const isBlogPost = context.url.pathname.startsWith('/blog/') && !isBlogIndex;

	setMeta(route.head, 'property', 'og:type', isBlogPost ? 'article' : 'website');
	setMeta(route.head, 'property', 'og:image', SOCIAL_IMAGE);
	setMeta(route.head, 'property', 'og:image:alt', 'pegainfer — Pure Rust + CUDA LLM Inference Engine');
	setMeta(route.head, 'property', 'og:image:type', 'image/png');
	setMeta(route.head, 'property', 'og:image:width', '1200');
	setMeta(route.head, 'property', 'og:image:height', '630');
	setMeta(route.head, 'name', 'twitter:image', SOCIAL_IMAGE);
	setMeta(
		route.head,
		'name',
		'twitter:image:alt',
		'pegainfer — Pure Rust + CUDA LLM Inference Engine',
	);

	if (isBlogPost && data.publishedDate && data.authors?.length) {
		const datePublished = data.publishedDate.toISOString();
		setMeta(route.head, 'property', 'article:published_time', datePublished);
		for (const author of data.authors) {
			route.head.push({
				tag: 'meta',
				attrs: { property: 'article:author', content: author.url },
			});
		}

		route.head.push({
			tag: 'script',
			attrs: { type: 'application/ld+json' },
			content: JSON.stringify({
				'@context': 'https://schema.org',
				'@type': 'BlogPosting',
				headline: data.title,
				description: data.description,
				datePublished,
				author: data.authors.map((author) => ({
					'@type': 'Person',
					name: author.name,
					url: author.url,
				})),
				image: data.seoImage ? new URL(data.seoImage, SITE).href : SOCIAL_IMAGE,
				mainEntityOfPage: { '@type': 'WebPage', '@id': canonical },
				isPartOf: { '@type': 'Blog', '@id': `${SITE}/blog/`, name: 'pegainfer Blog' },
				publisher: {
					'@type': 'Organization',
					name: 'pegainfer',
					url: SITE,
					logo: { '@type': 'ImageObject', url: `${SITE}/favicon.png` },
				},
				inLanguage: 'en',
			}),
		});
	} else if (isHomepage) {
		route.head.push({
			tag: 'script',
			attrs: { type: 'application/ld+json' },
			content: JSON.stringify(softwareApplicationJsonLd()),
		});
	} else if (isBlogIndex) {
		route.head.push({
			tag: 'script',
			attrs: { type: 'application/ld+json' },
			content: JSON.stringify({
				'@context': 'https://schema.org',
				'@type': 'Blog',
				'@id': `${SITE}/blog/`,
				name: 'pegainfer Blog',
				description: data.description,
				url: canonical,
				publisher: { '@type': 'Organization', name: 'pegainfer', url: SITE },
				inLanguage: 'en',
			}),
		});
	}
});
