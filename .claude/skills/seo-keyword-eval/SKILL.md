---
name: seo-keyword-eval
description: Evaluate SEO keyword demand and competition with measured data before choosing a blog post title or topic for open-infer.org. Use when drafting a new post, comparing title candidates, deciding if a trending topic is worth covering, or reviewing Search Console results.
---

# SEO Keyword Evaluation

Titles are keyword claims. Before choosing one, measure both sides of the
market: **demand** (are people searching this?) and **supply** (who already
ranks for it?). Never rank title candidates by gut feeling — every step below
produces a number you can compare.

## Why this works for this site

open-infer.org wins on *content gaps*: deep technical topics where demand
exists but third-party content is near zero. Validated examples:

- **green-ctx post**: "green context" had NVIDIA docs and nothing else.
  One post → ~40 impressions/2wk from a cold site, the only non-brand traffic.
- **DSpark (July 2026)**: 9 days after release, Google Trends showed `dspark`
  at parity with the years-old category term `speculative decoding`, while
  autocomplete for "dspark" still returned unrelated brands — demand had
  outrun supply. That mismatch is the signal to publish.

Chasing established head terms ("llm inference optimization") loses to
framework docs and review sites. Skip those.

## Step 1 — Measure demand

Three free, scriptable signals. Run all three; they measure different things.

**Google Trends** — relative heat and trajectory. Always include one
established category term as the yardstick:

```bash
uv run --with pytrends --no-project python - <<'PY'
from pytrends.request import TrendReq
pt = TrendReq(hl='en-US', tz=0)
pt.build_payload(['CANDIDATE_TERM', 'CATEGORY_BASELINE'], timeframe='today 3-m')
print(pt.interest_over_time().tail(10).to_string())
PY
```

Values are relative (0–100 within the compared set), not absolute volume.
Read the *ratio to the baseline* and the *direction* (climbing / decaying).

**Google Autocomplete** — has demand solidified into query variants?

```bash
curl -s "https://suggestqueries.google.com/complete/search?client=firefox&q=TERM"
```

- Many topical variants → mature demand, mature competition.
- No topical variants but Trends shows volume → **window open**: the term
  exists in search boxes but no content has claimed the completions yet.

**Hacker News (Algolia)** — leading indicator; HN heat precedes Google
search volume by days to weeks:

```bash
curl -s "https://hn.algolia.com/api/v1/search?query=TERM&tags=story"
```

Hundreds of points / comments on a launch story = a search wave is coming.
The comment threads are also your FAQ source — the questions asked there
are the queries people will type next week.

**Keyword Planner / Ahrefs volume data lags 2–3 months. For terms younger
than that, their "0 volume" is meaningless — ignore them.**

## Step 2 — Measure supply

Manual, five minutes:

1. Google the exact phrase you want to rank for (e.g. "dspark benchmark").
   Classify the top 10: news announcements / official docs / independent
   technical content. All-announcements = content-type gap → strong signal.
2. Search `intitle:"exact phrase"` — how many pages have claimed the phrase
   in their title? Near zero = unclaimed.

## Step 3 — Compose the title

Rules derived from the measurements:

- **Pair one emerging term with one established category phrase.** The
  emerging term rides the wave; the category phrase is the long-term floor
  if the wave dies. Both must appear as complete phrases — "speculative
  decoder" does not match "speculative decoding".
- **One H2 per long-tail variant** you found in autocomplete / HN comments
  ("X vs Y", "what is X", "run X"). Each H2 is a future search entry point.
- Site title style: `Descriptive Statement: Subtitle` (see existing posts).
- Frontmatter `description` is the meta description — write it for the
  searcher, front-load the keywords.

## Step 4 — Timing

Emerging-term windows are measured in **weeks**. The window closes when the
official framework docs (vLLM / SGLang / vendor) claim the term. If Trends
shows the spike already decaying, publishing this week still captures the
full tail; publishing next month captures nothing.

After publishing, post to r/LocalLLaMA and HN — backlinks from those
threads are what move a new domain's ranking.

## Step 5 — Verify with Search Console (ground truth)

~28 days after publishing, check GSC query data:

- Impressions confirm demand estimates; position >20 with impressions means
  Google matched the page but the content is too thin — strengthen it.
- Queries you rank for but didn't target = free topic ideas for the next post.

Feed real numbers back into this file when they confirm or refute a call —
the two examples at the top should not stay the only ones.
