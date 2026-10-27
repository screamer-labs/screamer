# Docs redesign: one Guide, one Reference

Date: 2026-10-27. Status: draft for review.

## Problem

A reader who wants to do one task (for example, show that a series mean-reverts) cannot find a short answer. The docs offer eight kinds of entry: recipes, notebooks, user guides, concept pages, "Choosing an operator", "What you can build", the gallery, and function groups. The home page tells seven stories (intro, install, badges, tabbed example with a three-panel chart, example tiles, speed chart, causality text) and then adds 10 function-group cards and two duplicate cards. The top bar has six overlapping sections. The content mostly exists; the structure hides it.

## Goal

A calm, Apple-like surface: one clear guide, read in order, made of short task pages, with links into depth for those who need it. Success: a reader can go from the home page to a runnable answer for a common task (denoise, detect trend, detect mean reversion, backtest a signal on OHLCV bars, test significance, go live) in two clicks, and the page fits on about one screen.

## Structure

The site has two kinds of content and nothing else at the top level.

**Guide**: the only place to learn. A book of chapters, each made of short task pages.

**Reference**: the place to look things up. Function pages by group (the gallery is a view inside it), "How screamer works" (the long concept pages), and release notes.

Top bar: **Guide · Reference · Release notes**, plus search. The sidebar shows the current section only, one level deep, with plain chapter titles.

The words "recipe", "examples", "capabilities" and "user guide" disappear from navigation.

## Home page

Four parts, in order, nothing else:

1. Hero: one sentence on what screamer is, `pip install screamer`, a "Start the Guide" link.
2. One example: the existing trend-sign example (Python, with a JavaScript tab) and one single-panel chart.
3. The Guide's table of contents: parts as small headings, chapters as a list with one line each. This replaces all tiles and cards.
4. Three facts, one line each with a link: fast (to Performance), causal (to Conventions), same code for batch and live (to Streaming).

Removed from the home page: badges (stay in the README), the speed chart (stays on the Performance page), the causality essay, the "What you can build" tiles, the "Functions by group" cards, the duplicate Gallery / What-you-can-build cards.

## Guide table of contents

Each chapter has a landing page: two or three sentences, the list of its task pages as "How do I ...?" links, and at the bottom a compact table "All functions for this chapter" (taken from the matching section of today's "Choosing an operator").

Part 0, Basics
- 0. First steps: install, one operator on an array, on a DataFrame column, on a live stream; NaN and warmup in one page.
- 1. Get your data in: pandas OHLCV, migrating from pandas rolling and TA-Lib, bars from ticks.

Part I, Understand a series
- 2. Smooth and denoise: moving averages, low-lag smoothers, Kalman, Butterworth, despiking.
- 3. Measure volatility: close-to-close, OHLC estimators, ATR, intraday seasonality.
- 4. Detect a trend: slope and regression, DMI/ADX, trend mode, crossovers.
- 5. Detect mean reversion: variance ratio, Hurst, OU half-life, z-score bands; "is this series mean-reverting?" and "trade the reversion".
- 6. Find cycles.
- 7. Detect regimes and change points: HMM, Bayesian changepoint, CUSUM events.
- 8. Read order flow and liquidity.

Part II, Build a strategy
- 9. From signal to position: sign, bands and hysteresis, holding, volatility sizing.
- 10. Backtest a signal on OHLCV bars: costs, report, equity curve, event-driven backtests.
- 11. Strategy templates: trend following, breakout, RSI/Bollinger reversion, pairs, cross-sectional momentum, carry, volatility targeting.
- 12. Stops and exits.
- 13. Portfolios of many assets.

Part III, Is it real?
- 14. Significance, walk-forward, parameter grids.
- 15. Labels and ML features.

Part IV, Go live
- 16. Streaming, pipelines, a live loop, JavaScript in the browser.

## Task page shape

Every task page has four parts and fits on about one screen:

1. Title as the task ("Detect mean reversion with the variance ratio").
2. One-sentence answer naming the functions.
3. A runnable example of 10 to 20 lines on the data already committed with the docs, executed at build time, followed by one plot.
4. "Go deeper": a short list of links (the notebook, function pages, the concept page).

Writing follows the existing docs style guide (lead with the point, plain language, no em-dashes).

## What happens to existing content

| Today | After |
|:------|:------|
| 17 recipes | Each becomes a task page in its chapter, trimmed to the task-page shape. |
| 25 notebooks | Kept as they are; linked only from "Go deeper" on the task pages they expand. No notebook list in navigation. |
| Choosing an operator | Split into the "All functions" tables on chapter landing pages; the page itself is removed. |
| What you can build (capabilities.md, examples.yml, examples_tiles.inc, build_examples.py) | Removed; the Guide table of contents does its job. |
| Concept pages (usage, polymorphic API, NaN and warmup, streaming, multistream, pipelines, conventions, performance, microstructure, labels, backtesting best practices, portfolio backtesting, agents, javascript) | Short versions become Guide pages where a chapter needs them; the full pages move to Reference, "How screamer works". |
| Gallery, function groups | Reference. |
| 326 function pages | Content kept. Light reorder: one-line summary, signature, minimal example first; long parameter notes and references lower. |

Mapping of existing material to chapters (go-deeper links or source for task pages):

- 0: notebook 01, usage, nan_and_warmup, notebook 05.
- 1: recipes pandas-ohlcv-indicators, migrating, ticks-to-backtest; notebooks 09, 10.
- 2: notebooks 04, 17. 3: notebook 02, recipe volatility-sizing (shared with 9). 4: notebook 03.
- 5: notebook 18. 6: notebook 19. 7: notebook 22 where it fits; otherwise new pages only.
- 8: notebooks 11, 12, 13; microstructure.
- 9: recipe volatility-sizing. 10: recipe first-strategy, notebooks 14, 15, backtesting_best_practices.
- 11: recipes pairs-trade, cross-sectional-momentum, funding-carry. 12: recipe stops-and-exits.
- 13: recipe multi-asset, portfolio_backtesting.
- 14: recipes significance-test, walk-forward, parameter-grid; notebooks 20, 21.
- 15: recipe ml-features, labels; notebooks 16, 23, 24, 25.
- 16: recipes live-loop, strategy-to-live; notebooks 06, 07, 08; streaming, multistream, pipelines, javascript.

## Redirects and tooling

- Every removed or moved URL gets an orphan stub page with a meta refresh to its new location (the pattern already used by recipes/index.md). No new Sphinx extension.
- A `docs/guide.yml` registry lists parts, chapters and task pages. A build script generates the chapter landing pages' task lists and the home table of contents from it, replacing build_examples.py.
- tests/test_examples_page.py is replaced by a test that fails when a task page or notebook is not reachable from the Guide, when a generated file is stale, or when a task page's example block is missing.
- llms.txt and the `screamer.docs` CLI keep working and list Guide pages.

## Phases

Phase 1: new navigation and home page; guide.yml and its build script and test; Part 0, Part I and Part II chapters and task pages (mean reversion included); redirects for moved pages.

Phase 2: Parts III and IV; remove Choosing an operator and What you can build; move concept pages to Reference; function-page reorder.

Each phase ends with a clean docs build (warnings as errors where they are today) and a check of the built site in a browser.

## Out of scope

Rewriting notebooks; new operators; changing the docs theme.
