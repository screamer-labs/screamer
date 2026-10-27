# Docs Guide, Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the Guide (Parts III and IV as short task pages), retire the last old entry points, tidy Reference, move each function page's example up, and close the Phase 1 leftovers.

**Architecture:** Same as Phase 1: `docs/guide.yml` is the registry, `devtools/build_guide.py` generates landings and the home table of contents, `tests/test_guide.py` gates shape, reachability and redirects. Task pages are executed MyST notebooks under `docs/guide/<slug>/`; moved pages leave orphan meta-refresh stubs.

**Tech Stack:** Sphinx with pydata-sphinx-theme, myst-nb, sphinx-design, matplotlib/plotly, PyYAML; pytest.

**Spec:** docs/superpowers/specs/2026-10-27-docs-guide-design.md (Phase 2 section). Phase 1 plan: docs/superpowers/plans/2026-10-27-docs-guide-phase1.md (its Global Constraints still hold).

## Global Constraints

- Everything in the Phase 1 plan's Global Constraints, unchanged (naming rule, task page shape, style guide, no em-dashes, screamer functions for every calculation, causal examples, orphan stubs for moved URLs, `.venv/bin/python`, `.venv/bin/sphinx-build -b html docs docs/_build/html`).
- Conventions settled in Phase 1, binding on every new page:
  - Backtests: signal at a bar's close fills at the next bar's open through the engine (no manual lag); state fees and slippage in prose; default costs `taker_fee=1e-3, slippage=5e-4` on daily bars.
  - Zero the first 100 days of the 20/100 trend rule (`direction[:100] = 0.0`), as on strategies/trend-following.md.
  - Max drawdown in percent of the peak via `MaxDrawdown()(1 + equity)[-1]`.
  - Money as "N USD" (never a dollar sign: MyST renders it as math).
  - US spelling; "warmup"; "bp".
  - Chapter function tables list only functions the chapter's pages use (ruling R3).
  - Honest results: never tune or cherry-pick silently; a look-ahead used only for scoring sits in its own block under `# Scoring only, looks ahead`.
- Compute cells 10 to 20 lines; one plot per page.
- A page whose example cannot execute at build (JavaScript, a live socket) sets `exempt_shape: true` in its front matter and still has a `## Go deeper` section.

## Rulings carried from Phase 1

- R6 (gallery entries for moved recipes) is resolved by Task 4 here.
- No hour-of-day volatility page: no committed intraday dataset spans enough days. The volatility chapter line already omits it.

## Review Focus

1. A recipe moved in Tasks 1 to 3 that a notebook, `javascript.md` or the README still links: the stub keeps it working, but the guide's own links must point at the new page (grep `docs/guide` and `docs/*.md`).
2. `javascript.md` quotes the daily rule's p-value 0.151 and says its pnl and Sharpe match the Python page: after Task 1 the p-value is on a Python page again; the two must agree.
3. A function page whose Examples section is long plotly code: moving it above the Description must not push the summary and signature off the first screen (Task 5 checks screenshots).
4. Deleting tile thumbnails: no page or test may still reference a deleted file (grep plus build).
5. Old URL of `choosing_an_operator`: must redirect (test_moved_pages_redirect's required list gains it).

---

### Task 1: Part III, chapter 14 (not luck)

Move and trim each recipe to the task-page shape, replace its guide.yml entry, leave a stub, repoint links.

- `guide/not-luck/timing-test`: "Test whether a strategy's timing beats chance", from recipes/significance-test. Use the 20/100 daily trend rule of backtest/ohlcv.md (so the p-value matches the one `javascript.md` quotes; update `javascript.md` if the number changes and say why).
- `guide/not-luck/walk-forward`: "Choose parameters with walk-forward testing", from recipes/walk-forward.
- `guide/not-luck/parameter-grid`: "Check that a result holds across nearby parameters", from recipes/parameter-grid.
- Landing: two or three sentences and a trimmed function table. backtest/ohlcv.md's Go deeper link to chapter 14 points at the timing-test page.

Commit: `docs(guide): not-luck chapter`.

### Task 2: Part III, chapter 15 (machine learning)

- `guide/machine-learning/features`: "Build a feature matrix for a model", from recipes/ml-features.
- `guide/machine-learning/labels`: "Label each bar with its future move" (fixed-horizon and triple-barrier labels; source: docs/labels.md and notebook 24; the label is a target, so its look-ahead is by design and the page says so in one sentence).
- `guide/machine-learning/training-set`: "Pair features with labels without look-ahead" (`forecast_pairs` or whatever labels.md uses; check with `python -m screamer.docs search forecast pairs`).
- Landing: intro and trimmed function table.

Commit: `docs(guide): machine-learning chapter`.

### Task 3: Part IV, chapter 16 (live)

- `guide/live/one-value-at-a-time`: "Process values one at a time" (the same function on an array and value by value, same result; source: streaming.md, notebook 06).
- `guide/live/align-streams`: "Combine streams that tick at different times" (`CombineLatest`; source: multistream.md, notebook 07).
- `guide/live/pipeline`: "Wire functions into a pipeline" (`Pipeline`; source: pipelines.md, notebook 08).
- `guide/live/live-loop`: "Run a strategy in a live loop", from recipes/live-loop.
- `guide/live/research-to-live`: "Move a strategy from research to a live loop", from recipes/strategy-to-live.
- `guide/live/browser`: "Run screamer in the browser" (`exempt_shape: true`; a short JavaScript example from javascript.md, not executed).
- Landing: intro and trimmed function table.

Commit: `docs(guide): live chapter`.

### Task 4: Retire old entry points and tile thumbnails, tidy Reference

**Files:** `docs/choosing_an_operator.md` (becomes an orphan stub to `guide/index.html`), `docs/references_index.rst`, `docs/_ext/llms_txt.py`, `devtools/build_topic_pages.py`, `docs/by_group_cards.inc`, `devtools/build_gallery.py`, `docs/_static/gallery/` (manifest and non-function thumbnails), `devtools/gallery_snippets/`, `tests/test_guide.py`.

- [ ] `choosing_an_operator` becomes a stub; remove its card and toctree entry from references_index; remove it from llms_txt GUIDES; add it to the required-stub list in `test_moved_pages_redirect`.
- [ ] llms.txt headings: the concept list becomes `## How screamer works`, the Guide list stays `## Guide`, and the Guide list includes chapter landings before their tasks.
- [ ] Stop generating `docs/by_group_cards.inc` and delete it (nothing includes it); keep build_topic_pages.py's other outputs unchanged.
- [ ] Remove `recipe_pages`, `notebook_pages` and `snippet_pages` from build_gallery.py (their thumbnails only fed the deleted tiles), delete the `recipes__*`, `notebooks__*` and `capability__*` thumbnails and their manifest entries, and delete `devtools/gallery_snippets/` if nothing else uses it. Do not regenerate function thumbnails; edit the manifest by removing keys if a full regeneration is slow.
- [ ] Update build_gallery.py's docstrings that mention tiles or the recipe index.
- [ ] Verify: `git grep` finds no reference to a deleted file; tests/test_guide.py and any gallery or topic-page test pass; Sphinx build has no new warnings.

Commit: `docs: retire Choosing an operator, tile thumbnails and unused cards`.

### Task 5: Function pages lead with the example

**Files:** `docs/_ext/function_page.py` only (the 326 page files stay unchanged).

- [ ] At read time, move the page's `## Examples` section (up to the next `## ` heading) to directly after the rendered header (summary and signatures), before `## Description`. Keep the parameter table after Description, then NaN handling, Implementation, See also, References.
- [ ] Leave a page unchanged when it has no `## Examples` section.
- [ ] Unit test (new `tests/test_function_page_order.py`, skipped without docs sources): feed the transform a minimal page with Description, NaN handling and Examples sections and assert the output order is header, Examples, Description, NaN handling; and a page without Examples is unchanged.
- [ ] Screenshot five pages of different kinds (RollingZscore, BacktestOHLCTarget, KalmanFilter, OFI, Sign) at 1400 px. If on any of them the example's code pushes the plot below the first screen so the reader sees only code, report it with the screenshot rather than guessing a fix.

Commit: `docs: function pages show the example first`.

### Task 6: Phase 1 leftovers and notebook 12

- [ ] Notebook 12 (`docs/notebooks/12-microstructure-price-impact.ipynb`): the Deribit `volume` column is already in USD, so notional is `|volume|`, not `|volume| * price`. Fix the computation and any prose that interprets the Amihud values or the BTC vs ETH comparison; re-execute the notebook if it stores outputs (check `nb_execution_mode = "auto"`: stored outputs are not re-run, so clear or refresh them). Update the illiquidity page's Go deeper description if it warned about this.
- [ ] Task pages longer than about one desktop screen before "Go deeper": for each task page, move any paragraph that explains internals or repeats the Go deeper links below the plot or into Go deeper, so that the one-sentence answer, the example and the plot come first. Do not cut facts; do not change code.
- [ ] Re-wrap ragged source lines in docs/guide (paragraphs wrapped at about 80 columns).

Commit: `docs(guide): shorter task pages, notebook 12 units fix`.

### Task 7: Final pass

- [ ] Full Sphinx build with `-W --keep-going` to list warnings: only the known `sphinx_exec_code` one.
- [ ] `tests/test_guide.py`, `tests/test_function_page_order.py`, and the full `tests/` once.
- [ ] Screenshots at 1400 px: home, Guide index, each Part III/IV chapter landing, one task page per new chapter, two function pages.
- [ ] Grep for leftovers: links to moved recipes inside docs/guide, "recipe" or "What you can build" in navigation text, em-dashes in docs/guide.

Commit (only if something changed): `docs(guide): phase 2 final pass`.
