# Docs Guide, Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the docs' eight entry points with one Guide (Parts 0, I, II written; Parts III, IV as chapter pages over existing recipes) and one Reference, and rebuild the home page as hero, one example, the Guide's table of contents, and three facts.

**Architecture:** `docs/guide.yml` is the single registry of parts, chapters, task pages and go-deeper notebooks. `devtools/build_guide.py` generates the Guide index, the home table of contents and each chapter's task list and hidden toctree from it, with a `--check` mode run by a pytest gate. Task pages are executed MyST notebooks (`file_format: mystnb`, like today's recipes) under `docs/guide/<chapter-slug>/`. Moved pages leave orphan meta-refresh stubs at their old paths.

**Tech Stack:** Sphinx with pydata-sphinx-theme, myst-nb, sphinx-design, plotly/matplotlib, PyYAML; pytest.

**Spec:** docs/superpowers/specs/2026-10-27-docs-guide-design.md

## Global Constraints

- Top bar is exactly: Guide · Reference · Release notes (toctree captions in `docs/index.rst`), plus search.
- Home page has exactly four parts in this order: hero (one sentence, `pip install screamer`, "Start the Guide" link), one example (existing trend-sign example, Python plus JavaScript tab, one single-panel chart), the Guide's table of contents, three one-line facts (fast, causal, same code for batch and live). No tiles, cards or badges.
- Naming rule: part, chapter and task titles say the action and its object in plain words, never a pronoun or a slogan.
- Task page shape: title as the task; one-sentence answer naming the functions; a runnable example of 10 to 20 lines on data committed under `docs/notebooks/data/`, executed at build, then one plot; a `## Go deeper` section with a short list of links.
- Writing follows the user's docs style guide (memory `feedback_writing_style`): lead with the point, plain language, say it once. No em-dashes anywhere (ASCII hyphens only).
- Examples use screamer functions for every calculation (no hand-rolled rolling numpy/pandas), and are causal.
- Every moved or removed URL keeps working through an orphan stub with a meta refresh, using the pattern of `docs/recipes/index.md`. No new Sphinx extension.
- Notebooks are not edited.
- Python for the build and tests: `.venv/bin/python` (system python cannot import the bindings). Docs build: `.venv/bin/sphinx-build -b html docs docs/_build/html` (same command as `.github/workflows/docs.yml`).
- Function names to use (verified on 3.8.0): `RollingRSI` (not RSI), `PlusDI`, `MinusDI`, `ADX`, `AmihudIlliquidity`, `BollingerBands`, `DonchianChannels`, `RollingOU`, `RollingVarianceRatio`, `RollingHurst`, `BandSignal`, `SchmittTrigger`, `TrendMode`, `RollingHMM`, `BayesianChangepoint`, `CusumFilter`, `HullMA`, `KAMA`, `SuperSmoother`, `KalmanFilter`, `Butter`, `Hampel`, `RollingYangZhangVol`, `ATR`, `RollingZscore`, `DominantCycle`, `CycleSine`, `TickRuleSign`, `OFI`, `RollingKyleLambda`, `Hold`, `RollingMax`, `RollingPoly1`. Check any other name with `.venv/bin/python -m screamer.docs search <words>`.

## Rulings on spec details (recorded here so tasks agree)

- "What you can build" is removed in Phase 1, not Phase 2: once the Guide's toctrees hold every recipe and notebook, a second toctree over the same pages breaks the sidebar. `choosing_an_operator` stays as a Reference page until Phase 2.
- Parts III and IV get chapter landing pages in Phase 1 whose task links point at the existing, untrimmed recipes, so the home table of contents is complete. Phase 2 trims them.
- Notebook 22 goes to chapter 14 (it tests whether a feature carries information), not chapter 7.
- Task page URLs use chapter slugs, not numbers (`guide/mean-reversion/variance-ratio.html`), so inserting a chapter never moves a URL. Numbers come from guide.yml order.

## File structure

- `docs/guide.yml`: registry (schema in Task 1).
- `devtools/build_guide.py`: generator and checker.
- `tests/test_guide.py`: gate.
- `docs/guide/index.md`: Guide landing (generated block).
- `docs/guide/<slug>/index.md`: chapter landing, hand-written intro and function table, generated task list block.
- `docs/guide/<slug>/<task>.md`: task pages.
- `docs/guide_toc.inc`: generated home table of contents.
- `docs/reference.md`: Reference landing.
- `docs/index.rst`: home page and the three top-level toctrees.
- Removed: `docs/capabilities.md` (becomes stub), `docs/examples.yml`, `docs/examples_tiles.inc`, `devtools/build_examples.py`, `tests/test_examples_page.py`.
- Modified: `docs/_ext/llms_txt.py` (reads guide.yml), `docs/references_index.rst` (title "How screamer works", adds the old User Guide pages), `docs/conf.py` (`html_sidebars` if needed), `devtools/build_gallery.py` only if it reads examples.yml.

## Review Focus

1. A recipe or notebook that is in no chapter: a reader can never reach it and Sphinx warns "document isn't included in any toctree". `--check` must fail on it (Task 1 test).
2. An old URL (any recipe moved in Tasks 4-8, `installation`, `capabilities`) opened from a bookmark: it must redirect, not 404 (Task 9 test).
3. A task page whose code fails at build: myst-nb raises (`nb_execution_raise_on_error = True`); each content task builds the docs before committing.
4. A chapter title in guide.yml that drifts from the task page's own `# ` heading: the home TOC and the page disagree. `--check` compares them (Task 1 test).
5. A data path written relative to the old recipe location (`../notebooks/data/...`) after a recipe moves one level deeper: it must become `../../notebooks/data/...`. Each content task's build catches it.

---

### Task 1: guide.yml registry, build script and gate

**Files:**
- Create: `docs/guide.yml`, `devtools/build_guide.py`, `tests/test_guide.py`
- Create (generated): `docs/guide/index.md`, `docs/guide_toc.inc`, one `docs/guide/<slug>/index.md` per chapter (stub intro "This chapter is being written." until its content task)

**Interfaces:**
- Produces: guide.yml schema:
  ```yaml
  parts:
  - title: Basics            # part heading, plain words
    chapters:
    - slug: first-steps      # folder under docs/guide/
      title: First steps     # must equal the chapter landing's "# " heading
      line: Install screamer, apply a function, and handle missing values.
      tasks:                 # doc paths relative to docs/, in reading order
      - guide/first-steps/install
      deeper:                # notebooks or long pages owned by this chapter's toctree
      - notebooks/01-quickstart-polymorphic-api
  ```
- Produces: `devtools/build_guide.py` with `load() -> dict`, `build() -> dict[Path, str]` (path to full generated text) and `main(argv=None) -> int`; `--check` returns 1 and prints each problem. Mirror the shape of the current `devtools/build_examples.py`.
- Generated block in a chapter landing sits between `<!-- tasks:start -->` and `<!-- tasks:end -->`: a numbered list of task links using each task's `# ` heading as text, then a hidden ```` ```{toctree} ```` of `tasks` and `deeper` with `:hidden:`. Text outside the markers is hand-written and preserved; if the file does not exist, the script writes a stub with `# <title>`, the line, and the markers.
- `docs/guide/index.md`: `# Guide`, one intro sentence, then per part a `## Part N: <title>` heading and the numbered chapter list (`N. [title](slug/index.md): line`), then a hidden toctree of all chapter landings. Chapters are numbered continuously from 0.
- `docs/guide_toc.inc`: the same parts-and-chapters list in reStructuredText for inclusion in index.rst, links via `:doc:`.

Fill guide.yml with all 17 chapters now (slugs and titles below), with `tasks` for chapters 14 to 16 pointing at existing recipes and `tasks: []` for chapters 0 to 13 (content tasks fill them). Put every notebook in exactly one chapter's `deeper`:

| # | Part | slug | title | deeper |
|--:|:-----|:-----|:------|:-------|
| 0 | Basics | first-steps | First steps | notebooks 01, 05 |
| 1 | Basics | get-data-in | Get your data in | notebooks 09, 10 |
| 2 | Understand a series | denoise | Smooth and denoise a series | notebooks 04, 17 |
| 3 | Understand a series | volatility | Measure volatility | notebook 02 |
| 4 | Understand a series | trend | Detect a trend | notebook 03 |
| 5 | Understand a series | mean-reversion | Detect mean reversion | notebook 18 |
| 6 | Understand a series | cycles | Find cycles | notebook 19 |
| 7 | Understand a series | regimes | Detect regimes and change points | (none) |
| 8 | Understand a series | order-flow | Read order flow and liquidity | notebooks 11, 12, 13 |
| 9 | Build a strategy | signal-to-position | Turn a signal into a position | (none) |
| 10 | Build a strategy | backtest | Backtest a signal on OHLCV bars | notebooks 14, 15 |
| 11 | Build a strategy | strategies | Strategy templates | (none) |
| 12 | Build a strategy | stops | Stops and exits | (none) |
| 13 | Build a strategy | portfolios | Portfolios of many assets | (none) |
| 14 | Test a strategy | not-luck | Check that a backtest result is not luck | notebooks 20, 21, 22; tasks: recipes/significance-test, recipes/walk-forward, recipes/parameter-grid |
| 15 | Test a strategy | machine-learning | Prepare data for machine learning | notebooks 16, 23, 24, 25; tasks: recipes/ml-features |
| 16 | Run a strategy live | live | Run on live data | notebooks 06, 07, 08; tasks: recipes/live-loop, recipes/strategy-to-live |

Part titles: "Basics", "Understand a series", "Build a strategy", "Test a strategy", "Run a strategy live".

- [ ] **Step 1: Write the failing tests** in `tests/test_guide.py` (skip if `docs/guide.yml` is absent, like test_examples_page.py):
  - `test_guide_is_fresh`: `build_guide.py --check` exits 0.
  - `test_every_notebook_and_recipe_is_in_one_chapter`: every `docs/notebooks/*.ipynb` and every non-stub `docs/recipes/*.md` (a stub has `orphan: true`) appears in exactly one chapter's `tasks` or `deeper`.
  - `test_task_titles_match_pages`: for each task under `guide/`, the page's first `# ` heading is non-empty; for each chapter, the landing's `# ` heading equals the chapter `title`.
  - `test_task_page_shape`: each task under `guide/` (not recipes) has `file_format: mystnb`, at least one ```` ```{code-cell} ````, at most 35 code-cell lines in total, and a `## Go deeper` heading followed by at least one markdown link.
  - `test_check_reports_unlisted_notebook` (unit): call the checker on a temp copy where one notebook is dropped from guide.yml and assert it returns 1 and names the notebook.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_guide.py -v`. Expected: FAIL (script missing).
- [ ] **Step 3: Implement** `devtools/build_guide.py` and `docs/guide.yml` as specified; run `.venv/bin/python devtools/build_guide.py` to write the generated files.
- [ ] **Step 4: Run** the tests. Expected: PASS (the shape test passes vacuously while chapters 0 to 13 have no tasks).
- [ ] **Step 5: Commit** `docs/guide.yml devtools/build_guide.py tests/test_guide.py docs/guide/` with message `docs: guide registry, generator and gate`.

### Task 2: Navigation, Reference and home page

**Files:**
- Modify: `docs/index.rst`, `docs/references_index.rst`, `docs/_ext/llms_txt.py`, `docs/conf.py` (only if `html_sidebars` references a moved page)
- Create: `docs/reference.md`
- Delete: `docs/examples.yml`, `docs/examples_tiles.inc`, `devtools/build_examples.py`, `tests/test_examples_page.py`
- Replace with stub: `docs/capabilities.md` (orphan, meta refresh to `guide/index.html`)
- Check: `devtools/build_gallery.py` for reads of examples.yml or capabilities; if it reads them, switch it to guide.yml task and deeper lists.

**Interfaces:**
- Consumes: `docs/guide_toc.inc`, `docs/guide/index.md` from Task 1.
- Produces: top-level toctrees in index.rst with captions exactly `Guide` (entry `guide/index`), `Reference` (entry `reference`), `Release notes` (entry `changelog`).

- [ ] **Step 1: Rewrite `docs/index.rst`** to the four parts in Global Constraints. Hero text: "Screamer is a fast C++ library for time series in finance and signal processing: indicators, filters, order flow, costed backtests and significance tests, with the same functions on arrays, on live streams and in JavaScript." Keep the existing trend-sign code tabs; replace the three-row plotly figure with one row: BTC closes as a line colored by `Sign()(RollingPoly2(window_size=50, derivative_order=1)(close))` (green up, red down). Then `.. include:: guide_toc.inc`. Facts, one line each: "Fast: rolling statistics in constant time per sample, see Performance." / "Causal: every output uses only current and past inputs, see Conventions." / "One code path: the same function runs on a stored array and on a live feed, see Streams in depth." Remove badges, the speed chart, the causality section, all tiles and cards.
- [ ] **Step 2: Write `docs/reference.md`**: `# Reference`, one sentence, then three short sections, each one line plus link: "Functions" (by_group_index), "Gallery" (gallery), "How screamer works" (references_index). Hidden toctree of those three.
- [ ] **Step 3: Update `docs/references_index.rst`**: title "How screamer works"; add `usage`, `choosing_an_operator`, `agents`, `backtesting_best_practices`, `portfolio_backtesting`, `labels`, `javascript` to its hidden toctree and cards, one card each with a one-sentence description, after the existing ones.
- [ ] **Step 4: Update `docs/_ext/llms_txt.py`**: the Examples section reads guide.yml and lists every chapter's tasks as `- [<chapter title>: <page title>](<source url>)`; keep GUIDES as is (choosing_an_operator stays a page until Phase 2).
- [ ] **Step 5: Delete the What-you-can-build files** listed above and write the capabilities stub.
- [ ] **Step 6: Build** with `.venv/bin/sphinx-build -b html docs docs/_build/html 2>&1 | tee /tmp/sphinx.log` (use the session scratchpad instead of /tmp if one exists) and compare warnings with a build of `main` before the task: no new warnings. Open `docs/_build/html/index.html` in a browser (headless Chrome screenshot is fine) and check the top bar shows exactly three items and the home page has the four parts.
- [ ] **Step 7: Run** `.venv/bin/python -m pytest tests/test_guide.py -q`. Expected: PASS.
- [ ] **Step 8: Commit** the files above with message `docs: Guide and Reference navigation, calm home page`.

### Content tasks: shared rules (Tasks 3 to 8)

Each content task writes its chapters' task pages, fills `tasks` in guide.yml in reading order, writes the hand part of each chapter landing (two or three sentences, then a compact `## All functions for this chapter` table copied from the matching section of `docs/choosing_an_operator.md`, Intent / Function columns only), runs `devtools/build_guide.py`, builds the docs, runs `tests/test_guide.py`, and commits.

A task page:
```markdown
---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# <Task title, action and object>

<One sentence: what to use and why.>

```{code-cell} ipython3
<10 to 20 lines: load committed data, compute with screamer>
```

```{code-cell} ipython3
<plot, matplotlib, one figure>
```

<Optional: one or two sentences reading the plot.>

## Go deeper

- [<page>](<relative link>): <a few words>
```

Data available (paths relative to a task page are `../../notebooks/data/<file>`): `binance_daily_20.csv` (daily OHLCV of 20 coins 2022 to 2026, columns date, symbol, open, high, low, close, volume), `binance_btc_funding.csv`, `deribit_btc_perp_6h.csv` and `deribit_eth_perp_6h.csv` (six hours of trades). When a page moves from a recipe, keep its content but trim it to the shape (one code cell for computation, one for the plot, at most 35 code lines), move extra material to "Go deeper" links, then replace the old recipe file with an orphan meta-refresh stub to the new page.

The verification for every content task: `.venv/bin/python devtools/build_guide.py && .venv/bin/python -m pytest tests/test_guide.py -q && .venv/bin/sphinx-build -b html docs docs/_build/html` with no new warnings and no execution errors; then open two of the new pages in the built HTML and check the plot renders.

### Task 3: Part 0, Basics (chapters 0 and 1)

Task pages:
- `guide/first-steps/install`: "Install screamer". Moved from `installation.md` (stub left at `installation.md`). This page is prose and a console block, so it is exempt from the code-cell rule: add `exempt_shape: true` to its front matter and make `test_task_page_shape` honor it.
- `guide/first-steps/first-function`: "Apply a function to an array, a column and a live stream" (`RollingMean` on a NumPy array, a pandas column, and value by value).
- `guide/first-steps/missing-values`: "Handle missing values and warmup" (`FillNa`, what an operator returns before its window fills). Go deeper: nan_and_warmup, notebook 05.
- `guide/get-data-in/ohlcv-dataframe`: "Compute indicators on an OHLCV DataFrame", moved from recipes/pandas-ohlcv-indicators.
- `guide/get-data-in/from-pandas-talib`: "Replace pandas rolling and TA-Lib calls", moved from recipes/migrating.
- `guide/get-data-in/bars-from-ticks`: "Build bars from trades", moved from recipes/ticks-to-backtest.

Commit message: `docs(guide): basics chapters`.

### Task 4: Part I, chapters 2 and 3 (denoise, volatility)

- `guide/denoise/moving-averages`: "Smooth a series with a moving average" (`RollingMean`, `EwMean`).
- `guide/denoise/low-lag`: "Smooth a series with less lag" (`HullMA`, `KAMA`, `SuperSmoother` against `RollingMean`).
- `guide/denoise/kalman`: "Track a noisy level with a Kalman filter" (`KalmanFilter`).
- `guide/denoise/lowpass`: "Remove fast wiggles with a low-pass filter" (`Butter`).
- `guide/denoise/despike`: "Remove spikes from a series" (`Hampel`, on BTC trade prices with a few injected spikes).
- `guide/volatility/close-to-close`: "Measure volatility from closing prices" (`RollingStd` and `EwStd` of log returns, annualized).
- `guide/volatility/ohlc`: "Measure volatility from OHLC bars" (`RollingYangZhangVol` against close-to-close).
- `guide/volatility/atr`: "Measure the average true range" (`ATR`).

Commit message: `docs(guide): denoise and volatility chapters`.

### Task 5: Part I, chapters 4 and 5 (trend, mean reversion)

- `guide/trend/slope`: "Measure a trend with a rolling slope" (`RollingPoly1` slope, `Sign`).
- `guide/trend/crossover`: "Detect a trend with a moving-average crossover" (two `EwMean`, `Sign`).
- `guide/trend/adx`: "Measure trend strength with ADX" (`ADX`, `PlusDI`, `MinusDI`).
- `guide/trend/trend-mode`: "Tell trending periods from cycling periods" (`TrendMode`).
- `guide/mean-reversion/variance-ratio`: "Test whether a series reverts with the variance ratio" (`RollingVarianceRatio`, `output="zscore"`; below 1 reverts, above 1 trends).
- `guide/mean-reversion/hurst`: "Estimate the Hurst exponent" (`RollingHurst`; below 0.5 reverts).
- `guide/mean-reversion/half-life`: "Estimate how fast a series reverts" (`RollingOU`, mean-reversion rate turned into a half-life in bars).
- `guide/mean-reversion/zscore-bands`: "Trade the reversion with z-score bands" (`RollingZscore`, `BandSignal`, `BacktestPriceTarget`, `backtest_report`; use a spread or a reverting series from the committed data, for example the ETH/BTC log price ratio).

The four mean-reversion pages must use the same series so a reader sees one story: detect, measure speed, trade.

Commit message: `docs(guide): trend and mean reversion chapters`.

### Task 6: Part I, chapters 6 to 8 (cycles, regimes, order flow)

- `guide/cycles/dominant-cycle`: "Measure the dominant cycle length" (`DominantCycle`, after `RoofingFilter`).
- `guide/cycles/turning-points`: "Mark cycle turning points" (`CycleSine`).
- `guide/regimes/hmm`: "Estimate the probability of a volatile regime" (`RollingHMM`).
- `guide/regimes/changepoint`: "Detect a change in mean or spread" (`BayesianChangepoint` on a standardized series).
- `guide/regimes/cusum`: "Sample events at large cumulative moves" (`CusumFilter`).
- `guide/order-flow/sign-trades`: "Sign trades as buys or sells" (`TickRuleSign` on Deribit trades).
- `guide/order-flow/imbalance`: "Measure order-flow imbalance" (`OFI` or `RollingOrderImbalance`).
- `guide/order-flow/price-impact`: "Measure price impact" (`RollingKyleLambda`, `AmihudIlliquidity`).

Commit message: `docs(guide): cycles, regimes and order-flow chapters`.

### Task 7: Part II, chapters 9 and 10 (signal to position, backtest)

- `guide/signal-to-position/sign`: "Go long or short on the sign of a score" (`Sign`).
- `guide/signal-to-position/bands`: "Enter and exit at different levels" (`BandSignal`, `SchmittTrigger`).
- `guide/signal-to-position/hold`: "Hold a position for a fixed number of bars" (`Hold`, `cooldown=`).
- `guide/signal-to-position/volatility-sizing`: "Size a position by volatility", moved from recipes/volatility-sizing.
- `guide/backtest/ohlcv`: "Backtest a signal on OHLCV bars", moved from recipes/first-strategy (the p-value part moves to a Go deeper link to chapter 14).
- `guide/backtest/costs`: "See how fees and slippage change a result" (same strategy at three cost levels, one equity plot).

The old Get-started link to `recipes/first-strategy` disappears with Task 2; the stub keeps the URL.

Commit message: `docs(guide): signal-to-position and backtest chapters`.

### Task 8: Part II, chapters 11 to 13 (strategy templates, stops, portfolios)

- `guide/strategies/trend-following`: "Follow a trend with a moving-average crossover".
- `guide/strategies/breakout`: "Trade a breakout of the recent high or low" (`DonchianChannels`).
- `guide/strategies/rsi-reversion`: "Buy oversold and sell overbought with RSI" (`RollingRSI`, `BandSignal`).
- `guide/strategies/bollinger-reversion`: "Fade moves outside the Bollinger Bands" (`BollingerBands`).
- `guide/strategies/pairs`: "Trade a pair", moved from recipes/pairs-trade.
- `guide/strategies/cross-sectional-momentum`: "Rank assets by momentum", moved from recipes/cross-sectional-momentum.
- `guide/strategies/carry`: "Earn the funding carry", moved from recipes/funding-carry.
- `guide/strategies/volatility-targeting`: "Target a constant volatility" (`EwStd`, position scaled to a target).
- `guide/stops/stops-and-exits`: "Add stops and exits", moved from recipes/stops-and-exits.
- `guide/portfolios/multi-asset`: "Backtest a portfolio of many assets", moved from recipes/multi-asset.

Every strategy page reports the same three numbers from `backtest_report` (Sharpe, max drawdown, turnover or trades) under its plot, so pages compare at a glance.

Commit message: `docs(guide): strategy, stops and portfolio chapters`.

### Task 9: Redirects check and final pass

**Files:**
- Modify: `tests/test_guide.py`
- Modify: any page with a broken internal link found by the build

- [ ] **Step 1: Write `test_moved_pages_redirect`**: for every stub (front matter `orphan: true` with an `http-equiv=refresh` meta) the target path, resolved against the stub's folder with `.html` mapped to the source `.md`/`.ipynb`/`.rst`, exists. Also assert stubs exist at `installation.md`, `capabilities.md` and each recipe moved in Tasks 3, 7 and 8.
- [ ] **Step 2: Run** the test. Expected: PASS (fix any stub that fails).
- [ ] **Step 3: Full build** with `-W --keep-going` to list every warning (the CI build does not use `-W`, so this is a listing, not a new gate) and fix each warning this plan introduced and each broken `:doc:` or markdown link.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest tests/test_guide.py tests/ -q -k "docs or guide or llms or gallery"` and the full `tests/` once. Expected: PASS.
- [ ] **Step 5: Browser check**: screenshots of home, Guide index, the mean-reversion chapter landing and `guide/mean-reversion/zscore-bands`, at desktop and phone width; check the top bar has three items and pages fit about one screen of reading before "Go deeper".
- [ ] **Step 6: Commit** with message `docs(guide): redirects gate and final pass`.
