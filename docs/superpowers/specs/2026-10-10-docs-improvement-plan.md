# Documentation improvement plan

Status: phases 1-3 done (2026-10-10), except the follow-ups under "Remaining
work" at the end.

## Problem

New users struggle to find out what screamer can do. The material is good, but
it is organised as a taxonomy of functions, not as things a user can do. The
plots exist, but they are spread across 252 reference pages, one per page. The
notebooks a newcomer opens first have almost no plots.

## Findings

### First five minutes today

1. The landing page (`docs/index.rst`) shows one trend example on random data,
   a speed chart, and nine cards: eight function groups plus "Examples".
2. The first card, "Statistics", opens a long table of function names.
3. `usage.md` explains how calls work using printed arrays. It has no plots.
4. Nothing answers "what can I build with this?" The landing page never
   mentions bars from ticks, backtesting, order flow, Kalman filtering, pairs
   trading, ML datasets, or running in a browser.

### Where the visual content is

- 219 of 252 function pages (87%) have a plotly example, built at doc build
  time. 20 have printed output only. 13 have no example at all: the twelve
  Ehlers cycle ops (`CycleAmplitude`, `CycleFrequency`, `CyclePhase`,
  `CycleSine`, `Decycler`, `DominantCycle`, `HilbertPhasor`,
  `InstantaneousTrendline`, `MAMA`, `RoofingFilter`, `SuperSmoother`,
  `TrendMode`) and `functions_fin/choosing_a_backtest_engine.md`.
- About 80 function-page examples use unseeded random data, so the plot changes
  on every build.
- Most examples plot an indicator over `cumsum(normal)` noise. On noise, an
  indicator does not show what it is for.

### Notebooks

| Notebook | Plots | Assessment |
|---|---|---|
| 01 quickstart | 0 | 10 lines; no visual result |
| 02 window statistics | 4 | OK |
| 03 financial indicators | 3 | OK |
| 04 signal processing | 6 | Strongest visual notebook |
| 05 NaN handling | 0 | Very short |
| 06 streaming | 1 | Thin |
| 07 multi-stream | 0 | Printed tables only |
| 08 pipelines | 1 (graph) | Thin |
| 09, 10 bars from ticks | 5, 2 | Long |
| 11-13 microstructure | 3, 4, 4 | Strong, real data |
| 14 backtesting a signal | 2 | Short |
| 15 event-driven backtests | 3 | Good |
| 16-19 | 3-6 each | Good |

The foundation notebooks (01, 05, 06, 07, 08, and 14) are the ones a new user
opens first, and they have the fewest plots.

### Discoverability gaps

- No capability overview and no visual gallery.
- No task recipes. Common tasks ("pandas OHLCV to indicators", "ticks to bars to
  costed backtest", "run live on a websocket") are spread across notebooks 06,
  08, 09, and 14.
- Backtesting is one card among nine. The engine-selection guide is buried in
  `functions_fin`.
- JavaScript/WASM is not mentioned anywhere in the Python docs, including the
  live-trades browser demo.
- The concept guides (`polymorphic_api`, `nan_and_warmup`, `multistream`,
  `pipelines`, `microstructure`, `performance`, `conventions`) sit in a hidden
  toctree. `references_index.rst` names them but does not link to them.
- `choosing_an_operator.md` is the best discovery page, but it does not cover
  microstructure, cycles, bars, or ML.

### Defects

- `examples/lag_example.py` imports `lag_generator`, which no longer exists.
- `docs/index.rst` and `docs/installation.md` say Python 3.11+.
  `pyproject.toml` and `README.md` say 3.12+.
- `multistream.md` has a migration section that belongs in the changelog.

## Plan

The plan has three phases. Each item lists its files and its scope and risk.

### Phase 1: quick wins

1. **Fix the defects.** Rewrite or delete `examples/lag_example.py`. Change the
   Python version to 3.12 in `index.rst` and `installation.md`. Move the
   migration section to the changelog. Trivial.
2. **Make References visible.** Replace the hidden toctree in
   `references_index.rst` with a card grid that has one line per guide. Move
   `conventions.md` into the User Guide. Trivial.
3. **Fill the 13 empty function pages.** Use seeded plots and reuse cells from
   notebook 19. For `choosing_a_backtest_engine`, add a decision diagram that
   maps data type and strategy type to an engine. Small.
4. **Extend `choosing_an_operator.md`** with sections for microstructure,
   cycles and regimes, bars and resampling, ML datasets, risk and drawdown, and
   data cleaning. Link every row to its function page. Small.

### Phase 2: give users an overview

5. **Add a capability page and redesign the landing page.** The new
   `docs/capabilities.md` has about eight tiles: Indicators, Filtering and
   cycles, Bars from ticks, Microstructure, Backtesting, Forecasting and ML,
   Live streaming and pipelines, JavaScript in the browser. Each tile has a
   thumbnail, one sentence, and links to a notebook and the relevant
   functions. These tiles go on `index.rst` above the function-group cards.
   Change the landing example from random data to real data. Small to medium;
   low risk.
6. **Add an auto-generated gallery.** A new script,
   `devtools/build_gallery.py`, runs each page's plot block, exports a static
   thumbnail with kaleido, and writes `docs/gallery.rst`: a grid grouped by
   topic, with each thumbnail linking to its function page. The same
   thumbnails go in the `by_group/*.rst` tables (`build_topic_pages.py`) and on
   the notebook cards. Medium; the risks are build time and the kaleido
   dependency, so cache thumbnails by a hash of the page source.
7. **Use shared, realistic example data.** Add a small committed dataset (a
   daily OHLCV slice and the existing Deribit trades slice) and a loader,
   `screamer.datasets` or a docs-only helper. Switch the function-page examples
   to that data and seed the rest. An RSI on a real price series explains
   itself; an RSI on noise does not. This touches many files but each change is
   mechanical; low risk.

### Phase 3: show complete tasks

8. **Add a recipes section** (`docs/recipes/`). It has 8 to 12 short,
   end-to-end pages, each with one plot:
   - pandas OHLCV DataFrame to indicators
   - ticks to bars to signal to costed backtest
   - live websocket loop
   - multi-asset `(T, K)` arrays
   - pairs trade
   - feature matrix for ML
   - significance test of a backtest (see the randomization spec)
   - migrating from TA-Lib or `pandas.rolling`
   Medium; low risk.
9. **Upgrade the foundation notebooks.**
   - 01: a visual result.
   - 05: shade the warmup region.
   - 06: overlay the batch result and the live result in one plot.
   - 07: a `CombineLatest` alignment timeline.
   - 08: the pipeline graph next to its outputs.
   - 14: equity, drawdown, and position panels, plus the cost breakdown.
   Medium; low risk.
10. **Surface JavaScript/WASM.** Add a "JavaScript and browser" capability tile
    and a `docs/javascript.md` page that links to the JS docs, the live-trades
    demo, and the parity guarantee. Optionally show the landing example in
    Python and JS side by side using sphinx-design tabs. Small.

## Order and verification

Do phase 1, then items 5 and 7, which unlock 6 and 8, then 9 and 10. After
each item, run a clean `make docs` build. Check that it has no new warnings,
that the gallery thumbnails resolve, and that all notebooks execute.

## What was delivered (phases 2 and 3)

- Item 5: `docs/capabilities.md` (eight tiles) and the same tiles on
  `index.rst` above the function-group cards. The landing example runs on
  one-minute BTC closes from the committed Deribit slice and shows the Python
  and JavaScript code in tabs.
- Item 6: `devtools/build_gallery.py` and `docs/gallery.rst`. kaleido could not
  start a browser in the build environment, so the script redraws each
  captured plotly figure with matplotlib instead (no new dependency). It covers
  the trace types the docs use; the thumbnails are sketches, not pixel copies.
  Thumbnails are WebP, 320 x 200, about 7 KB each, cached by a hash of the plot
  code in `docs/_static/gallery/manifest.json`. Re-run the script after adding
  or changing a function-page plot and commit the output.
- Item 7 (partial): the recipes, capability thumbnails, and landing example use
  the committed Deribit slice. The 36 function pages that drew unseeded random
  data are now seeded.
- Item 8: eight recipes under `docs/recipes/`, MyST notebooks executed at build
  time on the Deribit slice.
- Item 9: plots added to notebooks 01, 05, 06, 07, 08, and 14.
- Item 10: `docs/javascript.md` and the JavaScript capability tile.

## Remaining work

- Item 6: the thumbnails in the `by_group/*.rst` tables and on the notebook
  cards are not done. Notebook cards would need thumbnails from executed
  notebook outputs, which the committed notebooks do not store.
- Item 7: switching the function-page examples from `cumsum(normal)` noise to
  real data is not done. It needs a daily OHLCV slice (the committed tape
  covers six hours, too short for daily-scale indicators) and a docs-only
  loader so each page stays short. Real data helps most on the indicator,
  volatility, and backtest pages; the math and logic pages are fine on
  synthetic input.
