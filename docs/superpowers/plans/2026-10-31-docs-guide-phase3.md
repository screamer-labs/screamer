# Docs Guide, Phase 3 Implementation Plan: deeper chapters and findability

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make most of screamer's capabilities findable through the one Guide: about 35 new task pages in existing chapters plus two new chapters, searchable words in titles and chapter lines, search synonyms on function pages, "Used in" links back to the Guide, and a coverage check.

**Architecture:** Unchanged from Phases 1-2: `docs/guide.yml` registry, `devtools/build_guide.py` generator (`--check`), task pages as executed MyST notebooks under `docs/guide/<slug>/`, `tests/test_guide.py` gate, `docs/_ext/function_page.py` for function pages. No new entry points: no "popular tasks" list, no capability index page, no separate "coming from" section (user ruling 2026-10-31).

**Tech Stack:** Sphinx, myst-nb, pydata-sphinx-theme, matplotlib, PyYAML, pytest.

**Spec:** docs/superpowers/specs/2026-10-27-docs-guide-design.md, plus the user's ruling of 2026-10-31 (this plan's Goal and Architecture lines). Phase 1 and 2 plans' Global Constraints still hold.

## Global Constraints

- Everything in the Phase 1 and Phase 2 Global Constraints (task-page shape, naming rule, style guide, no em-dashes, screamer for every calculation, causal, committed data only, orphan stubs for moved URLs, Guide conventions: next-open fills, stated costs, drawdown in percent of peak, "N USD", US spelling, honest results, look-ahead scoring in its own cell, batch == live stated once only on guide/live/one-value-at-a-time).
- Findability rule: a page title says the task in plain words and, where people search by a term of art, adds it in brackets: "Shuffle trades in blocks to test a strategy's timing (block randomization test)". Chapter `line`s in guide.yml name the key searchable terms of their pages.
- Data: only files under docs/notebooks/data (binance_daily_20.csv, binance_btc_funding.csv, deribit_btc_perp_6h.csv, deribit_eth_perp_6h.csv, deribit_ethbtc_ratio_1m.csv). A page that needs quotes (L1) or another shape builds it from these with screamer (e.g. bars from trades) or, if impossible, uses a small clearly labelled synthetic series and says so in one sentence.
- After any function-page frontmatter change, regenerate `screamer/data/help.json` (devtools/build_help_registry.py) and the JS docs (devtools/wasm/gen_js_docs.py); read and write text files with encoding="utf-8".
- Python: /Users/thijs/screamer/.venv/bin/python (worktree: the bindings .so is symlinked into screamer/). Sphinx: /Users/thijs/screamer/.venv/bin/sphinx-build -b html docs docs/_build/html; baseline 1 known sphinx_exec_code warning.

## Review Focus

1. A new chapter inserted in guide.yml renumbers later chapters: every hard-coded chapter number in prose ("see chapter 14") must be found and fixed (grep "chapter [0-9]").
2. A synonym tag added to a function page must not break the topics/tags validation (CI validates frontmatter).
3. "Used in" links must only list Guide task pages that actually import or call the function (no substring false positives, e.g. `Sign` inside `SignedVolume`).
4. Pages on order-based engines (limit/stop, L1 quotes) must state fill rules exactly as the function page defines them.
5. The coverage check must not fail the build on intentionally uncovered math helpers (an explicit allow-list).

---

### Task 1: Structure, findability plumbing, coverage check

- Add two chapters to guide.yml with empty task lists and landings: `indicators` ("Compute classic technical indicators", after `trend`) and `performance` ("Measure performance and risk", after `backtest`). Regenerate.
- Rewrite every chapter `line` in guide.yml to name its key searchable terms (e.g. not-luck: "Randomization and permutation tests, block shuffles, walk-forward testing, deflated Sharpe").
- Function pages: add search synonyms (tags) where people search by another name, at least: position_test, strategy_test, segment_shuffle, circular_shift, permute_bars, permute_returns (block bootstrap, block randomization, Monte Carlo permutation test, White's reality check, data snooping), time_splits (purged cross-validation, embargo, time series split), deflated_sharpe / probabilistic_sharpe (multiple testing, selection bias), CorwinSchultzSpread / AbdiRanaldoSpread / RollSpread (bid-ask spread estimator, effective spread from OHLC), FracDiff (fractional differencing, stationarity with memory), VPIN (order flow toxicity), HawkesIntensity (self-exciting point process). Regenerate help.json and JS docs; topics validation must pass.
- "Used in" links: in docs/_ext/function_page.py, render a short "Used in the Guide" list on each function page from the Guide task pages whose code cells use the function as a whole word (import or call), from guide.yml order. Unit test with a `Sign` vs `SignedVolume` case.
- Coverage check: tests/test_guide.py gains `test_guide_covers_functions`: every function page outside an explicit allow-list (elementwise math helpers and similar, kept in tests or guide.yml) is used on at least one Guide task page; the test is marked xfail-strict-false or reports the current gap list until Tasks 2-5 land, then becomes a hard gate in Task 6.
Commit: `docs(guide): phase 3 structure, search synonyms, used-in links`.

### Task 2: Chapter 14 and 15 pages (randomization and validation)

- not-luck/block-shuffle: "Shuffle trades in blocks to test a strategy's timing (block randomization test)" (position_test method="segments", segment_shuffle)
- not-luck/permutation-test: "Run a Monte Carlo permutation test on a whole strategy" (strategy_test)
- not-luck/synthetic-paths: "Generate synthetic price paths with the same return distribution" (permute_returns, permute_bars)
- not-luck/shifted-copies: "Test a signal against time-shifted copies of itself" (circular_shift)
- not-luck/deflated-sharpe: "Correct a Sharpe ratio for the number of strategies tried (deflated Sharpe ratio)" (deflated_sharpe, probabilistic_sharpe)
- machine-learning/time-series-cv: "Cross-validate on time series without leakage (purged time series split)" (time_splits)
- machine-learning/fractional-differencing: "Make a series stationary without losing its memory (fractional differencing)" (FracDiff)
Commit: `docs(guide): randomization and validation pages`.

### Task 3: Chapter 10 pages and the new performance chapter

- backtest/limit-stop-orders: "Backtest with limit and stop orders" (BacktestOHLCOrders)
- backtest/tick-trades: "Backtest on tick trades instead of bars" (BacktestTradesTarget, BacktestTradesOrders)
- backtest/quotes: "Backtest against bid and ask quotes (L1)" (BacktestL1Target, BacktestL1Orders)
- backtest/volume-cap: "Cap trades at a share of each bar's volume" (BacktestOHLCVTarget)
- backtest/portfolio-report: "Report a portfolio backtest as one book" (backtest_report on (T, K, 4))
- backtest/drawdown-liquidation: "Track drawdown and distance to liquidation live" (Drawdown, LiquidationDistance)
- performance/rolling-ratios: "Compute a rolling Sharpe, Sortino and Calmar ratio" (RollingSharpe, RollingSortino, RollingCalmar, RollingOmega, RollingInfoRatio)
- performance/tail-risk: "Measure tail risk with rolling CVaR and downside deviation" (RollingCVaR, RollingDownsideDeviation, RollingMaxDrawdown)
- performance/beta-alpha: "Estimate beta and alpha against a benchmark" (RollingBeta, RollingAlpha, EwBeta, RollingTLSBeta)
- performance/factor-regression: "Regress returns on several factors in a moving window" (RollingMultiRegression, EwMultiRegression, ExpandingMultiRegression)
Commit: `docs(guide): order-based backtests and performance chapter`.

### Task 4: Chapter 8 and 13 pages

- order-flow/spread-from-bars: "Estimate the bid-ask spread from OHLC bars (Corwin-Schultz, Abdi-Ranaldo, Roll)"
- order-flow/vpin: "Measure order-flow toxicity (VPIN)"
- order-flow/micro-price: "Compute the micro-price and queue imbalance from quotes"
- order-flow/fill-quality: "Measure the effective and realized spread of your fills"
- order-flow/hawkes: "Model clustered trade arrivals (Hawkes intensity)"
- portfolios/cross-sectional-scores: "Z-score, demean and winsorize across assets each day" (CrossZscore, CrossDemean, CrossWinsorize, CrossStd)
- portfolios/per-group: "Restart a calculation each day, contract or group" (Segment)
Commit: `docs(guide): microstructure and cross-section pages`.

### Task 5: Indicators chapter and chapters 1, 2, 3, 7

- indicators/oscillators: "Momentum oscillators: Stochastic RSI, Williams %R, CCI, TRIX, Ultimate Oscillator"
- indicators/volume: "Volume indicators: OBV, MFI, accumulation/distribution and VWAP"
- indicators/keltner: "Keltner Channels next to Bollinger Bands"
- denoise/adaptive-trend: "Track an adaptive trend (MAMA, instantaneous trendline)"
- denoise/robust-outliers: "Detect outliers robustly (median absolute deviation, sigma clipping)" (RollingMad, RollingMedianAD, RollingSigmaClip)
- volatility/jumps: "Separate jumps from normal volatility (bipower variation)" (RollingBipowerVar, RollingJumpShare)
- volatility/seasonality: "Compare each hour with the same hour on past days (seasonal z-score)" (RollingSeasonal*): needs a regular intraday clock; build minute or hourly bars from the committed Deribit trades (6 hours) only if that shows the idea honestly, else use the daily data with weekday slots and say so.
- regimes/permutation-entropy: "Measure how predictable a series is (permutation entropy)"
- get-data-in/from-mlfinlab: "Replace mlfinlab and AFML functions" (map: fractional differencing, CUSUM filter, triple-barrier labels, purged CV, deflated Sharpe, bet sizing where screamer has it; each row links its screamer function and Guide page)
Commit: `docs(guide): indicators chapter and more task pages`.

### Task 6: Final pass

- Make the coverage check a hard gate with the final allow-list; fix any gaps it shows by adding a page or a justified allow-list entry.
- grep for hard-coded chapter numbers and stale links; full Sphinx build with -W --keep-going (only the known warning); tests/test_guide.py, tests/test_function_page_order.py, docs freshness tests, the full tests/ once; screenshots of home, the Guide index, chapter 14 landing and two new pages.
- Apply deferred minor findings from Tasks 1-5.
Commit: `docs(guide): phase 3 final pass`.
