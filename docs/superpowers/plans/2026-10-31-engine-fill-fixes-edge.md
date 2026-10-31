# Backtest engine fill fixes and the EDGE spread estimator

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix two backtest-engine defects found while writing the Guide (resting-fill `cost` measured against the mark; crossing orders filling as makers at prices outside the market), document the known limits of the bar-based spread estimators, and add the EDGE spread estimator.

**Architecture:** All logic in the C++ core (include/screamer/), Python and JavaScript are thin bindings compiled from the same C++ (JS via WASM). Shared fill accounting is `PnLAccount::step` in include/screamer/detail/pnl_account.h; fills are produced by `OHLCFill`, `TradesFill`, `L1Fill` and `BacktestL1TradesOrders`.

**Tech Stack:** C++17, nanobind, Embind/WASM (js/), pytest, node test runner, Sphinx.

**Spec:** the investigation findings and the user's approval of 2026-10-31 (user: "1 yes, 2 yes Fill at the open as a taker, 3 yes"), recorded here as the binding requirements.

### Findings (facts established on main @ 36b4bd16 with probes)
- `PnLAccount::step`: `fees = |dpos| * (fee_per_contract + fill * mult * fee_rate)`; `pnl = mark_pnl + dpos * (close - fill) * mult - fees`; `cost = dpos * (fill - reference) * mult + fees`, `reference` defaulting to the mark. `cost` is a reported column only: pnl and equity are computed from actual fills and fees (no double counting).
- Commit 8ee9a3c0 ("Fixed: Backtest cost on OHLC engines") passed `reference` only from `OHLCFill`; `TradesFill`, `L1Fill`, `BacktestL1TradesOrders` still use the mark, so a resting (maker) fill's cost includes its distance from the last print or the mid.
- `choosing_a_backtest_engine.md` says a resting fill's cost is its fee alone (correct intent) and lists the L1 taker reference as "the opposite touch" while the code uses the mid.
- Crossing orders today: OHLC fills every finite limit at its limit as maker when low/high reach it, so fills can lie outside the bar (bid 110 on 95-105 fills at 110; bid 100 on a gap bar O95/H97/L93 fills at 100). Trades is stateless: a new or re-priced bid at 102 with a print at 100 fills as maker at 102. L1 / L1Trades: a resting bid re-priced 99 -> 102 against 100/101 fills as maker at 102 with no taker fee; a new bid equal to the ask does not fill in breach mode and fills as a maker partial in touch mode.
- Spread estimators: CorwinSchultzSpread matches the authors' reference R package `bidask` (CS2, negatives floored then averaged) to 1.5e-9; AbdiRanaldoSpread matches the paper's monthly estimator to 8e-16; RollSpread matches 2*sqrt(-cov) to 2e-16. Implausible values on liquid data are the estimators' documented bias when the spread is small relative to per-bar volatility (Ardia, Guidotti and Kroencke 2024, JFE). Not a bug.

### Approved rules
1. Resting (maker) fills use their limit price as `cost` reference in every engine, so their cost is the fee alone. L1 taker fills keep the mid as reference (standard effective-spread cost); the docs table is corrected to say so. Remove the defaulted `reference` overloads so every call site names its reference.
2. Arrival rule: a side arrives on an event when it was not quoted on the previous event or its price changed (a size change alone is not arrival). Only an arriving order can be marketable.
   - L1 and L1Trades: track `prev_price` per side; an arriving bid with `bid >= market_ask` (ask: `ask <= market_bid`) is a taker: it fills from the opposite touch, walking one tick past the displayed size, with walk and slippage capped at the limit, and pays the taker fee. A resting, unchanged order run over by the market fills as maker at its own price (unchanged); a touch lock on a resting order gives a maker partial (unchanged). In L1Trades the arrival check runs before the print logic.
   - Trades: track `prev_price` per side; an arriving finite bid with a print below it fills at `min(bid, slippage.buy(print))` as a taker (like the existing market-order path, using the current print). A resting bid with a print through it fills as maker at its limit (unchanged). A bid exactly at the print stays resting.
   - OHLC (stays stateless): each bar's quote is placed at the open. If `open < bid` (sell: `open > ask`) the order fills at `min(bid, slippage.buy(open))` as a taker; this also covers a gap through a resting limit (user ruling: fill at the open as a taker). Otherwise it fills at the limit as maker when the low (high) reaches it (unchanged). Invariant: every fill before slippage lies within `[low, high]`.
   - Target engines (±inf prices, market orders) are unaffected; keep the +inf walk uncapped.
3. Spread estimators: add a `## Limitations` section to CorwinSchultzSpread, AbdiRanaldoSpread and RollSpread pages (CS biased upward when the spread is small relative to per-bar volatility; AR returns 0 in most windows in that regime; Roll is NaN in many windows and its non-NaN values are a selected, upward-biased subset; use daily bars of less liquid assets and relative comparisons; prefer EffectiveSpread / RealizedSpread when quotes exist; cite Ardia, Guidotti and Kroencke 2024). Add a new C++ operator `EdgeSpread` (the EDGE estimator of Ardia, Guidotti and Kroencke 2024, rolling over a window of bars, O(1) per bar), with Python and JavaScript bindings and a function page; test against the authors' MIT reference implementation and its published test data (repository eguidotti/bidask: pseudocode/ohlc.csv and the reference outputs), committing only a small test fixture with its license note.

## Global Constraints

- All computation in C++; Python and JS thin bindings; Python-JS parity (JS gets the same fixes via WASM; EdgeSpread added to the JS manifest/codegen).
- Batch == streaming; causal; existing operator conventions (constructor args, start_policy, NaN policy as similar operators).
- Compatibility rules: results change only where approved (cost column for maker fills; fills of crossing orders). CHANGELOG `## Unreleased`: `### Fixed` entries with before/after examples for 1 and 2, `### Added` for EdgeSpread, docs-only note for the limitations. Before finishing, diff results against the released 3.9.1 wheel on a broad set of engine runs and confirm every difference is one of the approved changes.
- Never edit version files. After C++ changes: `make install-dev POETRY_RUN= PY=/Users/thijs/screamer/.venv/bin/python PIP="/Users/thijs/screamer/.venv/bin/python -m pip"`. After frontmatter changes regenerate help.json (devtools/build_help_registry.py) and JS docs (devtools/wasm/gen_js_docs.py); rebuild WASM per js/README; read/write text files as UTF-8.
- Docs follow the style guide; Guide pages that describe the old behaviour (docs/guide/backtest/limit-stop-orders.md, quotes.md, tick-trades.md, choosing_a_backtest_engine.md, the engines' function pages) are updated in the task that changes the behaviour, numbers re-derived from the new output; never auto re-wrap pages.
- Efficiency: no extra allocation or pass in the hot loops; per-side `prev_price` state is a few doubles.
- Commits with explicit paths; never `git add -A` or `git stash`.

## Review Focus

1. A quote that was NaN (side off) and returns at a crossing price is an arrival and must be a taker.
2. Size-only changes of a resting order must not make it an arrival (no spurious taker fills).
3. Walk and slippage of a marketable limit must never fill beyond the limit.
4. OHLC: a sell limit below the open fills at max(ask, slippage.sell(open)); NaN or zero-volume bars keep today's behaviour.
5. Streaming (value by value) and `reset()` give the same fills as batch for the new per-side state.

---

### Task 1: Resting-fill cost reference

Files: include/screamer/detail/pnl_account.h (remove defaulted reference overloads), the fill helpers (TradesFill, L1Fill, BacktestL1TradesOrders), docs/functions_fin/choosing_a_backtest_engine.md (reference table: L1 taker = the mid; one-line `cost` definition) and each engine page (one line defining `cost`), tests.
- Tests first: maker cost equals fee for TradesOrders (through print), L1Orders (run over and lock), L1TradesOrders (print-driven and run-over fallback, mid != bid); a parametrized invariant over all engines `cost == dpos * (fill - documented reference) * mult + fees`; update tests/test_backtest.py::test_trades_fill_and_adverse_selection (through-fill cost now equals the fee); pnl and equity unchanged against the 3.9.1 behaviour on the same inputs; one JS spot check.
- CHANGELOG Fixed entry.
Commit: `fix(backtest): resting fills report their fee as cost in every engine`.

### Task 2: Crossing orders fill as takers

Files: the fill helpers for OHLC, Trades, L1 and L1Trades (per-side prev-price state where specified, reset() clears it), engine function pages and choosing_a_backtest_engine.md (price-value table: "marketable on arrival = taker"), Guide pages docs/guide/backtest/limit-stop-orders.md, quotes.md, tick-trades.md (re-derive numbers), tests (Python and JS).
- Tests first, per engine, from the Approved rules and Review Focus: OHLC bid 110 on 95-105 fills at 100 as taker; gap bar O95/H97/L93 with bid 100 fills at 95 as taker; sell mirror; property test low <= fill <= high (before slippage); Trades new bid 102 with print 100 fills taker at 100, re-price 99 -> 102 taker, resting through-print maker at limit with cost = fee; L1 and L1Trades re-price 99 -> 102 against 100/101 fills at 101 as taker, new bid equal to ask is taker in both fill modes, resting lock maker partial, walk stops at the limit, NaN-then-crossing is taker, size-only change is not arrival; batch == streaming and reset(); rewrite tests/test_backtest.py::test_l1_breach_no_fill_on_lock as a resting lock.
- CHANGELOG Fixed entry with before/after examples.
Commit: `fix(backtest): orders that cross on arrival fill as takers at the market`.

### Task 3: Spread estimator limitations and EdgeSpread

Files: docs/functions_micro/CorwinSchultzSpread.md, AbdiRanaldoSpread.md, RollSpread.md (Limitations sections); new include/screamer/edge_spread.h (or the micro module's layout), bindings, JS codegen/manifest, docs/functions_micro/EdgeSpread.md, tests and a small reference fixture; Guide page docs/guide/order-flow/spread-from-bars.md gains EdgeSpread in its example and prose (numbers re-derived).
- Tests first: EdgeSpread against the authors' reference outputs on their test data; a known-spread Monte Carlo showing EDGE much less biased than floored CS where the spread is small relative to volatility; batch == streaming; NaN handling as the other spread estimators; JS parity test.
- CHANGELOG Added entry; regenerate help.json and JS docs.
Commit: `feat(micro): EdgeSpread (EDGE estimator) and spread-estimator limitations`.

### Task 4: Final pass

- make install-dev; full tests/; npm test in js/; fresh sphinx-build -E -W --keep-going (only the known warning); tests/test_guide.py and the coverage gate (EdgeSpread must be used on a Guide page).
- Compatibility diff against the released 3.9.1 wheel (install it in a scratch venv; run a broad set of engine configurations on the committed data and synthetic edge cases; list every difference and confirm each is an approved change).
- CHANGELOG Unreleased reads as one coherent section.
Commit (if anything changed): `chore: final pass for engine fill fixes and EdgeSpread`.
