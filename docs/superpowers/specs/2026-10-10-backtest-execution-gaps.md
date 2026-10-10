# Backtest engines: current state and missing execution modes

Status: F1-F5 implemented (2.6.0 and later): the cost fix, `slippage` /
`slippage_unit`, `fill_at` / `delay`, risk exits, `BacktestStats` and
`periods_per_year`, and `BacktestCarry` (carry as an overlay node rather than an
engine input). F6 and F7 wait for the decisions in
`2026-10-10-backtest-open-decisions.md`.

## Current state

There are eight engines and two reports, all in C++. Every engine returns
`[equity, pnl, position, cost]`. Every engine also accepts `(T, K)` inputs and
returns `(T, K, 4)`, so K strategies or K assets run in one call.

| Engine | Inputs | Execution |
|---|---|---|
| `BacktestPriceTarget` | target, price | Same-step fill at `price * (1 +/- spread/2)` plus a `fee`. Target from step t is held from t to t+1. |
| `BacktestOHLCTarget` | target, O, H, L, C | Target from bar t fills at **bar t+1's open +/- `tick_size`** (taker). |
| `BacktestOHLCOrders` | bid p/s, ask p/s, O, H, L, C | Resting limits fill on touch or breach of low or high. A marketable order fills at open +/- tick. |
| `BacktestTradesTarget` / `Orders` | target or quotes, trade p/s | Fills against prints. |
| `BacktestL1Target` / `Orders` | target or quotes, L1 book | Fills against the displayed book. |
| `BacktestL1TradesOrders` | quotes, L1 book, trades | Market-making engine. Trades drive fills. |
| `BacktestReport`, `PortfolioReport` | engine output | Drawdown, cumulative cost, turnover, trade count, max drawdown, running Sharpe. |

Shared parameters: `maker_fee` and `taker_fee` (fraction of notional),
per-contract fees, `multiplier`, `min_position`/`max_position`, `fill`
(`"touch"` or `"breach"`), and `participation_ratio`.

The engines handle order-type fill logic carefully, and the fill logic is well
tested (57 tests). Coverage is thinner in three areas: when and at what price a
target order executes, risk exits, and the statistics in the report.

## "Buy at the next bar open with a bit of slippage"

This is already possible:

```python
BacktestOHLCTarget(taker_fee=0.0005, tick_size=0.01)(target, o, h, l, c)
```

Verified: a target of 1 decided on bar 0 fills at `open[1] + tick_size`.
However, users will not find it, for three reasons.

- Slippage is called `tick_size` and is an **absolute** price offset. A
  relative slippage in basis points, which is the common way to state it,
  does not exist.
- Fill and cost parameters are named differently across engines:
  `BacktestPriceTarget(spread, fee)` but
  `BacktestOHLCTarget(taker_fee, tick_size)`.
- Next-bar open is the only option. Next-bar close, a typical-price proxy,
  and a delay of more than one bar are all missing.

## Findings

### F1. On OHLC engines, the `cost` column includes the intrabar price move (correctness)

`PnLAccount::step` defines cost as `dpos * (fill_price - mark) + fees`. On OHLC
engines the mark is the bar's close, so the move from the open to the close is
booked as cost, not as PnL. Verified: a buy at `open + tick = 101.05` on a bar
that closes at 101.5 reports `cost = -0.45`. As a result, `cum_cost` and
`total_cost` in `backtest_report` can be negative, or much larger than the true
cost. Equity is still correct; only the split between PnL and cost is wrong.

The same definition is correct for `BacktestPriceTarget`, because there the fill
reference and the mark are the same price.

**Fix:** measure cost against a reference price: the bar's open for OHLC, the
mid for L1, the print for trades. Then
`cost = dpos * (fill - reference) + fees`, and the move from reference to mark
goes into PnL. Equity does not change, but the meaning of the `cost` column does
change, and tests encode the current meaning (`test_backtest.py:239, 381,
483`). This needs a version decision (see Q1).

### F2. Missing execution timing and price modes

Add these parameters to `BacktestOHLCTarget`, and the ones that apply to
`BacktestPriceTarget`:

- `fill_at`: which price of the execution bar to fill at.
  - `"open"`: the default and the current behaviour.
  - `"close"`: market-on-close of the next bar.
  - `"typical"` (H+L+C)/3 or `"ohlc4"`: average-price proxies, for when a
    VWAP is not available.
- `delay`: the number of bars between the decision and the execution
  (default 1). A delay of 0 is allowed only with `fill_at="close"`; that is
  market-on-close on the signal bar, which is what `BacktestPriceTarget` does
  now.
- `slippage`: a fractional adverse price offset applied on top of the fill
  price (for example `slippage=5e-4` is 5 bps). It applies alongside the
  absolute offset, which is renamed from `tick_size` to `slippage_ticks`, with
  `tick_size` kept as a deprecated alias.

### F3. No risk exits

The engines have no stop-loss, take-profit, or trailing stop. Most backtest
libraries offer them as standard options (for example vectorbt's `sl_stop`,
`tp_stop`, and `tsl_stop`). On bars, a stop must handle gaps correctly:

- A sell stop at S triggers when `low <= S`.
- It fills at `min(open, S) - slippage`. If the bar opens below the stop, the
  fill is the open.

Proposal: add fractional `stop_loss`, `take_profit`, and `trailing_stop`
parameters to `BacktestOHLCTarget` (and to the price and trades target
engines). An exit flattens the position. It stays flat until the target changes
sign or returns to zero, so the engine does not re-enter on the next bar. If
the stop and the take-profit both trigger inside one bar, the order of events
is unknown, so take the conservative case and fill the stop. This is all state
inside the engine, so it streams normally.

### F4. Report statistics are minimal

Missing: annualisation (`periods_per_year`), Sortino, Calmar, exposure (time in
market), hit rate, profit factor, average win and loss, average holding period,
and per-round-trip trade statistics. The Sharpe ratio is per step and not
annualised, which users misread. All of these are causal accumulators, so they
fit as extra output columns on `BacktestReport`, or as a sibling node
`BacktestTradeStats` that detects round trips from the position column.

### F5. No carry or financing

Perpetual funding, short borrow cost, and interest on cash are not modelled.
Proposal: an optional per-step `carry` rate input, charged on position
notional. For crypto perpetuals this is essential; without it, PnL on held
positions is overstated.

### F6. No sizing in notional or weight terms

Targets are in units. A target stated as a weight (fraction of equity) needs the
running equity. Users cannot compute that upstream without a circular
dependency. Proposal: `size="units" | "notional" | "weight"` with an
`initial_capital` parameter. This also gives a returns-based equity curve, so a
percentage drawdown becomes meaningful.

### F7. No volume participation limit for target engines

`BacktestOHLCTarget` fills the whole order at the open regardless of bar
volume. Proposal: an optional volume input and `max_participation`. The engine
fills at most `max_participation * volume` per bar and carries the remainder
to the next bar. Lower priority.

## Priority

| # | Item | Why | Scope and risk |
|---|---|---|---|
| 1 | F1 cost reference fix | Reported cost is wrong on OHLC | Small code change; changes the meaning of an output |
| 2 | F2 `fill_at`, `delay`, `slippage` | This is the user's request; consistent naming | Small to medium; additive |
| 3 | F3 stop, take-profit, trailing stop | Standard backtest feature | Medium; gap and same-bar ordering need careful tests |
| 4 | F4 report statistics and annualisation | Users misread the per-step Sharpe | Small to medium; additive |
| 5 | F5 carry | Needed for perpetuals | Small; adds an input |
| 6 | F6 sizing modes | Compounding and percentage metrics | Medium |
| 7 | F7 volume participation | Realism for large orders | Medium |

Every item goes in C++ in the shared fill cores (`detail/ohlc_fill.h` and
`detail/target_front.h`). Python and JS stay thin bindings, and each new
parameter gets the JS codegen entry so the two bindings stay in parity. Each
item needs tests for batch == stream, a causality check (changing future inputs
must not change past rows), reset, and NaN skipping.

## Open questions

- **Q1.** F1 changes the meaning of the `cost` column. Should it ship as a
  bug fix in a minor version, or wait for 3.0? Recommendation: a minor
  version. Equity is unchanged, and the current numbers are wrong.
- **Q2.** Should `tick_size` be renamed to `slippage_ticks` with a deprecated
  alias, or kept as it is?
