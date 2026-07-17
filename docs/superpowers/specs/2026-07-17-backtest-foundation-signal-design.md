# screamer: Backtest family - foundation and the signal engine (sub-project 1)

**Status:** design, pending review
**Date:** 2026-07-17
**Topic:** risk & performance (alongside `Drawdown`, `MaxDrawdown`, `RollingInfoRatio`)

## Context

screamer produces signals but cannot turn one into a costed profit-and-loss
curve. Backtesting fits its model: mark-to-market PnL is causal, streaming, and
batch == stream. But "backtest" is not one thing. Because screamer is an async,
multi-clock event system (`merge` / `combine_latest` / `Pipeline`), the most
realistic backtest is event-driven, and a bar-sampled one is a simpler special
case. This is a **family**, so we design a shared spine first and add engines as
their own sub-projects.

**This spec is sub-project 1: the spine plus the simplest engine** (`BacktestSignal`).
It ships the common directional case now and locks the interfaces every later
engine reuses.

## The family and its spine

Every engine, whatever its clock or strategy interface, ends the same way:

```
<fill model>  ->  detail::PnLAccount  ->  [equity, pnl, position, cost]  ->  same stats + backtest_report
```

One accounting core, a **pluggable fill model**, one output schema. Engines are
named by their **input data model**, and the interface / fill fidelity match the
data richness (the sophisticated order model only appears where the data supports
it). This is a **suite** so users at every data level get a sensible backtest:

| Engine | Input data | Interface | Fill model | Sub-project |
|---|---|---|---|---|
| **`BacktestSignal`** | value series `(signal, price)` | target position | full, market (taker) | **1 (this spec)** |
| **`BacktestL1`** | L1 quotes `(bid, ask, bid_size, ask_size)` events | two-sided quotes + market | crossing, partial up to L1 size, exact maker/taker | 2 (own spec, the prize) |
| **`BacktestTrades`** | trade tape `(price, size)` events + order stream | limit / market / cancel | crossing on a print, partial up to trade size, touch/breach | 3 (own spec) |
| **`BacktestOHLC`** | bars `(open, high, low, close)` | directional limit + market | limit maker (touch/breach at the limit price, gap-aware), market taker at next open | 4 (own spec, lean) |

**Fill stream (extra output).** The three market-data engines emit, in addition
to the shared account, a **fill-event stream** `(price, size, side, maker/taker,
fee)`. The shared `[equity, pnl, position, cost]` is the running aggregate of
those fills through `detail::PnLAccount`; fill-level detail is an engine-specific
extra, not a schema change.

**Notes fixed during design:**
- **`BacktestL1` is the realistic, market-making engine** (real bid/ask/sizes,
  exact maker vs taker, partial fills, two-sided quoting). It is the top of the
  fidelity range; deeper book / queue-position is documented, not built.
- **`BacktestOHLC` stays lean and directional** (limit + market orders, no
  two-sided market-making): a bar has no intra-bar path, so maker/taker is
  inferred from order type against the `spread` param, market orders fill at the
  **next open** (no lookahead), and limits fill on touch/breach at the limit
  price. Two-sided MM fidelity belongs in `BacktestL1`.

**Build order:** sub-project 1 (this) proves the spine on the directional engine
and fixes `detail::PnLAccount`, the output schema (including the optional fill
stream), the statistics, and `backtest_report`, so 2-4 are additive (a new fill
model feeding the same core), not rewrites. Priority after 1 is `BacktestL1`.

## Design principles

1. **Causal, no lookahead.** The position held into bar `t` is decided from data
   at or before `t`.
2. **batch == stream.** One per-sample step drives both.
3. **Logic in C++.** Engines and accounting are C++; `backtest_report` is a thin
   Python `Pipeline` wrapper that reads final values (the latitude of `to_pandas`),
   adding no operator logic.
4. **Compose, do not duplicate.** Statistics are existing running operators on the
   outputs; engines share one accounting core.

## Shared accounting core (`detail::PnLAccount`)

The C++ primitive every engine calls to advance the account, given the mark price
`close`, the position change `dpos` executed, its `fill_price`, and a signed
per-notional `fee_rate`:

```
mark_pnl     = has_prev ? position_prev * (close - close_prev) : 0.0   # M2M on the held position
trade_cost   = dpos * (fill_price - close)          # adverse (+) or favorable (-) fill vs mark
             + |dpos| * fill_price * fee_rate        # explicit fee (signed; rebate is negative)
position_now = position_prev + dpos
pnl          = mark_pnl - trade_cost
cum_equity  += pnl
emit           [ cum_equity , pnl , position_now , trade_cost ]
position_prev = position_now ; close_prev = close ; has_prev = true
```

`dpos * (fill_price - close)` carries the spread economics with the correct sign
for both sides and any fill type (a market buy fills above the mark, a passive
limit buy at or below it), so the same core serves the future order-based engines.
Outputs are always `[equity, pnl, position, cost]`.

## The engine (this sub-project): `BacktestSignal`

`FunctorBase<BacktestSignal, 2, 4>`.

**Inputs:** `(signal, price)`. `signal` is the target position in units (sign is
long / short / flat, any magnitude); `price` is the mark (mid).

**Parameters:** `spread` (fractional bid-ask, default `0.0`) and `fee` (fractional
taker fee, default `0.0`). This engine is directional and always crosses to reach
the target, so it is a taker; maker economics belong to the order-based engines.

**Per bar:**
```
dpos       = signal_t - position_prev
side       = sign(dpos)
fill_price = price_t * (1 + side * spread/2)          # cross half the spread
account.step(close = price_t, dpos, fill_price, fee_rate = fee)
```
Default `spread = 0, fee = 0` is frictionless. Position always equals the signal
(no fill uncertainty; that is `BacktestEvents`' job). First bar: start flat, mark
PnL `0`, so only the entry cost is charged. There is no warmup NaN.

**Causality.** `signal_t` enters PnL only at `t+1` (through `position_prev`), so a
future signal never changes a past row. This is the signal-at-close,
hold-next-bar convention.

**NaN (`nan_policy: ignore`).** A NaN `signal` or `price` emits an all-NaN row and
leaves the state untouched (the bar is skipped, the position held across the gap).

## Output schema and statistics

Every engine emits `[equity, pnl, position, cost]` (positional columns). Each
summary statistic is an existing running operator on these; because screamer is
streaming, you get its whole evolution, and the last value is the summary.

| Statistic | Composition (existing ops) |
|---|---|
| Total PnL | last of `equity` |
| Max drawdown (dollars) | `CumMin(equity - CumMax(equity))` |
| Cumulative transaction cost | `CumSum(cost)` |
| Volume turnover (units) | `CumSum(Abs(Diff(position)))` |
| Notional turnover | `CumSum(Abs(Diff(position)) * price)` |
| Number of trades | `CumSum(Sign(Abs(Diff(position))))` |
| Sharpe-like ratio | `RollingInfoRatio(pnl)`, or mean/std of `pnl` |

The equity curve is dollar PnL starting at `0`, so the ratio-based `Drawdown`
(`price/peak - 1`) does not apply; dollar drawdown `equity - CumMax(equity)` (min
via `CumMin`) is the correct form and uses only existing operators.

## The convenience helper: `backtest_report`

`backtest_report(outputs)` takes any engine's `[equity, pnl, position, cost]` and
returns the common statistics. It is a thin Python wrapper around a `Pipeline`, so
the shared streams are consumed once:

```
[equity, pnl, position, cost]
    -> equity                                    (curve)
    -> CumMin(equity - CumMax(equity))           (running max drawdown)
    -> CumSum(cost)                              (running cost)
    -> CumSum(Abs(Diff(position)))               (running turnover)
    -> CumSum(Sign(Abs(Diff(position))))         (running trade count)
```
It returns the running series (co-indexed, as a `pandas` frame) and the final-row
summary (a labeled `pandas` Series). It contains no operator logic and works for
every engine in the family.

## Correctness and testing

- **Shared accounting:** hand-computed `mark_pnl`, `trade_cost`, equity for a short
  path; the `dpos * (fill_price - close)` sign is correct for buy/sell and above /
  below the mark.
- **`BacktestSignal`:** frictionless (`spread=fee=0`) equals
  `CumSum(prev_position * price_change)`; taker cost equals `turnover * price *
  spread/2` plus the fee on notional; long and short; causality (a changed future
  signal leaves past rows unchanged); batch == stream; reset; NaN skip.
- **`backtest_report`:** its statistics match composing the operators by hand; the
  summary equals the final row.
- **Docs:** a `BacktestSignal` reference page with a plotted example (a signal to an
  equity curve, with and without cost, plus running drawdown) and a
  `backtest_report` page. One demo notebook that backtests a screamer signal end to
  end (signal -> equity -> the statistics table), asserting batch == stream.

## Scope

**In (this sub-project):** `detail::PnLAccount`, `BacktestSignal`,
`backtest_report`, the statistics compositions (documented, using existing ops),
docs pages with a plotted example, tests, and one demo notebook.

**Out (their own specs / not built):**
- `BacktestL1`, `BacktestTrades`, `BacktestOHLC`: separate sub-projects, each its
  own spec. The spine here (accounting core, output schema, optional fill stream,
  statistics, report) is designed so they are additive.
- Queue-position modelling and multi-level book depth (`BacktestL1` is the
  top-of-book ceiling); multi-asset / portfolio backtests (a portfolio is a sum of
  per-instrument PnLs the caller builds); financing / funding carry; fixed
  per-trade commissions; position sizing (Kelly, vol targeting), which the caller
  applies to the signal upstream.

## Resolved decisions

1. **A backtest suite with one spine.** `detail::PnLAccount` + the
   `[equity, pnl, position, cost]` schema (+ an optional fill-event stream for the
   market-data engines) + composed statistics + `backtest_report` are shared;
   engines are pluggable fill models named by input data model (`BacktestSignal`,
   `BacktestL1`, `BacktestTrades`, `BacktestOHLC`), with interface / fidelity
   matched to the data.
2. **This sub-project ships `BacktestSignal`** (value series, directional, taker).
3. **Signal is a position in units; PnL is dollars;** equity is a running sum.
4. **Cost via the fill price plus a signed fee;** `BacktestSignal` is a taker
   (`spread` + `fee`). Maker economics and rebates live in the market-data engines.
5. **`BacktestL1` is the realistic prize** (real bid/ask/sizes, exact maker/taker,
   partial fills, two-sided quoting) and is sub-project 2. `BacktestTrades` (tape,
   sub-project 3) and a lean directional `BacktestOHLC` (bars, sub-project 4)
   follow, each its own spec.
