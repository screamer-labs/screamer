# screamer: Backtest PnL engines (signal and market-maker) - Design

**Status:** design, pending review
**Date:** 2026-07-17
**Topic:** risk & performance (alongside `Drawdown`, `MaxDrawdown`, `RollingInfoRatio`)

## Context

screamer produces signals but cannot turn one into a costed profit-and-loss
curve. Backtesting a signal (equity curve, transaction costs, drawdown,
turnover, trade count) is the natural next step and fits screamer's model:
mark-to-market PnL is causal, streaming, and batch == stream.

Two levels of realism are wanted, both in this release:

- a **signal** backtest: the common case, few parameters, small learning curve.
  You hold the position your signal asks for, filled at market.
- a **market-maker** backtest: market *and* limit orders, an explicit limit
  price, and a fill model, so passive (maker) strategies and rebates can be
  modelled.

The signal backtest is a subset of the market-maker one. Rather than duplicate
the money math, both engines share one C++ accounting core and differ only in how
fills are determined. Both emit the same four output streams, so the statistics
and the report helper are identical across the two.

## Goal

Two causal C++ functors, `BacktestSignal` and `BacktestMarketMaker`, over a shared
`detail::` accounting primitive, each emitting `[equity, pnl, position, cost]`.
Every performance statistic composes from those streams with existing operators. A
thin `backtest_report` helper wires the common bundle for either engine.

## Design principles

1. **Causal, no lookahead.** The position held into bar `t` is decided from data
   at or before `t`; a signal at `t` cannot earn PnL before it is known.
2. **batch == stream.** One per-sample step function drives both.
3. **Logic in C++.** The engines and the accounting are C++. The report helper is
   a thin Python wrapper (a `Pipeline` of existing operators plus reading final
   values), adding no operator logic (the latitude of `to_pandas`).
4. **Compose, do not duplicate.** Statistics are existing running operators on the
   outputs; the two engines share one accounting core.

## Shared accounting core (`detail::PnLAccount`)

A small C++ primitive that both engines call each bar. Given the mark price
`close_t`, the position change `dpos` executed this bar, its fill price
`fill_price`, and a per-notional `fee_rate`, it advances the account:

```
mark_pnl     = has_prev ? position_prev * (close_t - close_prev) : 0.0   # M2M on the held position
trade_cost   = dpos * (fill_price - close_t)          # adverse (or favorable) fill vs mark
             + |dpos| * fill_price * fee_rate         # explicit exchange fee (signed)
position_now = position_prev + dpos
pnl_t        = mark_pnl - trade_cost
cum_eq      += pnl_t
emit           [ cum_eq , pnl_t , position_now , trade_cost ]
position_prev = position_now ; close_prev = close_t ; has_prev = true
```

`dpos * (fill_price - close)` captures the spread economics with the right sign
for both sides and both fill types: a market buy fills above the mark (a cost); a
limit buy fills at or below it (a gain); selling is symmetric. `fee_rate` is
signed, so a maker rebate is negative. Outputs are always
`[equity, pnl, position, cost]`.

## Engine 1: `BacktestSignal` (the common case)

`FunctorBase<BacktestSignal, 2, 4>`.

**Inputs:** `(signal, price)`. `signal` is the target position in units (sign is
long / short / flat); `price` is the mark (mid).

**Parameters:** `spread` (fractional bid-ask, default `0.0`) and `fee` (fractional
taker fee, default `0.0`).

**Per bar:** you always achieve the target as a market order.
```
dpos       = signal_t - position_prev
side       = sign(dpos)
fill_price = price_t * (1 + side * spread/2)          # cross half the spread
account.step(close = price_t, dpos, fill_price, fee_rate = fee)
```
Default `spread=0, fee=0` is frictionless. Position always equals the signal, so
there is no fill uncertainty (that is the market-maker engine's job). First bar:
start flat, mark PnL 0, so only the entry cost is charged.

## Engine 2: `BacktestMarketMaker` (market and limit orders)

`FunctorBase<BacktestMarketMaker, 5, 4>`.

**Inputs:** `(target_position, limit_price, high, low, close)`.
- `target_position`: the position the strategy wants this bar.
- `limit_price`: the order's limit. `NaN` is a **market** order (the ergonomic
  spelling of an infinitely marketable price); a finite value is a limit order,
  which is a taker or a maker depending on whether it crosses (below).
- `high`, `low`: the bar range, used to decide passive (maker) fills.
- `close`: the mark price for M2M.

**Parameters:** `spread` (fractional bid-ask, default `0.0`), `taker_fee`
(fractional, default `0.0`), `maker_fee` (fractional, signed, default `0.0`;
negative is a rebate), and `fill` (`"touch"` or `"breach"`, default `"touch"`).

**Maker vs taker is a rule, not a mode.** A single order type (a limit at
`limit_price`) expresses both. With `h = close * spread/2` (so ask = `close + h`,
bid = `close - h`):
- **Taker** (cross the spread, pay `taker_fee`): a market order (`limit_price`
  NaN), or a *marketable* limit (buy `L >= close + h`, sell `L <= close - h`).
  Fills this bar at the ask (buy) or bid (sell), `close +/- h`.
- **Maker** (rest, earn the spread, `maker_fee`): a *passive* limit (buy
  `L < close + h`, sell `L > close - h`). Fills only if the bar reaches it, at
  `limit_price`.

**Per bar** (cancel-replace: each bar submits a fresh order, nothing rests
across bars):
```
dpos = target_position_t - position_prev
h    = close_t * spread/2
if dpos == 0:
    account.step(close_t, 0, close_t, 0)                     # no order, mark only
elif dpos > 0:                                               # buy
    if isnan(limit_price) or limit_price >= close_t + h:     # market / marketable -> taker
        account.step(close_t, dpos, close_t + h, taker_fee)  # lift the offer at the ask
    elif (touch: low_t <= limit_price) or (breach: low_t < limit_price):  # passive -> maker fill
        account.step(close_t, dpos, limit_price, maker_fee)
    else:
        account.step(close_t, 0, close_t, 0)                 # resting order not hit: hold
else:                                                        # sell (symmetric)
    if isnan(limit_price) or limit_price <= close_t - h:     # market / marketable -> taker
        account.step(close_t, dpos, close_t - h, taker_fee)  # hit the bid
    elif (touch: high_t >= limit_price) or (breach: high_t > limit_price):
        account.step(close_t, dpos, limit_price, maker_fee)
    else:
        account.step(close_t, 0, close_t, 0)
```
A taker fill crosses the spread (fills at `close +/- h`, so `dpos*(fill-close) =
|dpos|*h`, the half-spread cost) and pays `taker_fee`. A maker fill trades at the
favorable `limit_price` (a gain versus the mark) and pays `maker_fee` (a rebate if
negative). A passive order the bar never reaches simply holds the position.

**Causality.** Every decision at `t` uses only `t`'s inputs and the prior state;
fills use `t`'s high/low, which are known once the bar closes. No future data.

## NaN policy

`nan_policy: ignore` for both: a NaN in any input emits an all-NaN row and leaves
the state untouched (the bar is skipped, the position held across the gap). Batch
and streaming stay identical.

## Statistics compose from the outputs

Both engines emit `[equity, pnl, position, cost]`, so the same compositions serve
either. Each is a running series whose last value is the summary.

| Statistic | Composition (existing ops) |
|---|---|
| Total PnL | last of `equity` |
| Max drawdown (dollars) | `CumMin(equity - CumMax(equity))` |
| Cumulative transaction cost | `CumSum(cost)` |
| Volume turnover (units) | `CumSum(Abs(Diff(position)))` |
| Notional turnover | `CumSum(Abs(Diff(position)) * close)` |
| Number of trades | `CumSum(Sign(Abs(Diff(position))))` |
| Sharpe-like ratio | `RollingInfoRatio(pnl)`, or mean/std of `pnl` |

The equity curve is dollar PnL starting at `0`, so the ratio-based `Drawdown` does
not apply; dollar drawdown `equity - CumMax(equity)` (min via `CumMin`) is the
correct form and uses only existing operators.

## The convenience helper: `backtest_report`

`backtest_report(outputs)` takes an engine's `[equity, pnl, position, cost]`
outputs (from either engine) and returns the common statistics. It is a thin
Python wrapper around a `Pipeline`, so the shared streams are consumed once:

```
[equity, pnl, position, cost]
    -> equity                                    (curve)
    -> CumMin(equity - CumMax(equity))           (running max drawdown)
    -> CumSum(cost)                              (running cost)
    -> CumSum(Abs(Diff(position)))              (running turnover)
    -> CumSum(Sign(Abs(Diff(position))))        (running trade count)
```
It returns the running series (co-indexed, as a `pandas` frame) and the final-row
summary (a labeled `pandas` Series). It contains no operator logic.

## Correctness and testing

- **Shared accounting:** hand-computed example of `mark_pnl`, `trade_cost`, equity
  for a short path; the `dpos * (fill_price - close)` sign is correct for
  buy/sell and above/below the mark.
- **`BacktestSignal`:** frictionless (`spread=fee=0`) equals
  `CumSum(prev_position * price_change)`; taker cost equals `turnover * price *
  spread/2 + fee notional`; long and short; causality (a changed future signal
  leaves past rows unchanged); batch == stream; reset; NaN skip.
- **`BacktestMarketMaker`:** a market order (NaN limit) reproduces
  `BacktestSignal` given the same cost; a marketable limit is charged the taker
  fee and fills at `close +/- h` (not the limit); a passive limit fills at its
  price exactly when the range reaches it under `touch`, only when it trades
  through under `breach`, and pays the maker fee; an unfilled passive order holds
  position and marks only; a maker rebate (`maker_fee < 0`) increases equity on
  fills; causality and batch == stream.
- **`backtest_report`:** its statistics match composing the operators by hand; the
  summary equals the final row.
- **Docs:** reference pages for both engines with a plotted example (a signal to
  an equity curve, with and without cost, plus running drawdown for
  `BacktestSignal`; a limit strategy showing filled versus unfilled bars for
  `BacktestMarketMaker`) and a `backtest_report` page. One demo notebook that
  backtests a screamer signal end to end (both engines, the statistics table),
  asserting batch == stream.

## Scope

**In:** `detail::PnLAccount`, `BacktestSignal`, `BacktestMarketMaker`,
`backtest_report`, docs pages with plotted examples, tests, and one demo notebook.

**Out (documented, not built):** partial fills and queue-position modelling (fills
are all-or-nothing at bar level); multi-level book depth; multi-asset / portfolio
backtests (a portfolio is a sum of per-instrument PnLs the caller builds);
financing / borrow / funding carry; non-linear or per-trade fixed commissions;
position sizing (Kelly, vol targeting), which the caller applies to the signal
upstream.

## Resolved decisions

1. **Two engines over one accounting core.** `BacktestSignal` (simple, taker,
   position = signal) is the market-order subset of `BacktestMarketMaker`
   (market + limit, `touch`/`breach` fills); they share `detail::PnLAccount`.
2. **Signal is a position in units; PnL is dollars;** equity is a running sum.
3. **Cost via the fill price plus a signed fee.** `dpos * (fill_price - close)`
   carries the spread economics; `fee_rate` is signed so maker rebates are
   negative. `BacktestSignal`: `spread` + taker `fee`. `BacktestMarketMaker`:
   `spread`, `taker_fee`, `maker_fee` (signed), `fill` mode.
4. **One order type; maker vs taker is a marketability rule.** A market order
   (`limit_price` NaN) or a marketable limit (crosses the ask/bid) is a taker
   filled this bar at `close +/- h`; a passive limit is a maker filled at its
   price only if the bar reaches it. Fills are full, cancel-replace each bar;
   `touch` fills when the range reaches the limit, `breach` only when it trades
   through.
5. **Both emit `[equity, pnl, position, cost]`;** statistics compose from these,
   and `backtest_report` shares them through one `Pipeline` pass.
