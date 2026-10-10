# Backtest: open decisions (sizing and volume limits)

Status: decided and implemented, 2026-10-10. D1: option B plus the optional
band (`size`, `initial_capital`, `rebalance_band` on all target engines). D2:
option A (`BacktestOHLCVTarget` with `max_participation`), plus
`max_participation` per print on `BacktestTradesTarget`; square-root impact
deferred. Everything else in
`2026-10-10-backtest-execution-gaps.md` is done (F1-F5).

## D1. Sizing by notional or by fraction of equity (F6)

Today a target is in units. A target stated as a notional amount or as a
fraction of equity has to be converted to units at execution, because the
conversion uses that bar's price (and, for a fraction, the running equity, which
only the engine knows).

The open question is what happens on bars where the target does not change but
the price does. A constant 50% weight is a different number of units every bar.

| Option | Behaviour | Pros | Cons |
|---|---|---|---|
| A. Rebalance every bar | Units follow the weight exactly | Matches "hold 50%" literally | A trade every bar: costs and turnover on noise |
| B. Rebalance only when the target changes (recommended) | Units are fixed when the target value changes and held until it changes again | No churn; matches how signals are usually traded | Weight drifts with the price between signals |
| C. Rebalance when the weight drifts by more than a band | `rebalance_band=0.05` trades back when the weight is 5 points off | Realistic for portfolios | One more parameter |

Recommendation: B by default, with C as an optional `rebalance_band`
(0 disables it). API: `size="units" | "notional" | "weight"` and
`initial_capital` on the target engines; equity for `"weight"` is
`initial_capital + equity`.

## D2. Volume participation for target engines (F7)

A large order filled entirely at one bar's open is optimistic. A participation
limit caps each bar's fill at a fraction of that bar's volume and carries the
rest to the next bar. That needs the bar's volume as an input, which changes the
engine's input count.

| Option | API | Pros | Cons |
|---|---|---|---|
| A. New engine `BacktestOHLCVTarget` (recommended) | 6 inputs: target, open, high, low, close, volume; `max_participation` | Existing engines and their callers unchanged; clear name | One more engine in the grid |
| B. Optional sixth input on `BacktestOHLCTarget` | Arity depends on a parameter | One engine | Input count that changes with a parameter breaks the fixed-arity node contract and the DAG |
| C. Square-root impact cost instead of a hard cap | `impact=k`: cost grows with sqrt(order / volume) | Smooth, common in practice | Also needs volume; models cost, not partial fills |

Recommendation: A, with `max_participation` (hard cap and carry-over), and C's
square-root impact as a later optional parameter on the same engine.
