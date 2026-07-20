---
name: choosing_a_backtest_engine
title: Choosing a backtest engine
kind: guide
short: A coverage matrix of all backtest engines by data model and order strategy.
topics:
- backtesting
---

# Choosing a backtest engine

Screamer ships seven backtest engines. They share one accounting core, one output
schema (`[equity, pnl, position, cost]`), and a common fill-cap and order-encoding
contract. The only difference between engines is the market data they consume. Pick
the engine whose data model matches your feed.

## Coverage matrix

Each cell names the engine to use for that combination of data model and order
strategy. An engine listed under "market" and "limit (directional)" handles both
order types; the `MARKET`/NaN encoding selects which one at call time (see below).

| Data model     | Market orders              | Limit (directional)        | Market-making              |
|:---------------|:---------------------------|:---------------------------|:---------------------------|
| Value series   | `BacktestPriceTarget`      | `BacktestPriceTarget` (coarse) | n/a                    |
| OHLC bars      | `BacktestOHLCTarget`       | `BacktestOHLCOrders`       | `BacktestOHLCOrders`       |
| Trade tape     | `BacktestTrades`           | `BacktestTrades`           | `BacktestTradesMaker`      |
| L1 quotes      | `BacktestL1`               | `BacktestL1`               | `BacktestL1`               |
| L1 + trades    | `BacktestL1Trades`         | `BacktestL1Trades`         | `BacktestL1Trades`         |

**Value series** (`BacktestPriceTarget`) takes a raw position signal and a scalar
price. Limit-order fidelity is coarse: the signal can encode a directional target,
but there is no bar range or tape to test a resting limit against.

**OHLC bars** split into two engines: `BacktestOHLCTarget` for directional
strategies that post a target position per bar (executes as a market order at the
next bar's open, causal), and `BacktestOHLCOrders` for two-sided strategies that
post simultaneous bid and ask quotes and earn the spread.

**Trade tape** splits the same way: `BacktestTrades` for directional resting orders,
`BacktestTradesMaker` for two-sided market-making driven by crossing prints.

**L1 quotes** and **L1 + trades** each cover all three strategies within a single
engine (`BacktestL1` and `BacktestL1Trades`). The quote stream marks the position;
`BacktestL1Trades` prefers explicit prints to drive fills and falls back to a
quote-cross heuristic only when no explaining trade arrives.

## Market-order encoding

Every engine accepts the same price encoding to distinguish a resting limit order
from a marketable order.

| Price value                 | Meaning                                                    |
|:----------------------------|:-----------------------------------------------------------|
| Finite number               | Resting limit order at that price (maker fill, `maker_fee`)|
| `+inf` / `screamer.MARKET`  | Market buy (taker); never-fill sell                        |
| `-inf`                      | Never-fill buy; market sell (taker)                        |
| `NaN`                       | Side-agnostic market order (taker, regardless of direction)|

`screamer.MARKET` equals `+math.inf` and is provided as a readable constant for
passing a market buy without writing `float('inf')` or `math.inf` directly.

For `BacktestOHLCTarget`, the target decided on bar t is always executed as a
market order at bar t+1's open. For `BacktestOHLCOrders`, a `NaN` bid or ask
price is a market order that fills at the bar's open; a finite price is a resting
limit that waits for the bar's range.

## Fill-cap rule

Every engine accepts `min_position` and `max_position` constructor parameters
(both default to unbounded). All fills are capped by a three-way minimum: the
order size, the participation limit (where applicable), and the remaining room
to the bound. For a buy, the fill cannot exceed `max_position - current_position`;
for a sell it cannot exceed `current_position - min_position`. A fill that would
breach the bound is truncated to exactly the remaining room. Passing
`min_position=-8.0, max_position=8.0` enforces a hard inventory limit of eight
units on each side, independent of order size.

<!-- HELP_END -->
