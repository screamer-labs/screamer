---
name: choosing_a_backtest_engine
title: Choosing a backtest engine
kind: guide
short: The 5x2 engine grid by data model and order definition, with fill rules and encoding.
topics:
- backtesting
---

# Choosing a backtest engine

Screamer's backtest engines are arranged in a `Backtest<DataModel><OrderDef>` grid.
The data model (rows) is the market feed you have. The order definition (columns)
is the strategy's output contract. All eight engines share one accounting core,
the `[equity, pnl, position, cost]` output schema, a common MARKET/NaN/inf
encoding, and the static fill-cap rule.

## The grid

| Data model    | Target                        | Orders                          |
|:--------------|:------------------------------|:--------------------------------|
| Price         | `BacktestPriceTarget`         | not provided (*)                |
| OHLC          | `BacktestOHLCTarget`          | `BacktestOHLCOrders`            |
| Trades        | `BacktestTradesTarget`        | `BacktestTradesOrders`          |
| L1            | `BacktestL1Target`            | `BacktestL1Orders`              |
| L1 + Trades   | not provided (**)             | `BacktestL1TradesOrders`        |

(*) `BacktestPriceOrders` is not provided. A raw value series carries no bid/ask
book, so there is no credible fill model for a two-sided quote. Use
`BacktestOHLCOrders` or `BacktestL1Orders` when a richer feed is available.

(**) `BacktestL1TradesTarget` is not provided. A Target engine is a pure taker: it
hits the book immediately on each event. For L1 + trades data,
`BacktestL1Target` already does this with only the quote stream, and adding the
trade feed does not improve it (trades are taker events, not maker fills).
Use `BacktestL1Target` when you have an L1 + trades feed and want a Target engine.

## Order definition interfaces

**Target** engines receive `(target_position, <market cols>)`. On each event the
engine computes `delta = clamp(target, cap) - position` and submits a marketable
order for that delta, taking liquidity at the current market price. The target is
the desired inventory level, not an order size. The engine handles the sizing and
direction internally.

**Orders** engines receive `(bid_price, bid_size, ask_price, ask_size, <market cols>)`.
The strategy posts a two-sided resting quote each event. Either or both sides can
fill on the same event. Setting a side's size to zero or its price to NaN disables
that side. A quote submitted already crossing the spread (`bid_price >= market_ask`
or `ask_price <= market_bid`) is treated as a taker and fills immediately.

Input arities by engine:

| Engine                 | Inputs | Order columns                                     | Market columns               |
|:-----------------------|:-------|:--------------------------------------------------|:-----------------------------|
| `BacktestPriceTarget`  | 2      | `target`                                          | `price`                      |
| `BacktestOHLCTarget`   | 5      | `target`                                          | `open, high, low, close`     |
| `BacktestOHLCOrders`   | 8      | `bid_price, bid_size, ask_price, ask_size`        | `open, high, low, close`     |
| `BacktestTradesTarget` | 3      | `target`                                          | `trade_price, trade_size`    |
| `BacktestTradesOrders` | 6      | `bid_price, bid_size, ask_price, ask_size`        | `trade_price, trade_size`    |
| `BacktestL1Target`     | 5      | `target`                                          | `market_bid, market_ask, market_bid_size, market_ask_size` |
| `BacktestL1Orders`     | 8      | `bid_price, bid_size, ask_price, ask_size`        | `market_bid, market_ask, market_bid_size, market_ask_size` |
| `BacktestL1TradesOrders` | 10   | `bid_price, bid_size, ask_price, ask_size`        | `market_bid, market_ask, market_bid_size, market_ask_size, trade_price, trade_size` |

For `BacktestL1Orders` and `BacktestL1TradesOrders` the own-quote columns come
first, then the market book columns.

## MARKET/NaN/inf encoding

Every engine uses the same price encoding to distinguish a resting limit order from
a marketable order.

| Price value                | Meaning                                                     |
|:---------------------------|:------------------------------------------------------------|
| Finite number              | Resting limit order at that price (maker fill, `maker_fee`) |
| `+inf` / `screamer.MARKET` | Market buy (taker); never-fill sell                         |
| `-inf`                     | Never-fill buy; market sell (taker)                         |
| `NaN`                      | Side-agnostic market order (taker, regardless of direction) |

`screamer.MARKET` equals `+math.inf` and is provided as a readable constant for
submitting a market buy without writing `float('inf')` directly.

For `BacktestOHLCTarget`, the target decided on bar t always executes as a market
order at bar t+1's open (causal deferral; no manual lag is needed). For
`BacktestOHLCOrders`, a NaN bid or ask price is a market order that fills at the
bar's open; a finite price rests and waits for the bar's range.

## Fill-cap rule

Every engine accepts `min_position` and `max_position` constructor parameters
(default unbounded). All fills are capped by a three-way minimum: the order size,
the participation limit (where applicable), and the remaining room to the bound.
For a buy, the fill cannot exceed `max_position - current_position`; for a sell it
cannot exceed `current_position - min_position`. A fill that would breach the bound
is truncated to exactly the remaining room. Passing
`min_position=-8.0, max_position=8.0` enforces a hard inventory limit of eight
units on each side, independent of order size.

<!-- HELP_END -->
