---
name: choosing_a_backtest_engine
title: Choosing a backtest engine
kind: guide
short: The engine grid by data model and order definition, with fill rules, sizing, volume limits, and encoding.
topics:
- backtesting
---

# Choosing a backtest engine

Screamer's backtest engines are arranged in a `Backtest<DataModel><OrderDef>` grid.
For the practices that make a backtest trustworthy, and the tools for each, see
[Backtesting best practices](../backtesting_best_practices.md).
The data model (rows) is the market feed you have. The order definition (columns)
is the strategy's output contract. All nine engines share one accounting core,
the `[equity, pnl, position, cost]` output schema, a common MARKET/NaN/inf
encoding, and the static fill-cap rule.

## The grid

| Data model    | Target                        | Orders                          |
|:--------------|:------------------------------|:--------------------------------|
| Price         | `BacktestPriceTarget`         | not provided (*)                |
| OHLC          | `BacktestOHLCTarget`          | `BacktestOHLCOrders`            |
| OHLC + volume | `BacktestOHLCVTarget`         | not provided                    |
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
| `BacktestOHLCVTarget`  | 6      | `target`                                          | `open, high, low, close, volume` |
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

For `BacktestOHLCTarget`, the target decided on bar t executes as a market order
on bar t+1 at the open by default (causal deferral; no manual lag is needed).
`fill_at` and `delay` change the price and the bar, as described below. For
`BacktestOHLCOrders`, a NaN bid or ask price is a market order that fills at the
bar's open; a finite price rests and waits for the bar's range.

## Execution price, slippage, and cost

A taker (market) fill starts from a **reference price**: the fair price at
execution, before any friction.

| Engine family | Reference price of a taker fill |
|:--------------|:--------------------------------|
| Price         | `price`, then half of `spread` |
| OHLC          | The execution bar's `fill_at` price: `"open"` (default), `"close"`, `"typical"`, or `"ohlc4"` |
| Trades        | The print price |
| L1            | The opposite touch, walking one `tick_size` beyond the displayed size |

`slippage` then moves the fill against the trader: a buy fills higher and a sell
lower. One number is read in one unit, chosen with `slippage_unit`:

| `slippage_unit` | Offset | Example |
|:----------------|:-------|:--------|
| `"fraction"` (default) | `reference * slippage` | `slippage=5e-4` is 5 bps |
| `"ticks"` | `slippage * tick_size` | `slippage=2, tick_size=0.01` is 0.02 |
| `"price"` | `slippage` | `slippage=0.05` is 0.05 |

`tick_size` is the instrument's price grid (0 means continuous prices). When it is
set, the offset rounds up to whole ticks. A resting limit order never slips: it
fills at its own limit price. The examples at the end of this page show both
styles of call.

The `cost` column is the friction of each trade: the fill-versus-reference
difference (spread and slippage) plus fees. The move from the fill price to the
mark is profit or loss, not cost. A resting limit fill has the limit price as its
reference, so its cost is its fee alone. The split between `pnl` and `cost` does
not change `equity`.

## Risk exits

The four Target engines take `stop_loss`, `take_profit`, and `trailing_stop`:
fractions of the average entry fill price (0 turns an exit off). On OHLC bars a
bar that opens past a level exits at the open, and otherwise at the level; when
the stop and the take-profit both fall inside one bar, the stop is taken. The
event engines (price, trades, L1) exit on the event that crosses a level, at that
event's price. Exits are market orders that pay the taker fee and slippage. After
an exit, same-side targets are ignored until the target turns flat or reverses.

## Sizing and volume

The Target engines read the target as units by default. With `size="notional"`
it is a currency amount and with `size="weight"` a fraction of equity
(`initial_capital` plus the backtest's pnl), converted to units at the execution
price when the target value changes; `rebalance_band` trades back to the target
when the holding drifts. Two engines limit fills by volume, carrying the rest of
the order to later events: `BacktestOHLCVTarget` (`max_participation` of each
bar's volume) and `BacktestTradesTarget` (`max_participation` of each print's
size). A `NaN` target keeps working toward the last target.

## Fill-cap rule

Every engine accepts `min_position` and `max_position` constructor parameters
(default unbounded). All fills are capped by a three-way minimum: the order size,
the participation limit (where applicable), and the remaining room to the bound.
For a buy, the fill cannot exceed `max_position - current_position`; for a sell it
cannot exceed `current_position - min_position`. A fill that would breach the bound
is truncated to exactly the remaining room. Passing
`min_position=-8.0, max_position=8.0` enforces a hard inventory limit of eight
units on each side, independent of order size.

## Examples

### Slippage and execution price

```python
# Next bar's open, 5 bps of slippage, 2 bps taker fee
BacktestOHLCTarget(taker_fee=2e-4, slippage=5e-4)(target, o, h, l, c)

# Market-on-close of the signal bar, one tick of slippage on a 0.25 grid
BacktestOHLCTarget(fill_at="close", delay=0,
                   slippage=1, slippage_unit="ticks", tick_size=0.25)(target, o, h, l, c)
```

<!-- HELP_END -->
