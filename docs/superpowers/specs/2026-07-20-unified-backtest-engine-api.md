# Unified backtest engine API

## Context

screamer ships five backtest engines, and they disagree on which trading
strategies they support:

| Engine | data model | market | limit (directional) | market-making | inventory cap |
|---|---|---|---|---|---|
| `BacktestSignal` | value series | yes | no | no | no |
| `BacktestOHLC` | OHLC bars | yes | yes | no | no |
| `BacktestTrades` | trade tape | via aggressive price | yes | no | no |
| `BacktestL1` | L1 quotes | yes | yes | yes | yes |
| `BacktestL1Trades` | L1 + trades | yes | yes | yes | yes |

Using them for pairs research (screamer-bots) surfaced the inconsistency
concretely: a two-sided passive maker on OHLC bars (post `ref - half` bid and
`ref + half` ask, fill on the bar's low/high breach, bound the inventory) has no
engine. Market-making and the inventory cap exist only where there is quote data,
and each engine exposes a different order interface.

The engines already agree on one thing: they all return
`[equity, pnl, position, cost]` and share `detail::PnLAccount`. The redesign
extends that agreement to the inputs, so every engine accepts the same order and
risk controls and differs only in its market-data columns. screamer is pre-1.0, so
this is an allowed breaking change.

## Goals

Make four concerns orthogonal, each expressed identically on every engine:

1. **Data model** (which fill simulation runs): the only thing that differs
   between engines.
2. **Order intent** (what the strategy posts): one universal two-sided quote that
   spells market, limit, and market-making.
3. **Position management** (target and inventory cap): one `[min, max]` band.
4. **Cost and fill fidelity**: fees, slippage, and the touch/breach and
   participation knobs.

The output stays `[equity, pnl, position, cost]`.

## Non-goals

- L2 / L3 (full-depth) engines. Future work; the primitive here is two-sided
  (one bid, one ask), not a ladder.
- Changing `detail::PnLAccount`, `backtest_report`, or the `BacktestReport` node.
- The `SchmittTrigger` warmup NaN (a separate, related finding; see below).

## Axis 2: the universal order primitive

Per event the strategy posts at most two resting orders, a buy and a sell:

```
bid_price, bid_size,   ask_price, ask_size
```

**Size gates presence.** `size <= 0` or `NaN` size means no order on that side;
the price is then ignored.

**A market order is a limit at the maximally aggressive price.** This is the key
idea: market is not a flag, it is a price, so the direction falls out of the
normal fill comparison and needs no special case.

| `size` | `price` | bid (buy) | ask (sell) |
|---|---|---|---|
| `0` / `NaN` | any | no order | no order |
| `> 0` | finite | limit buy at `price` | limit sell at `price` |
| `> 0` | `+inf` | market buy (clears any offer) | limit at `+inf`, never fills |
| `> 0` | `-inf` | limit at `-inf`, never fills | market sell (hits any bid) |
| `> 0` | `NaN` | market buy | market sell |

A buy limit at `price` fills when the market reaches `<= price`, so `+inf` always
fills (market) and `-inf` never does; the sell side is the mirror. `+inf`/`-inf`
therefore need no handling at all. The only substitution is the side-agnostic
`NaN`: on a bid it becomes `+inf`, on an ask it becomes `-inf`. A wrong-direction
infinity (`-inf` bid, `+inf` ask) is a harmless never-fill limit, not a surprise
market order.

`screamer.MARKET` is a convenience constant (backed by `inf`) for the common
spelling; `NaN` and `+inf`/`-inf` are all accepted. `screamer.MARKET` on a bid is
`+inf`, on an ask is `-inf` (the constant resolves per side, or use `NaN` to avoid
thinking about direction).

**Maker vs taker follows from the price.** An order that is marketable on
submission (crosses immediately) is a taker and pays `taker_fee`; a resting limit
filled later is a maker and pays `maker_fee`.

## Axis 3: position management as a band

Two more inputs carry the position controls, per event so they can vary in time:

```
min_position, max_position
```

Fills are clamped so the position stays inside `[min_position, max_position]`; a
fill that would breach a bound is truncated to the room left. The band expresses
both position concerns that used to be separate features:

| intent | band |
|---|---|
| **Directional target `X`** | `min = max = X` (no room once reached; a moving `X` retargets) |
| **Market-maker inventory limit** | `min = -cap, max = +cap` (a range the quotes fill within) |
| **Unbounded** | `min = -inf, max = +inf` |

So "target position" and "inventory cap", previously a `BacktestSignal` idea and a
`BacktestL1` parameter, are the same `[min, max]` band.

## Axis 4: cost and fill fidelity (static config)

Every engine takes the same static parameters:

- `maker_fee`, `taker_fee`: fractional fees; negative for a rebate.
- `tick_size`: price a marketable order walks per unit of size beyond the
  displayed quote (the taker slippage model).
- `fill`: `"touch"` or `"breach"`, the optimistic/conservative limit-fill rule.
- `participation_ratio`: the fraction of the at-price volume a limit captures.

## The unified contract

Every engine is:

```
[market-data columns] + [bid_price, bid_size, ask_price, ask_size,
                         min_position, max_position]
    ->  [equity, pnl, position, cost]
```

with the static config above. Only the market-data prefix differs:

| Engine | market-data columns | total inputs |
|---|---|---|
| `BacktestValue` | `price` | 7 |
| `BacktestOHLC` | `open, high, low, close` | 10 |
| `BacktestTape` | `trade_price, trade_size` | 8 |
| `BacktestL1` | `market_bid, market_ask, market_bid_size, market_ask_size` | 10 |
| `BacktestL1Trades` | `market_bid, market_ask, market_bid_size, market_ask_size, trade_price, trade_size` | 12 |

The order/bounds columns are named to avoid colliding with the L1 market quote
(the market's book is `market_bid`/`market_ask`; the strategy's orders are
`bid`/`ask`).

Each engine's fill simulation is the only thing that differs:

- **Value**: a limit fills if the price crosses it between events; a market fills
  at the price. (One-price series, so limit fidelity is coarse.)
- **OHLC**: market fills at the open (plus half `spread`); a limit fills on the
  bar's low/high reaching it (`touch`/`breach`).
- **Tape**: a limit fills when a print crosses it, up to `participation_ratio` of
  the print; a through-print sweeps the order.
- **L1**: a resting quote fills when the opposite side of the market reaches it; a
  marketable order takes the displayed size plus `tick_size` overflow.
- **L1+trades**: trades drive the fills; a quote cross with no explaining trade is
  the run-over fallback.

Market-making on bars and on the tape now exist by construction: they are the same
two-sided order primitive, only the fill simulation changes.

## Convenience wrappers

The full form is verbose for everyday research, so thin wrappers expand to it and
keep the common cases one-liners:

- `BacktestSignal(signal, price)`: sets `min = max = signal`, `bid_price = MARKET`,
  `ask_price = MARKET` (market-to-target). Unchanged for the caller.
- `BacktestOHLC.directional(target, limit, o, h, l, c)`: `min = max = target`, one
  limit side toward the target.
- `BacktestL1.maker(...)`, `BacktestOHLC.maker(...)`: pass a two-sided quote and a
  static inventory band.

Wrappers are the only place the "market = aggressive price" and "target = band"
conventions are hidden; the raw engines are uniform.

## Coverage, before and after

The redesign closes the matrix. Rows are the data model; columns are the order
strategy; every cell is now covered:

| Data model | market | limit (directional) | market-making |
|---|---|---|---|
| Value series | yes | (coarse) | yes |
| OHLC bars | yes | yes | **yes (was gap)** |
| Trade tape | yes | yes | **yes (was gap)** |
| L1 quotes | yes | yes | yes |
| L1 + trades | yes | yes | yes |

This matrix, plus a short "choosing a backtest engine" overview, goes on a
hand-written docs page linked from the Backtesting group (the group page is
auto-generated, so the matrix lives beside it).

## Migration

- The five engines adopt the unified input contract. Their current signatures
  become the convenience wrappers where they map (`BacktestSignal`,
  `BacktestOHLC` directional), or are re-expressed in the new form.
- `BacktestTrades` is renamed `BacktestTape` for consistency with the data-model
  naming; `BacktestSignal` may become `BacktestValue` with `BacktestSignal` kept
  as the wrapper name.
- `detail::PnLAccount`, `backtest_report`, and `BacktestReport` are unchanged.
- The unreleased `[Unreleased]` changelog entry for the backtest suite is rewritten
  to describe the unified contract rather than the five divergent engines.

## Related finding (separate spec)

`SchmittTrigger` emits `NaN` during warmup when the signal starts inside its
hysteresis band, forcing a `nan_to_num` at the call site. A small follow-up gives
it an `initial` state (or the existing `start_policy`) so warmup resolves to the
low/flat state. Out of scope here; noted so it is not lost.

## Validation

- Each engine reproduces its old behavior through the convenience wrappers
  (parity tests on the pre-redesign outputs where they overlap).
- The new market-making-on-bars and market-making-on-tape paths get hand-computed
  fill tests (two-sided quotes, breach fills, inventory-cap truncation), plus the
  standard causality, batch==stream, NaN, and reset checks.
- One worked example per data model in the docs, and the coverage matrix renders.
