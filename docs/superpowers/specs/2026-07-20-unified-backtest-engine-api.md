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
extends that agreement to the inputs, so every engine accepts the same order
strategies and the same risk controls and differs only in its market-data columns.
screamer is pre-1.0, so this is an allowed breaking change.

## Goals

Make four concerns orthogonal, each expressed identically on every engine:

1. **Data model** (which fill simulation runs): the only thing that differs
   between engines.
2. **Order intent** (what the strategy posts): the same two forms on every engine,
   directional (a target position) and market-making (a two-sided quote).
3. **Fill cap** (inventory limit): one static `[min_position, max_position]` that
   truncates fills on every engine.
4. **Cost and fill fidelity**: fees, slippage, and the touch/breach and
   participation knobs.

The output stays `[equity, pnl, position, cost]`.

## Axis 2: order intent, two forms

A strategy is either directional (I want to *be* at a position) or a market maker
(I *post* quotes and take what fills). These are genuinely different intents, so
each is its own input form. Both are available on every data model, and both share
the market-order price encoding below.

**Directional (target).** Per event:

```
target_position, limit_price
```

The engine trades toward `target_position` (sizing `target - position` itself,
since only it knows the live position). `limit_price` sets how: a finite price is a
resting limit, a non-finite price is a market order (see the encoding). This is
`BacktestSignal(signal, price)` and `BacktestOHLC(target, limit, ...)` as they
already work.

**Market-making (quote).** Per event, up to two resting orders, a buy and a sell:

```
bid_price, bid_size,   ask_price, ask_size
```

`size <= 0` or `NaN` size means no order on that side. The position emerges from
whichever side fills, capped by the static fill cap (axis 3).

### Market order = a limit at the maximally aggressive price

Market is not a flag, it is a price, so direction falls out of the normal fill
comparison and needs no special case. This encoding applies to `limit_price` and to
each quote price:

| `price` | buy side (bid / target up) | sell side (ask / target down) |
|---|---|---|
| finite | limit at `price` | limit at `price` |
| `+inf` | market (clears any offer) | limit at `+inf`, never fills |
| `-inf` | limit at `-inf`, never fills | market (hits any bid) |
| `NaN` | market | market |

A buy limit at `price` fills when the market reaches `<= price`, so `+inf` always
fills (market) and `-inf` never does; the sell side is the mirror. `+inf`/`-inf`
therefore need no handling. The only substitution is the side-agnostic `NaN`: on a
buy it becomes `+inf`, on a sell `-inf`. A wrong-direction infinity (`-inf` buy,
`+inf` sell) is a harmless never-fill limit, not a surprise market order.

`screamer.MARKET` is a convenience constant (backed by `inf`) for the common
spelling; `NaN`, `+inf`, and `-inf` are all accepted. An order that is marketable
on submission is a taker and pays `taker_fee`; a resting limit filled later is a
maker and pays `maker_fee`.

## Axis 3: the static fill cap

Two static parameters, `min_position` and `max_position` (default `-inf` / `+inf`),
cap the inventory. They are a fill limit, not a target: a fill is the minimum of

1. the order size,
2. the counterparty volume available (the print size, the displayed quote size, or
   the bar's reachable range), and
3. the room left to the cap: `max_position - position` when buying,
   `position - min_position` when selling.

Whichever is smallest truncates the fill. This matters most for market-making,
where the position emerges from the quotes and must be bounded; it applies under
the directional form too (a target beyond the cap is truncated). This is exactly
how `BacktestL1`'s cap already works, made static config on every engine.

## Axis 4: cost and fill fidelity (static config)

Every engine takes the same static parameters:

- `maker_fee`, `taker_fee`: fractional fees; negative for a rebate.
- `tick_size`: price a marketable order walks per unit of size beyond the
  displayed quote (the taker slippage model).
- `fill`: `"touch"` or `"breach"`, the optimistic/conservative limit-fill rule.
- `participation_ratio`: the fraction of the at-price volume a limit captures.

## The unified contract

Every engine returns `[equity, pnl, position, cost]` and takes the market-data
columns for its data model plus one of the two order forms, with the static cap and
cost config above. Only the market-data prefix differs:

| Engine | market-data columns |
|---|---|
| `BacktestSignal` (value series) | `price` |
| `BacktestOHLC` (bars) | `open, high, low, close` |
| `BacktestTrades` (tape) | `trade_price, trade_size` |
| `BacktestL1` (quotes) | `market_bid, market_ask, market_bid_size, market_ask_size` |
| `BacktestL1Trades` (quotes + trades) | `+ trade_price, trade_size` |

The quote form's order columns are named to avoid colliding with the L1 market book
(the market is `market_bid`/`market_ask`; the strategy's orders are `bid`/`ask`).

Each engine's fill simulation is the only thing that differs:

- **value**: a limit fills if the price crosses it between events; a market fills at
  the price (one-price series, so limit fidelity is coarse).
- **bars**: market fills at the open (plus half `spread`); a limit fills when the
  bar's low/high reaches it (`touch`/`breach`).
- **tape**: a limit fills when a print crosses it, up to `participation_ratio` of
  the print; a through-print sweeps the order.
- **L1**: a resting quote fills when the opposite side of the market reaches it; a
  marketable order takes the displayed size plus `tick_size` overflow.
- **L1+trades**: trades drive the fills; a quote cross with no explaining trade is
  the run-over fallback.

Market-making on bars and on the tape now exist by construction: they are the
quote form on those data models, only the fill simulation changes.

### Directional and market-making per engine

Each data model supports both order forms. Where an engine already covers a form it
keeps its name; the missing market-making forms on bars and the tape are the new
work. Whether a data model exposes the two forms as one engine with both input sets
or as a directional engine plus a `*Maker` engine is an implementation choice for
the plan; the contract (same cap, cost, and output) is fixed here.

## Convenience wrappers

The directional form is a one-liner already (`BacktestSignal(signal, price)`); it
stays. The market-making form is more verbose, so a wrapper takes a static
inventory cap and a two-sided quote and expands to the full column set. The
market-order price encoding and the fill cap are the only conventions a caller
learns.

## Coverage, before and after

The redesign closes the matrix. Rows are the data model; columns are the order
strategy; every cell is covered:

| Data model | market | limit (directional) | market-making |
|---|---|---|---|
| value series | yes | (coarse) | yes |
| OHLC bars | yes | yes | **yes (was gap)** |
| trade tape | yes | yes | **yes (was gap)** |
| L1 quotes | yes | yes | yes |
| L1 + trades | yes | yes | yes |

This matrix, plus a short "choosing a backtest engine" overview, goes on a
hand-written docs page linked from the Backtesting group (the group page is
auto-generated, so the matrix lives beside it).

## Migration

- The engines keep their current names (`BacktestSignal`, `BacktestOHLC`,
  `BacktestTrades`, `BacktestL1`, `BacktestL1Trades`). No renames.
- Each engine gains the static `min_position`/`max_position` cap and the full cost
  config, so the four axes are uniform.
- The market-making form is added on bars and the tape (the two gaps).
- `detail::PnLAccount`, `backtest_report`, and `BacktestReport` are unchanged.
- The unreleased backtest-suite changelog entry is rewritten to describe the
  unified contract rather than the five divergent engines.

## Related finding (separate spec)

`SchmittTrigger` emits `NaN` during warmup when the signal starts inside its
hysteresis band, forcing a `nan_to_num` at the call site. A small follow-up gives
it an `initial` state (or the existing `start_policy`) so warmup resolves to the
low/flat state. Out of scope here; noted so it is not lost.

## Validation

- Each engine reproduces its old behavior (parity tests on the pre-redesign
  outputs where they overlap; the directional forms are unchanged).
- The new market-making-on-bars and market-making-on-tape paths get hand-computed
  fill tests (two-sided quotes, breach fills, fill-cap truncation via the three-way
  minimum), plus the standard causality, batch==stream, NaN, and reset checks.
- One worked example per data model in the docs, and the coverage matrix renders.
