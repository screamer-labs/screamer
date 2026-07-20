# Backtest engine suite: data model x order definition

## Context

screamer ships a family of backtest engines that today mix two unrelated ideas in
their names and their decomposition. Some engines are named for their market-data
type (`BacktestOHLC`, `BacktestL1`, `BacktestTrades`), one is named for its order
format (`BacktestSignal`), and the market-making variants carry a `Maker` suffix
(`BacktestOHLCMaker`, `BacktestTradesMaker`). The suffix split runs along the wrong
axis: `BacktestOHLC` and `BacktestOHLCMaker` share a data model and differ only in
how you hand them orders, so they read as unrelated engines when they are two faces
of one thing.

Two axes are actually in play, and they are orthogonal:

1. **Data model**: the market data the fill is simulated against (a price series,
   OHLC bars, the trade tape, an L1 book, an L1 book plus trades). This is the only
   thing that changes the fill simulation.
2. **Order definition**: how the strategy expresses intent. Either it names a
   **target** position and the engine sizes the trade, or it posts explicit
   **orders** (bid and ask, price and size) and takes what fills.

The engines already agree on the output (`[equity, pnl, position, cost]`) and share
`detail::PnLAccount`. This redesign makes the two axes explicit in the naming and
the code, so every engine is one (data model, order definition) cell, the suite is
a clean grid, and the missing cells are obvious. screamer is pre-1.0, so the
renames are acceptable breaking changes.

## Goals

- Name every engine `Backtest<DataModel><OrderDef>`, so both axes are visible.
- One fill core per data model, reused by both order definitions.
- One target front, reused across data models.
- Keep the output schema, `PnLAccount`, `backtest_report`, and `BacktestReport`
  unchanged.
- Populate the cells that are useful; document the ones that are not.

## The naming grid

`DataModel` is one of `Price`, `OHLC`, `Trades`, `L1`, `L1Trades`. `OrderDef` is
`Target` or `Orders`.

| data model \ order def | `Target` (name a position) | `Orders` (post bid/ask) |
|---|---|---|
| `Price` (value series) | `BacktestPriceTarget` | not provided |
| `OHLC` (bars) | `BacktestOHLCTarget` | `BacktestOHLCOrders` |
| `Trades` (tape) | `BacktestTradesTarget` | `BacktestTradesOrders` |
| `L1` (quotes) | `BacktestL1Target` | `BacktestL1Orders` |
| `L1Trades` (quotes + trades) | not provided | `BacktestL1TradesOrders` |

Eight engines. The mapping from today's names:

| new name | replaces |
|---|---|
| `BacktestPriceTarget` | `BacktestSignal` |
| `BacktestOHLCTarget` | `BacktestOHLC` (the directional form) |
| `BacktestOHLCOrders` | `BacktestOHLCMaker` |
| `BacktestTradesTarget` | `BacktestTrades` (the directional form) |
| `BacktestTradesOrders` | `BacktestTradesMaker` |
| `BacktestL1Target` | new (the one genuinely useful gap) |
| `BacktestL1Orders` | `BacktestL1` |
| `BacktestL1TradesOrders` | `BacktestL1Trades` |

### Cells not provided, and why

- `BacktestPriceOrders`: a bare value series has no book to quote against; posting a
  bid and ask around one price is a toy simulation already covered better by
  `OHLC`/`L1`. Use `BacktestOHLCOrders` or `BacktestL1Orders`.
- `BacktestL1TradesTarget`: reaching a target position means taking liquidity, which
  is a taker action, so the trade-driven fill
  refinement (fill a resting quote only when a trade explains it) does not change
  the result. Use `BacktestL1Target`.

The naming grid stays complete and orthogonal even though two cells are empty; the
"choosing an engine" doc lists the empty cells with the redirect.

## Axis: order definition

### `Target`: name a position, always market

A `Target` engine takes a **target position** and reaches it by taking liquidity.
Per event the order input is a single column:

```
target_position
```

The engine sizes the trade itself (`target_position - position`, since only it
holds the live position) and fills it as a marketable order against the data model,
paying `taker_fee`. Because fills can be partial (see the fill cap and
participation below), the position may trail the target and catch up over later
events; the engine re-sizes toward the target on every event.

`Target` is deliberately market only. "Be at position X" is a take-liquidity intent;
price control (rest a limit at a chosen price) belongs to `Orders`. This keeps the
two order definitions crisply separated and keeps `BacktestPriceTarget` a two-input
engine `(target_position, price)`, preserving today's `BacktestSignal(signal,
price)` ergonomics with no new column.

Cost of this choice: the current `BacktestOHLC` can reach a target through a
**resting limit** (`target`, `limit`). That "work toward a target passively"
behavior is dropped. To rest a passive order you use the `Orders` interface (post a
one-sided limit) and size it yourself. This is called out for review.

### `Orders`: post explicit bid and ask

An `Orders` engine takes a two-sided quote. Per event the order input is four
columns:

```
bid_price, bid_size, ask_price, ask_size
```

`size <= 0` or `NaN` size means no order on that side, so a one-sided order (a plain
directional buy or sell) is the degenerate case of zeroing the other side. The
position emerges from whatever fills, bounded by the static fill cap. A resting
limit pays `maker_fee`; an order that is marketable on submission pays `taker_fee`.

### Market order = a limit at the maximally aggressive price

Both interfaces share one price encoding (a market order is a price, not a flag), so
direction falls out of the normal fill comparison:

| `price` | buy side | sell side |
|---|---|---|
| finite | limit at `price` | limit at `price` |
| `+inf` | market | never fills |
| `-inf` | never fills | market |
| `NaN` | market | market |

`screamer.MARKET` (backed by `inf`) is the convenience spelling; `NaN`, `+inf`,
`-inf` are all accepted. The only substitution is the side-agnostic `NaN` (to `+inf`
on a buy, `-inf` on a sell). A wrong-direction infinity is a harmless never-fill
limit. This is the existing `market_limit(price, buy)` helper. `Target` uses it
implicitly (it always submits the aggressive price); `Orders` applies it to each
quote price.

## Axis: data model (the fill cores)

Each data model has one fill core: given a canonical order (a marketable trade for
`Target`, a two-sided quote for `Orders`) it simulates fills against that data,
updates `PnLAccount`, applies the fill cap and cost, and emits
`[equity, pnl, position, cost]`. The market-data columns follow the order columns.

| data model | market-data columns | fill rule |
|---|---|---|
| `Price` | `price` | a limit fills when the price crosses it; a market fills at `price` (one series, so limit fidelity is coarse) |
| `OHLC` | `open, high, low, close` | a market fills at the next bar's open; a limit fills when the bar's low/high reaches it (`touch`/`breach`); marks to close |
| `Trades` | `trade_price, trade_size` | a limit fills when a print crosses it, up to `participation_ratio` of the print; a through-print sweeps it; marks to the last trade |
| `L1` | `market_bid, market_ask, market_bid_size, market_ask_size` | a resting quote fills when the opposite side of the market reaches it; a marketable order takes the displayed size plus `tick_size` overflow |
| `L1Trades` | `+ trade_price, trade_size` | trades drive the fills; a quote cross with no explaining trade is the run-over fallback |

Causality note: on bars, a `Target` is typically derived from the close, so
`BacktestOHLCTarget` decides on bar t and executes on bar t+1's open (as
`BacktestOHLC` does today). `BacktestOHLCOrders` fills the quote you pass against the
current bar and assumes you supplied a quote knowable at that bar's open (the engine
cannot enforce causal inputs). This asymmetry is intrinsic to the two intents and is
documented per engine.

## Shared concerns (identical on every engine)

- **Fill cap**: static `min_position` / `max_position` (default `-inf` / `+inf`). A
  fill is the minimum of the order size, the counterparty volume, and the room to
  the cap (`max_position - position` buying, `position - min_position` selling).
- **Cost and fidelity**: `maker_fee`, `taker_fee`, `tick_size`, `fill`
  (`"touch"`/`"breach"`), `participation_ratio`. Partial fills are real on the
  volume-aware data models (`Trades`, `L1`, `L1Trades`) through
  `participation_ratio` and the cap; `Price` and `OHLC` fill the full triggered
  amount because they carry no per-level volume.
- **Output**: `[equity, pnl, position, cost]` through the shared, unchanged
  `detail::PnLAccount`.

## Architecture

Two reusable pieces compose into the eight engines:

- **Five fill cores**, one per data model. A core takes a canonical order plus its
  market-data columns and produces the four outputs. This is where each data model's
  fill rule lives, and it is shared by both order definitions of that data model.
- **One target front.** Given a `target_position` and the engine's live position it
  produces the canonical marketable order (`target - position` on the aggressive
  side). It composes onto any fill core to form the `Target` engine for that data
  model.

An `Orders` engine is the fill core taking the two-sided quote directly. A `Target`
engine is the target front feeding the same fill core. Each of the eight classes is
a thin `FunctorBase<Derived, N, 4>` wrapper over a shared core, so the fixed input
arity is satisfied cleanly (one schema per class) with no runtime dispatch, and the
per-data-model fill logic is written once.

## Migration

- Rename the seven existing engines to their grid names; add `BacktestL1Target`.
- The in-flight `feat/unified-backtest-api` branch (the static fill cap on every
  engine, the `MARKET` encoding, and the two-sided making on bars and the tape) is
  superseded as a standalone change but its work is reused: its maker engines become
  the `OHLCOrders` / `TradesOrders` cells, and the cap and `MARKET` work carries
  into every cell. The redesign is implemented fresh from `main` so the renames and
  the new `Target`/`Orders` split are coherent, cherry-picking that branch's fill
  logic rather than building on its names.
- `detail::PnLAccount`, `backtest_report`, and `BacktestReport` are unchanged.
- The changelog entry is rewritten to describe the grid rather than the old set.
- Docs: a "choosing a backtest engine" page carries the naming grid (the two empty
  cells with their redirects), the two order-definition interfaces, and the market
  order encoding. Notebooks and any `BacktestSignal`/`BacktestOHLC` references are
  updated to the new names (screamer-bots is a downstream follow-up, not part of
  this spec).

## Related follow-up (separate spec)

`SchmittTrigger` emits `NaN` during warmup when the signal starts inside its
hysteresis band, forcing a `nan_to_num` at the call site. A small follow-up gives it
an initial state so warmup resolves to a definite side. Out of scope here; noted so
it is not lost.

## Validation

- Renamed engines reproduce their pre-rename behavior (parity tests on the existing
  outputs; the default cap is unbounded so behavior is preserved).
- The shared target front and each fill core get hand-computed fill tests: market to
  target, two-sided quotes, `touch`/`breach`, the three-way fill-cap minimum, and
  the `MARKET`/`NaN`/`inf` encoding.
- `BacktestL1Target` (the new cell) gets its own hand-computed taker-to-target
  tests.
- Every engine gets the standard causality, batch-equals-stream, NaN-policy, and
  reset checks.
- The naming grid renders in the docs and every populated cell links to a page.
