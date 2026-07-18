# Backtest fill models: participation, touch/breach, and the L1 engines

## Context and scope

The backtest suite already ships four engines that share `detail::PnLAccount`
and the `[equity, pnl, position, cost]` schema: `BacktestSignal`, `BacktestOHLC`,
`BacktestTrades`, `BacktestL1`. Their accounting core is sound. Their fill
models are not: the committed `BacktestTrades` and `BacktestL1` size fills in
ways that are wrong (haircutting a trade that sweeps through your price, filling
a fraction of the remainder rather than of the available volume, and inferring
fills from L1 quote-size changes that cannot be told apart from cancels).

This spec fixes the fill models for the market-data engines, adds one new engine
(`BacktestL1Trades`), and settles the fill-related parameters. The accounting
core, the output schema, `backtest_report`, and `BacktestSignal` are unchanged.

## Principles

1. **Orders are counterfactual.** A simulated order carries zero volume, is not
   in the exchange feed, and has no market impact. The feed reflects the real
   market that never included us. A fill therefore may never exceed the volume
   that actually traded or was displayed at the level; anything we cannot bound
   against observed volume is a heuristic, and is documented as one.
2. **Causal and deterministic.** No lookahead. No randomness. Batch and
   streaming give identical results. Every fill knob is a deterministic
   parameter, never a random draw.
3. **A fill is driven by an execution event, never by a static-price size
   change.** A trade print is an execution event. A quote crossing your price is
   an execution event. A change in displayed size at an unchanged price is not,
   because it is fill, cancel, and re-quote all at once, and acting on it
   double-counts.

## Parameters

| Parameter | Range | Default | Meaning |
|---|---|---|---|
| `participation_ratio` | `(0, 1]` | `1.0` | Fraction of the volume at your level that you capture on an at-touch fill. `1.0` is front-of-queue. |
| `tick_size` | `>= 0` | `0.0` | Price step a market/marketable order walks for the size beyond the displayed quote. |
| `fill` | `"touch"` / `"breach"` | see per engine | Optimistic (fill when the price is reached) vs conservative (fill only when the price is traded through). |

At-touch fill size, everywhere it applies:

```
fill = min(remaining, participation_ratio * available_size)
```

`available_size` is `trade_size` (tape) or `ask_size` (L1). Scaling the
*available volume*, capped by the *remaining order*, is the Zipline `volume_limit`
semantics: it converges to a full fill as volume accrues and never leaves an
asymptotic remainder.

## Per-engine fill rules

Notation: a resting buy at price `L` for `remaining` lots. Sells are symmetric.

### BacktestOHLC (bars, 6 inputs)

Bars carry no intra-bar path or per-level volume, so fills are full. **Causal by
design**: the `target_position` and `limit_price` passed on a bar are decided from
that bar's close, and the engine defers them one bar and executes on the next bar
(a target from a bar's close cannot trade within that same bar, whose open already
happened). No manual lag is needed, matching `BacktestSignal`'s timing. The
deferred order executes on the next bar:

- Market order (`limit_price` NaN): full fill at that bar's `open`, crossing half
  `spread`, pays `taker_fee`.
- Limit order: `fill = "touch"` fills when `low <= L`, `"breach"` when `low < L`;
  full fill at `L`, pays `maker_fee`.

No `participation_ratio` (no volume to participate in). A `NaN` target places no
order for the next bar (the position holds).

### BacktestTrades (tape, 4 inputs)

Inputs `(order_price, order_size, trade_price, trade_size)`; `order_size` signed.
On each print:

- **Through** (`trade_price < L`): the market swept your level. **Full remaining**
  at `L`. The small print at the through-price is the residual after the sweep;
  it does not bound your fill.
- **At** (`trade_price == L`): `fill = min(remaining, participation_ratio *
  trade_size)` at `L`, front-of-queue.
- `fill = "breach"` keeps only the through branch.

Fills pay `maker_fee` (negative for a rebate). Marks to the last trade.

Correction from the committed engine: through-fills were capped at
`min(remaining, trade_size)` and at-fills fractioned the remainder. Both change.

### BacktestL1 (top-of-book quotes only, 8 inputs)

Inputs `(bid, ask, bid_size, ask_size, my_bid, my_bid_size, my_ask, my_ask_size)`.
Quotes are as-of state (forward-filled upstream). No trade feed, so fills are a
documented heuristic driven by the quote crossing your price.

```
resting buy at my_bid:

  breach (conservative, DEFAULT):
      ask <  my_bid  -> full remaining at my_bid            # swept; completes
      ask >= my_bid  -> no fill

  touch (optimistic, opt-in):
      ask <  my_bid  -> full remaining at my_bid
      ask == my_bid  -> ONE fill on entering the lock:
                          min(remaining, participation_ratio * ask_size) at my_bid
                        no further fills while ask stays == my_bid
      ask >  my_bid  -> lock ends; keep the partial; remainder rests
```

The touch partial is edge-triggered on entry to the lock. Size changes while the
ask price is unchanged are ignored, because they are exactly the fill/cancel
ambiguous ones. When the ask price makes a new lock later, that is a fresh
episode and another partial is allowed.

`breach` is fully clean: through-fills self-limit to zero remaining, so there is
no partial and no size-update ambiguity at all. It is the honest default; it can
only under-fill (miss a fill that happened at a touch without the quote
crossing). `touch` is the opt-in optimistic mode and carries all the irreducible
heuristic error, documented in a Limitations box.

Fills pay `maker_fee`, mark to the mid.

### BacktestL1Trades (quotes + trades, 10 inputs), NEW

Inputs: the eight L1 fields plus `(trade_price, trade_size)`. The preferred
market-making engine, because trades are unambiguous execution events.

Input contract: quotes are as-of state (forward-filled); **trades are NOT
forward-filled**, they appear only on their own event rows and are NaN
otherwise. A NaN trade is a quote-only update (re-mark, update queue estimate, no
fill), which is `nan_policy: ignore`. Each real trade appears on exactly one row,
so there is no double-counting.

```
resting buy at my_bid:

  passive fill (from the trade tape):
      sell-print at/through my_bid -> as BacktestTrades:
        through -> full remaining at my_bid
        at      -> min(remaining, participation_ratio * trade_size) at my_bid

  run-over fill (from the quote, no explaining trade):
      ask < my_bid -> full remaining at my_bid
      (guarded: if a trade on the same event already explains the move, the
       trade takes precedence; the cross is only a fallback)
```

Fills mark to the mid. This engine has neither the Zeno problem nor the
size-update ambiguity: passive fills come from trades, and the run-over path is
the only quote-driven fill (a bounded, one-shot completion).

### Market and marketable orders

A market order, or a resting quote submitted already crossing the spread
(`my_bid >= ask`), is a taker. It fills its full size immediately, walking the
book by `tick_size` for the part beyond the displayed quote:

```
market buy Q, displayed ask_size A at price ask:
  min(Q, A)       fills at ask
  max(Q - A, 0)   fills at ask + tick_size
  VWAP = (min(Q,A)*ask + max(Q-A,0)*(ask+tick_size)) / Q
```

Pays `taker_fee`. Documented limitation: one tick assumes the next level has
enough depth, which L1 cannot confirm; a large order understates impact. L2 is
the real fix and is out of scope.

## Fill price and fee side

- A **resting** limit order that fills later (passive or run-over) fills at **its
  own limit price**, never better. (The opening-rotation exception does not apply
  here.)
- A **marketable** order submitted already crossing fills at the market (`ask`
  plus `tick_size` overflow) and pays `taker_fee`.
- A resting quote the market **runs through** is a maker fill (pays `maker_fee`).
- The engine classifies run-over (maker, fill at limit) versus submitted-crossing
  (taker, fill at market) from the prior quote state: a quote resting passive
  below the ask that the ask then crosses is run-over; a quote that first appears
  already marketable is submitted-crossing.

## Transparency (documentation requirements)

Each engine's reference page gets a "Limitations" section stating plainly:

- `BacktestL1` (quotes only): fills are a heuristic. `touch` can **over-fill**
  when a lock's size came from cancels rather than trades, and its per-touch
  partial is a proxy because real traded volume is not observed. `breach` can
  **under-fill**. Prefer `BacktestL1Trades` when a trade feed is available.
- `tick_size` overflow assumes depth at the next level; large market orders
  understate impact.
- All engines: orders are counterfactual (zero volume, no market impact).

## Change scope

- **Modify** `include/screamer/backtest_trades.h`: through = full remaining;
  at = `min(remaining, participation_ratio * trade_size)`; add
  `participation_ratio`.
- **Modify** `include/screamer/backtest_l1.h`: breach/touch with the lock-episode
  logic; `min(remaining, participation_ratio * ask_size)`; add
  `participation_ratio`, `tick_size`, `taker_fee`; marketable/taker path;
  run-over vs submitted-crossing classification.
- **Confirm** `include/screamer/backtest_ohlc.h`: full-fill semantics (no change
  expected beyond a comment audit).
- **Create** `include/screamer/backtest_l1_trades.h`: the 10-input engine.
- **Update** `bindings/bindings_fin.cpp`: new params on the modified engines, and
  the new engine.
- **Docs**: reference pages for the changed engines plus `BacktestL1Trades`, each
  with a Limitations section; update notebook 15 to reflect corrected fills.
- **Tests**: hand-computed cases for through vs at, participation sizing (no
  Zeno), lock-episode edge-triggering (no double-count on static size updates),
  market overflow VWAP, run-over vs submitted-crossing, plus the standard
  causality / batch==stream / NaN / reset checks.

## Decisions (settled)

- `participation_ratio` scales available volume, capped by remaining; default 1.0.
- `BacktestL1` defaults to `breach` (conservative); `touch` is opt-in.
- Fill price is always the resting limit price; marketable submission is a taker.
- `BacktestL1Trades` is the preferred engine; quotes-only `BacktestL1` is kept as
  a documented least-wrong fallback for feeds without trades.
