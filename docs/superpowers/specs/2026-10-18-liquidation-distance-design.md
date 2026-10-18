# LiquidationDistance operator design

Date: 2026-10-18. Status: agreed in conversation (a small liquidation-distance
operator rather than a liquidation engine); details are rulings made while the
user is away, for review before merge.

## Why an operator and not an engine

A backtest cannot price a liquidation reliably: exchanges trigger on a mark
price, the liquidation fill goes through the exchange's engine and insurance
fund, and auto-deleveraging cannot be modelled. What a backtest can answer is
how close a position came to liquidation. `LiquidationDistance` reports that
distance bar by bar, so a user can choose a leverage with a buffer. It does
not close positions or change pnl.

## Semantics

Inputs (four streams, in this order): `position` (signed size), `open`,
`high`, `low` (the trigger price series, ideally mark-price bars). The
position is the one a backtest engine reports (column 2 of
`[equity, pnl, position, cost]`), already filled at that bar's open; a raw
target would place each fill one bar early (look-ahead).

Parameters: `leverage` (> 1), `maintenance` (maintenance-margin rate,
0 <= maintenance < 1 / leverage, default 0.005).

Position changes follow exchange accounting, with every fill at the bar's
open (ruling, fix round 1):

- from flat, or a sign flip: a fresh entry at the open;
- same-sign increase: the entry blends,
  `entry = (|p0| * e0 + (|p1| - |p0|) * open) / |p1|`, and margin is posted on
  the added size;
- same-sign reduce: the entry, and so the liquidation price, is unchanged.

Under isolated margin with no top-up, the position holds
|size| * entry / leverage. The liquidation price is

- long: `entry * (1 - 1 / leverage + maintenance)`
- short: `entry * (1 + 1 / leverage - maintenance)`

Outputs per bar (output_names order):

- `distance`: the adverse move still needed from the bar's worst price to
  reach liquidation, as a fraction of that price: long `(low - liq) / low`,
  short `(liq - high) / high`. Negative or zero means the bar breached.
  NaN when flat.
- `liq_price`: the liquidation price of the current position; NaN when flat.
- `breached`: 1.0 from the first breaching bar until the position goes flat
  or flips, 0.0 before it, NaN when flat. A resize does not clear it, since
  the position was in reality liquidated.

NaN inputs: a NaN position counts as unchanged (hold); NaN prices give NaN
outputs for that bar without changing the state, except that a position
change on that bar is still taken. A fill that needs the open (a fresh entry
or an increase) happens at the next finite open; going flat before it cancels
it.

## Shape and bindings

A C++ node in the backtest family (header under `include/screamer/`), with
the standard push / flush / reset contract, batch == live, 1-D and 2-D
(column-wise) inputs like the other multi-input operators, Python and JS
bindings with identical numbers, a docs page entry with a short example, and
`see_also` links to `BacktestCarry` and `backtest_trades`.

## Tests

Hand-computed long and short positions, a sign flip, a blended increase, a
reduce that keeps the entry, a breach that latches through a resize, flat
bars, going flat while a fill is pending, a flip on a NaN-open bar, NaN inputs, leverage / maintenance validation errors,
batch vs streaming equality, Python vs JS parity in the existing parity
harness.
