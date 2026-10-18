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

Inputs (four streams, in this order): `position` (signed size; only its sign
and its changes matter), `open`, `high`, `low` (the trigger price series,
ideally mark-price bars).

Parameters: `leverage` (> 1), `maintenance` (maintenance-margin rate,
0 <= maintenance < 1 / leverage, default 0.005).

Segments: a segment starts on a bar where the position changes to a nonzero
value (including a sign flip or a resize) and fills at that bar's open. Under
isolated margin with no top-up, the segment posts |size| * entry / leverage.
The liquidation price is

- long: `entry * (1 - 1 / leverage + maintenance)`
- short: `entry * (1 + 1 / leverage - maintenance)`

Outputs per bar (output_names order):

- `distance`: the adverse move still needed from the bar's worst price to
  reach liquidation, as a fraction of that price: long `(low - liq) / low`,
  short `(liq - high) / high`. Negative or zero means the bar breached.
  NaN when flat.
- `liq_price`: the segment's liquidation price; NaN when flat.
- `breached`: 1.0 from the first breaching bar to the end of the segment,
  0.0 before it in a segment, NaN when flat.

A resize starts a new segment at the new fill (the conservative reading:
margin is re-posted at the current price, with no credit for earlier gains).
NaN inputs: a NaN position counts as unchanged (hold); NaN prices give NaN
outputs for that bar without ending the segment.

## Shape and bindings

A C++ node in the backtest family (header under `include/screamer/`), with
the standard push / flush / reset contract, batch == live, 1-D and 2-D
(column-wise) inputs like the other multi-input operators, Python and JS
bindings with identical numbers, a docs page entry with a short example, and
`see_also` links to `BacktestCarry` and `backtest_trades`.

## Tests

Hand-computed long and short segments, a sign flip, a resize, a breach that
latches, flat bars, NaN inputs, leverage / maintenance validation errors,
batch vs streaming equality, Python vs JS parity in the existing parity
harness.
