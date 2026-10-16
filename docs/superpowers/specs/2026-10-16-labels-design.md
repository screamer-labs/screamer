# Labels: design

Status: draft for review, 2026-10-16. Branch `feat/labels`.

## Goal

A "Labels" section in screamer with ready-to-use target builders for
supervised research: regression targets and classification labels that a user
picks from a table, pairs with features in one call, and trains on. Labels are
computed in C++ like every other operator, run in Python and JavaScript, and
follow the hard rules: causal, and the same result on stored arrays and on a
live feed.

Out of scope (agreed 2026-10-16; these live in the research project or in
other libraries): data collection, train/test splitting with purging and
embargo, sample weights, models, feature importance, experiment tracking.
Trend-scanning labels are out (flawed statistics, heavy compute).

## The catalog

| Operator | Task | Inputs | Outputs |
|---|---|---|---|
| `ForwardReturn` | regression | `price` | `return` |
| `ForwardVolatility` | regression | `price` | `volatility` |
| `ForwardExcursion` | regression | `price` | `up`, `down` |
| `ForwardExcursionOHLC` | regression | `high`, `low`, `close` | `up`, `down` |
| `HorizonLabel` | classification | `price` | `label` |
| `TripleBarrier` | classification + return | `side`, `price` | `label`, `return`, `holding` |
| `TripleBarrierOHLC` | classification + return | `side`, `open`, `high`, `low`, `close` | `label`, `return`, `holding` |

Cross-sectional labels need no new operator: `ForwardReturn` on a `(T, K)`
panel followed by `CrossRank` or `CrossTopBottom`. Excess return over a
benchmark is `ForwardReturn` of the asset minus `ForwardReturn` of the
benchmark (`Sub`). Both are recipes in the guide.

The price-only and OHLC variants follow the backtest engines'
convention (`BacktestPriceTarget`, `BacktestOHLCTarget`): an operator has a
fixed number of inputs, so a second input layout is a second operator.

## The central decision: labels are released `horizon` steps after entry

Every labeler has an integer `horizon`, the longest time a label can take to
resolve. **The output at step `t` is the label of the entry at step
`t - horizon`.** It is NaN for the first `horizon` steps.

This makes every labeler an ordinary one-in, one-out operator (`FunctorBase`),
and that buys a lot:

- It is causal by construction: at step `t` the label of entry `t - horizon`
  is fully known, because its horizon has passed.
- Batch equals live with no new machinery: it is a normal `EvalOp`, so it
  works eagerly, lazily, inside a `Pipeline`, on `(T, K)` panels, and in
  JavaScript through the existing code generation.
- Pairing with features needs nothing new:
  `forecast_pairs(X, y, count=horizon)` already lags the features by
  `horizon` and puts each entry's features next to its label.

What it costs: a triple-barrier label that resolves early (a barrier touched
after 3 of 20 steps) is still released at step `entry + 20`. On a live feed a
training example arrives up to `horizon - holding` steps later than it could.
For research this changes nothing, and the `holding` column still says when
the label resolved.

Rejected alternative (proposed in chat on 2026-10-16): emit each label at the
step it resolves, as an event stream with an entry-offset column. It releases
labels sooner on a live feed, but several labels can resolve in one step, so
it needs a variable-output node instead of an `EvalOp`, an extension of
`forecast_pairs` to variable offsets, separate JavaScript work, and it loses
`(T, K)` panel support. The saving (at most `horizon` steps of latency on a
live feed) does not pay for that.

Horizons are counts of steps. A time-based horizon on irregular data is
handled by resampling to a clock first (`Resample(every=...)`), which the guide
shows. A `duration=` form is a possible later addition.

## Shared parameters

One vocabulary across the family, so a user learns it once.

- `horizon` (int, min 1): steps from entry to the latest resolution, and the
  output delay.
- `scale`: `"absolute"` or `"vol"`. With `"vol"`, thresholds, barriers, and
  (optionally) outputs are in units of the volatility expected over the
  horizon at entry:

  `sigma_h(e) = EwStd(span=vol_span) of the 1-step log returns, as of step e, times sqrt(horizon)`

  This is the usual square-root-of-time scaling of a per-step volatility.
  `sigma_h` uses data up to and including the entry step, so it is known at
  entry.
- `vol_span` (float, default 100): span of that `EwStd`. Used only when
  `scale="vol"`. Follows the `span` convention of `EwStd`.

All returns are log returns unless an operator says otherwise. A non-positive
or NaN price at entry or in the window gives a NaN label.

## Operators

### `ForwardReturn(horizon=1, kind="log", scale="absolute", vol_span=100)`

`return(e) = log(price[e + horizon] / price[e])`, or the simple return with
`kind="simple"`. With `scale="vol"` the value is divided by `sigma_h(e)`, a
z-score of the move. Raw `ForwardReturn(h)` equals `LogReturn(h)` read at the
same step; the operator exists so the target has a name that says what it is,
and for the `scale` option. The docs say so.

### `ForwardVolatility(horizon=1, scale="absolute", vol_span=100)`

Realized volatility over the horizon, the square root of the sum of squared
1-step log returns from `e + 1` to `e + horizon` (no demeaning, the
realized-variance convention). With `scale="vol"` it is divided by
`sigma_h(e)`: above 1 the period was more volatile than expected at entry.

### `ForwardExcursion(horizon=1, scale="absolute", vol_span=100)` and `ForwardExcursionOHLC`

The largest move up and down from the entry close within the horizon:

- `up(e) = log(max(p[e+1 .. e+horizon]) / p[e])`
- `down(e) = log(min(p[e+1 .. e+horizon]) / p[e])`

Neither is clipped at 0: when every later price is lower than the entry, `up`
is negative too.

The OHLC variant uses the highs for `up` and the lows for `down`, relative to
`close[e]`. With `scale="vol"` both are divided by `sigma_h(e)`. Typical use:
choosing stop and target levels, and the maximum adverse excursion of a
strategy.

### `HorizonLabel(horizon=1, threshold=0.0, scale="vol", vol_span=100, classes=3)`

The direction of the forward log return `r(e)` with a dead zone:

- `classes=3`: `+1` if `r > threshold`, `-1` if `r < -threshold`, else `0`.
- `classes=2`: `1` if `r > threshold`, else `0`.

With `scale="vol"` (the default here) the threshold is in units of
`sigma_h(e)`, so `threshold=0.5` labels moves larger than half the expected
move. With `scale="absolute"` it is a log return.

### `TripleBarrier(horizon=20, take=2.0, stop=2.0, scale="vol", vol_span=100, vertical="zero")` and `TripleBarrierOHLC`

The first barrier touched after entry, following López de Prado (2018), with
the side input of meta-labeling built in.

- Entries come from `side`: `+1` opens a long label at the close of that step,
  `-1` a short label, and `0` or NaN no label. Only the sign is used. To label
  every step by direction, pass `side = 1` everywhere: the label is then `+1`
  for the upper barrier and `-1` for the lower.
- Barriers, as log returns from the entry price in the trade's direction:
  profit at `+take * w`, loss at `-stop * w`, where `w = sigma_h(e)` with
  `scale="vol"` and `w = 1` with `scale="absolute"`. `take=0` or `stop=0`
  turns that barrier off, as in the engines. The vertical barrier is
  `horizon` steps after entry.
- Outputs for the entry `e = t - horizon` (NaN for steps without an entry):
  - `label`: `+1` profit barrier first, `-1` loss barrier first, at the
    vertical barrier `0` (`vertical="zero"`) or the sign of the return
    (`vertical="sign"`). For meta-labeling, the label to learn is
    `label == +1`.
  - `return`: the trade's log return to its exit, signed by `side`.
  - `holding`: steps from entry to exit, 1 to `horizon`. The label's span is
    `[e, e + holding]`, which is what purging and sample weights in the
    research project need.
- `TripleBarrier` (one price): a barrier counts as touched when a later price
  crosses it, and the exit is at that price.
- `TripleBarrierOHLC`: barriers are tested on each bar's high and low,
  mirroring `detail::risk_exits` in the engines: a gap through a barrier at
  the open exits at the open; when both barriers lie inside one bar the loss
  barrier wins, because the order of high and low within the bar is unknown;
  the exit price is the barrier level clamped to the bar. The implementation
  reuses `risk_exits` where it fits, so labels and backtests agree on the
  same bars.
- Several entries can be open at once (every step can be an entry); each is
  tracked independently. State is a ring of the open entries of the last
  `horizon` steps.

## Pairing and the guide's main example

```python
import screamer as sm

h = 12
y = sm.TripleBarrierOHLC(horizon=h, take=2.0, stop=1.0)(side, open_, high, low, close)
X_train, y_train = sm.forecast_pairs(X, y[:, 0], count=h, dropna=True)
```

Row `t` of the pair holds the features of entry `t - h` and its label. The
guide states the rule once: a labeler's output is released `horizon` steps
after entry, and `forecast_pairs(..., count=horizon)` lines it up.

## Documentation

- A new topic `labels` ("Labels") in the "Forecasting & ML" group
  (`docs/topics.yml`), next to `supervised`.
- A guide page `docs/labels.md`: what a label is in screamer (released after
  `horizon`), the chooser table below, `scale="vol"` explained once, pairing,
  the label span for purging, and the two recipes (cross-sectional, excess
  return). Linked from the Forecasting & ML area of `docs/examples.yml`, and a
  "Label a training set" section in `docs/choosing_an_operator.md`.

  | I want to predict | Use |
  |---|---|
  | how much the price moves | `ForwardReturn` |
  | how volatile the next period is | `ForwardVolatility` |
  | how far it runs for and against me | `ForwardExcursion(OHLC)` |
  | up, down, or flat | `HorizonLabel` |
  | whether a trade with a stop and a target works | `TripleBarrier(OHLC)` |
  | which assets do best | `ForwardReturn` + `CrossRank` |

- A function page per operator (`docs/functions_labels/`), each with a plot of
  its labels drawn on a price chart.
- Notebook 23, "Labels": one synthetic series labeled every way, side by side,
  then one label paired and fit with a simple model to show the full path.
- `forecast_pairs` page and notebook 16 point to the new section.
- CHANGELOG "Added" entry.

## Implementation notes

- New headers `include/screamer/forward_return.h` etc., `FunctorBase`-derived;
  bindings in a new `bindings/bindings_labels.cpp` (picked up by the WASM
  manifest scan), integer minimums in `common/int_settings.h`.
- One shared helper for `sigma_h` (an `EwStd` of log returns plus a ring of
  the last `horizon` values, so each label reads the value at its entry).
- Ring buffers of size `horizon + 1`; `ForwardExcursion` uses a monotonic
  deque for the window max and min, so every operator is O(1) amortized per
  step. `TripleBarrier` checks each open entry against the new step, which is
  O(open entries) per step, at most `horizon`.
- All generated files regenerate through the existing scripts (help registry,
  `__init__`, WASM manifest, TS API, JS docs).

## Testing

- A plain numpy reference per operator (`devtools/baselines`), compared on
  random walks, including NaN gaps and non-positive prices.
- Batch equals streaming, and `(T, K)` equals K separate runs.
- No look-ahead: changing prices after step `t` leaves every output at or
  before `t` unchanged.
- `TripleBarrierOHLC` agrees with `BacktestOHLCTarget` stop and take exits on
  the same bars (gap at the open, both barriers in one bar, clamping).
- Edge cases: `horizon=1`; side changing every step; side NaN; `take=0` or
  `stop=0`; `scale="vol"` during the `EwStd` warmup (NaN labels).
- JavaScript parity fixtures as for every operator.
- Docs: the guide's examples run (`tests/test_docs_module.py` pattern), the
  notebook executes in the docs build.

## Review focus

1. A user reading `y[t]` as the label of step `t`: the docs and every
   function page lead with "released `horizon` steps after entry", and the
   pairing example uses `count=horizon`.
2. Overlapping triple-barrier labels when every step is an entry: each entry
   resolves on its own, and a test covers 1000 consecutive entries.
3. A price of zero or below, or NaN, inside an open label's window: that label
   is NaN, others are unaffected.
4. `scale="vol"` before `vol_span` has data: NaN labels, never infinite.
5. Ties at a barrier (a price exactly at the level): counts as touched, the
   same rule as the engines.

## Open questions for the reviewer

1. `scale` defaults: `"absolute"` for the three regression targets,
   `"vol"` for `HorizonLabel` and the triple barrier. Mixed defaults follow
   what each target is used for; the alternative is one default everywhere.
2. Default `TripleBarrier` values: `horizon=20, take=2.0, stop=2.0`.
