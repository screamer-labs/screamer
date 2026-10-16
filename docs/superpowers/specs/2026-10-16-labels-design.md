# Labels and training sets: design

Status: draft for review, 2026-10-16 (second version). Branch `feat/labels`.

## Goal

Make supervised research in screamer easy to understand and hard to get
wrong, with a design that does not need to change later. A user should be
able to say "these are my features, this is what I want to predict, over this
horizon" and get a training set whose rows look exactly like what the model
sees live.

Out of scope (agreed 2026-10-16; these belong in the research project or in
other libraries): data collection, train/test splitting with purging and
embargo, sample weights, models, feature importance, experiment tracking.
Trend-scanning labels are out (flawed statistics, heavy compute).

## The idea in one picture

A training row is **the features known at time t, next to what happened
between t and t+h.** Live, the same row exists with the target still unknown:

| | features | target |
|---|---|---|
| training, row t | `X[t]` | outcome of (t, t+h] |
| live, row t | `X[t]` | NaN, not known yet |

So the feature matrix is built by the same code, at the same timestamps, in
research and live. The model learns "features now, outcome next".

## What changes, and what is added

1. **`forecast_pairs` gets `index="features" | "target"`**, which chooses the
   timestamp each row carries. The new default is `"features"`.
2. **One new operator family: `TripleBarrier` and `TripleBarrierOHLC`.**
3. **Every other label is an existing operator used as a target**, shown in a
   guide with a chooser table. The first version of this spec proposed
   `ForwardReturn`, `ForwardVolatility`, `ForwardExcursion`, and
   `HorizonLabel`; they are dropped, because each was an existing operator
   under a second name (`LogReturn(h)`, a rolling std, `RollingMax`/`RollingMin`
   relative to a lagged price, `Sign`).

## `forecast_pairs(X, y, *, count=None, duration=None, index="features", dropna=False)`

`y` is a causal series, as today: its value at step t is known at t, so a
target over a horizon h is stamped at the end of that horizon. For example
`LogReturn(h)(price)` at step t is the return over (t-h, t].

`index` chooses whose timestamp a row carries:

| | row holds | NaN padding |
|---|---|---|
| `index="features"` (new default) | `X[t]` and the target over (t, t+h] | the last h targets |
| `index="target"` (today) | `X[t-h]` and `y[t]` | the first h feature rows |

- **Count mode.** Both settings produce the same pairs; only the stamp and the
  position of the padding differ. With `dropna=True` the two return identical
  arrays. Eager output keeps one row per input row in both settings.
- **Duration mode.** Rows follow the clock of the side whose timestamp they
  carry. `index="target"` is today's as-of join on the y clock: one row per y
  event, holding the last X at or before `t - duration`. `index="features"` is
  the mirror image on the X clock: one row per X event at t, holding the last
  y at or before `t + duration`. When no y event exists at or before
  `t + duration` after the X event, the target is NaN.
- **Lazy and graph regimes (streaming).** With `index="features"` the row for
  t is emitted once the input has passed `t + h` (count) or the watermark has
  passed `t + duration` (duration), still stamped t. Stamps stay in increasing
  order, so it fits the event-time watermark model. At end of stream the rows
  still waiting are emitted with a NaN target, so streaming output equals the
  eager output. Memory is the h pending rows, not the dataset: large datasets
  that do not fit in memory stream through it.
- **The same option in JavaScript:** `forecastPairs(X, y, { count, index:
  "features" })`.
- The setting is checked like any other: a value other than the two strings
  raises a `ValueError` naming both.

This is a change of default with a "Changed" entry in the CHANGELOG: code that
relied on the padding at the start, or on lazy stamps at the target time,
passes `index="target"`. Code using `dropna=True` in count mode gets the same
arrays as before.

Implementation: the existing graph in `include/screamer/dag/forecast_pairs.h`
gets the second form; the stamp of a pending row is kept with it, so the
output event carries the features' index. Python and JavaScript adapt their
arguments only.

## `TripleBarrier` and `TripleBarrierOHLC`

The first barrier a trade touches after entry, following López de Prado
(2018), with the `side` input of meta-labeling built in.

Signatures:

- `TripleBarrier(horizon=20, take=2.0, stop=2.0, scale="vol", vol_span=100)`,
  inputs `side`, `price`.
- `TripleBarrierOHLC(...)`, same settings, inputs `side`, `open`, `high`,
  `low`, `close`.
- Outputs: `label`, `return`, `holding`.

The price-only and OHLC variants follow the backtest engines' convention
(`BacktestPriceTarget`, `BacktestOHLCTarget`): an operator has a fixed number
of inputs, so a second input layout is a second operator.

**Stamping.** Like any causal target, the output at step t is the outcome of
the trade entered at `t - horizon`, the step at which it is certain to be
resolved. It is NaN for the first `horizon` steps and for steps whose entry
had no side. `forecast_pairs(X, y, count=horizon)` then puts it next to the
features of its entry step, with the default `index="features"`. The guide
and the function page state the rule once, in that form: "use
`count=horizon`".

**Entries.** `side` at the entry step: `+1` opens a long trade at that step's
close, `-1` a short, `0` or NaN no trade. Only the sign is used. To label
every step by direction, pass `side = 1` everywhere: the label is then `+1`
when the upper barrier comes first and `-1` for the lower.

**Barriers**, as log returns from the entry price in the trade's direction:

- profit at `+take * w`, loss at `-stop * w`, with `w = sigma_h(e)` when
  `scale="vol"` and `w = 1` when `scale="absolute"`;
- `take=0` or `stop=0` turns that barrier off, as in the engines;
- the vertical barrier is `horizon` steps after entry.

`sigma_h(e)` is the volatility expected over the horizon, known at entry:
`EwStd(span=vol_span)` of the one-step log returns up to and including step
e, times `sqrt(horizon)`. Before the `EwStd` has data the labels are NaN.

**Outputs** for the entry `e = t - horizon`:

- `label`: `+1` profit barrier first, `-1` loss barrier first, `0` at the
  vertical barrier. For meta-labeling, the class to learn is `label == +1`.
- `return`: the trade's log return to its exit, signed by `side`. Its sign at
  the vertical barrier gives the alternative "sign at the vertical" label
  without another setting.
- `holding`: steps from entry to exit, 1 to `horizon`. The label's span is
  `[e, e + holding]`, which is what purging in the research project needs.

**Touching a barrier.**

- `TripleBarrier`: a barrier is touched when a later price reaches it (a price
  exactly at the level counts), and the exit is at that price.
- `TripleBarrierOHLC`: barriers are tested on each bar's high and low with the
  rules of `detail::risk_exits` in the engines: a gap through a barrier at the
  open exits at the open; when both barriers lie inside one bar the loss
  barrier wins, because the order of high and low within the bar is unknown;
  the exit price is the barrier level clamped to the bar. The implementation
  reuses `risk_exits`, so labels and backtests agree on the same bars.

**State.** Several trades can be open at once (every step can be an entry),
each resolved on its own. State is a ring of the open entries of the last
`horizon` steps; each step checks the open entries, at most `horizon`.

A non-positive or NaN price at entry or inside a trade's window gives that
trade a NaN label; other trades are unaffected.

## The guide: `docs/labels.md`, "Labels and training sets"

Opens with the training-versus-live table above, then the one rule (a target
is a causal series, stamped when its horizon ends; `forecast_pairs` with
`count=h` puts it at the features' time), then a chooser table. Each row is a
short, runnable example.

| I want to predict | Target |
|---|---|
| how much the price moves | `LogReturn(h)(price)` |
| up or down | `Sign` of that return |
| up, down, or flat beyond a dead zone | the return against a threshold, applied after pairing |
| a move in units of expected volatility | the paired return divided by a volatility feature from `X` |
| how volatile the next period is | `RollingStd(h)` of one-step returns |
| how far it runs for and against me | `RollingMax(h)` and `RollingMin(h)` of the price, over the price h steps ago |
| whether a trade with a stop and a target works | `TripleBarrier` or `TripleBarrierOHLC` |
| which assets do best | `LogReturn(h)` on a `(T, K)` panel, then `CrossRank` |

The volatility row explains why scaling after pairing is the simple way:
every row is stamped at the features' time, so a volatility column in `X` is
the volatility known at entry. No lag is needed.

## Example notebooks

Synthetic data with a known edge, as in notebooks 16 and 20, so the reader can
see that the method finds what is there. Each notebook is short and covers one
learning problem.

- **16, "Supervised forecasting" (rewritten).** Regression on the next-h
  return. The training-versus-live picture, `forecast_pairs` with the new
  default, a train/test split in time order, the same features used live.
- **23, "Classifying direction".** Up, down, or flat with a dead zone in
  volatility units; class balance; hit rate; turning predicted probabilities
  into positions; a backtest.
- **24, "Triple-barrier labels".** Labels drawn on a price chart; choosing
  take and stop from the excursion recipe; label counts and holding times;
  `TripleBarrierOHLC` against `BacktestOHLCTarget` on the same bars.
- **25, "Meta-labeling".** A simple rule gives the side; the triple barrier
  labels whether each trade worked; a second model decides which trades to
  take; backtest with and without the filter.

## Other documentation

- `forecast_pairs`: docstring, function page (`docs/functions_supervised/`),
  JavaScript docs, and help registry rewritten around the training-versus-live
  picture, with `index=` explained once. The current page leads with "lags X,
  never leads y", which no longer describes the default.
- Function pages for `TripleBarrier` and `TripleBarrierOHLC`, each with a plot
  of labels on a price chart. Both in the existing `supervised` topic.
- `docs/choosing_an_operator.md`: a short "Build a training set" section that
  links the guide.
- `docs/recipes/ml-features.md`, `docs/backtesting_best_practices.md`,
  `docs/capabilities.md`, and the agent guide: updated where they describe the
  old alignment.
- CHANGELOG: "Added" (`TripleBarrier`, `TripleBarrierOHLC`, `index=`, the
  guide, notebooks 23 to 25) and "Changed" (the `forecast_pairs` default).

## Testing

- `forecast_pairs`:
  - count mode, both settings, against a numpy reference; with `dropna=True`
    the two settings return identical arrays;
  - duration mode, `index="features"`, on an irregular index against a
    reference as-of join, including targets past the last y event;
  - eager equals lazy equals a graph in a `Pipeline`, for both settings,
    including the NaN rows flushed at the end of a stream;
  - streaming memory stays bounded: a long generator runs with a buffer of h
    rows;
  - a bad `index` value raises; JavaScript parity for both settings.
- `TripleBarrier`, `TripleBarrierOHLC`:
  - a plain numpy reference (`devtools/baselines`) on random walks, with NaN
    gaps and non-positive prices;
  - batch equals streaming, and `(T, K)` equals K separate runs;
  - no look-ahead: changing prices after step t leaves every output at or
    before t unchanged;
  - `TripleBarrierOHLC` exits agree with `BacktestOHLCTarget` stop and take
    exits on the same bars (gap at the open, both barriers in one bar,
    clamping);
  - edge cases: `horizon=1`; side changing every step; side NaN; `take=0` or
    `stop=0`; `scale="vol"` during the `EwStd` warmup;
  - JavaScript parity fixtures as for every operator.
- Docs: the guide's examples run; the notebooks execute in the docs build.

## Review focus

1. A user pairing a triple-barrier label with `count` different from its
   `horizon`: the function page and guide say "use `count=horizon`", and the
   notebooks always do.
2. Existing code that relied on the old `forecast_pairs` padding or lazy
   stamps: the CHANGELOG names the change and the one-word fix.
3. Overlapping triple-barrier trades when every step is an entry: each
   resolves on its own; a test covers 1000 consecutive entries.
4. A price of zero or below, or NaN, inside an open trade's window: that
   trade's label is NaN, others are unaffected.
5. Ties at a barrier (a price exactly at the level): counts as touched, the
   same rule as the engines.

## Open questions for the reviewer

1. `TripleBarrier` defaults: `horizon=20, take=2.0, stop=2.0, scale="vol"`.
2. The notebooks need a classifier. Options: scikit-learn (what users will
   use in practice; added to the docs extra only), or a few lines of numpy
   as notebook 16 does for regression. Recommendation: scikit-learn.
