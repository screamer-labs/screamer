# Delay and forecast_pairs: time-based shift and supervised label pairing

## Context

Trading-bot research constantly needs a supervised training set: features at time t
paired with a target realized in t's future ("do these features predict the next-h
move?"). Building that pairing by hand is error-prone. In screamer-bots it showed up
as hand-rolled `forward_slope` and `forward_return` labels, each with its own
fencepost (which array to shift, is "now" inside the window, drop the leading or
trailing rows), and the recurring risk is leakage: accidentally letting the
forward-looking target reach back into the features.

screamer already owns the causal operator layer, so the fix is not a new class of
operators. It is a small pairing utility plus one missing shift primitive. The key
observation that shapes the whole design:

**A forward label is a causal trailing quantity paired with lagged features.** You
never need an anti-causal operator. `forward_return(ret, h)[s]` equals the causal
`RollingSum(h)(ret)[s+h]`, so pairing `X[s]` with `RollingSum(h)(ret)[s+h]` gives the
forward-return training pair with no operator that sees the future. The
"looking forward" lives entirely in the pairing, never in a transform. This means
the entire forward-label category dissolves into (a causal trailing quantity screamer
already computes) plus (a shift and a join).

The shift has two honest forms, and screamer only has one of them today.

## Two faces of shift: count and time

Shifting a series backward has two distinct meanings that coincide on a regular grid
and diverge on irregular (async) event streams:

- **Count**: "N observations ago." Event-based, ignores wall-time gaps, needs no
  timestamp. This is today's `Lag(window_size)` (a fixed-size ring buffer).
- **Time**: "delta of wall-time ago." Time-based, ignores event count, requires a
  timestamp index. screamer has no primitive for this.

On a regular grid with spacing s, `Lag(N)` and a time shift of `N*s` produce the
same numbers, which is why the distinction never surfaced. On an async feed (multi
venue ticks at different cadences) they are genuinely different operations with
different mechanisms and different requirements, so they get two names, not one
dual-mode function.

The time shift itself has a single truthful meaning: a **latency line**. Re-stamp
every event `(t, v)` to `(t + delta, v)`. It preserves every event, is 1:1 in event
count, is order-preserving, and simply starts `delta` late (no NaN warmup). The other
thing one might reach for, an "as-of value sampled on some clock," is not a
primitive: it is the latency line composed with `combine_latest` (or `Resample`), and
its lossiness (superseded values disappearing) belongs to that resample step, not to
the shift. We therefore do not build a monolithic as-of operator; we build the pure
latency line and let the existing `combine_latest` do the alignment.

## Design

Three pieces, in dependency order.

### 1. `Lag(window_size)` (unchanged)

Count-based shift, the existing fixed-ring delay. No API change, no index required.
It answers "N events ago."

### 2. `Delay(duration)` (new)

The time-based latency line, a **stream op** (it manipulates the timestamp index, not
just values, so it belongs with `Resample` / `CombineLatest`, not the plain 1 -> 1
functors).

- Signature: `Delay(duration)`, `duration` positional and numeric in **index units**
  (like `Resample`'s numeric window; no string or calendar parsing in the core).
  `Delay(600_000)` on a millisecond index is a 10-minute delay.
- Semantics: re-stamp each event `(t, v)` to `(t + duration, v)`. Lossless, 1:1,
  order-preserving. The first output event is at `t0 + duration`; there is no NaN
  warmup, the stream just starts late.
- Requires a timestamp index. Called without one it is an error (there is nothing to
  shift against). This requirement is structural, not a runtime "did you pass an
  index" branch: `Delay` is a stream op that consumes `(values, index)`.
- Batch implementation is trivial: `(values, index + duration)`.
- Streaming implementation is the real work: emitting an event at a **future** index
  is a new engine capability. Nothing currently emits future-dated events, so `Delay`
  needs a reorder / pending buffer that holds each event until logical time reaches
  its shifted index. That buffer is variable-size (it spans one `duration` window of
  events, so its footprint is `duration * event_rate`, unlike `Lag`'s fixed ring).
  batch and stream must produce identical `(index, value)` pairs.

### 3. `forecast_pairs(X, y, count=|duration=)` (new)

Thin Python that builds the supervised training set. It lives in a fenced supervised
namespace (`screamer.supervised`), separate from the causal-op core, because it is
offline training-data preparation with an sklearn-shaped `(X, y)` contract.

- It composes existing C++ nodes and adds no new compute:
  `combine_latest(shift(X), y)` then drop the warmup rows, where `shift` is `Lag(N)`
  for `count=N` or `Delay(delta)` for `duration=delta`.
- Exactly one of `count=` / `duration=` is given (a one-of, like `Resample`'s
  `freq=`/`count=`). `count=` always works; `duration=` inherits `Delay`'s
  requirement that X carries an index.
- Semantics: pairs features from the past with the target now, i.e. "features predict
  the target `horizon` ahead." Because it lags X (causal) rather than leading y
  (anti-causal), every operation is causal; it is streamable, and it is structurally
  impossible to feed y back as a live feature (at decision time the paired target does
  not exist yet).
- Contract on the caller: **y must be causal (known as-of its own index).** The
  intended use is to build y from a causal trailing screamer op (e.g.
  `RollingSum(h)(ret)`) and hand it in. A forward y would leak, but the whole point is
  that you never need one.
- Output: the paired `(X_shifted, y)`. Default leaves the leading warmup rows of
  `X_shifted` as NaN (the screamer idiom: do not slice, emit NaN, let the caller
  drop) and preserves alignment; `dropna=True` returns NaN-free arrays for a direct
  sklearn hand-off. It also returns the `as_of` index (the completion timestamp of
  each example, when its y is realized), which is exactly the boundary a
  leakage-aware splitter/embargo needs.

Why the dual-kwarg here but two functions for `Lag`/`Delay`: the split rule is "split
when the operations differ, dual-kwarg when it is one operation spelled two ways."
`Lag` and `Delay` are different operations (different buffers, different index
requirement), so overloading them would hide a behavioral fork. `forecast_pairs` has
one behavior (shift X, join to y, drop warmup); `count` vs `duration` only selects the
shift primitive, so a single function with a one-of is honest, and it matches
`Resample`'s own `count`/`freq` shape.

## Naming rationale

- `Delay`, not `Lag(freq=...)`: "freq" imports a repeating-clock meaning into a
  one-shot displacement. `Delay` is the standard delay-line term and reads correctly
  for a latency line. `Lag` keeps its discrete-shift (autocorrelation-lag, z^-1)
  connotation.
- `forecast_pairs`, not `make_supervised` / `lookahead` / `oracle` / `predict`: the
  name should describe the task (build forecasting pairs), not the mechanism (shift),
  and must not imply the transform peeks into the future (it does not) or collide.
  `predict` collides with sklearn inference; `oracle` collides with the codebase's
  existing "oracle tests" and implies cheating; `lookahead` / `info-leak` mis-teach a
  causal operation as anti-causal (they are good docstring warnings, wrong as names).
- `duration`: the clearest single word for the shift amount; numeric, index units.

## Problems and policies

Settled behaviors the implementation must pin down:

- **Ties (same timestamp):** coalesce, matching `combine_latest`'s "one row per
  distinct index" (same-index events settle into one row).
- **As-of boundary:** inclusive (`<=`), "the most recent value at or before the
  shifted time."
- **Warmup:** for `Delay`, the stream starts `duration` late (a time span), not a
  fixed number of rows; do not paper over this with a fixed-count NaN prefix.
- **Buffer bound:** `Delay`'s reorder buffer is data-dependent (`duration *
  event_rate`). A burst inside one `duration` window grows it. Provide a cap with a
  documented drop-or-error policy rather than unbounded growth.
- **Out-of-order / late input:** the as-of/index model assumes a non-decreasing index.
  Define the behavior for a late event (assume sorted and document, or reject); do not
  silently produce a retroactively wrong shift.
- **End-of-stream tail (streaming):** events still pending (shifted index beyond the
  last seen time) at end of stream. Inherit `Resample`'s flush rule so batch and
  stream agree on whether the tail is emitted or withheld.
- **Index requirement:** `Delay` and `forecast_pairs(duration=)` require an index and
  error clearly without one; `Lag` and `forecast_pairs(count=)` never need one.

## C++ / Python split

- `Delay` is a **C++ stream op** (in the streams layer, alongside `Resample` /
  `CombineLatest`). The batch path is index arithmetic; the streaming path is the
  reorder buffer. All compute in C++.
- `forecast_pairs` is **thin Python** in `screamer.supervised`. It orchestrates
  `Lag` / `Delay` / `CombineLatest` / a drop-NaN step, all of which are already C++
  nodes, so no operator logic lives in Python. (It may equivalently be expressed as a
  `Pipeline` factory to run entirely through the C++ driver; either way Python only
  wires.)

## Validation

- `Delay` batch: `(values, index + duration)`, lossless, order-preserving; hand
  checked on a regular grid (equals the matching `Lag` numbers) and on an irregular
  feed (the 7s-feed / 5s-delay worked example: value observed at t=7 becomes current
  at t=12).
- `Delay` batch == stream on the `(index, value)` pairs, including ties, the pending
  tail under flush, and a burst that exercises the reorder buffer.
- `forecast_pairs(count=)` reproduces the hand-rolled `forward_return` /
  `forward_slope` pairing exactly (the screamer-bots rewrites are the reference), for
  the fencepost and the short-series (n <= horizon) edge.
- `forecast_pairs(duration=)` on a regular grid equals `forecast_pairs(count=)` with
  the matching sample count; on an irregular feed it pairs by wall-time.
- `forecast_pairs` errors on `duration=` without an index, and on both/neither of
  `count`/`duration`.
- The `as_of` index equals the target's completion time per row.

## Migration and scope

- No breaking change: `Lag` is untouched; `Delay` and `forecast_pairs` are additive.
- The screamer-bots `forward_return` / `forward_slope` helpers become one-line
  `forecast_pairs` calls once this ships (a downstream follow-up in that repo, not part
  of this spec).
- Out of scope, noted so they are not lost: the footprint-aware embargo (deriving the
  purge size from the operator graph's warmup plus the label horizon) and the
  scheduled-refit walk-forward node. Both build naturally on `forecast_pairs`'s
  `as_of` index and are separate specs.
