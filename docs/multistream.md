# Streams, keys, and alignment

screamer's single-series operators (`RollingMean`, `RollingCorr`, …) assume
lockstep alignment: row `i` of one input pairs with row `i` of another. Real
multi-stream data breaks that assumption — feeds tick at different rates, arrive
out of step, and drop samples. The `screamer.streams` module adds a small,
composable layer for combining, splitting, filtering, and replaying streams that
do **not** tick together, while keeping every existing operator unchanged.

The whole design rests on four principles.

## 1. Every stream has an order key

A stream is a sequence of `(key, value)` events. The **key** is whatever you
already use to order data — a `datetime64` timestamp, an `int64` tick count, a
`float64` second, or, when you supply none, the **row number**. screamer only
ever *orders* and *compares* keys; it never interprets them.

Two capability tiers:

- **Comparable key** (can be ordered) — enough for `merge`, `combine_latest`,
  and backtest replay. Row numbers and any numeric key qualify.
- **Metric key** (differences are meaningful) — additionally enables wall-clock
  replay (`pace`), because a sleep duration only exists when key deltas convert
  to time.

Keys are numeric (`int64` or `double`), chosen from your array's dtype;
`datetime64` is carried losslessly as its underlying `int64`. The lockstep
behavior of the core operators is exactly the degenerate "no key → row number"
case, so nothing you already rely on changes.

## 2. Compute preserves shape; combinators change cardinality

| Layer | Cardinality | Examples |
|---|---|---|
| **Compute functors** | preserved (output length == input length) | `RollingMean`, `RollingCorr`, `FillNa`, `Ffill` |
| **Combinators** | may change it | `merge`, `combine_latest`, `dropna`, `filter`, `split`, `pace` |

Compute functors handle `NaN` internally via their `nan_policy` (see
[NaN policy](nan_policy.md)) and never add or drop rows. Combinators own all
time alignment and stream shaping. `dropna`/`filter`/`split` are the
cardinality-changing tools; `fillna`/`ffill` are shape-preserving and belong to
both worlds.

## 3. Alignment is a separate layer from computation

Time-aware combinators do the key handling and hand *aligned* data to the
unchanged compute functors. The idiom is:

```python
from screamer import combine_latest, RollingCorr

# Two async price feeds, each a (timestamps, prices) pair.
keys, aligned = combine_latest((t_a, p_a), (t_b, p_b))   # as-of latest-value join
corr = RollingCorr(20)(aligned[:, 0], aligned[:, 1])      # functor, untouched
```

`combine_latest` emits an aligned row whenever any input advances, carrying each
input's most recent value (forward-fill). `emit="when_all"` (default) waits
until every input is warm; `emit="on_any"` emits from the first event with
`NaN` for inputs not yet seen. Feed the aligned columns to any existing functor.

Other combinators:

- `merge(*series)` → one key-sorted, source-tagged stream (`keys, values, sources`).
- `split(keys, values, sources, n=None)` → the inverse of `merge`.
- `dropna(keys, values, how="any")` / `filter(keys, values, predicate)` → drop events.
- `pace(*series, speed=1.0)` → async replay; `speed=inf` is a max-speed backtest.

The batch combinators each have a streaming twin (`merge_iter`, `combine_latest_iter`,
`dropna_iter`, `filter_iter`) that yields events one at a time.

## 4. Causal, and identical across modes

- **Causal**: an output at key `t` depends only on events at keys `≤ t`. There
  is no backward-fill and no lookahead operator, ever.
- **Batch == streaming == replay**: the batch form and its streaming twin emit
  byte-identical event sequences; `pace` changes only *when* events are emitted,
  never their values or order. This is what lets you validate a pipeline on
  stored data and run the identical pipeline live. It is enforced by the
  identity matrix in `tests/test_streams_identity.py`.

## See also

- [Polymorphic API](polymorphic_api.md) — the single-series input/output
  contract; lockstep is the row-number-key special case of this page.
- [NaN policy](nan_policy.md) — how compute functors treat `NaN`; `ffill` is the
  same forward-fill carry that `combine_latest` uses.
