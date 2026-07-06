# NaN and warmup

Two closely related situations produce `NaN` in a streaming statistic: a `NaN`
in the *input*, and the *warmup* period before a function has seen enough data
to produce a value. This page defines both — how every function responds to a
`NaN` input, and how the `start_policy` argument controls warmup.

The first half is the **NaN policy**: every function declares one of three NaN
policies in its frontmatter, the build refuses to publish a function that doesn't
declare one, and the test suite verifies that the runtime behavior matches the
declaration. There are no "it depends" answers - the function's documentation
page tells you what happens, and CI guarantees the page is not lying. The second
half is **warmup and `start_policy`**, which is the authoritative definition that
the individual function pages refer to.

> For dropping vs filling `NaN` **across streams** (`dropna`, `fillna`/`ffill`
> in the stream operator layer), see [Streams, values, and alignment](multistream.md).
> `ffill` there is the same forward-fill carry that `combine_latest` uses.

## The contract

A `NaN` in the input never corrupts internal state. Output may be `NaN` at and around the input `NaN` index depending on the function's policy, but the function always recovers - there is no "sticky `NaN`" that poisons subsequent outputs forever.

`process_scalar` always consumes one sample and emits one sample. The output array is the same length as the input array. `NaN` inputs do not collapse, expand, or shift the timeline.

## The three policies

### `ignore`

Used by every summary-statistic function (rolling means/variances/quantiles/poly fits, EW family, bands, oscillators, volatility estimators, smoothers, cumulative reductions).

**Rule.** When the input at index `t` is `NaN`:

1. Emit `NaN` at output index `t`.
2. Leave internal state exactly as it was at the end of step `t − 1`. The sample is not stored in the lookback buffer, not added to running sums, not used to advance EW decay.

The function behaves at step `t + 1` as if step `t` had not happened - except that one output slot has been filled with `NaN` to preserve the 1:1 length invariant.

This is implemented as a one-line early return at the top of `process_scalar`:

```cpp
double process_scalar(double x) override {
    if (isnan2(x)) return std::numeric_limits<double>::quiet_NaN();
    // ... normal logic, guaranteed to never see NaN ...
}
```

**Worked example.** `RollingMean(3)` on `[1, 1, 1, 1, NaN, 1, 1, 1, 1]`:

| index | input | output | buffer after step |
|---|---|---|---|
| 0 | 1   | `NaN` (warmup) | `[1]` |
| 1 | 1   | `NaN` (warmup) | `[1, 1]` |
| 2 | 1   | 1   | `[1, 1, 1]` |
| 3 | 1   | 1   | `[1, 1, 1]` |
| 4 | `NaN` | `NaN` | `[1, 1, 1]` (unchanged) |
| 5 | 1   | 1   | `[1, 1, 1]` |
| 6 | 1   | 1   | `[1, 1, 1]` |
| 7 | 1   | 1   | `[1, 1, 1]` |
| 8 | 1   | 1   | `[1, 1, 1]` |

Single input `NaN` → single output `NaN`. Recovery is immediate.

**Semantic consequence.** Under `ignore`, the window length is measured in *finite samples*, not in input positions. If an input stream has many `NaN` gaps, the windowed statistic effectively spans more positions than the nominal window size. This is the right default for a streaming library - but if a downstream consumer needs strictly time-positional semantics, they should fill or drop the `NaN` upstream (with [`FillNa`](functions_preprocessing/FillNa.md) or [`Ffill`](functions_preprocessing/Ffill.md)) before feeding the statistic.

### `propagate`

Used by functions whose output references the input by an explicit positional offset: [`Lag`](functions_misc/Lag.md), [`Diff`](functions_misc/Diff.md), [`Diff2`](functions_misc/Diff2.md), [`Momentum`](functions_misc/Momentum.md), [`ROC`](functions_fin/ROC.md), [`ROCP`](functions_fin/ROCP.md), [`ROCR`](functions_fin/ROCR.md), [`LogReturn`](functions_fin/LogReturn.md), [`Return`](functions_fin/Return.md).

**Rule.** The function stores every input - including `NaN` - in its lookback. Output at index `t` is computed by the function's positional formula, with IEEE arithmetic propagating `NaN` naturally. Output recovers once the `NaN` slides out of the lookback.

**Why these functions use propagate and not ignore.** `Lag(3)` is contractually "the value 3 positions ago." If we silently dropped `NaN` from the buffer, `Lag(3)` would become "the value 3 *finite samples* ago" - silently shorter than 3 positions whenever the stream has gaps. Same argument for `Diff`, `ROC`, etc.: their semantics require honest positional offsets. Propagate gives the user a visible `NaN` that says "I can't answer that" rather than a wrong-but-finite answer.

**Worked example.** `Diff(1)` on `[1, 2, 3, NaN, 5, 6]`:

| index | x[t] | x[t-1] | output |
|---|---|---|---|
| 0 | 1   | -    | `NaN` (warmup, no previous) |
| 1 | 2   | 1    | 1   |
| 2 | 3   | 2    | 1   |
| 3 | `NaN` | 3  | `NaN` (input is `NaN`) |
| 4 | 5   | `NaN` | `NaN` (previous is `NaN`) |
| 5 | 6   | 5    | 1   |

One input `NaN` → two output `NaN`s (the input's own index, and one more while it sits in the lag-1 lookback), then full recovery.

**Generalization.** For lookback `n`:

- `Lag(n)`: 1 in → 1 out (just shifted by `n`).
- `Diff(n)`, `Momentum(n)`, `ROC(n)`, `ROCP(n)`, `ROCR(n)`, `LogReturn(n)`, `Return(n)`: 1 in → 2 out, at positions `t` and `t + n`.

### `nan-aware`

Used by functions whose **purpose** is to handle `NaN`: [`FillNa`](functions_preprocessing/FillNa.md), [`Ffill`](functions_preprocessing/Ffill.md), and any future `Bfill`-style additions.

**Rule.** The function's own documentation defines what happens - `NaN` input is the input it was designed to process, and the output is by design *not* `NaN`. These functions are the only place in the library where input `NaN` can become a finite output.

## Which functions use which policy

The full enumeration. This list is regenerated from frontmatter; if it drifts from the per-function pages, that's a bug.

### `ignore`

All `Rolling*` functions, all `Ew*` functions, all `Cum*` functions, all moving-average / smoothing functions (`WMA`, `TRIMA`, `HullMA`, `DEMA`, `TEMA`, `KAMA`, `MovingAverage`, `MACD`, `Trix`), all bands (`BollingerBands`, `KeltnerChannels`, `DonchianChannels`), all volatility estimators (`TrueRange`, `ATR`, `NATR`, `Parkinson`, `GarmanKlass`, `RogersSatchell`, `YangZhang`, and their `Rolling*` / `Ew*` variants), all oscillators (`Stoch`, `StochRSI`, `UltimateOscillator`, `WilliamsR`, `CCI`, `BOP`, `MFI`), all volume indicators (`AD`, `ADOSC`, `ADX`, `OBV`, `VWAP`), `Drawdown`, `MaxDrawdown`, and all IIR/FIR filters (`Butter*`, `KalmanFilter`, `WilderSmoother`).

### `propagate`

`Lag`, `Diff`, `Diff2`, `Momentum`, `ROC`, `ROCP`, `ROCR`, `LogReturn`, `Return`.

### `nan-aware`

`FillNa`, `Ffill`.

## Edge cases

### Leading `NaN` warmup from another function

The original motivation for this policy. Chaining `RollingPoly1(window=21)` on top of `EwKurt(span=20)`'s output:

- `EwKurt` emits a handful of leading `NaN`s during its own warmup, then finite values.
- Under `ignore`, those leading `NaN`s do not enter `RollingPoly1`'s state. `RollingPoly1` simply doesn't see them.
- `RollingPoly1` runs its own warmup over the finite samples that follow. First valid output appears at index `(EwKurt warmup) + (RollingPoly1 window − 1)`.

Concretely, on a 200-sample finite input, the chain emits roughly 23 `NaN` outputs followed by 177 finite values - versus the previous behavior of 200 `NaN`s out of 200.

### Single mid-stream `NaN`

- `ignore` policy: exactly 1 `NaN` output at the same index. Full recovery at the next step.
- `propagate` policy: depends on the function's lookback (see `Diff` example above). Recovery is bounded by the lookback length.

### All-`NaN` input

- `ignore`: all outputs are `NaN`. No exception, no infinity.
- `propagate`: all outputs are `NaN`.
- `nan-aware`: behavior is function-specific (e.g. `Ffill` returns its initial `NaN` until any finite sample arrives).

### `NaN` tail

`NaN`s at the end of the stream behave the same as `NaN`s in the middle. The function does not "freeze" or hold the last finite output.

## What we don't do

Two policies are deliberately out of scope for now. Both are reasonable in some contexts, both are exclude-by-default because they introduce subtle silent failure modes.

### Decay-aware EW (pandas's `ignore_na=False`)

Pandas's `ewm` supports two NaN modes:

- `ignore_na=True`: equivalent to our `ignore`. `NaN` positions are invisible to the decay; before-and-after-gap outputs are computed as if the gap didn't exist.
- `ignore_na=False` (pandas default): the exponential decay still ages over `NaN` positions, even though no value contribution accumulates.

Screamer EW functions implement the `ignore_na=True` semantics. A future opt-in flag could expose `ignore_na=False`-style behavior; the choice was to start with the simpler, more streaming-natural default.

### "Treat as zero"

It is tempting to say "if `x[t]` is `NaN`, just pretend it's 0 and keep going." For any mean/sum-like statistic, this silently biases the output toward zero. The library refuses to do this anywhere; if you want zeros, write them explicitly with [`FillNa(0.0)`](functions_preprocessing/FillNa.md).

## Verifying compliance

`tests/test_nan_input_compliance.py` enumerates every entry in `screamer/data/help.json`, reads each function's `nan_policy` declaration, constructs the function with documented defaults, and asserts:

1. **No state poisoning.** After any `NaN` input, the function must produce a finite output within at most `lookback + warmup` finite samples (where `lookback` is `window_size` or the relevant order parameter).
2. **`ignore` invariant.** For an `ignore`-policy function, output at index `t` is `NaN` *if and only if* either (a) input at `t` is `NaN`, or (b) `t` is in the function's intrinsic warmup region computed over the finite-sample subsequence.
3. **`propagate` invariant.** For a `propagate`-policy function, output at index `t` is `NaN` if any input the function would reference (per its positional formula) is `NaN`.
4. **`nan-aware` invariant.** No invariant beyond what the function's own page documents.

These tests are deterministic and run on every commit. If a function's runtime behavior diverges from its declared policy, CI fails.

## Warmup and `start_policy`

Separate from input `NaN`s, most windowed and recurrence-based functions have a
**warmup** period at the start of a stream, before they have seen enough samples
to produce a defined value. A `RollingMean(20)` has no mean to report until 20
samples have arrived. What happens during that period is controlled by the
`start_policy` argument.

This section is the canonical definition; individual function pages refer here
rather than repeating it.

### The three policies

`start_policy` accepts one of three values. The default everywhere is
`"strict"`.

- **`"strict"`** (default) — return `NaN` for every step until the full window
  has been seen (`window_size` samples for a rolling function). Nothing is
  reported until the statistic is fully defined.
- **`"expanding"`** — compute with whatever samples are available, starting from
  the first one and growing the effective window until it reaches
  `window_size`. Early outputs are defined but based on fewer samples. Some
  functions need a minimum count before any output is meaningful (for example a
  correlation needs at least two samples); those return `NaN` until the minimum
  is met.
- **`"zero"`** — behave as if the stream were pre-filled with `window_size`
  zeros before the real data, so the window is "full" from the first real
  sample. Early outputs are defined but biased toward zero by the padding.

Warmup is measured in *finite* samples. Under the `ignore` NaN policy, a `NaN`
input is skipped and does not advance warmup — so a stream with gaps reaches the
end of warmup after the same number of *finite* samples, not the same number of
positions. (See the [`ignore`](#ignore) policy above.)

### Which value to choose

`"strict"` is the honest default: no output until the statistic is genuinely
defined, which is what you want for a backtest that must not act on
half-formed values. `"expanding"` trades some statistical stability for earlier
output, useful when you cannot afford a long dead period at the start of a
stream. `"zero"` is rarely the right choice for analysis — the zero padding
biases the early window — but it is occasionally convenient when a downstream
consumer requires a value at every index and you will discard the early region
anyway.

### Warmup interacts with chained functions

When one function feeds another, warmup regions stack, and the `ignore` policy
keeps them honest. A leading run of `NaN`s from an upstream function's warmup
does not enter the downstream function's state; the downstream function simply
runs its own warmup over the finite samples that follow. The worked example is
in [Leading `NaN` warmup from another function](#leading-nan-warmup-from-another-function)
above.

## Implementation notes for contributors

When writing a new function, pick the policy that matches the function's mathematical meaning, declare it in the frontmatter (`nan_policy: ignore | propagate | nan-aware`), and implement it. The `ignore` case is almost always one line of code (`if (isnan2(x)) return NaN;` at the top of `process_scalar`). The `propagate` case typically needs no `NaN` handling at all - IEEE arithmetic does the right thing - but the lookback buffer must store `NaN` faithfully.

If a function genuinely needs different semantics from any of the three policies, that's a signal to discuss the API before merging. We have not yet found a case where one of the three doesn't fit.
