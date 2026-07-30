# screamer: FracDiff (fractional differentiation) - Design

Date: 2026-07-30
Status: approved, ready to implement

## Context

Mohammadjavad Vakili wrote a `RollingFracDiff` operator for an early version of
screamer that was never merged. It implements Marcos Lopez de Prado's fractional
differentiation (*Advances in Financial Machine Learning*, chapter 5): a filter
that makes a series stationary while keeping more of its memory than an integer
difference does.

The original is a working sketch against an older base class. It computes the
weight recursion correctly. Everything around the recursion predates the
conventions the library now enforces: it validates arguments after using them,
buffers with `std::deque`, has no `NaN` policy, has no `start_policy`, and
duplicates a convolution the library already owns in `MovingAverage`.

This is his only contribution, and the attribution should say so precisely: a
contributor of one operator rather than a co-author of the library.

## Goal

Ship `FracDiff` as a C++ node that meets the current definition of done, and
record the attribution in the places that describe who wrote what.

## API

```python
FracDiff(d=0.4, window_size=100, threshold=1e-5, start_policy="strict")
```

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `d` | float | 0.4 | Fractional order. `d=0` is identity, `d=1` the first difference, `d=2` the second. Negative values are fractional integration and are allowed. |
| `window_size` | int >= 1 | 100 | Hard cap on the number of taps, so memory and per-step cost are bounded. |
| `threshold` | float >= 0 | 1e-5 | Drop the tail once `abs(w_k) < threshold`. |
| `start_policy` | str | `"strict"` | `strict`, `expanding`, or `zero`, as defined in `docs/nan_and_warmup.md`. |

The effective filter length is `L = min(window_size, first k with abs(w_k) < threshold)`.
`L`, not `window_size`, is what warmup waits for, because `L` is the point at
which the filter is fully defined.

Family `signal`: it is an FIR filter, so it belongs beside `MovingAverage`,
`Butter*`, and `KalmanFilter` rather than among the windowed statistics. The
`Rolling` prefix in this library marks a trailing-window statistic, which this is
not, so the name drops it.

## Weights

```
w_0 = 1
w_k = -w_{k-1} * (d - k + 1) / k
```

which is `w_k = (-1)^k * binom(d, k)`. The recursion is the one part of the
original code that carries over unchanged.

The output is the convolution

```
y[t] = sum_{k=0}^{L-1} w_k * x[t-k]
```

`w_0` multiplies the current sample.

## Implementation

`include/screamer/frac_diff.h`, class `FracDiff : public ScreamerBase`.

- Arguments are validated in the constructor before the weight loop runs. The
  original called `compute_weights()` first and then checked `window_size`,
  so an invalid window ran the loop before raising.
- Weights are computed once in the constructor and stored in a `std::vector`.
- The convolution and its ring buffer move into `include/screamer/detail/fir_core.h`,
  shared by `MovingAverage` and `FracDiff`. The library forbids a second
  implementation of a behavior it already has, and a shared core means the `NaN`
  fix below is written once.
- `process_scalar` returns `NaN` immediately on a `NaN` input, leaving state
  untouched.

Cost is O(L) per step with a single `L`-element buffer, matching `MovingAverage`.

## Warmup and NaN

`nan_policy: ignore`, matching every other filter in the library. A `NaN` input
is skipped, internal state is unchanged, and one `NaN` appears at that index.
Warmup is counted in finite samples.

- `strict` (default): `NaN` until `L` finite samples have arrived.
- `expanding` and `zero`: the truncated convolution from the first sample, which
  is what the original code did.

For a linear filter, using only the available samples is the same arithmetic as
padding the missing past with zeros, so `expanding` and `zero` produce identical
output. Both values are accepted so that `start_policy` means the same thing
across the library, and the page says plainly that they coincide here.

## Verification

The identities come first because they check the weights against code that was
written independently of this operator:

- `FracDiff(d=0)` reproduces the input exactly.
- `FracDiff(d=1, window_size>=2)` equals `Diff(1)`.
- `FracDiff(d=2, window_size>=3)` equals `Diff2()`.

The recursion gives `[1]`, `[1, -1]`, and `[1, -2, 1]` for those three orders.

Then:

- `devtools/baselines/FracDiff.py` with a numpy transcription of Lopez de Prado's
  `getWeights_FFD` / `fracDiff_FFD`, which `tests/test_baselines.py` picks up.
- Weights against the closed form `(-1)^k * binom(d, k)` via `scipy.special.binom`.
- Identical output in all three regimes (eager, `Pipeline`, lazy iterator) and
  batch against streaming, via `tests/regime_helpers.py`.
- A threshold above `abs(w_1)` collapses the filter to identity; a tighter
  threshold changes only the tail contribution.
- `NaN` compliance: one `NaN` in gives one `NaN` out, and the next output matches
  the run where that sample was never `NaN`.
- `make tidy` clean.

## MovingAverage NaN fix

`MovingAverage` declares `nan_policy: ignore` and behaves as `propagate`. On
`[1, 1, 1, 1, NaN, 1, ...]` it emits two `NaN`s where `WMA` emits one, because
`process_scalar` writes the `NaN` into its buffer instead of returning early.

`tests/test_nan_input_compliance.py` misses this: it asserts that a `NaN` does
not poison state forever, and that the output at the `NaN` index is `NaN`, but
never that the *following* output is unaffected.

Fix both:

- The shared FIR core returns early on `NaN`, which corrects `MovingAverage` and
  keeps `FracDiff` from inheriting the same trap.
- Add a property to `tests/test_nan_input_compliance.py`: for every function
  declaring `ignore`, a single mid-stream `NaN` changes the output at that index
  only. Any other operator this catches goes into a `KNOWN_*` xfail set, which is
  the pattern that file already uses, rather than growing this change.

## Attribution

- `AUTHORS.md`: Mohammadjavad Vakili moves from "created and maintained by" to a
  `Contributors` entry naming `FracDiff`.
- `pyproject.toml`: the poetry `authors` list names him too, and is updated to
  agree with `AUTHORS.md`.
- `CHANGELOG.md`: an Unreleased entry that credits him.
- The implementation commit carries
  `Co-Authored-By: Mohammadjavad Vakili <mohammadjavad.vakili@gmail.com>`, which
  is what puts him on the contributors graph.

## Definition of done

From `CONTRIBUTING.md`:

1. C++ node plus a thin binding in `bindings/bindings_signal.cpp`.
2. `docs/functions_signal/FracDiff.md` with frontmatter, topics `filtering` and
   `data-prep`, and `nan_policy: ignore`.
3. A baseline in `devtools/baselines/`.
4. `make build`, then `devtools/build_help_registry.py` and
   `devtools/build_topic_pages.py`.
5. Tests proving eager, graph, and lazy regimes agree.
6. `pytest -q` green with no skips, `make tidy` clean, docs build clean.

## Out of scope

- The expanding-window (non-fixed-width) fracdiff variant from the same chapter.
  The threshold-truncated fixed-width filter is the one that streams.
- Automatic selection of `d`, for example by searching for the smallest `d` that
  passes an ADF test. That is a study, not a streaming operator, and it needs a
  hypothesis test the library does not provide.
- Exposing the resolved tap count `L` as an attribute. The page documents how to
  derive it; add it if a user asks.

## Resolved decisions

| Question | Decision |
|---|---|
| Name and family | `FracDiff`, signal family |
| Window parameterisation | `window_size` required, `threshold` trims the tail |
| `expanding` for a linear filter | Accepted, documented as identical to `zero` |
| `MovingAverage` mismatch | Fixed in this change, with a compliance property to match |
