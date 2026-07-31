# Performance

There are two questions, and they have different answers. Over a whole array,
screamer is as fast as or faster than the equivalent numpy and pandas code, and
many times faster for rolling-window statistics. Over a live stream, where a
value is needed before the next event arrives, the gap is much wider, because a
batch library has no way to update a result: it recomputes the trailing window
every event.

In both regimes the reason is the algorithms, not just the C++. Most functions
update their result in constant time per sample and make a single pass, so their
cost does not grow with the window. The rank-based statistics (the rolling
median, quantiles, IQR, and CVaR) keep a sorted structure and update in
O(log window). numpy rebuilds each window, so its cost grows linearly with the
window; pandas carries per-call overhead on top of its own pass.

## Batch speedups

The chart compares screamer against the fastest of numpy, pandas, and scipy on a
one-million-element array. A bar past `1x` means screamer is faster; the colour
marks which library came second.

![screamer speedup versus the fastest alternative](img/speed_chart.png)

screamer is faster on 29 of the 32 functions and level on the other three:
`EwMean` at 0.96x, `Abs` and `Sqrt` at 0.99x, which is parity within measurement
noise. Those three are a single pass over memory with nothing to improve on.

The rolling statistics run 1.8x to 14x faster than pandas, and far faster than
numpy, whose sliding-window cost grows with the window while screamer's does
not. Full numbers, against numpy and pandas separately:

| function | vs numpy | vs pandas | vs fastest |
|---|---|---|---|
| `RollingMax` | 7.5x | 14x | **14x** |
| `RollingMin` | 7.0x | 13x | **13x** |
| `Clip` | 1.8x | 13x | **13x** |
| `Sign` | 12x | - | **12x** |
| `Return` | 1.9x | 8.5x | **8.5x** |
| `RollingMean` | 6.1x | 5.6x | **5.6x** |
| `RollingZscore` | 7.9x | 5.2x | **5.2x** |
| `RollingStd` | 6.1x | 3.4x | **3.4x** |
| `RollingVar` | 6.1x | 3.3x | **3.3x** |
| `RollingQuantile` | 1.7x | 2.8x | **2.8x** |
| `RollingSkew` | 29x | 2.2x | **2.2x** |
| `RollingKurt` | 24x | 2.2x | **2.2x** |
| `RollingPoly1` | 2.0x | 2.0x | **2.0x** |
| `Diff` | 1.0x | 1.9x | **1.9x** |
| `RollingMedian` | 1.2x | 1.9x | **1.9x** |
| `EwZscore` | - | 1.9x | **1.9x** |
| `RollingSum` | 1.7x | 1.8x | **1.8x** |
| `RollingRms` | 2.9x | 1.8x | **1.8x** |
| `FillNa` | 2.6x | 1.4x | **1.4x** |
| `Erfc` | - | - | **1.3x** |
| `Erf` | - | - | **1.2x** |
| `Lag` | 1.0x | 1.2x | **1.2x** |
| `EwStd` | - | 1.1x | **1.1x** |
| `LogReturn` | 1.0x | 1.1x | **1.1x** |
| `Butter2` | - | - | **1.1x** |
| `Exp` | 1.1x | - | **1.1x** |
| `Ffill` | 5.0x | 1.1x | **1.1x** |
| `EwVar` | - | 1.0x | **1.0x** |
| `Log` | 1.0x | - | **1.0x** |
| `Sqrt` | 1.0x | - | **1.0x** |
| `Abs` | 1.0x | - | **1.0x** |
| `EwMean` | 0.7x | 1.0x | **1.0x** |

## Streaming

Batch numbers describe one regime. The other is a feed: an event arrives and the
current value of the statistic is needed before the next one does.

screamer updates its state and returns the new value, which is O(1) or
O(log window) per event. numpy, pandas and TA-Lib have no incremental entry
point, so the only way to get the current value is to hand them the trailing
window again. That is O(window) per event however fast the library is.

Cost per event, on a stream of 200,000 samples of `RollingMean`:

| window | screamer | TA-Lib | numpy | pandas |
|---|---|---|---|---|
| 10 | **93 ns** | 557 ns | 1530 ns | 12601 ns |
| 100 | **92 ns** | 613 ns | 1532 ns | 12689 ns |
| 1000 | **92 ns** | 1373 ns | 1675 ns | 13496 ns |
| 5000 | **92 ns** | 4324 ns | 2230 ns | 15604 ns |

screamer is 6x to 47x faster than TA-Lib here, 16x to 24x faster than numpy, and
135x to 170x faster than pandas. It is also the only one whose cost does not grow
with the window: going from a window of 10 to 5000, TA-Lib goes from 557 to 4324
ns per event and screamer does not move.

The loop in that table is compiled, so the interpreter is not part of the
measurement. Compiling the event loop is worth having, and it helps every
library: screamer goes from about 115 ns per event under a plain Python loop to
about 90. It does not change the ordering, because what separates the libraries
is whether they can update a result rather than recompute one.

### Let screamer own the loop

Nearly all of those 92 nanoseconds are the call into the extension rather than
the operator. Every library pays that on every event, and for all of them it is
the dominant cost.

The way past it is to stop calling per event and let the loop run inside
screamer, which is what the array call and [`Pipeline`](pipelines.md) do. The
same recurrence, over the same stream, then costs **1 to 2 ns per event**, some
sixty times less:

```python
# 92 ns per event: the loop is yours, and every event crosses into the extension
op = screamer.RollingMean(100)
for value in feed:
    current = op(value)

# 1 to 2 ns per event: the loop is screamer's, and events stay in C++
current = screamer.RollingMean(100)(values)
```

A `Pipeline` does the same for a graph of operators over a live feed, so the
per-event boundary is crossed once for the whole graph rather than once per
operator. There is no equivalent for a batch library: TA-Lib has no streaming
entry point to build such a loop around.

## Why it is fast

- **Constant or logarithmic work per sample.** A rolling mean keeps a running sum,
  a rolling standard deviation keeps running moments, and a rolling maximum keeps a
  monotonic deque; each new sample updates that state in O(1). The rank-based
  statistics (rolling median, quantiles, IQR, CVaR) keep a balanced search tree and
  update in O(log window). Neither rebuilds the window, so a full pass is O(n) or
  O(n log window), against numpy's O(n * window).
- **One pass, no window rebuild.** The numpy sliding-window approach scans every
  window, which is O(n * window). screamer never looks back over the window.
- **C++ with thin bindings.** The compute runs in C++, the same code the batch,
  streaming, and pipeline paths all share, so there is no per-element Python.

The same cost applies live: a value computed on a stream runs the same code as the
batch pass, so the streaming path does the same per-event work, O(1) or
O(log window).

## Reproduce it

The suite lives in `benchmarks/`. Regenerate the numbers and this chart on your
own hardware with:

```
make benchmark
```

It times a screamer variant of each function against numpy, pandas, scipy and
TA-Lib references across a range of array and window sizes, writes per-function
CSVs to `benchmarks/experiments/`, and rebuilds the plots.

The streaming numbers come from a second suite:

```
make benchmark-streaming
```

It measures cost per event against window size, and gives each reference library
the strongest form available rather than a naive one: a preallocated ring buffer
of the trailing window, no allocation per event, and the fastest reduction the
library offers. It reports the same operator under a plain Python loop, under a
compiled loop, and through screamer's own array path, so the cost of the
per-event boundary is visible rather than folded into the result. The compiled
loops need Cython; without it those rows are skipped and the rest still runs.

The numbers on this page are from a single machine, so treat them as ratios
rather than absolutes; the shape holds across hardware.
