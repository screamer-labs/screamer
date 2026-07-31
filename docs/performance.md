# Performance

There are two questions, and they have different answers. Over a whole array,
screamer is in the same class as the fastest alternative for simple statistics
and pulls ahead on the ones that need a data structure. Over a live stream,
where a value is needed before the next event arrives, the gap is much wider,
because a batch library has no way to update a result: it recomputes the
trailing window every event.

Both tables below use the same rows, so the two regimes can be read against each
other. A dash means the library has no implementation of that function, which is
as much the comparison as the numbers are.

## Batch

Cost per sample over a one-million-sample array, window 100.

| function | | screamer | numpy | pandas | TA-Lib |
|---|---|---|---|---|---|
| `RollingMean` | O(1) update | **1.5** | 16.7 | 5.2 | 1.3 |
| `RollingStd` | O(1) update | **3.1** | 87.3 | 11.1 | 1.6 |
| `RollingMax` | O(1) update | **1.3** | 12.5 | 16.5 | 1.1 |
| `RollingMedian` | O(log window) update | **140** | 890 | 249 | - |
| `RollingQuantile` | O(log window) update | **124** | 861 | 248 | - |
| `RollingIqr` | O(log window) update | **145** | - | 485 | - |
| `RollingSkew` | O(1) update | **3.5** | - | 7.3 | - |
| `EwMean` | O(1) update | **4.0** | - | 3.8 | 1.7 |
| `RollingRSI` | indicator | **6.8** | - | - | 4.2 |
| `ATR` | indicator, 3 inputs | **7.0** | - | - | 4.5 |
| `RollingHurst` | screamer only | **536** | - | - | - |
| `FracDiff` | screamer only | **65.9** | - | - | - |
| `Hampel` | screamer only | **1,004** | - | - | - |

Nanoseconds per sample, window 100, 1,000,000 samples. A dash means the library has no implementation.

For the O(1) statistics screamer and TA-Lib are close, and both are well ahead
of pandas and numpy. TA-Lib is quicker on several of them; those are single
passes with little state, and the remaining difference is constant factors
rather than algorithms.

The O(log window) rows are where the algorithm shows. screamer keeps a sorted
structure and updates it per sample; numpy rebuilds every window, which is
O(n * window); pandas does its own per-window work. The margin over numpy grows
with the window, and neither TA-Lib nor numpy offers a rolling interquartile
range at all.

![screamer speedup versus the fastest alternative](img/speed_chart.png)

The chart covers a wider set of 32 functions, comparing screamer against the
fastest of numpy, pandas and scipy on a one-million-element array. screamer is
faster on 29 of them and level on the other three: `EwMean` at 0.96x, `Abs` and
`Sqrt` at 0.99x, which is parity within measurement noise on a single pass over
memory.

## Streaming

The same functions, now fed one event at a time, with the current value needed
before the next event arrives.

screamer updates its state and returns the new value, which is O(1) or
O(log window) per event. numpy, pandas and TA-Lib have no incremental entry
point, so the only way to get the current value is to hand them the trailing
window again. That is O(window) per event however fast the library is.

| function | | screamer | numpy | pandas | TA-Lib |
|---|---|---|---|---|---|
| `RollingMean` | O(1) update | **93.9** | 1,267 | 12,539 | 623 |
| `RollingStd` | O(1) update | **92.8** | 3,922 | 17,962 | 645 |
| `RollingMax` | O(1) update | **104** | 556 | 12,655 | 606 |
| `RollingMedian` | O(log window) update | **251** | 6,081 | 23,290 | - |
| `RollingQuantile` | O(log window) update | **227** | 16,618 | 88,056 | - |
| `RollingIqr` | O(log window) update | **251** | - | 175,943 | - |
| `RollingSkew` | O(1) update | **96.6** | - | 18,308 | - |
| `EwMean` | O(1) update | **90.2** | - | 28,415 | 626 |
| `RollingHurst` | screamer only | **684** | - | - | - |
| `FracDiff` | screamer only | **181** | - | - | - |
| `Hampel` | screamer only | **1,168** | - | - | - |

Nanoseconds per event, window 100, 200,000 events. A dash means the library has no implementation.

The ordering from the batch table does not survive. TA-Lib leads several batch
rows and is 6 to 7 times slower here, because it has to be given the window
back every event. The rank statistics diverge furthest: a rolling interquartile
range costs 251 ns per event in screamer and 176 microseconds in pandas.

The event loop in that table is compiled, so the interpreter is not part of the
measurement. Compiling the loop is worth having and it helps every library, but
it does not change the ordering, because what separates them is whether a result
can be updated or must be recomputed.

### Let screamer own the loop

Nearly all of screamer's ~90 ns above is the call into the extension rather than
the operator. Every library pays that on every event, and for all of them it is
the dominant cost.

The way past it is to stop calling per event and let the loop run inside
screamer, which is what the array call and [`Pipeline`](pipelines.md) do. The
same recurrence over the same stream then costs **1 to 2 ns per event**, some
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
- **A batch path where the batch algorithm differs.** The sliding-window extremum
  is computed by block decomposition over an array, which has no data-dependent
  branching, and by a monotonic deque on a stream, which needs no lookahead. Both
  give the same answers; the tests assert it.

## Reproduce it

The suite lives in `benchmarks/`. Both tables on this page come from:

```
make benchmark
make benchmark-streaming
```

The first times a screamer variant of each function against numpy, pandas, scipy
and TA-Lib references across a range of array and window sizes, writes
per-function CSVs to `benchmarks/experiments/`, and rebuilds the chart. The
second measures cost per event against window size.

Both give each reference library the strongest form available rather than a
naive one: for streaming that means a preallocated ring buffer of the trailing
window, no allocation per event, and the fastest reduction the library offers.
The compiled event loops need Cython; without it those rows are skipped and the
rest still runs.

The numbers on this page are from a single machine, so treat them as ratios
rather than absolutes; the shape holds across hardware.
