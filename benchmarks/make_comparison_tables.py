"""Build the two comparison tables on the Performance page.

One table per regime, batch and streaming. Columns are screamer, numpy, pandas
and TA-Lib; rows are a curated set of screamer functions; cells are cost per
sample or per event in nanoseconds. A blank cell means that library has no
implementation, which is as much the point as the numbers are.

    python benchmarks/make_comparison_tables.py

Writes plots/comparison_batch.md and plots/comparison_streaming.md, and prints
both.

Row selection. The rows are chosen to separate three different claims, because
a table of moving averages alone would only show the least interesting one:

  * O(1) updates where every library is also O(1) per output (`RollingMean`,
    `EwMean`). Here screamer wins on constant factors, not algorithms, and the
    margin is small.
  * O(1) or O(log window) updates where the alternatives must rebuild the
    window (`RollingMax`, `RollingMedian`, `RollingQuantile`, `RollingIqr`).
    numpy is O(n * window) and pandas carries its own per-window work. This is
    where the algorithmic difference shows, and it grows with the window.
  * Functions the alternatives do not have at all. The blank cells are the
    coverage argument.

These are speed references, not correctness references: the numbers say how
long each library takes to compute its own version of the statistic. Whether
screamer agrees with them numerically is the job of `tests/test_baselines.py`,
which compares against the same libraries on every run.
"""
import os
import timeit

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

import screamer as sc

try:
    import talib
except ImportError:                                  # pragma: no cover
    talib = None

try:
    import streaming_compiled as compiled
except ImportError:                                  # pragma: no cover
    compiled = None

HERE = os.path.dirname(os.path.abspath(__file__))

N_BATCH = 1_000_000
N_STREAM = 200_000
WINDOW = 100
REPEAT = 5

_rng = np.random.default_rng(0)
VALUES = _rng.normal(size=N_BATCH)
PRICES = 100.0 * np.exp(np.cumsum(_rng.normal(0, 0.001, N_BATCH)))
HIGH = PRICES * np.exp(np.abs(_rng.normal(0, 0.002, N_BATCH)))
LOW = PRICES * np.exp(-np.abs(_rng.normal(0, 0.002, N_BATCH)))


def _sliding(values, window):
    return swv(values, window)


# Each row: label, and one entry per library. An entry is either None, meaning
# the library has no implementation, or a callable taking (values, window).
BATCH_ROWS = [
    ("RollingMean", "O(1) update",
     lambda v, w: sc.RollingMean(w)(v),
     lambda v, w: _sliding(v, w).mean(axis=1),
     lambda v, w: pd.Series(v).rolling(w).mean().to_numpy(),
     (lambda v, w: talib.SMA(v, w)) if talib else None),

    ("RollingStd", "O(1) update",
     lambda v, w: sc.RollingStd(w)(v),
     lambda v, w: _sliding(v, w).std(axis=1, ddof=1),
     lambda v, w: pd.Series(v).rolling(w).std().to_numpy(),
     (lambda v, w: talib.STDDEV(v, w)) if talib else None),

    ("RollingMax", "O(1) update",
     lambda v, w: sc.RollingMax(w)(v),
     lambda v, w: _sliding(v, w).max(axis=1),
     lambda v, w: pd.Series(v).rolling(w).max().to_numpy(),
     (lambda v, w: talib.MAX(v, w)) if talib else None),

    ("RollingMedian", "O(log window) update",
     lambda v, w: sc.RollingMedian(w)(v),
     lambda v, w: np.median(_sliding(v, w), axis=1),
     lambda v, w: pd.Series(v).rolling(w).median().to_numpy(),
     None),

    ("RollingQuantile", "O(log window) update",
     lambda v, w: sc.RollingQuantile(w, 0.75)(v),
     lambda v, w: np.quantile(_sliding(v, w), 0.75, axis=1),
     lambda v, w: pd.Series(v).rolling(w).quantile(0.75).to_numpy(),
     None),

    ("RollingIqr", "O(log window) update",
     lambda v, w: sc.RollingIqr(w)(v),
     None,
     lambda v, w: (pd.Series(v).rolling(w).quantile(0.75)
                   - pd.Series(v).rolling(w).quantile(0.25)).to_numpy(),
     None),

    ("RollingSkew", "O(1) update",
     lambda v, w: sc.RollingSkew(w)(v),
     None,
     lambda v, w: pd.Series(v).rolling(w).skew().to_numpy(),
     None),

    ("EwMean", "O(1) update",
     lambda v, w: sc.EwMean(span=w)(v),
     None,
     lambda v, w: pd.Series(v).ewm(span=w).mean().to_numpy(),
     (lambda v, w: talib.EMA(v, w)) if talib else None),

    ("RollingRSI", "indicator",
     lambda v, w: sc.RollingRSI(w)(PRICES),
     None,
     None,
     (lambda v, w: talib.RSI(PRICES, w)) if talib else None),

    ("ATR", "indicator, 3 inputs",
     lambda v, w: sc.ATR(w)(HIGH, LOW, PRICES),
     None,
     None,
     (lambda v, w: talib.ATR(HIGH, LOW, PRICES, w)) if talib else None),

    ("RollingHurst", "screamer only",
     lambda v, w: sc.RollingHurst(w)(v), None, None, None),

    ("FracDiff", "screamer only",
     lambda v, w: sc.FracDiff(d=0.4, window_size=w)(v), None, None, None),

    ("Hampel", "screamer only",
     lambda v, w: sc.Hampel(w)(v), None, None, None),
]


def _time(fn, values, window, n, repeat=REPEAT):
    if fn is None:
        return None
    fn(values, window)
    best = min(timeit.timeit(lambda: fn(values, window), number=1)
               for _ in range(repeat))
    return best / n * 1e9


def batch_table():
    rows = []
    for label, note, s, n_, p, t in BATCH_ROWS:
        # numpy's sliding-window view is O(n * window); cap the array so a
        # single measurement stays bounded, then scale to per-sample cost.
        numpy_n = min(N_BATCH, 10_000_000 // WINDOW)
        rows.append({
            "function": label,
            "note": note,
            "screamer": _time(s, VALUES, WINDOW, N_BATCH),
            "numpy": _time(n_, VALUES[:numpy_n], WINDOW, numpy_n, repeat=3),
            "pandas": _time(p, VALUES, WINDOW, N_BATCH, repeat=3),
            "talib": _time(t, VALUES, WINDOW, N_BATCH),
        })
    return pd.DataFrame(rows)


STREAM_VALUES = VALUES[:N_STREAM]

# Streaming rows: screamer updates its state per event; the others have no
# incremental entry point, so they are handed the trailing window each event.
STREAM_ROWS = [
    ("RollingMean", "O(1) update", lambda w: sc.RollingMean(w),
     lambda b, w: b.mean(), lambda b, w: pd.Series(b).mean(),
     (lambda b, w: talib.SMA(b, w)) if talib else None),
    ("RollingStd", "O(1) update", lambda w: sc.RollingStd(w),
     lambda b, w: b.std(ddof=1), lambda b, w: pd.Series(b).std(),
     (lambda b, w: talib.STDDEV(b, w)) if talib else None),
    ("RollingMax", "O(1) update", lambda w: sc.RollingMax(w),
     lambda b, w: b.max(), lambda b, w: pd.Series(b).max(),
     (lambda b, w: talib.MAX(b, w)) if talib else None),
    ("RollingMedian", "O(log window) update", lambda w: sc.RollingMedian(w),
     lambda b, w: np.median(b), lambda b, w: pd.Series(b).median(), None),
    ("RollingQuantile", "O(log window) update", lambda w: sc.RollingQuantile(w, 0.75),
     lambda b, w: np.quantile(b, 0.75),
     lambda b, w: pd.Series(b).quantile(0.75), None),
    ("RollingIqr", "O(log window) update", lambda w: sc.RollingIqr(w),
     None,
     lambda b, w: pd.Series(b).quantile(0.75) - pd.Series(b).quantile(0.25), None),
    ("RollingSkew", "O(1) update", lambda w: sc.RollingSkew(w),
     None, lambda b, w: pd.Series(b).skew(), None),
    ("EwMean", "O(1) update", lambda w: sc.EwMean(span=w),
     None, lambda b, w: pd.Series(b).ewm(span=w).mean().iloc[-1],
     (lambda b, w: talib.EMA(b, w)) if talib else None),
    ("RollingHurst", "screamer only", lambda w: sc.RollingHurst(w), None, None, None),
    ("FracDiff", "screamer only",
     lambda w: sc.FracDiff(d=0.4, window_size=w), None, None, None),
    ("Hampel", "screamer only", lambda w: sc.Hampel(w), None, None, None),
]


def streaming_table():
    if compiled is None:
        raise SystemExit(
            "the compiled event loops are needed for the streaming table; "
            "build them with `python benchmarks/build_compiled.py build_ext --inplace`")

    rows = []
    for label, note, factory, n_, p, t in STREAM_ROWS:
        def screamer_call(values, window, factory=factory):
            return compiled.screamer_loop(values, factory(window))

        def recompute(fn):
            if fn is None:
                return None
            return lambda values, window: compiled.window_recompute_loop(
                values, lambda b: fn(b, window), window)

        rows.append({
            "function": label,
            "note": note,
            "screamer": _time(screamer_call, STREAM_VALUES, WINDOW, N_STREAM),
            "numpy": _time(recompute(n_), STREAM_VALUES, WINDOW, N_STREAM, repeat=3),
            "pandas": _time(recompute(p), STREAM_VALUES, WINDOW, N_STREAM, repeat=2),
            "talib": _time(recompute(t), STREAM_VALUES, WINDOW, N_STREAM, repeat=3),
        })
    return pd.DataFrame(rows)


def to_markdown(frame, unit):
    lines = [f"| function | | screamer | numpy | pandas | TA-Lib |",
             "|---|---|---|---|---|---|"]
    for _, r in frame.iterrows():
        def cell(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "-"
            return f"{v:,.0f}" if v >= 100 else f"{v:.1f}"
        lines.append(
            f"| `{r['function']}` | {r['note']} | **{cell(r['screamer'])}** | "
            f"{cell(r['numpy'])} | {cell(r['pandas'])} | {cell(r['talib'])} |")
    lines.append("")
    lines.append(f"{unit}. A dash means the library has no implementation.")
    return "\n".join(lines)


def main():
    out = os.path.join(HERE, "plots")
    os.makedirs(out, exist_ok=True)

    batch = batch_table()
    text = to_markdown(batch, f"Nanoseconds per sample, window {WINDOW}, "
                              f"{N_BATCH:,} samples")
    open(os.path.join(out, "comparison_batch.md"), "w").write(text + "\n")
    print("BATCH\n")
    print(text)

    stream = streaming_table()
    text = to_markdown(stream, f"Nanoseconds per event, window {WINDOW}, "
                               f"{N_STREAM:,} events")
    open(os.path.join(out, "comparison_streaming.md"), "w").write(text + "\n")
    print("\n\nSTREAMING\n")
    print(text)


if __name__ == "__main__":
    main()
