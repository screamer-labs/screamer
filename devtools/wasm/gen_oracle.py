#!/usr/bin/env python3
"""Generate wasm/smoke/oracle.json: a Python-oracle parity fixture for the WASM smoke test.

Run with the abi3 build loaded:

    poetry run python devtools/wasm/gen_oracle.py

Bare `python3` uses pyenv 3.11 and cannot import the cp312 abi3 module.

For a curated sample of ops spanning ScreamerBase and FunctorBase and the
plain / transform / ew_optional ctor kinds, this drives a FIXED input series
through the Python op one event at a time (the scalar streaming path, op(x) per
event) and records the single-event output at a post-warmup index. The emitted
`args` use the WASM/JS ctor convention (positional doubles, NaN for a missing
optional slot, written as JSON null) so smoke.mjs can construct `new M[name](...args)`
and reproduce the same op.

A second fixture, wasm/smoke/oracle_reducer.json, covers the dynamic-width
reducer path that the fixed-width event ABI above cannot express: a
(time, assets, 4) backtest engine output reduced by PortfolioReport, recorded
both as one batch call and as a row-at-a-time stream.

A third fixture, wasm/smoke/oracle_micro.json, covers the five microstructure
operators that reuse another operator's kernel (SignedVolume, QueueImbalance,
RollingOrderImbalance, RollingKyleLambda, EwKyleLambda): separate 1-D inputs,
one packed (T, K, N) array, iterators and a stream of scalar events, on inputs
with NaN values and with each start policy.

A fourth fixture, wasm/smoke/oracle_backtest.json, holds backtest_report's
running columns and summary for a few engine outputs (NaN and infinite
values, annualized ratios, one row, no rows), which backtestReport must
reproduce exactly. A non-finite number is written as null (NaN), "inf" or
"-inf".

A fifth fixture, wasm/smoke/oracle_forecast_pairs.json, holds forecast_pairs
in count and duration mode (1-D and (T, K) features, NaN values, dropna, a y
on its own clock): the inputs, the eager result, and the rows and index of the
same graph as a Pipeline node, which forecastPairs must reproduce exactly.

A sixth fixture, wasm/smoke/oracle_split.json, holds split's per-source
values and index for tagged streams (positional and indexed, NaN and infinite
values and index, an n larger than max(sources)+1, no events, values with rows
of shape (T, 3), (T, 1), (T, 2, 2) and (0, 3)), which split must reproduce
exactly. A non-finite number is written as null (NaN), "inf" or "-inf".
"""

import json
import math
import os

import screamer

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "wasm", "smoke", "oracle.json")
OUT_REDUCER = os.path.join(REPO, "wasm", "smoke", "oracle_reducer.json")
OUT_MICRO = os.path.join(REPO, "wasm", "smoke", "oracle_micro.json")
OUT_BACKTEST = os.path.join(REPO, "wasm", "smoke", "oracle_backtest.json")
OUT_FORECAST = os.path.join(REPO, "wasm", "smoke", "oracle_forecast_pairs.json")
OUT_SPLIT = os.path.join(REPO, "wasm", "smoke", "oracle_split.json")

# Fixed, deterministic, all-positive input series (positive keeps Log/Sqrt finite).
SERIES = [
    5.0, 3.0, 8.0, 2.0, 9.0, 4.0, 7.0, 1.0, 6.0, 10.0,
    5.5, 3.5, 8.5, 2.5, 9.5, 4.5, 7.5, 1.5, 6.5, 10.5,
    5.25, 3.25, 8.25, 2.25, 9.25, 4.25, 7.25, 1.25, 6.25, 10.25,
    5.75, 3.75, 8.75, 2.75, 9.75, 4.75, 7.75, 1.75, 6.75, 10.75,
    5.1, 3.1, 8.1, 2.1, 9.1, 4.1, 7.1, 1.1, 6.1, 10.1,
    5.6, 3.6, 8.6, 2.6, 9.6, 4.6, 7.6, 1.6, 6.6, 10.6,
]
CHECK_INDEX = 40  # comfortably past the warmup of every sampled op

# Each entry: name, a zero-arg factory building the Python op, and the WASM/JS
# ctor args (None == NaN sentinel for a missing optional slot). The two must
# describe the SAME op; only the calling convention differs (kwargs vs positional
# NaN-sentinel for the ew_optional decay parameters).
NAN = float("nan")
SAMPLE = [
    ("RollingMean",   lambda: screamer.RollingMean(3, "strict"),   [3, "strict"]),
    ("RollingStd",    lambda: screamer.RollingStd(3, "strict"),    [3, "strict"]),
    ("RollingSum",    lambda: screamer.RollingSum(3, "strict"),    [3, "strict"]),
    ("RollingMin",    lambda: screamer.RollingMin(3),              [3]),
    ("RollingMax",    lambda: screamer.RollingMax(3),              [3]),
    ("RollingMinMax", lambda: screamer.RollingMinMax(3),           [3]),          # n_out == 2
    ("RollingZscore", lambda: screamer.RollingZscore(3, "strict"), [3, "strict"]),
    ("RollingSkew",   lambda: screamer.RollingSkew(4, "strict"),   [4, "strict"]),
    ("RollingMedian", lambda: screamer.RollingMedian(3),           [3]),
    ("Abs",           lambda: screamer.Abs(),                      []),           # transform
    ("Sign",          lambda: screamer.Sign(),                     []),           # transform
    ("Sqrt",          lambda: screamer.Sqrt(),                     []),           # transform
    ("Log",           lambda: screamer.Log(),                      []),           # transform
    ("Diff",          lambda: screamer.Diff(1, "strict"),          [1, "strict"]),
    ("Lag",           lambda: screamer.Lag(2, "strict"),           [2, "strict"]),
    ("Return",        lambda: screamer.Return(1),                  [1]),
    ("RollingQuantile", lambda: screamer.RollingQuantile(5, 0.5),  [5, 0.5]),
    ("EwMean",        lambda: screamer.EwMean(span=5),             [None, 5.0, None, None]),  # ew_optional
    ("EwStd",         lambda: screamer.EwStd(span=5),              [None, 5.0, None, None]),  # ew_optional
    ("MACD",          lambda: screamer.MACD(12, 26, 9),            [12, 26, 9]),  # FunctorBase, n_out == 3
]


def as_list(out):
    """Normalize a single-event op output to a list of floats."""
    if isinstance(out, (tuple, list)):
        return [float(v) for v in out]
    return [float(out)]


def jsonify(x):
    """NaN -> None (JSON null); leave everything else as-is."""
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


def engine_output():
    """A deterministic stand-in for a (time, assets, 4) backtest engine output.

    Columns are [equity, pnl, position, cost]. Positions change on a different
    cadence per asset, so the fixture contains rows that trade one asset, rows
    that trade several, and rows that trade none, which is what separates the
    per-asset turnover the reducer computes from the net-position turnover a
    naive implementation would produce. Equity rises and falls so drawdown and
    max_drawdown are both exercised, and one row carries a NaN so the ignore
    policy (skip the row, hold the position state, emit NaN) is covered too.
    """
    time_steps, assets = 12, 3
    equity = [100.0, 200.0, 50.0]
    positions = [0.0, 0.0, 0.0]
    rows = []
    for t in range(time_steps):
        row = []
        for a in range(assets):
            if t % (a + 2) == 0:
                positions[a] += 1.0 if (t + a) % 3 else -2.0
            pnl = ((t + 1) * (a + 1) % 7) - 3.0
            equity[a] += pnl
            row.append([equity[a], pnl, positions[a], 0.05 * abs(positions[a])])
        rows.append(row)
    rows[5][1][0] = float("nan")
    return rows


def reducer_entry():
    """Batch and per-row streaming outputs of PortfolioReport over that fixture.

    Both are recorded so the JavaScript side can check its batch path, its
    per-event path, and that the two agree, against one Python reference.
    """
    import numpy as np

    rows = engine_output()
    batch = screamer.PortfolioReport()(np.asarray(rows, dtype=np.float64))

    op = screamer.PortfolioReport()
    stream = [as_list(op(np.asarray(r, dtype=np.float64))) for r in rows]

    return {
        "name": "PortfolioReport",
        "shape": [len(rows), len(rows[0]), 4],
        "input": [[[jsonify(v) for v in group] for group in row] for row in rows],
        "batch": [[jsonify(v) for v in row] for row in batch.tolist()],
        "stream": [[jsonify(v) for v in row] for row in stream],
    }


def micro_entries():
    """Python results of the five microstructure operators on fixed inputs.

    The inputs hold NaN values, and the windowed operators run with each
    start policy. Each entry records the JavaScript constructor arguments,
    the inputs (1-D series and one packed (T, K, N) array) and the Python
    results for the batch call on separate inputs, the packed call, and a
    scalar stream.
    """
    import numpy as np

    rng = np.random.default_rng(5)
    T, K = 40, 3
    flow = rng.normal(size=(T, K))
    ret = 0.8 * flow + rng.normal(scale=0.3, size=(T, K))
    size = rng.exponential(1.0, size=(T, K, 2))
    flow[[3, 17], [0, 1]] = np.nan
    ret[[9, 17], [0, 2]] = np.nan
    size[[5, 22], [0, 0], [0, 1]] = np.nan
    ops = [
        ("SignedVolume", screamer.SignedVolume, (), [], [np.sign(flow), np.abs(ret)]),
        ("QueueImbalance", screamer.QueueImbalance, (), [], [size[..., 0], size[..., 1]]),
        ("EwKyleLambda", lambda: screamer.EwKyleLambda(span=6.0), (), [None, 6.0, None, None],
         [flow, ret]),
    ]
    for policy in ("strict", "expanding", "zero"):
        ops += [
            ("RollingOrderImbalance", screamer.RollingOrderImbalance, (5, policy), [5, policy], [flow]),
            ("RollingKyleLambda", screamer.RollingKyleLambda, (8, policy), [8, policy], [flow, ret]),
        ]

    def nums(a):
        return [jsonify(float(v)) for v in np.ravel(a)]

    out = []
    for name, make, py_args, js_args, cols in ops:
        series = [c[:, 0] for c in cols]
        packed = np.stack(cols, axis=-1)
        op = make(*py_args)
        stream = [float(op(*[float(c[t]) for c in series])) for t in range(T)]
        out.append({
            "name": name,
            "args": [jsonify(a) for a in js_args],
            "series": [nums(c) for c in series],
            "packed": {"shape": list(packed.shape), "data": nums(packed)},
            "batch": nums(make(*py_args)(*series)),
            "packed_expect": {"shape": [T, K], "data": nums(make(*py_args)(packed))},
            "stream": [jsonify(v) for v in stream],
        })
    return out


def backtest_entries():
    """backtest_report on fixed engine outputs: the input rows, the options,
    and Python's running columns (in order) and summary."""
    import numpy as np

    def num(v):
        v = float(v)
        if math.isnan(v):
            return None
        return v if math.isfinite(v) else ("inf" if v > 0 else "-inf")

    rng = np.random.default_rng(11)
    T = 60
    pos = np.round(rng.normal(size=T)).clip(-1, 1)
    pnl = rng.normal(size=T)
    cost = np.abs(np.diff(pos, prepend=0.0)) * 0.05
    base = np.column_stack([np.cumsum(pnl - cost), pnl, pos, cost])
    holes = base.copy()
    holes[4, 1] = np.nan
    holes[20, 3] = np.inf
    holes[-2:, 0] = np.nan
    winners = base.copy()
    winners[:, 1] = np.abs(winners[:, 1])        # no losing trade: profit_factor is inf
    winners[:, 0] = np.cumsum(winners[:, 1])
    cases = [
        ("base", base, {}),
        ("annualized", base, {"periods_per_year": 252}),
        ("holes", holes, {"periods_per_year": 24 * 365}),
        ("winners", winners, {}),
        ("one row", base[:1], {}),
        ("no rows", base[:0], {}),
    ]
    out = []
    for name, values, kw in cases:
        running, summary = screamer.backtest_report(values, **kw)
        out.append({
            "name": name,
            "shape": list(values.shape),
            "values": [num(v) for v in np.ravel(values)],
            "periods_per_year": kw.get("periods_per_year", 1),
            "running": {k: [num(v) for v in a] for k, a in running.items()},
            "summary": {k: num(v) for k, v in summary.items()},
        })
    return out


def forecast_entries():
    """forecast_pairs on fixed inputs: the inputs, the options, the eager
    (X_shifted, y), and the Pipeline node's output rows and index."""
    import numpy as np
    from screamer.supervised import forecast_pairs

    def nums(a):
        return [None if math.isnan(v) else float(v) for v in np.ravel(a)]

    rng = np.random.default_rng(21)
    T = 40
    xi = np.cumsum(rng.integers(1, 4, size=T)).astype(np.int64)
    yi = np.cumsum(rng.integers(1, 6, size=T // 2)).astype(np.int64) + 1
    xv = rng.standard_normal(T)
    xv[[3, 12]] = np.nan
    yc = rng.standard_normal(T)
    yc[[5, 20]] = np.nan
    yv = rng.standard_normal(T // 2)
    yv[[4]] = np.nan
    X = rng.standard_normal((T, 3))
    X[[2, 9], [0, 2]] = np.nan
    rows = np.arange(T, dtype=np.int64)
    cases = [
        ("count", xv, rows, yc, rows, {"count": 3}),
        ("count dropna", xv, rows, yc, rows, {"count": 2, "dropna": True}),
        ("count (T,K) dropna", X, rows, yc, rows, {"count": 2, "dropna": True}),
        ("duration", xv, xi, yv, yi, {"duration": 4}),
        ("duration dropna", xv, xi, yv, yi, {"duration": 7, "dropna": True}),
        ("duration (T,K)", X, xi, yv, yi, {"duration": 0}),
        ("duration empty", xv[:0], xi[:0], yv[:0], yi[:0], {"duration": 1}),
    ]
    out = []
    for name, x, ix, y, iy, kw in cases:
        if "count" in kw:
            xs, ys = forecast_pairs(x, y, **kw)
        else:
            xs, ys = forecast_pairs((x, ix), (y, iy), **kw)
        X_in, y_in = screamer.Input("X"), screamer.Input("y")
        node = forecast_pairs(X_in, y_in, **kw)
        vals, idx = screamer.Pipeline([X_in, y_in], [node])((x, ix), (y, iy))
        out.append({
            "name": name,
            "options": kw,
            "X": {"shape": list(x.shape), "values": nums(x), "index": [int(i) for i in ix]},
            "y": {"values": nums(y), "index": [int(i) for i in iy]},
            "X_shifted": {"shape": list(xs.shape), "values": nums(xs)},
            "y_out": nums(ys),
            "node": {"values": nums(vals), "index": [int(i) for i in idx]},
        })
    return out


def split_entries():
    """split on fixed tagged streams: the inputs (values with their shape), n
    (None for the default) and each output's values (with their shape) and
    index (None when positional)."""
    import numpy as np

    def nums(a):
        return [None if math.isnan(v) else ("inf" if v == math.inf else "-inf" if v == -math.inf
                                            else float(v)) for v in np.ravel(a)]

    rng = np.random.default_rng(22)
    T = 30
    values = rng.standard_normal(T)
    values[[4, 17]] = np.nan
    values[9] = np.inf
    sources = rng.integers(0, 4, size=T)
    int_index = np.cumsum(rng.integers(1, 4, size=T))
    float_index = int_index * 0.25
    odd_index = float_index.copy()
    odd_index[[2, 11]] = [np.nan, -np.inf]
    rows = rng.standard_normal((T, 3))
    rows[[5, 6], [0, 2]] = np.nan
    cube = rng.standard_normal((T, 2, 2))
    cases = [
        ("positional", values, sources, None, None),
        ("int index", values, sources, int_index, None),
        ("float index", values, sources, float_index, None),
        ("NaN and -inf index", values, sources, odd_index, None),
        ("n larger than max(sources)+1", values, sources, int_index, 7),
        ("one source", values[:5], np.zeros(5, dtype=np.int64), None, None),
        ("no events", values[:0], sources[:0], None, None),
        ("no events, n=2", values[:0], sources[:0], int_index[:0], 2),
        ("(T, 3) rows", rows, sources, int_index, None),
        ("(T, 1) rows", rows[:, :1], sources, None, None),
        ("(T, 2, 2) rows", cube, sources, float_index, 5),
        ("(0, 3) rows", rows[:0], sources[:0], None, 2),
    ]
    out = []
    for name, v, src, idx, n in cases:
        parts = screamer.split(v, src, index=idx, n=n)
        out.append({
            "name": name,
            "shape": list(v.shape),
            "values": nums(v),
            "sources": [int(x) for x in src],
            "index": None if idx is None else nums(idx),
            "n": n,
            "parts": [{"shape": list(pv.shape), "values": nums(pv),
                       "index": None if pi is None else nums(pi)}
                      for pv, pi in parts],
        })
    return out


def main():
    entries = []
    for name, factory, wasm_args in SAMPLE:
        op = factory()
        expect = None
        for i, x in enumerate(SERIES):
            out = as_list(op(x))
            if i == CHECK_INDEX:
                expect = out
        if expect is None:
            raise RuntimeError(f"{name}: CHECK_INDEX {CHECK_INDEX} out of range")
        if any(isinstance(v, float) and math.isnan(v) for v in expect):
            raise RuntimeError(
                f"{name}: expected output at index {CHECK_INDEX} is NaN "
                f"(pick a later post-warmup index): {expect}"
            )
        entries.append(
            {
                "name": name,
                "args": [jsonify(a) for a in wasm_args],
                "inputs": [[x] for x in SERIES],
                "check_index": CHECK_INDEX,
                "expect": [jsonify(v) for v in expect],
            }
        )
        print(f"  {name:16s} args={wasm_args} expect@{CHECK_INDEX}={expect}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(entries, f, indent=2, allow_nan=False)
        f.write("\n")
    print(f"\nWrote {len(entries)} oracle entries -> {OUT}")

    reducer = reducer_entry()
    with open(OUT_REDUCER, "w") as f:
        json.dump(reducer, f, indent=2, allow_nan=False)
        f.write("\n")
    print(
        f"Wrote the {reducer['name']} reducer fixture "
        f"(shape {reducer['shape']}) -> {OUT_REDUCER}"
    )

    micro = micro_entries()
    with open(OUT_MICRO, "w") as f:
        json.dump(micro, f, indent=1, allow_nan=False)
        f.write("\n")
    print(f"Wrote {len(micro)} microstructure entries -> {OUT_MICRO}")

    backtest = backtest_entries()
    with open(OUT_BACKTEST, "w") as f:
        json.dump(backtest, f, indent=1, allow_nan=False)
        f.write("\n")
    print(f"Wrote {len(backtest)} backtest_report entries -> {OUT_BACKTEST}")

    forecast = forecast_entries()
    with open(OUT_FORECAST, "w") as f:
        json.dump(forecast, f, indent=1, allow_nan=False)
        f.write("\n")
    print(f"Wrote {len(forecast)} forecast_pairs entries -> {OUT_FORECAST}")

    split = split_entries()
    with open(OUT_SPLIT, "w") as f:
        json.dump(split, f, indent=1, allow_nan=False)
        f.write("\n")
    print(f"Wrote {len(split)} split entries -> {OUT_SPLIT}")


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    main()
