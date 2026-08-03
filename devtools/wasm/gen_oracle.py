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
"""

import json
import math
import os

import screamer

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "wasm", "smoke", "oracle.json")
OUT_REDUCER = os.path.join(REPO, "wasm", "smoke", "oracle_reducer.json")

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


if __name__ == "__main__":
    main()
