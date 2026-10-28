"""Write wasm/smoke/oracle_book_report.json: Python's backtest_report on the
(T, K, 4) output of a portfolio engine, which js/test/backtest_book.test.ts
replays in JavaScript.

Each case holds the input (shape and flat values, NaN as null, infinities as
"inf" and "-inf"), the options, and Python's running columns, summary (without
per_asset), per-asset summaries, and by_group table when `by` is given.

Modes:
    (no args)  write the fixture
    --stdout   print the fixture to stdout (freshness gate)
"""
import argparse
import json
import math
import os
import sys

import numpy as np

import screamer as s

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "wasm", "smoke",
                   "oracle_book_report.json")


def _num(v):
    v = float(v)
    if math.isnan(v):
        return None
    return v if math.isfinite(v) else ("inf" if v > 0 else "-inf")


def _book(T, K, seed):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal((T, K)), axis=0))
    target = np.resize(np.repeat(rng.integers(-1, 2, (T // 10 + 1, K)), 10, axis=0), (T, K))
    out = s.BacktestPortfolioOHLCTarget(fee=1e-3)(0.3 * target.astype(float), c, c, c, c)
    return np.round(np.asarray(out), 12)


def fixture():
    base = _book(80, 3, 5)
    late = base.copy()
    late[:20, 2, :] = np.nan                   # asset 2 starts at row 20
    late[40, 1, 1] = np.nan                    # asset 1 skips row 40
    late[50, :, 0] = np.nan                    # every asset skips row 50
    cases = [
        ("base", base, {}),
        ("late asset", late, {"periods_per_year": 365}),
        ("grouped", late, {"periods_per_year": 365, "by": (np.arange(80) // 30).tolist()}),
        ("one asset", base[:, :1, :], {"periods_per_year": 252}),
        ("no rows", base[:0], {}),
    ]
    out = []
    for name, values, kw in cases:
        running, summary = s.backtest_report(values, **kw)
        per_asset = summary.pop("per_asset")
        groups = summary.pop("by_group", None)
        entry = {
            "name": name,
            "shape": list(values.shape),
            "values": [_num(v) for v in np.ravel(values)],
            "periods_per_year": kw.get("periods_per_year", 1),
            "running": {k: [_num(v) for v in a] for k, a in running.items()},
            "summary": {k: _num(v) for k, v in summary.items()},
            "per_asset": [{k: _num(v) for k, v in p.items()} for p in per_asset],
        }
        if groups is not None:
            entry["by"] = kw["by"]
            entry["by_group"] = [[label, {k: _num(v) for k, v in g.items()}]
                                 for label, g in groups.items()]
        out.append(entry)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stdout", action="store_true", help="print the fixture instead of writing it")
    args = p.parse_args()
    text = json.dumps(fixture(), allow_nan=False) + "\n"
    if args.stdout:
        sys.stdout.write(text)
        return
    with open(OUT, "w") as f:
        f.write(text)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
