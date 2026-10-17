#!/usr/bin/env python3
"""Write wasm/smoke/oracle_research.json: the operators and helpers of the
research toolkit, computed by the Python binding, for js/test/research.test.ts.

Each operator case names the operator, its settings in Python names, and one
of the input series below; js/test/research.test.ts builds the same operator
in JavaScript (settings in camelCase), runs it on the same series, and
compares the result, to 1e-9 relative unless the case gives a tolerance. The
helper cases hold the inputs and results of backtest_trades, backtest_report
with by=, and walk_forward with carry= and portfolio engines. A non-finite
number is written as null (NaN), "inf" or "-inf".

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

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "wasm", "smoke", "oracle_research.json")


def _inputs():
    rng = np.random.default_rng(20261017)
    T = 600
    events = np.where(rng.random(T) < 0.15, np.round(rng.standard_normal(T), 3), 0.0)
    events[[40, 41, 300]] = np.nan
    scale = np.where((np.arange(T) // 150) % 2 == 0, 0.5, 2.0)
    returns = rng.standard_normal(T) * scale
    returns[[100, 450]] = np.nan
    return {"events": events, "returns": returns}


# (operator, settings, input series[, relative tolerance]). Each operator
# task adds its cases here.
CASES = [
    ("Hold", {"n": 3}, "events"),
    ("Hold", {"n": 3, "cooldown": 4}, "events"),
    ("Hold", {"n": 1, "release": -1.0, "cooldown": 2}, "events"),
    ("CusumFilter", {"threshold": 1.0}, "returns"),
    ("CusumFilter", {"threshold": 2.5, "drift": 0.2}, "returns"),
    ("RollingSeasonalMean", {"period": 24, "window_size": 5}, "returns"),
    ("RollingSeasonalStd", {"period": 24, "window_size": 5, "start_policy": "expanding"}, "returns"),
    ("RollingSeasonalZscore", {"period": 7, "window_size": 6, "start_policy": "zero"}, "returns"),
    ("RollingUpsideDeviation", {"window_size": 20}, "returns"),
    ("RollingUpsideDeviation", {"window_size": 15, "mar": 0.2, "start_policy": "expanding"}, "returns"),
    ("RollingUpsideDeviation", {"window_size": 15, "start_policy": "zero"}, "returns"),
    ("RollingBipowerVar", {"window_size": 30}, "returns"),
    ("RollingJumpShare", {"window_size": 30, "start_policy": "expanding"}, "returns"),
    ("RollingJumpShare", {"window_size": 12, "start_policy": "zero"}, "returns"),
    ("RollingVarianceRatio", {"window_size": 60, "lag": 4}, "returns"),
    ("RollingVarianceRatio", {"window_size": 60, "lag": 4, "output": "zscore", "start_policy": "expanding"}, "returns"),
    ("RollingPermutationEntropy", {"window_size": 50, "order": 3}, "returns"),
    ("RollingPermutationEntropy", {"window_size": 40, "order": 5, "delay": 2, "start_policy": "expanding"}, "returns"),
    ("RollingPermutationEntropy", {"window_size": 20, "order": 4, "start_policy": "zero"}, "events"),
    ("BayesianChangepoint", {"expected_run": 100.0, "max_run": 300}, "returns", 1e-7),
    ("BayesianChangepoint", {"expected_run": 20.0, "max_run": 40, "prior_var": 4.0}, "returns", 1e-7),
    ("RollingHMM", {"window_size": 150, "refit_every": 40, "max_iter": 20, "tol": 0.0}, "returns", 1e-6),
    ("RollingHMM", {"window_size": 150, "refit_every": 40, "emission": "ar1", "max_iter": 20, "tol": 0.0},
     "returns", 1e-6),
]


def _engine_output():
    """A BacktestPriceTarget output with long, short, flat, flipping, and
    resized stretches, two NaN rows, and a trade open on the last row."""
    rng = np.random.default_rng(7)
    target = np.repeat([0, 1, 1, -2, 0, 3, -1, 0, 2, 3, 0, -1], 10).astype(float)
    price = 100 + np.cumsum(rng.standard_normal(len(target)))
    out = np.asarray(s.BacktestPriceTarget(fee=1e-3)(target, price), dtype=float)
    out[[33, 34]] = np.nan
    return out


def _market(T):
    """The deterministic OHLC market of tests/test_walk_forward.py."""
    o, h, l, c = [], [], [], []
    p = 100.0
    for t in range(T):
        prev = p
        p *= np.exp(0.01 * np.sin(1.7 * t) * np.cos(0.31 * t * t))
        o.append(prev)
        c.append(p)
        h.append(max(prev, p) * 1.001)
        l.append(min(prev, p) * 0.999)
    return [np.array(x) for x in (o, h, l, c)]


def _candidates(T):
    t = np.arange(T)
    return np.column_stack([np.sign(np.sin(t / 6.0)), np.sign(np.cos(t / 17.0)),
                            ((t // 5) % 3) - 1.0]).astype(float)


def _helpers():
    """Inputs and results of the backtest helpers; Tasks 12 to 14 add entries."""
    out = _engine_output()
    trades = s.backtest_trades(out)
    labels = (np.arange(len(out)) // 30) * 10
    _, summary = s.backtest_report(out, periods_per_year=252.0, by=labels)
    by = {"labels": labels.tolist(), "periods_per_year": 252.0,
          "groups": [[int(k), {n: _num(v) for n, v in g.items()}]
                     for k, g in summary["by_group"].items()]}
    T = 260
    o, h, l, c = _market(T)
    X = _candidates(T)
    rate = 1e-3 * np.sin(np.arange(T) / 7.0)
    wf = s.walk_forward(s.BacktestOHLCTarget(taker_fee=2e-4), X, o, h, l, c, train=100, test=40,
                        carry=rate, carry_multiplier=2.0)
    walk_forward_carry = {
        "market": [_flat(x) for x in (o, h, l, c)], "targets": _flat(X), "K": X.shape[1],
        "carry": _flat(rate), "carry_multiplier": 2.0, "train": 100, "test": 40,
        "train_scores": _flat(wf.train_scores), "choice": wf.choice.tolist(),
        "output": _flat(wf.output), "stats": {k: _num(v) for k, v in wf.stats.items()}}
    return {
        "walk_forward_carry": walk_forward_carry,
        "by": by,
        "backtest_trades": {
            "values": _flat(out), "T": len(out),
            "result": {k: _flat(np.asarray(v, dtype=float)) for k, v in trades.items()},
        },
    }


def _num(v):
    v = float(v)
    if math.isnan(v):
        return None
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return v


def _flat(a):
    return [_num(v) for v in np.asarray(a, dtype=float).ravel()]


def build():
    inputs = _inputs()
    cases = []
    for case in CASES:
        name, settings, key = case[:3]
        out = np.asarray(getattr(s, name)(**settings)(inputs[key]), dtype=float)
        entry = {"name": name, "settings": settings, "input": key,
                 "shape": list(out.shape), "out": _flat(out)}
        if len(case) > 3:
            entry["tol"] = case[3]
        cases.append(entry)
    return {"inputs": {k: _flat(v) for k, v in inputs.items()}, "cases": cases,
            "helpers": _helpers()}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stdout", action="store_true", help="print instead of writing")
    args = p.parse_args()
    text = json.dumps(build(), separators=(",", ":")) + "\n"
    if args.stdout:
        sys.stdout.write(text)
    else:
        with open(OUT, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
