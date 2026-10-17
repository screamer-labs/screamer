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


def _helpers():
    """Inputs and results of the backtest helpers; Tasks 11 to 14 add entries."""
    return {}


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
