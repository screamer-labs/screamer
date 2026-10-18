#!/usr/bin/env python3
"""Write wasm/smoke/oracle_ew_flat.json: the exponentially weighted ratios of a
series that stops varying, computed by the Python binding, for
js/test/ew_flat.test.ts.

The operators that divide by an EW variance (EwBeta and the operators built on
it, EwCorr, EwZscore, EwSkew, EwKurt) give NaN once that variance has fallen
far below a decaying peak of it (include/screamer/detail/ew_flat.h). Each case
names the operator, its settings in Python names, and the input series below;
the JavaScript test runs the same operator on the same series and compares the
result, NaN for NaN, to 1e-9 relative. A NaN is written as null.

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
OUT = os.path.join(HERE, "..", "..", "wasm", "smoke", "oracle_ew_flat.json")
ALPHA = 0.3


def _inputs():
    """flat: 100 varying rows, 200 at 3.7, 300 varying again. rand: varying.
    tick: returns of spread 0.01 with one 1e10 tick at row 100."""
    rng = np.random.default_rng(20261018)
    T = 600
    flat = rng.standard_normal(T)
    flat[100:300] = 3.7
    rand = rng.standard_normal(T)
    tick = 0.01 * rng.standard_normal(T)
    tick[100] = 1e10
    small = 0.01 * rng.standard_normal(T)
    return {"flat": flat, "rand": rand, "tick": tick, "small": small}


# (operator, input series).
CASES = [
    ("EwBeta", ["rand", "flat"]),
    ("EwBeta", ["flat", "rand"]),
    ("EwKyleLambda", ["flat", "rand"]),
    ("EwAlpha", ["rand", "flat"]),
    ("EwSpread", ["rand", "flat"]),
    ("EwResidualStd", ["rand", "flat"]),
    ("EwCorr", ["flat", "rand"]),
    ("EwCorr", ["rand", "flat"]),
    ("EwZscore", ["flat"]),
    ("EwSkew", ["flat"]),
    ("EwKurt", ["flat"]),
    ("EwBeta", ["small", "tick"]),
    ("EwCorr", ["tick", "small"]),
    ("EwZscore", ["tick"]),
    ("EwKurt", ["tick"]),
]


def _num(v):
    return None if math.isnan(v) else float(v)


def build():
    inputs = _inputs()
    cases = []
    for name, cols in CASES:
        out = getattr(s, name)(alpha=ALPHA)(*(inputs[c] for c in cols))
        cases.append({"name": name, "settings": {"alpha": ALPHA}, "inputs": cols,
                      "out": [_num(v) for v in out]})
    return {"inputs": {k: [_num(v) for v in x] for k, x in inputs.items()}, "cases": cases}


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
        print(f"wrote {os.path.relpath(OUT)}")


if __name__ == "__main__":
    main()
