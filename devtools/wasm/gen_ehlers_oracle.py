"""Write wasm/smoke/oracle_ehlers.json: Python's results for the Ehlers
operators that Task 33 changed, which js/test/ehlers_parity.test.ts replays
in JavaScript. Both packages run the one C++ kernel; the values agree up to
the rounding of the math library (sin, cos, atan), so the JavaScript test
compares with a relative tolerance of 1e-9, and exactly where no library
function enters the result (TrendMode, InstantaneousTrendline).

Each case holds an input series (NaN written as null) and, per operator, the
flat output and its shape. The inputs are a random walk on a level of 100 with
three NaN gaps and a sine of period 20 plus a ramp on a level of 100.

Modes:
    (no args)  write the fixture
    --stdout   print the fixture to stdout (freshness gate)
"""
import argparse
import json
import os
import sys

import numpy as np

import screamer as s

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "wasm", "smoke",
                   "oracle_ehlers.json")
OPS = ["CycleSine", "TrendMode", "InstantaneousTrendline", "MAMA", "HilbertPhasor"]
N = 300


def _nums(a):
    return [None if np.isnan(x) else float(x) for x in np.asarray(a, dtype=float).ravel()]


def inputs():
    rng = np.random.default_rng(33)
    walk = np.round(100.0 + np.cumsum(rng.normal(size=N)), 6)
    walk[[90, 91, 200]] = np.nan
    t = np.arange(N, dtype=float)
    trend = np.round(100.0 + np.sin(2.0 * np.pi * t / 20.0) + t / 20.0, 9)
    return {"random walk with NaN": walk, "sine plus ramp": trend}


def cases():
    out = []
    for name, x in inputs().items():
        results = {}
        for op in OPS:
            y = np.asarray(getattr(s, op)()(x))
            results[op] = {"values": _nums(y), "shape": list(y.shape)}
        out.append({"name": name, "input": _nums(x), "results": results})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stdout", action="store_true", help="print the fixture instead of writing it")
    args = p.parse_args()
    text = json.dumps(cases(), allow_nan=False) + "\n"
    if args.stdout:
        sys.stdout.write(text)
        return
    with open(OUT, "w") as f:
        f.write(text)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
