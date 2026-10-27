"""Write wasm/smoke/oracle_iir_start.json: Python's results for the seven IIR
filters under each start_policy, which js/test/iir_start.test.ts replays in
JavaScript.

Each case names an operator, its settings in the JavaScript options form
(camelCase keys), and the flat output on one input series: a random walk on a
level of 100 with two leading NaN values and one NaN gap, so the steady start
happens at the first non-NaN value.

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
                   "oracle_iir_start.json")
N = 120

# operator, Python keyword settings (snake_case)
FILTERS = [
    ("Butter", {"order": 3, "cutoff_freq": 0.1}),
    ("ButterHighpass", {"order": 2, "cutoff_freq": 0.1}),
    ("ButterBandpass", {"order": 2, "low_cutoff": 0.05, "high_cutoff": 0.2}),
    ("ButterBandstop", {"order": 2, "low_cutoff": 0.05, "high_cutoff": 0.2}),
    ("SuperSmoother", {"period": 10.0}),
    ("Decycler", {"period": 20.0}),
    ("RoofingFilter", {"hp_period": 48.0, "lp_period": 10.0}),
]


def _camel(name):
    head, *rest = name.split("_")
    return head + "".join(w.title() for w in rest)


def _nums(a):
    return [None if np.isnan(x) else float(x) for x in np.asarray(a, dtype=float).ravel()]


def series():
    rng = np.random.default_rng(41)
    x = np.round(100.0 + np.cumsum(rng.normal(size=N)), 6)
    x[[0, 1, 60]] = np.nan
    return x


def fixture():
    x = series()
    cases = []
    for op, kw in FILTERS:
        for policy in ("steady", "zero"):
            settings = dict(kw, start_policy=policy)
            y = getattr(s, op)(**settings)(x)
            cases.append({"op": op, "options": {_camel(k): v for k, v in settings.items()},
                          "values": _nums(y)})
    return {"input": _nums(x), "cases": cases}


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
