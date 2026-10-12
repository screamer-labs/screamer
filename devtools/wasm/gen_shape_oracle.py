"""Write wasm/smoke/shape_oracle.json: shape-contract cases computed by the
Python binding, which js/test/shape_contract.test.ts reproduces in WASM."""
import json
import os

import numpy as np

import screamer as s

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "wasm", "smoke", "shape_oracle.json")

T, K = 40, 3
rng = np.random.default_rng(11)
X = rng.normal(size=(T, K)).cumsum(axis=0) + 100
y = rng.normal(size=T).cumsum() + 100
tgt = np.sign(rng.normal(size=(T, K)))


def nd(a):
    a = np.asarray(a, dtype=float)
    return {"shape": list(a.shape), "data": [None if np.isnan(v) else float(v) for v in a.ravel()]}


CASES = [
    ("RollingMean", [5], [X]),
    ("RollingMean", [5], [X[:, :, None]]),
    ("BollingerBands", [5], [X]),
    ("RollingCorr", [10], [X, y]),
    ("RollingCorr", [10], [X, y[:, None]]),
    ("Sub", [], [np.stack([X, y[:, None] + 0 * X], axis=-1)]),
    ("BacktestPriceTarget", [], [tgt, y]),
]


def main():
    out = []
    for name, args, inputs in CASES:
        result = getattr(s, name)(*args)(*inputs)
        out.append({"name": name, "args": args, "inputs": [nd(a) for a in inputs], "expect": nd(result)})
    with open(OUT, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    main()
