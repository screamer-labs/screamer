"""Write wasm/smoke/oracle_time_splits.json: Python's time_splits folds, which
js/test/time_splits.test.ts replays in JavaScript.

Each case holds the arguments in the JavaScript form (camelCase option keys)
and the folds as [train, test] lists of row indices. The cases cover the
default test size, an explicit one, max_train_size, and an integer and an
array label_length.

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
                   "oracle_time_splits.json")


def _label(n):
    rng = np.random.default_rng(5)
    return [float(v) for v in rng.integers(0, 9, n)]


# n, n_splits, Python keyword options (snake_case)
CASES = [
    (12, 3, {}),
    (101, 5, {}),
    (50, 3, {"test_size": 7}),
    (60, 4, {"test_size": 5, "max_train_size": 8, "label_length": 2}),
    (80, 4, {"label_length": _label(80)}),
    (80, 3, {"label_length": _label(80), "max_train_size": 12}),
]


def _camel(name):
    head, *rest = name.split("_")
    return head + "".join(w.title() for w in rest)


def fixture():
    cases = []
    for n, n_splits, kw in CASES:
        folds = s.time_splits(n, n_splits, **kw)
        cases.append({"n": n, "nSplits": n_splits,
                      "options": {_camel(k): v for k, v in kw.items()},
                      "folds": [[tr.tolist(), te.tolist()] for tr, te in folds]})
    return {"cases": cases}


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
