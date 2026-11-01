"""Write wasm/smoke/save_oracle.json: Python's save files, which
js/test/save_restore.test.ts reads and writes in JavaScript.

"cases" has one entry per functor (tests/save_cases.py lists the operators,
their settings and their inputs):

    name                  the operator
    settings              its settings(), the input of fromSettings
    prefix, rest          the events before and after the save point, one
                          array per input
    json_text, cbor_hex   the file after feeding `prefix` one event at a time
    expected_rest_output  the outputs of feeding `rest` after loading, one
                          row per event

The save point is after two windows (tests/save_cases.py splits). "registry"
lists every functor with the settings it is built with in the save tests, so
JavaScript can check that fromSettings builds each one.

Numbers are written exactly; NaN is null, and an infinity is the string
"Infinity" or "-Infinity" (in "settings" null means not given, and only the
infinities are spelled as strings). The files hold the library version they were
written with; the tests compare them with it normalized.

Modes:
    (no args)  write the fixture
    --stdout   print the fixture to stdout (freshness gate)
"""
import argparse
import contextlib
import json
import math
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)

# The test modules print while they load; keep stdout for the fixture.
with contextlib.redirect_stdout(sys.stderr):
    import screamer
    from screamer import screamer_bindings
    from tests.save_cases import FUNCTOR_NAMES, REST, case, feed, kwargs_for, splits

OUT = os.path.join(REPO, "wasm", "smoke", "save_oracle.json")


def num(x):
    x = float(x)
    if math.isnan(x):
        return None
    if math.isinf(x):
        return "Infinity" if x > 0 else "-Infinity"
    return x


def nums(a):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 0:
        return num(a)
    return [nums(v) for v in a]


def setting(v):
    """A constructor argument as a settings value."""
    if isinstance(v, screamer_bindings.EvalOp):
        return {"type": type(v).__name__, "settings": v.settings()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [setting(x) for x in v]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return num(v)
    return v


def settings_tree(v):
    """A settings() tree with its infinities spelled as strings; null stays."""
    if isinstance(v, dict):
        return {k: settings_tree(x) for k, x in v.items()}
    if isinstance(v, list):
        return [settings_tree(x) for x in v]
    if isinstance(v, float) and math.isinf(v):
        return num(v)
    return v


def cases():
    out = []
    for name in FUNCTOR_NAMES:
        make, inputs = case(name)
        k = splits(name, kwargs_for(name))[2]
        a = make()
        feed(a, inputs, 0, k)
        out.append({
            "name": name,
            "settings": settings_tree(a.settings()),
            "prefix": [nums(x[:k]) for x in inputs],
            "rest": [nums(x[k:k + REST]) for x in inputs],
            "json_text": a.dumps(),
            "cbor_hex": a.dumps(format="cbor").hex(),
            "expected_rest_output": nums(feed(a, inputs, k, k + REST)),
        })
    return out


def registry():
    return [{"name": n, "settings": {k: setting(v) for k, v in kwargs_for(n).items()}}
            for n in FUNCTOR_NAMES]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stdout", action="store_true", help="print the fixture instead of writing it")
    args = p.parse_args()
    text = json.dumps({"cases": cases(), "registry": registry()}, allow_nan=False, indent=1) + "\n"
    if args.stdout:
        sys.stdout.write(text)
        return
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
