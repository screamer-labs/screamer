"""Write wasm/smoke/field_axis_oracle.json: every multi-output operator on
(T, K) inputs, computed by the Python binding.

Rule 1 puts the field axis last: an operator with F outputs gives (T, K, F)
on (T, K) inputs. js/test/field_axis.test.ts runs each case in WASM, as a
batch call and as a stream of (K,) events, and checks the shape and the
values. tests/test_field_axis_order.py checks the same in Python.

Each op is built from the manifest's defaults (with the null oracle's fill-ins
for required settings) and called on the fewest inputs it accepts. A series
operator reads each (T, K) input column by column; a row operator (the
cross-sectional and portfolio family) reads each row as one event of K assets.
K is 3, or 2 for an operator of 3 inputs, so that a (T, K) input is never read
as K packed input fields (rule 4). The fixture keeps the last TAIL rows of
each result, past the operators' warm-up, at 12 significant digits.

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

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "wasm_manifest.json")
OUT = os.path.join(HERE, "..", "..", "wasm", "smoke", "field_axis_oracle.json")

T, TAIL, MAX_INPUTS = 64, 8, 10
# Values for settings that an op cannot be built without (as gen_null_oracle).
FILL = {"n": 3, "lower": 1.0, "upper": 3.0}


def _builds(cls, kw):
    try:
        cls(**kw)
        return True
    except Exception:
        return False


def _base(cls, args):
    kw = {a["name"]: FILL[a["name"]] for a in args
          if a["default"] is None and a["name"] in FILL}
    if _builds(cls, kw):
        return kw
    for a in args:
        for v in (10.0, 0.5):
            if _builds(cls, dict(kw, **{a["name"]: v})):
                return dict(kw, **{a["name"]: v})
    for a in args:
        for b in args:
            if a is not b and _builds(cls, dict(kw, **{a["name"]: 48.0, b["name"]: 10.0})):
                return dict(kw, **{a["name"]: 48.0, b["name"]: 10.0})
    return None


def inputs():
    """MAX_INPUTS related (T, 3) random walks: an open, a high above it and
    the close, a low below both, the close, a positive volume, then more."""
    rng = np.random.default_rng(31)
    walk = lambda: np.cumsum(rng.standard_normal((T, 3)), axis=0) + 100.0  # noqa: E731
    xs = [walk() for _ in range(MAX_INPUTS)]
    o, c = xs[0], xs[3]
    xs[1] = np.maximum(np.maximum(o, c), xs[1]) + 1.0
    xs[2] = np.minimum(np.minimum(o, c), xs[2]) - 1.0
    xs[4] = np.abs(xs[4]) + 1.0
    # At the precision the fixture stores, so both sides read the same inputs.
    return [np.array(values(x)).reshape(T, 3) for x in xs]


def values(a):
    return [None if np.isnan(v) else float(f"{v:.12g}") for v in np.asarray(a, float).ravel()]


def case(cls, base, xs):
    """(n, K, result) for the fewest inputs with a (T, K, F) result, or None."""
    for k in (3, 2):
        for n in range(1, MAX_INPUTS + 1):
            try:
                res = cls(**base)(*[x[:, :k] for x in xs[:n]])
            except Exception:
                continue
            if isinstance(res, np.ndarray) and res.ndim == 3:
                return n, k, res
            break        # one output per asset, or the K columns read as fields
    return None


def compute():
    with open(MANIFEST) as f:
        manifest = json.load(f)
    xs = inputs()
    cases = []
    for op in manifest:
        cls = getattr(s, op["name"])
        base = _base(cls, op["ctor_args"])
        found = case(cls, base, xs) if base is not None else None
        if found:
            n, k, res = found
            cases.append({"name": op["name"], "base": base, "inputs": n, "assets": k,
                          "fields": int(res.shape[2]), "row_op": bool(op.get("row_op")),
                          "tail": values(res[T - TAIL:])})
    return {"T": T, "tail": TAIL, "inputs": [values(x) for x in xs], "cases": cases}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stdout", action="store_true", help="print the fixture to stdout")
    args = ap.parse_args()
    text = json.dumps(compute(), sort_keys=True) + "\n"
    if args.stdout:
        sys.stdout.write(text)
        return
    with open(OUT, "w") as f:
        f.write(text)
    print(f"wrote {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
