"""Write wasm/smoke/oracle_combine_latest.json: Python's combine_latest on a
fixed set of inputs, which js/test/combine_latest_parity.test.ts replays with
JavaScript's combineLatest. Both call the one C++ join
(include/screamer/streams/join_plan.h), so every value matches exactly.

Each case holds the inputs (values with their shape, and an index per
input; `plain` cases pass them as lists, which JavaScript replays as plain
Arrays: a list is always values, a tuple marks (values, index)), the layout and emit settings, and the result of four forms: the
positional batch, the indexed batch, the positional iterator and the indexed
iterator. A result is the rows (flat values, row shape, index) or the error
(its Python class and message). NaN is written as null.

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
                   "oracle_combine_latest.json")
LAYOUTS = {1: [None], 2: [None, "TA", "TF", "AT", "TAF->TA"], 3: [None, "TAF", "TFA", "TAA"]}


def _nums(a):
    return [None if np.isnan(x) else float(x) for x in np.asarray(a, dtype=float).ravel()]


def _run(f):
    try:
        r = f()
        rows = list(r) if hasattr(r, "__next__") else None
    except Exception as e:  # every error is a result to compare
        return {"error": type(e).__name__, "message": str(e)}
    if rows is not None:
        return {"rows": [{"value": _nums(v), "shape": list(np.shape(v)), "index": i} for v, i in rows]}
    v, i = r
    return {"values": _nums(v), "shape": list(v.shape), "index": None if i is None else _nums(i)}


def _case(name, ins, idxs, layout, emit, plain=False):
    """One case. With `plain`, the inputs, their indexes and their events are
    plain lists (JavaScript replays them as plain Arrays); else NumPy arrays."""
    cl = lambda *a: s.CombineLatest(emit=emit, layout=layout)(*a)   # noqa: E731
    vals = [x.tolist() for x in ins] if plain else ins
    index = [[int(t) for t in i] for i in idxs]
    return {
        "name": name,
        "plain": plain,
        "inputs": [{"values": _nums(x), "shape": list(x.shape)} for x in ins],
        "index": index,
        "layout": layout,
        "emit": emit,
        "batch": _run(lambda: cl(*vals)),
        "batch_indexed": _run(lambda: cl(*[(x, i if plain else np.asarray(i)) for x, i in zip(vals, index)])),
        "iter": _run(lambda: cl(*[iter(list(x)) for x in vals])),
        "iter_indexed": _run(lambda: cl(*[iter([(e, t) for e, t in zip(x, i)]) for x, i in zip(vals, index)])),
    }


def cases():
    out = []
    # Named cases: two-value rows, a time axis that is not first, a reordering
    # layout, an empty input, and lengths that differ.
    T = 4
    two = np.arange(T * 2.0).reshape(T, 2)
    at = np.arange(3.0 * T).reshape(3, T)
    tfa = np.arange(T * 6.0).reshape(T, 2, 3)
    sig = np.arange(T) * 10.0
    ix = np.arange(T)
    out.append(_case("two-value rows", [two, two + 100], [ix, ix + 1], None, "when_all"))
    out.append(_case("two-value rows, plain lists", [two, two + 100], [ix, ix + 1], None, "when_all", plain=True))
    out.append(_case("two-value rows and a signal, plain lists", [two, sig], [ix, ix * 2], None, "on_any",
                     plain=True))
    out.append(_case("layout AT, T", [at, sig], [ix, ix], ["AT", "T"], "when_all"))
    out.append(_case("layout AT, AT", [at, at + 1], [ix, ix], "AT", "on_any"))
    out.append(_case("layout TFA, T", [tfa, sig], [ix, ix * 2], ["TFA", "T"], "when_all"))
    out.append(_case("empty input", [sig, np.zeros(0)], [ix, ix[:0]], None, "on_any"))
    out.append(_case("lengths differ", [sig, sig[:3]], [ix, ix[:3]], None, "when_all"))
    out.append(_case("layouts do not broadcast", [two, np.zeros((T, 3))], [ix, ix], "TA", "when_all"))
    out.append(_case("layout count", [two, sig], [ix, ix], ["TA"], "when_all"))
    # Random cases: 1 to 3 inputs of rank 1 to 3, NaN, repeated indexes.
    rng = np.random.default_rng(24)
    for k in range(80):
        n = int(rng.integers(1, 4))
        T = int(rng.integers(1, 6))
        ins, lays = [], []
        for _ in range(n):
            nd = int(rng.integers(1, 4))
            shp = (T,) if nd == 1 else (T, int(rng.integers(1, 4))) if nd == 2 \
                else (T, int(rng.choice([1, 3])), int(rng.integers(1, 3)))
            x = np.round(rng.normal(size=shp), 3)
            x[rng.random(shp) < 0.1] = np.nan
            ins.append(x)
            lays.append(LAYOUTS[nd][int(rng.integers(len(LAYOUTS[nd])))])
        layout = None if all(lay is None for lay in lays) else lays
        idxs = [np.sort(rng.integers(0, 8, size=T)) for _ in range(n)]
        out.append(_case(f"random {k}", ins, idxs, layout, ["when_all", "on_any"][k % 2]))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stdout", action="store_true", help="print the fixture instead of writing it")
    args = p.parse_args()
    text = json.dumps(cases(), indent=1, allow_nan=False) + "\n"
    if args.stdout:
        sys.stdout.write(text)
        return
    with open(OUT, "w") as f:
        f.write(text)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
