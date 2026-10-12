"""Time every public operator in batch mode and compare two builds.

The sweep constructs every public screamer class that accepts default
arguments, or needs only ``span`` or ``period`` (set to 20), and calls it on
the first ``n`` of six synthetic price series, taking the smallest ``n`` the
operator accepts. Each operator runs in two shapes: a 1-D series of 200,000
values and a 2-D array of shape (2000, 100), the same values reshaped. A
timing is the best of 7 samples, and each sample repeats the call until it
lasts at least 2 ms. The JSON output holds the time in ns per element and the
last 2000 output values of every operator (``--values`` sets the count; the
committed baseline ``bench_baseline_pre_fpcontract.json`` keeps 50).

Record this build::

    .venv/bin/python devtools/bench_all_ops.py run out.json

Record another build, for example a scratch worktree built with ``cmake``.
``--src`` imports the ``screamer`` package of that tree (with the extension
``make build`` copies into it) instead of the editable install::

    .venv/bin/python devtools/bench_all_ops.py run ref.json --src /path/to/tree

Print the operators that are slower than 1.05x in either shape, and fail
when one is slower than the gate (1.10x by default). Several files per side
take the per-operator median. Some loops run at one of two speeds that
depend on the process (the memory layout), so use at least three runs per
side::

    .venv/bin/python devtools/bench_all_ops.py compare ref.json new.json
    .venv/bin/python devtools/bench_all_ops.py compare a1.json,a2.json,a3.json b1.json,b2.json,b3.json

``compare`` also reports the largest relative change of the output values.
"""
import argparse
import inspect
import json
import math
import statistics
import sys
import time

T = 200_000
SHAPE_2D = (2000, 100)
N_SERIES = 6
REPEATS = 7
MIN_SAMPLE_S = 0.002
N_VALUES = 2000
# Tried in order. Most operators have defaults; the exponentially weighted
# family needs a span and the Ehlers filters a period.
CONSTRUCT_KWARGS = ({}, {"span": 20}, {"period": 20})


def import_screamer(src):
    """Import screamer, from ``src`` instead of the editable install if given."""
    if src:
        # The editable install puts a redirecting finder at the front of
        # sys.meta_path, ahead of sys.path. Drop it so ``src`` wins.
        sys.meta_path[:] = [
            f for f in sys.meta_path
            if not type(f).__module__.startswith("_editable")
        ]
        sys.path.insert(0, src)
    import screamer
    return screamer


def construct(cls):
    """An instance with default arguments, or with the one required length."""
    for kwargs in CONSTRUCT_KWARGS:
        try:
            return cls(**kwargs)
        except Exception:
            pass
    return None


def make_series(np):
    rng = np.random.default_rng(0)
    base = np.cumsum(rng.standard_normal(T)) + 1000.0
    xs = [base + rng.random(T) for _ in range(N_SERIES)]
    xs[1] = np.maximum(xs[0], xs[1]) + 1.0  # high
    xs[2] = np.minimum(xs[0], xs[2]) - 1.0  # low
    return xs


def as_array(np, out):
    """The output as one float array, or None when it is not array data."""
    if isinstance(out, np.ndarray):
        return np.asarray(out, dtype=float)
    if isinstance(out, (tuple, list)) and out and all(isinstance(o, np.ndarray) for o in out):
        return np.concatenate([np.asarray(o, dtype=float).ravel() for o in out])
    return None


def time_call(op, args):
    reps = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(reps):
            op(*args)
        dt = time.perf_counter() - t0
        if dt >= MIN_SAMPLE_S:
            break
        reps = max(reps * 2, int(reps * MIN_SAMPLE_S / max(dt, 1e-9)) + 1)
    best = dt
    for _ in range(REPEATS - 1):
        t0 = time.perf_counter()
        for _ in range(reps):
            op(*args)
        best = min(best, time.perf_counter() - t0)
    return best / reps


def run(path, src, only, n_values):
    s = import_screamer(src)
    import numpy as np

    xs = make_series(np)
    shapes = {"1d": xs, "2d": [x.reshape(SHAPE_2D) for x in xs]}
    result = {"meta": {"screamer": s.__file__, "T": T, "shape_2d": SHAPE_2D},
              "shapes": {k: {"ns": {}, "vals": {}} for k in shapes}}
    names = sorted(n for n in dir(s) if not n.startswith("_"))
    if only:
        names = [n for n in names if n in only]
    for name in names:
        cls = getattr(s, name)
        if not inspect.isclass(cls) or not callable(cls):
            continue
        op = construct(cls)
        if op is None:
            continue
        n_args = None
        for n in range(1, N_SERIES + 1):
            try:
                out = as_array(np, op(*xs[:n]))
            except Exception:
                continue
            if out is not None:
                n_args = n
            break
        if n_args is None:
            continue
        for shape, data in shapes.items():
            args = data[:n_args]
            try:
                out = as_array(np, op(*args))
            except Exception:
                continue
            if out is None:
                continue
            ns = time_call(op, args) / data[0].size * 1e9
            result["shapes"][shape]["ns"][name] = ns
            tail = out.ravel()[max(out.size - n_values, 0):] if n_values else out[:0]
            result["shapes"][shape]["vals"][name] = tail.tolist()
        print(f"{name:32s} " + "  ".join(
            f"{k} {result['shapes'][k]['ns'].get(name, float('nan')):8.3f}" for k in shapes),
            flush=True)
    with open(path, "w") as f:
        json.dump(result, f)
    print(len(result["shapes"]["1d"]["ns"]), "ops ->", path)


def load_runs(spec):
    """Per-operator median time over a comma-separated list of JSON files."""
    files = [json.load(open(p)) for p in spec.split(",")]
    merged = {}
    for shape in files[0]["shapes"]:
        runs = {}
        for f in files:
            for k, v in f["shapes"].get(shape, {}).get("ns", {}).items():
                runs.setdefault(k, []).append(v)
        merged[shape] = {"ns": {k: statistics.median(v) for k, v in runs.items()},
                         "vals": files[0]["shapes"][shape]["vals"]}
    return merged


def rel_diff(a, b):
    """Largest relative difference over the values both runs kept (aligned at the end)."""
    n = min(len(a), len(b))
    worst = 0.0
    for x, y in zip(a[len(a) - n:], b[len(b) - n:]):
        if x == y or (x != x and y != y):
            continue
        if (x != x) != (y != y):
            return math.inf
        worst = max(worst, abs(x - y) / max(abs(x), abs(y), 1e-300))
    return worst


def compare(ref_spec, new_spec, gate, show):
    ref, new = load_runs(ref_spec), load_runs(new_spec)
    failed = False
    for shape in ref:
        r, n = ref[shape]["ns"], new.get(shape, {}).get("ns", {})
        common = sorted(set(r) & set(n))
        ratios = {k: n[k] / r[k] for k in common}
        logs = [math.log(v) for v in ratios.values()]
        geo = math.exp(sum(logs) / len(logs)) if logs else float("nan")
        med = sorted(ratios.values())[len(ratios) // 2] if ratios else float("nan")
        print(f"[{shape}] {len(common)} ops, geomean {geo:.3f}, median {med:.3f}")
        missing = sorted(set(r) ^ set(n))
        if missing:
            shown = ", ".join(missing[:10]) + (", ..." if len(missing) > 10 else "")
            print(f"  {len(missing)} ops in one side only: {shown}")
        for k, v in sorted(ratios.items(), key=lambda kv: -kv[1]):
            if v <= show:
                break
            flag = "  OVER GATE" if v > gate else ""
            failed |= v > gate
            print(f"  {k:32s} {r[k]:9.3f} -> {n[k]:9.3f} ns/elem  {v:5.2f}x{flag}")
        changed = {}
        for k in common:
            d = rel_diff(ref[shape]["vals"].get(k, []), new[shape]["vals"].get(k, []))
            if d > 0:
                changed[k] = d
        print(f"  values changed in {len(changed)} ops")
        for k, d in sorted(changed.items(), key=lambda kv: -kv[1])[:15]:
            print(f"    {k:32s} max rel diff {d:.3g}")
    print("FAIL" if failed else "PASS", f"(gate {gate:.2f}x)")
    return 1 if failed else 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run", help="time every operator and write JSON")
    pr.add_argument("out")
    pr.add_argument("--src", help="import screamer from this source tree")
    pr.add_argument("--only", help="comma-separated operator names")
    pr.add_argument("--values", type=int, default=N_VALUES,
                    help=f"output values kept per operator and shape (default {N_VALUES})")
    pc = sub.add_parser("compare", help="compare two runs")
    pc.add_argument("ref", help="reference JSON (comma-separated for several runs)")
    pc.add_argument("new", help="new JSON (comma-separated for several runs)")
    pc.add_argument("--gate", type=float, default=1.10)
    pc.add_argument("--show", type=float, default=1.05)
    a = p.parse_args()
    if a.cmd == "run":
        run(a.out, a.src, set(a.only.split(",")) if a.only else None, a.values)
        return 0
    return compare(a.ref, a.new, a.gate, a.show)


if __name__ == "__main__":
    sys.exit(main())
