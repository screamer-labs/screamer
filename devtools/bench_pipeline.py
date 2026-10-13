"""Time the Pipeline engine and the stream operators, and compare two builds.

``bench_all_ops.py`` times the compute operators on arrays. This sweep times
the paths around them: a Pipeline graph run in batch, lazily and live, the
stream operators (Resample, CombineLatest, Merge, split, forecast_pairs) and
the packed (T, ..., N) input of every multi-input operator next to its N
separate arrays. Every case uses only API that 2.8.0 has, so the same script
times the 2.8.0 wheel, an older tree and this tree.

A timing is the best of 7 samples, and each sample repeats the call until it
lasts at least 20 ms. The JSON output holds the time in ms per call of every
case.

Record this build, the 2.8.0 wheel (from a directory outside the repository,
so the wheel is imported), or another tree built with ``make build``::

    .venv/bin/python devtools/bench_pipeline.py run new.json
    /path/to/old/bin/python -I devtools/bench_pipeline.py run old.json
    .venv/bin/python devtools/bench_pipeline.py run ref.json --src /path/to/tree

Print every case of two builds with the ratio new / ref, and fail when a case
is slower than the gate (1.10x by default). Several files per side take the
per-case median; some loops run at one of two speeds that depend on the
process, so use at least three runs per side::

    .venv/bin/python devtools/bench_pipeline.py compare ref.json new.json
    .venv/bin/python devtools/bench_pipeline.py compare a1.json,a2.json,a3.json b1.json,b2.json,b3.json

The operator cases are compared within a build: ``compare`` prints the time of
the packed input over the time of the separate arrays in each build, and fails
when the new build's ratio is over the gate or missing (rule 4: two spellings
of one call cost the same). ``bench_all_ops.py`` compares the operators across
builds. ``compare`` also fails on an error of the new build, which ``run``
records: a case that raises, or a packed call that raises or gives other
values than the separate call; and on a case of the reference that the new
run lacks. Check the gate itself with::

    .venv/bin/python devtools/bench_pipeline.py selftest
"""
import argparse
import json
import os
import statistics
import sys
import time

REPEATS = 7
MIN_SAMPLE_S = 0.02
T = 1_000_000           # rows of a batch case
T_LAZY = 200_000        # events of a lazy or live case
T_OP = 200_000          # rows of a packed-versus-separate case
WIDE = 8                # columns of a wide case
OP_SHAPE_2D = (2000, 100)
# An operator call slower than this is checked (packed equals separate) but
# not timed: VPIN takes about 1 s on T_OP rows, which would take most of the
# sweep, and its speed differs between calls by about 12% in either spelling.
MAX_OP_S = 0.1
# Constructor arguments tried in order (see bench_all_ops.py).
CONSTRUCT_KWARGS = ({}, {"span": 20}, {"period": 20}, {"hp_period": 48, "lp_period": 10})


def import_screamer(src):
    """Import screamer, from ``src`` instead of the editable install if given."""
    if src:
        sys.meta_path[:] = [f for f in sys.meta_path
                            if not type(f).__module__.startswith("_editable")]
        sys.path.insert(0, src)
    import screamer
    return screamer


def time_call(fn):
    """Seconds per call: the best of REPEATS samples of at least MIN_SAMPLE_S."""
    reps = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        dt = time.perf_counter() - t0
        if dt >= MIN_SAMPLE_S:
            break
        reps = max(reps * 2, int(reps * MIN_SAMPLE_S / max(dt, 1e-9)) + 1)
    best = dt
    for _ in range(REPEATS - 1):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        best = min(best, time.perf_counter() - t0)
    return best / reps


# ---- the cases ---------------------------------------------------------------
# Each case is a function (s, np, d) -> a callable of no arguments, where d
# holds the shared data. A case that the build cannot run is skipped.

def data(np):
    rng = np.random.default_rng(0)
    d = {}
    d["x"] = np.cumsum(rng.standard_normal(T)) + 1000.0
    d["y"] = np.cumsum(rng.standard_normal(T)) + 1000.0
    d["i"] = np.arange(T, dtype=np.int64)
    # Two streams with interleaved, irregular indexes.
    d["ia"] = np.cumsum(rng.integers(1, 4, T)).astype(np.int64)
    d["ib"] = np.cumsum(rng.integers(1, 4, T)).astype(np.int64)
    d["w"] = np.cumsum(rng.standard_normal((T // WIDE, WIDE)), axis=0) + 1000.0
    d["iw"] = np.arange(T // WIDE, dtype=np.int64)
    d["clk"] = np.arange(0, int(d["ia"][-1]), 10, dtype=np.int64)
    d["src"] = (rng.random(T) < 0.5).astype(np.int64)
    d["xl"] = d["x"][:T_LAZY]
    d["il"] = d["i"][:T_LAZY]
    d["wl"] = d["w"][:T_LAZY // WIDE]
    return d


def pipe(s, ins, out_fn, **kw):
    nodes = [s.Input(n) for n in ins]
    outs = out_fn(*nodes)
    return s.Pipeline(nodes, outs if isinstance(outs, list) else [outs], **kw)


def c_pipe_lag(s, np, d):
    p = pipe(s, "x", lambda x: s.Lag(1)(x))
    return lambda: p((d["x"], d["i"]))


def c_pipe_chain(s, np, d):
    p = pipe(s, "x", lambda x: s.EwMean(span=20)(s.Diff()(x)))
    return lambda: p((d["x"], d["i"]))


def c_pipe_multi_input(s, np, d):
    p = pipe(s, "ab", lambda a, b: s.Sub()(s.CombineLatest()(a, b)))
    return lambda: p((d["x"], d["ia"]), (d["y"], d["ib"]))


def c_pipe_wide(s, np, d):
    p = pipe(s, "x", lambda x: s.Resample(count=4, agg="last")(s.Dropna()(x)))
    return lambda: p((d["w"], d["iw"]))


def c_pipe_resample_every(s, np, d):
    p = pipe(s, "x", lambda x: s.Resample(freq=10, agg="mean")(x))
    return lambda: p((d["x"], d["i"]))


def c_pipe_resample_count(s, np, d):
    p = pipe(s, "x", lambda x: s.Resample(count=10, agg="ohlc")(x))
    return lambda: p((d["x"], d["i"]))


def c_pipe_resample_clock(s, np, d):
    p = pipe(s, "xc", lambda x, c: s.Resample(x, c, clock=True, agg="last"))
    clk = (np.zeros(len(d["clk"])), d["clk"])
    return lambda: p((d["x"], d["ia"]), clk)


def c_pipe_aligned_multi_output(s, np, d):
    p = pipe(s, "x", lambda x: [s.EwMean(span=20)(x), s.Lag(1)(x)])
    return lambda: p((d["x"], d["i"]))


def c_pipe_forecast_pairs_count(s, np, d):
    p = pipe(s, "xy", lambda x, y: s.forecast_pairs(x, y, count=5))
    return lambda: p((d["x"], d["i"]), (d["y"], d["i"]))


def c_pipe_lazy(s, np, d):
    p = pipe(s, "x", lambda x: s.EwMean(span=20)(s.Diff()(x)))
    xl, il = d["xl"].tolist(), d["il"].tolist()
    return lambda: sum(1 for _ in p(zip(xl, il)))


def c_pipe_live_push(s, np, d):
    p = pipe(s, "x", lambda x: s.EwMean(span=20)(x))
    xl, il = d["xl"].tolist(), d["il"].tolist()

    def run():
        L = p.live()
        push = L.push
        for v, t in zip(xl, il):
            push("x", t, v)
        L.flush()
        L.result()
    return run


def c_resample_every(s, np, d):
    r = s.Resample(freq=10, agg="mean")
    return lambda: r((d["x"], d["i"]))


def c_resample_count(s, np, d):
    r = s.Resample(count=10, agg="ohlc")
    return lambda: r((d["x"], d["i"]))


def c_resample_clock(s, np, d):
    clk = (np.zeros(len(d["clk"])), d["clk"])
    return lambda: s.Resample((d["x"], d["ia"]), clk, clock=True, agg="last")


def c_resample_wide_every(s, np, d):
    r = s.Resample(freq=10, agg="last")
    return lambda: r((d["w"], d["iw"]))


def c_resample_lazy_every(s, np, d):
    r = s.Resample(freq=10, agg="mean")
    xl, il = d["xl"].tolist(), d["il"].tolist()
    return lambda: sum(1 for _ in r(zip(xl, il)))


def c_resample_lazy_count(s, np, d):
    r = s.Resample(count=10, agg="ohlc")
    xl, il = d["xl"].tolist(), d["il"].tolist()
    return lambda: sum(1 for _ in r(zip(xl, il)))


def c_resample_lazy_clock(s, np, d):
    xl, il = d["xl"].tolist(), d["ia"][:T_LAZY].tolist()
    ck = d["clk"][d["clk"] <= il[-1]].tolist()
    return lambda: sum(1 for _ in s.Resample(zip(xl, il), zip([0.0] * len(ck), ck),
                                             clock=True, agg="last"))


def c_resample_lazy_wide(s, np, d):
    r = s.Resample(freq=10, agg="last")
    rows, il = list(d["wl"]), d["iw"][:len(d["wl"])].tolist()
    return lambda: sum(1 for _ in r(zip(rows, il)))


def c_combine_latest_batch(s, np, d):
    c = s.CombineLatest()
    return lambda: c((d["x"], d["ia"]), (d["y"], d["ib"]))


def c_combine_latest_positional_iter(s, np, d):
    c = s.CombineLatest()
    a, b = d["xl"].tolist(), d["y"][:T_LAZY].tolist()
    return lambda: sum(1 for _ in c(iter(a), iter(b)))


def c_combine_latest_indexed_iter(s, np, d):
    c = s.CombineLatest()
    a = list(zip(d["xl"].tolist(), d["ia"][:T_LAZY].tolist()))
    b = list(zip(d["y"][:T_LAZY].tolist(), d["ib"][:T_LAZY].tolist()))
    return lambda: sum(1 for _ in c(iter(a), iter(b)))


def c_merge(s, np, d):
    m = s.Merge()
    return lambda: m((d["x"], d["ia"]), (d["y"], d["ib"]))


def c_split(s, np, d):
    return lambda: s.split((d["x"], d["i"]), d["src"])


def c_forecast_pairs_count(s, np, d):
    return lambda: s.forecast_pairs(d["x"], d["y"], count=5)


def c_forecast_pairs_duration(s, np, d):
    return lambda: s.forecast_pairs((d["x"], d["ia"]), (d["y"], d["ia"]), duration=5)


CASES = {name[2:]: fn for name, fn in globals().items() if name.startswith("c_")}


def construct(cls):
    for kwargs in CONSTRUCT_KWARGS:
        try:
            return cls(**kwargs)
        except Exception:
            pass
    return None


# Operators that take one array per input by design and refuse a packed
# argument ("this operator takes N inputs, got 1"): the portfolio engines read
# one (T, K) array per input.
NO_PACKED = {"BacktestPortfolioTarget", "BacktestPortfolioOHLCTarget"}

# Extra constructions timed besides each operator's default one (name, args).
EXTRA_OPS = (("WilliamsR", (20,)), ("WilliamsR", (50,)))


def packed_cases(s, np, errors, untimed):
    """packed:<op>:<shape> and separate:<op>:<shape> for every multi-input operator.

    A pair whose packed call raises or differs from the separate call is added
    to `errors` (the gate fails on it); a pair over MAX_OP_S a call to
    `untimed`. A shape the separate call refuses is not a case."""
    import inspect
    rng = np.random.default_rng(1)
    base = np.cumsum(rng.standard_normal(T_OP)) + 1000.0
    xs = [base + rng.random(T_OP) for _ in range(6)]
    xs[1] = np.maximum(xs[0], xs[1]) + 1.0
    xs[2] = np.minimum(xs[0], xs[2]) - 1.0
    out = {}
    ops = []
    # The operators: EvalOp classes (a stream combinator such as Filter is
    # not one), and of those, the ones that take a packed argument.
    evalop = getattr(getattr(s, "screamer_bindings", None), "EvalOp", None)
    if evalop is None:
        import importlib
        evalop = importlib.import_module(s.__name__ + ".screamer_bindings").EvalOp
    for name in sorted(n for n in dir(s) if not n.startswith("_")):
        cls = getattr(s, name)
        if inspect.isclass(cls) and issubclass(cls, evalop) and name not in NO_PACKED:
            ops.append((name, construct(cls)))
    for name, args in EXTRA_OPS:
        try:
            ops.append((f"{name}({', '.join(map(str, args))})", getattr(s, name)(*args)))
        except Exception:
            pass
    for name, op in ops:
        if op is None:
            continue
        n_in = None
        for n in range(1, len(xs) + 1):     # the inputs: the first count that runs
            try:
                op(*xs[:n])
            except Exception:
                continue
            n_in = n
            break
        if n_in is None or n_in < 2:
            continue
        for shape, cols in (("1d", xs[:n_in]),
                            ("2d", [x.reshape(OP_SHAPE_2D) for x in xs[:n_in]])):
            packed = np.ascontiguousarray(np.stack(cols, axis=-1))
            key = f"{name}:{shape}"
            try:
                t0 = time.perf_counter()
                a = op(*cols)
                slow = time.perf_counter() - t0 > MAX_OP_S
            except Exception:
                continue                        # the shape is not a case of this operator
            try:
                b = op(packed)
            except Exception as e:
                errors.append(f"packed:{key} raises {type(e).__name__}: {str(e)[:80]}")
                continue
            if not np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True):
                errors.append(f"packed:{key} differs from separate")
                continue
            if slow:
                untimed.extend([f"separate:{key}", f"packed:{key}"])
                print(f"{key}: values checked, not timed (over {MAX_OP_S} s a call)")
                continue
            out[f"separate:{name}:{shape}"] = (lambda op=op, cols=cols: op(*cols))
            out[f"packed:{name}:{shape}"] = (lambda op=op, p=packed: op(p))
    return out


def run(path, src, only, packed):
    s = import_screamer(src)
    import numpy as np
    d = data(np)
    result = {"meta": {"screamer": s.__file__, "version": getattr(s, "__version__", "?")},
              "ms": {}, "errors": [], "untimed": []}
    cases = {k: (lambda f=f: f(s, np, d)) for k, f in CASES.items()}
    if packed:
        cases.update({k: (lambda f=f: f) for k, f in
                      packed_cases(s, np, result["errors"], result["untimed"]).items()})
    for name, make in cases.items():
        if only and not any(o in name for o in only):
            continue
        try:
            fn = make()
            fn()
        except Exception as e:
            # A case this build cannot run: an error of this build, which the
            # gate fails on when this build is the new one.
            result["errors"].append(f"{name} raises {type(e).__name__}: {str(e)[:80]}")
            print(f"{name:44s} raises: {type(e).__name__}: {str(e)[:60]}", flush=True)
            continue
        ms = time_call(fn) * 1e3
        result["ms"][name] = ms
        print(f"{name:44s} {ms:9.3f} ms", flush=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    print(len(result["ms"]), "cases ->", path)


def load(spec):
    """The per-case median of several runs, and their errors and untimed cases."""
    runs = [json.load(open(p)) for p in spec.split(",")]
    keys = set().union(*(r["ms"] for r in runs))
    ms = {k: statistics.median(r["ms"][k] for r in runs if k in r["ms"]) for k in keys}
    errors = sorted(set().union(*(r.get("errors", []) for r in runs)))
    untimed = set().union(*(r.get("untimed", []) for r in runs))
    return ms, errors, untimed


def compare(ref_spec, new_spec, gate):
    """Prints the cases of two builds; returns 1 when the gate fails: a case
    over the gate, an error of the new build (a case that raises, a packed call
    that raises or differs), a case of the reference that the new build lacks,
    or a packed / separate ratio of the new build that is over the gate or
    missing."""
    (ref, _, _), (new, new_errors, new_untimed) = load(ref_spec), load(new_spec)
    fails = [f"error: {e}" for e in new_errors]
    print(f"{'case':36s} {'ref ms':>10s} {'new ms':>10s} {'new/ref':>8s}")
    for k in sorted(k for k in set(ref) | set(new) if ":" not in k):
        r, n = ref.get(k), new.get(k)
        if n is None:
            fails.append(f"{k} missing from the new run")
            print(f"{k:36s} {r:10.3f} {'missing':>10s}")
            continue
        if r is None:
            print(f"{k:36s} {'-':>10s} {n:10.3f}")
            continue
        slower = n / r > gate
        if slower:
            fails.append(k)
        print(f"{k:36s} {r:10.3f} {n:10.3f} {n / r:8.3f}{' SLOWER' if slower else ''}")
    # The operator cases compare the two spellings within each build; the
    # operators' own speed across builds is bench_all_ops.py's.
    pairs = sorted({k.split(":", 1)[1] for k in set(ref) | set(new) if ":" in k})
    if pairs:
        print(f"\n{'packed / separate':36s} {'ref':>10s} {'new':>10s} {'new ms':>8s}")
    for k in pairs:
        def ratio(d):
            p, s = d.get("packed:" + k), d.get("separate:" + k)
            return p / s if p and s else float("nan")
        if "packed:" + k in new_untimed:
            print(f"{k:36s} {ratio(ref):10.3f} {'untimed':>10s}")
            continue
        rn = ratio(new)
        bad = not (rn <= gate)                  # a NaN ratio (a side missing) fails too
        if bad:
            fails.append("packed:" + k + (" missing" if rn != rn else ""))
        p = new.get("packed:" + k, float("nan"))
        print(f"{k:36s} {ratio(ref):10.3f} {rn:10.3f} {p:8.3f}{' FAIL' if bad else ''}")
    if fails:
        print(f"{len(fails)} failure(s) of the {gate:.2f}x gate: {', '.join(fails)}")
        return 1
    print(f"all cases within the {gate:.2f}x gate")
    return 0


def selftest():
    """Checks that compare passes a clean run and fails each kind of failure."""
    import contextlib
    import io
    import tempfile
    base = {"ms": {"pipe_a": 10.0, "separate:Op:1d": 1.0, "packed:Op:1d": 1.0},
            "errors": [], "untimed": []}
    def case(**change):
        new = json.loads(json.dumps(base))
        for k, v in change.items():
            if k == "ms":
                new["ms"].update(v)
            else:
                new[k] = v
        return new
    scenarios = {
        "clean": (case(), 0),
        "slower": (case(ms={"pipe_a": 12.0}), 1),
        "error": (case(errors=["packed:Op:1d differs from separate"]), 1),
        "missing case": ({**base, "ms": {k: v for k, v in base["ms"].items() if k != "pipe_a"}}, 1),
        "missing packed": ({**base, "ms": {k: v for k, v in base["ms"].items()
                                           if k != "packed:Op:1d"}}, 1),
        "packed slower": (case(ms={"packed:Op:1d": 1.2}), 1),
        "untimed": ({**base, "ms": {"pipe_a": 10.0},
                     "untimed": ["separate:Op:1d", "packed:Op:1d"]}, 0),
    }
    ok = True
    with tempfile.TemporaryDirectory() as d:
        ref = os.path.join(d, "ref.json")
        json.dump(base, open(ref, "w"))
        for name, (new, want) in scenarios.items():
            path = os.path.join(d, "new.json")
            json.dump(new, open(path, "w"))
            with contextlib.redirect_stdout(io.StringIO()):
                got = compare(ref, path, 1.10)
            ok &= got == want
            print(f"{name:16s} exit {got}, expected {want}{'' if got == want else '  WRONG'}")
    print("selftest", "passed" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("out")
    r.add_argument("--src", help="import screamer from this source tree")
    r.add_argument("--only", nargs="*", help="run the cases whose name holds one of these")
    r.add_argument("--no-packed", action="store_true",
                   help="skip the packed-versus-separate operator cases")
    c = sub.add_parser("compare")
    c.add_argument("ref")
    c.add_argument("new")
    c.add_argument("--gate", type=float, default=1.10)
    sub.add_parser("selftest", help="check that compare fails each kind of failure")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.out, a.src, a.only, not a.no_packed)
    elif a.cmd == "selftest":
        sys.exit(selftest())
    else:
        sys.exit(compare(a.ref, a.new, a.gate))


if __name__ == "__main__":
    main()
