"""Randomized property: batch == lazy == live for multi-output Pipelines (B7).

Each case builds 2 to 4 outputs from random trees over the inputs a, b
(scalar, with NaN), m (a 0/1/NaN mask), c (a clock) and y (2 columns). The
nodes are Filter, Dropna, Select, CombineLatest, Delay, RollingMean and
Resample: every= with label left or right and fill skip, nan or carry; count=;
threshold= with a Dropna driver; and clock=. Indexes repeat and 15% of cases
are unaligned. Three live sessions per case drain after each event with
probability 1, 0.3 and 0.05, with advance(now) between events; the
concatenated drains must equal batch byte for byte, and lazy must equal batch.

advance(now) stays at or below every input's last event: past it, a fill
resample closes windows that batch never opens, which is what advance means.
Cases with a clock resample run without advance (tracked separately).
The seeds are fixed.
"""
import numpy as np
import pytest

import screamer as sc
from screamer import Input, Pipeline

def concat(parts, m):
    full = [p for p in parts if len(p[0][1])] or parts[-1:]
    out = []
    for j in range(m):
        out.append((np.concatenate([p[j][0] for p in full]),
                    np.concatenate([p[j][1] for p in full])))
    return tuple(out)


def concat_unaligned(parts, m):
    out = []
    for j in range(m):
        full = [p[j] for p in parts if len(p[j][1])] or [parts[-1][j]]
        out.append((np.concatenate([q[0] for q in full]), np.concatenate([q[1] for q in full])))
    return tuple(out)


def same(got, want):
    if len(got) != len(want):
        return f"count {len(got)} vs {len(want)}"
    for j, ((gv, gk), (wv, wk)) in enumerate(zip(got, want)):
        if not np.array_equal(np.asarray(gk), np.asarray(wk)):
            return f"out {j} index {list(gk)[:40]} vs {list(wk)[:40]}"
        gv = np.asarray(gv); wv = np.asarray(wv)
        if gv.size == 0 and wv.size == 0:
            continue
        if gv.shape != wv.shape or gv.tobytes() != wv.tobytes():
            return f"out {j} values {gv.ravel()[:20]} vs {wv.ravel()[:20]} shapes {gv.shape} {wv.shape}"
    return None


def make_case(rng, enable_clock=True):
    n = int(rng.integers(5, 120))
    def idx(k, start=0):
        return (start + np.cumsum(rng.integers(0, 4, k))).astype(np.int64)
    ins = {nm: Input(nm) for nm in "abmcy"}
    x = rng.normal(size=n); x[rng.random(n) < rng.uniform(0, 0.5)] = np.nan
    nb = int(rng.integers(1, 60))
    bv = rng.normal(size=nb); bv[rng.random(nb) < 0.2] = np.nan
    nm_ = int(rng.integers(1, n + 1))
    mv = (rng.random(nm_) > rng.uniform(0, 1)).astype(float)
    mv[rng.random(nm_) < 0.1] = np.nan
    nc = int(rng.integers(1, 15))
    ny = int(rng.integers(1, 40))
    yv = rng.normal(size=(ny, 2)); yv[rng.random((ny, 2)) < 0.1] = np.nan
    feeds = {"a": (x, idx(n, int(rng.integers(0, 30)))), "b": (bv, idx(nb, int(rng.integers(0, 60)))),
             "m": (mv, idx(nm_, int(rng.integers(0, 30)))), "c": (np.ones(nc), idx(nc, int(rng.integers(0, 80))) * 3),
             "y": (yv, idx(ny, int(rng.integers(0, 40))))}
    used = set()
    has_clock = []

    def inp(nm):
        used.add(nm); return ins[nm]

    def scalar_src(depth=0):
        r = rng.integers(0, 9 if depth < 2 else 3)
        if r == 0: return inp("a")
        if r == 1: return inp("b")
        if r == 2: return sc.Select(int(rng.integers(0, 2)))(inp("y"))
        if r == 3: return sc.Dropna()(scalar_src(depth + 1))
        if r == 4: return sc.Filter()(scalar_src(depth + 1), inp("m"))
        if r == 5: return sc.Delay(int(rng.integers(0, 6)))(scalar_src(depth + 1))
        if r == 6: return sc.RollingMean(int(rng.integers(1, 4)))(scalar_src(depth + 1))
        if r == 7: return sc.Select(int(rng.integers(0, 2)))(sc.CombineLatest()(scalar_src(depth + 1), scalar_src(depth + 1)))
        return resample(depth + 1)

    def resample(depth):
        r = rng.integers(0, 4 if enable_clock else 3)
        agg = ["mean", "last", "sum", "first", "count"][int(rng.integers(0, 5))]
        if r == 0:
            kw = dict(every=int(rng.integers(1, 30)), agg=agg)
            if rng.random() < 0.3: kw["label"] = "right"
            if rng.random() < 0.3: kw["fill"] = ["carry", "nan"][int(rng.integers(0, 2))]
            return sc.Resample(**kw)(scalar_src(depth))
        if r == 1:
            return sc.Resample(count=int(rng.integers(1, 6)), agg=agg)(scalar_src(depth))
        if r == 2:
            return sc.Resample(scalar_src(depth), sc.Dropna()(inp("b")), threshold=float(rng.uniform(0.5, 4)), agg="sum")
        has_clock.append(True)
        return sc.Resample(scalar_src(depth), inp("c"), clock=True, agg="last")

    def output():
        r = rng.integers(0, 4)
        if r == 0:
            return sc.CombineLatest()(scalar_src(1), scalar_src(1))
        if r == 1:
            return inp("y") if rng.random() < 0.5 else sc.Dropna()(inp("y"))
        return scalar_src(0)

    m = int(rng.integers(2, 5))
    while True:
        try:
            outs = [output() for _ in range(m)]
            break
        except Exception:
            used.clear()
    names = [nm for nm in "abmcy" if nm in used]
    return [ins[nm] for nm in names], outs, [feeds[nm] for nm in names], names, bool(has_clock)


def events(feeds, names):
    evs = []
    for i, (v, k) in enumerate(feeds):
        for j in range(len(k)):
            evs.append((int(k[j]), i, j, names[i], v[j]))
    evs.sort(key=lambda e: e[:3])
    return evs


def lazy(p, feeds, m):
    gens = [iter([(v[j].tolist() if np.ndim(v[j]) else float(v[j]), int(k[j]))
                  for j in range(len(k))]) for v, k in feeds]
    rows = list(p(*gens))
    k = np.array([r[-1] for r in rows], dtype=np.int64)
    out = []
    for j in range(m):
        vals = [r[j] for r in rows]
        out.append((np.array(vals, dtype=np.float64), k))
    return tuple(out)


def run_case(seed, enable_clock=True, use_advance=True):
    rng = np.random.default_rng(seed)
    inputs, outs, feeds, names, has_clock = make_case(rng, enable_clock)
    # advance(now) never passes an input's last event: past it, a fill resample
    # closes windows that batch never opens (the meaning of advance). With a
    # clock resample, advance is left out (tracked separately).
    use_advance = use_advance and not has_clock
    horizon = min(int(k[-1]) for _, k in feeds)
    align = rng.random() < 0.85
    p = Pipeline(inputs, outs, align_outputs=align)
    m = len(outs)
    try:
        want = p(*feeds)
    except Exception as e:
        return ("batch-raise", repr(e))
    evs = events(feeds, names)
    for trial in range(3):
        s = p.live()
        parts = []
        prob = [1.0, 0.3, 0.05][trial]
        last = None
        for q, (t, _, _, nm, v) in enumerate(evs):
            if use_advance and last is not None and rng.random() < 0.15 and last <= min(t, horizon):
                s.advance(int(rng.integers(last, min(t, horizon) + 1)))
            s.push(nm, t, v.tolist() if np.ndim(v) else float(v))
            last = t
            if rng.random() < prob:
                parts.append(s.result())
        s.flush()
        parts.append(s.result())
        got = concat(parts, m) if align else concat_unaligned(parts, m)
        d = same(got, want)
        if d:
            return ("live", f"trial {trial} align {align}: {d}")
    if align:
        try:
            lz = lazy(p, feeds, m)
        except Exception as e:
            return ("lazy-raise", repr(e))
        want_l = tuple((np.asarray(v, dtype=np.float64), k) for v, k in want)
        # lazy rows are tuples of scalars; skip wide outputs for lazy compare
        if all(np.ndim(v) == 1 for v, _ in want):
            d = same(lz, want_l)
            if d:
                return ("lazy", d)
    return None


N_CASES = 250
CHUNK = 25


@pytest.mark.parametrize("chunk", range(N_CASES // CHUNK))
def test_batch_lazy_live_agree_on_random_graphs(chunk):
    bad = []
    for seed in range(chunk * CHUNK, (chunk + 1) * CHUNK):
        r = run_case(seed, enable_clock=True, use_advance=True)
        if r:
            bad.append((seed, r))
    assert not bad, bad[:3]
