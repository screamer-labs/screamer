"""Randomized property: batch == lazy == live for multi-output Pipelines (B7).

Each case builds 2 to 4 outputs from random trees over the inputs a, b
(scalar, with NaN), m (a 0/1/NaN mask), c (a clock) and y (2 columns). The
nodes are Filter, Dropna, Select, CombineLatest, Delay, RollingMean,
RollingCov and Dropna given two producers (two streams that the node takes
in index order), and Resample: every= with label left or right and fill skip, nan or carry; count=;
threshold= with a Dropna driver; and clock= with the clock c, Dropna(c), a
Filter, an every=, count= or threshold= Resample of c, CombineLatest, or a
nested clock resample. Indexes repeat and 15% of cases are unaligned, and in
40% of cases the inputs end with rows that Dropna or Filter drop (a NaN tail,
a 0 mask tail). Three live sessions per case drain after each event with
probability 1, 0.3 and 0.05, with advance(now) between events; the first
pushes in index order and the other two let inputs run ahead of each other
(each input in its own order). The concatenated drains must equal batch byte
for byte, and lazy must equal batch.

advance(now) stays at or below every input's last event: past it, a fill
resample closes windows that batch never opens, which is what advance means.
It also stays at or below every input's next event, as a live session
requires. The seeds are fixed.
"""
import numpy as np
import pytest

import screamer as sc
from screamer import Input, Pipeline
from screamer.dag import make_operator_node

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
    if rng.random() < 0.4:                       # inputs that end with dropped rows
        x[n - int(rng.integers(1, n // 2 + 2)):] = np.nan
        bv[nb - int(rng.integers(1, nb // 2 + 2)):] = np.nan
        mv[nm_ - int(rng.integers(1, nm_ // 2 + 2)):] = 0.0
    feeds = {"a": (x, idx(n, int(rng.integers(0, 30)))), "b": (bv, idx(nb, int(rng.integers(0, 60)))),
             "m": (mv, idx(nm_, int(rng.integers(0, 30)))), "c": (np.ones(nc), idx(nc, int(rng.integers(0, 80))) * 3),
             "y": (yv, idx(ny, int(rng.integers(0, 40))))}
    used = set()
    has_clock = []

    def inp(nm):
        used.add(nm); return ins[nm]

    def scalar_src(depth=0):
        r = rng.integers(0, 11 if depth < 2 else 3)
        if r == 9:                               # an operator with two producers
            return sc.RollingCov(int(rng.integers(2, 4)))(wide_src(depth + 1), wide_src(depth + 1))
        if r == 10:                              # a Dropna with two producers
            return make_operator_node(sc.Dropna, (scalar_src(depth + 1), scalar_src(depth + 1)),
                                      {"how": "any"})
        if r == 0: return inp("a")
        if r == 1: return inp("b")
        if r == 2: return sc.Select(int(rng.integers(0, 2)))(inp("y"))
        if r == 3: return sc.Dropna()(scalar_src(depth + 1))
        if r == 4: return sc.Filter()(scalar_src(depth + 1), inp("m"))
        if r == 5: return sc.Delay(int(rng.integers(0, 6)))(scalar_src(depth + 1))
        if r == 6: return sc.RollingMean(int(rng.integers(1, 4)))(scalar_src(depth + 1))
        if r == 7: return sc.Select(int(rng.integers(0, 2)))(sc.CombineLatest()(scalar_src(depth + 1), scalar_src(depth + 1)))
        return resample(depth + 1)

    def wide_src(depth):
        """A 2-column stream."""
        r = rng.integers(0, 4)
        if r == 0: return inp("y")
        if r == 1: return sc.Dropna()(inp("y"))
        if r == 2: return sc.Delay(int(rng.integers(0, 6)))(inp("y"))
        return sc.CombineLatest()(scalar_src(depth + 1), scalar_src(depth + 1))

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
        return sc.Resample(scalar_src(depth), clock_src(depth), clock=True, agg="last")

    def clock_src(depth):
        r = rng.integers(0, 8 if depth < 2 else 3)
        if r == 0: return inp("c")
        if r == 1: return sc.Dropna()(inp("c"))
        if r == 2: return sc.Resample(every=int(rng.integers(2, 12)), agg="last")(inp("c"))
        if r == 3: return sc.Filter()(inp("c"), inp("m"))
        if r == 4: return sc.Resample(count=int(rng.integers(1, 4)), agg="last")(inp("c"))
        if r == 5: return sc.Resample(inp("c"), sc.Dropna()(inp("b")), threshold=float(rng.uniform(0.5, 3)), agg="last")
        if r == 6: return sc.Select(0)(sc.CombineLatest()(inp("c"), inp("b")))
        return sc.Resample(inp("c"), clock_src(depth + 1), clock=True, agg="last")   # a nested clock

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


def skewed(evs, rng, skew):
    """The events in a live push order: each input keeps its own order, and
    with probability `skew` the next event comes from a random input instead of
    the lowest index (ties by input order, as batch merges them), so inputs run
    ahead of each other. Each event carries `bound`, the lowest next index over
    the inputs before it is pushed: an advance(now) at or below it keeps every
    later push at or above now."""
    queues = {}
    for e in evs:
        queues.setdefault(e[1], []).append(e)
    heads = {i: 0 for i in queues}
    out = []
    while True:
        live = [i for i in queues if heads[i] < len(queues[i])]
        if not live:
            return out
        bound = min(queues[i][heads[i]][0] for i in live)
        if rng.random() < skew:
            i = live[int(rng.integers(0, len(live)))]
        else:
            i = min(live, key=lambda i: queues[i][heads[i]][:3])
        out.append(queues[i][heads[i]] + (bound,))
        heads[i] += 1


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
    # closes windows that batch never opens (the meaning of advance).
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
        # Trial 0 pushes in index order; trials 1 and 2 let inputs run ahead
        # of each other (finding 1 of the Task 30 review).
        for q, (t, _, _, nm, v, bound) in enumerate(skewed(evs, rng, [0.0, 0.3, 0.7][trial])):
            hi = min(bound, horizon)
            if use_advance and last is not None and rng.random() < 0.15:
                s.advance(int(rng.integers(min(last, hi), hi + 1)))
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


# ---------------------------------------------------------------------------
# Clock resamples (G19, G20). A clock Resample's bar at tick t holds the value
# events with index in (previous tick, t], whatever releases them late: the
# value stream comes through a Filter, a threshold or count Resample, an
# every= Resample, a CombineLatest, Dropna, RollingMean or a Delay of either
# sign, and the clock through Dropna, a Filter, an every=, count= or
# threshold= Resample, a CombineLatest, a Delay or a nested clock resample.
# Each case checks batch against that rule computed from the two streams
# alone, and lazy and three live sessions (drains with probability 1, 0.3 and
# 0.05, inputs pushed ahead of each other in the last two, advance(now)
# between events and past the end) against batch.
# ---------------------------------------------------------------------------

def _clock_reference(vk, vv, ticks, agg, fill):
    out_k, out_v, last, prev = [], [], None, None
    for t in np.unique(ticks):
        sel = (vk <= t) if prev is None else ((vk > prev) & (vk <= t))
        prev = t
        xs = [float(x) for x in vv[sel] if not np.isnan(x)]
        if xs:
            if agg == "sum":
                val = 0.0
                for x in xs:
                    val += x
            else:
                val = xs[-1] if agg == "last" else xs[0]
            out_k.append(t); out_v.append(val); last = val
        elif fill == "nan":
            out_k.append(t); out_v.append(np.nan)
        elif fill == "carry" and last is not None:
            out_k.append(t); out_v.append(last)
    return np.array(out_v, float), np.array(out_k, np.int64)


def run_clock_case(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 80))

    def idx(k, start):
        return (start + np.cumsum(rng.integers(0, 4, k))).astype(np.int64)
    xa = rng.normal(size=n)
    xa[rng.random(n) < 0.2] = np.nan
    nc = int(rng.integers(1, 25))
    feeds = {"a": (xa, idx(n, int(rng.integers(0, 20)))),
             "m": ((rng.random(n) > 0.4).astype(float), idx(n, int(rng.integers(0, 20)))),
             "d": (rng.uniform(0, 2, n), idx(n, int(rng.integers(0, 20)))),
             "c": (np.ones(nc), idx(nc, int(rng.integers(0, 40))) * 2)}
    ins = {k: Input(k) for k in feeds}
    used = set()

    def inp(k):
        used.add(k)
        return ins[k]

    def value(depth=0):
        r = int(rng.integers(0, 9 if depth < 2 else 1))
        if r == 0: return inp("a")
        if r == 1: return sc.Filter()(value(depth + 1), inp("m"))
        if r == 2: return sc.Resample(value(depth + 1), sc.Dropna()(inp("d")),
                                      threshold=float(rng.uniform(0.5, 3)), agg="last")
        if r == 3: return sc.Delay(int(rng.integers(-4, 5)))(value(depth + 1))
        if r == 4: return sc.Select(0)(sc.CombineLatest()(value(depth + 1), inp("d")))
        if r == 5: return sc.Resample(count=int(rng.integers(1, 4)), agg="last")(value(depth + 1))
        if r == 6: return sc.Resample(every=int(rng.integers(1, 9)), agg="last",
                                      label=["left", "right"][int(rng.integers(0, 2))])(value(depth + 1))
        if r == 7: return sc.Dropna()(value(depth + 1))
        return sc.RollingMean(2)(value(depth + 1))

    def clock(depth=0):
        r = int(rng.integers(0, 9 if depth < 2 else 1))
        if r == 0: return inp("c")
        if r == 1: return sc.Dropna()(inp("c"))
        if r == 2: return sc.Resample(every=int(rng.integers(2, 9)), agg="last")(inp("c"))
        if r == 3: return sc.Delay(int(rng.integers(-3, 4)))(inp("c"))
        if r == 4: return sc.Filter()(clock(depth + 1), inp("m"))
        if r == 5: return sc.Resample(count=int(rng.integers(1, 4)), agg="last")(clock(depth + 1))
        if r == 6: return sc.Resample(clock(depth + 1), sc.Dropna()(inp("d")),
                                      threshold=float(rng.uniform(0.5, 3)), agg="last")
        if r == 7: return sc.Select(0)(sc.CombineLatest()(clock(depth + 1), inp("d")))
        return sc.Resample(inp("d"), clock(depth + 1), clock=True, agg="last")   # a nested clock

    agg = ["last", "sum", "first"][int(rng.integers(0, 3))]
    fill = ["skip", "nan", "carry"][int(rng.integers(0, 3))]
    vnode, cnode = value(), clock()
    out = sc.Resample(vnode, cnode, clock=True, agg=agg, fill=fill)
    outs = [out] + ([sc.RollingMean(1)(inp("a"))] if rng.random() < 0.5 else [])
    names = sorted(used)
    inputs, fl = [ins[k] for k in names], [feeds[k] for k in names]
    p = Pipeline(inputs, outs)
    want = p(*fl)
    bars = Pipeline(inputs, outs, align_outputs=False)(*fl)
    bars = bars[0] if len(outs) > 1 else bars
    (vv, vk), (_, ck) = Pipeline(inputs, [vnode, cnode], align_outputs=False)(*fl)
    rv, rk = _clock_reference(np.asarray(vk), np.asarray(vv).reshape(-1), np.asarray(ck), agg, fill)
    if not (np.array_equal(rk, bars[1]) and np.array_equal(rv, bars[0], equal_nan=True)):
        return ("reference", f"{list(bars[1])[:12]} vs {list(rk)[:12]}")
    m = len(outs)
    if m == 1:
        want = (want,)
    if all(np.ndim(v) == 1 for v, _ in want):
        d = same(lazy(p, fl, m), want)
        if d:
            return ("lazy", d)
    evs = events(fl, names)
    horizon = max(int(k[-1]) for _, k in fl)
    for prob, skew in ((1.0, 0.0), (0.3, 0.3), (0.05, 0.7)):
        s = p.live()
        parts, last = [], None
        for t, _, _, nm, v, bound in skewed(evs, rng, skew):
            if last is not None and rng.random() < 0.3:
                s.advance(int(rng.integers(min(last, bound), bound + 1)))
            s.push(nm, t, float(v))
            last = t
            if rng.random() < prob:
                r = s.result()
                parts.append(r if m > 1 else (r,))
        if rng.random() < 0.5:
            s.advance(horizon + int(rng.integers(0, 50)))
        s.flush()
        r = s.result()
        parts.append(r if m > 1 else (r,))
        d = same(concat(parts, m), want)
        if d:
            return ("live", f"drain probability {prob}: {d}")
    return None


@pytest.mark.parametrize("chunk", range(8))
def test_clock_resample_follows_its_rule_in_batch_lazy_and_live(chunk):
    bad = []
    for seed in range(chunk * 25, (chunk + 1) * 25):
        r = run_clock_case(seed)
        if r:
            bad.append((seed, r))
    assert not bad, bad[:3]
