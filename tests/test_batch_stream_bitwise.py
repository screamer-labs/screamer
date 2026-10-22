"""Batch output equals per-event output bit for bit, for every operator.

Tracker row D5. Each operator is built the way ``devtools/bench_all_ops.py``
builds it (default arguments, or the one required length) and run twice on
the same data: once as a batch call on whole arrays, and once one event at a
time on a fresh instance. The two results must be equal bit for bit, with NaN
in the same places. Clang may fuse ``a * b + c`` into one instruction
(``-ffp-contract=on``), and the batch loop and the per-event call must not
differ in how they evaluate any step, so this is checked on every build.

The inputs are random walks at level 0 and 1e3, three seeds, with NaN injected
into every input. A 1-D run compares the batch call with the per-event loop.
A 2-D run of shape (T, K) goes through the column driver, and each of its
columns must equal the per-event loop on that column. Row operators (the
cross-sectional and portfolio family) take a row of K values per event, so
they run in 2-D only, one row per event. ``PortfolioReport`` takes one
(assets, 4) block per event.

A second test repeats the sweep for every choice of the string options
(start policy, output, sizing, fill rule and so on) and for the backtest
engines with nonzero costs, stops and bands, whose code paths the defaults
leave out.
"""
import inspect

import numpy as np
import pytest

import screamer as s
from devtools import get_constructor_arguments
from devtools.bench_all_ops import construct
from tests.param_cases import ROW_OPS
from tests.regime_helpers import assert_bitwise

T = 300
K = 10
SEEDS = (0, 1, 2)
LEVELS = (0.0, 1e3)
MAX_INPUTS = 10

# Not numeric operators: graph nodes and stream combinators that need an
# index, a graph or a second stream, and Filter, whose output length depends
# on the data.
EXCLUDED = {
    "Node", "Pipeline", "Delay", "Merge", "CombineLatest", "Dropna", "Select",
    "Resample", "Filter",
}

# Operators without defaults for every argument.
REQUIRED_KWARGS = {
    "Hold": {"n": 3},
    "CusumFilter": {"threshold": 1.0},
    "LiquidationDistance": {"leverage": 10.0},
    "RollingMultiRegression": {"window_size": 20, "n_regressors": 3},
    "ExpandingMultiRegression": {"n_regressors": 3},
    "EwMultiRegression": {"n_regressors": 3, "alpha": 0.1},
    "SchmittTrigger": {"lower": -1.0, "upper": 1.0},
    # Barriers on, so the labels are not all 0.
    "TripleBarrier": {"horizon": 5, "upper": 0.01, "lower": 0.01},
    "TripleBarrierOHLC": {"horizon": 5, "upper": 0.01, "lower": 0.01},
    # The label is a random walk here, so most segments are one row long;
    # tests/test_segment.py sweeps long segments.
    "Segment": {"op": s.RollingStd(5)},
}

# Takes one (assets, 4) block per event; batch input is (T, assets, 4).
BLOCK_OPS = {"PortfolioReport"}

# String options and their values, for the variant sweep.
CHOICES = {
    "start_policy": ("expanding", "zero"),
    "fill": ("touch", "breach"),
    "slippage_unit": ("ticks", "price"),
    "size": ("units", "notional", "weight"),
    "fill_at": ("close", "typical", "ohlc4"),
    "rebalance": ("all",),
    "scale": ("unit",),
    "method": ("cutler",),
    "barrier_unit": ("price", "vol"),
    "emission": ("ar1",),
}
OUTPUT_CHOICES = {
    "Hampel": ("flag", "nan"),
    "ImpulseClip": ("flag", "nan"),
    "RollingOU": ("mean", "relmean", "std"),
    "RollingSigmaClip": ("mean", "std", "nan"),
    "RollingVarianceRatio": ("zscore",),
}
# Backtest costs, stops and bands (all default to 0), set together.
COSTS = {
    "fee": 1e-3, "maker_fee": 2e-4, "taker_fee": 5e-4, "fee_per_contract": 0.01,
    "maker_fee_per_contract": 0.01, "taker_fee_per_contract": 0.02,
    "slippage": 1e-3, "spread": 0.05, "tick_size": 0.01, "stop_loss": 0.02,
    "take_profit": 0.03, "trailing_stop": 0.02, "rebalance_band": 0.01,
}


def default_kwargs(name):
    """The keyword arguments ``construct`` uses for this operator."""
    if name in REQUIRED_KWARGS:
        return REQUIRED_KWARGS[name]
    from devtools.bench_all_ops import CONSTRUCT_KWARGS
    for kwargs in CONSTRUCT_KWARGS:
        try:
            getattr(s, name)(**kwargs)
            return kwargs
        except Exception:
            pass
    return None


def operator_names():
    names = []
    for name in sorted(n for n in dir(s) if not n.startswith("_")):
        cls = getattr(s, name)
        if name in EXCLUDED or not inspect.isclass(cls) or not callable(cls):
            continue
        if name in REQUIRED_KWARGS or construct(cls) is not None:
            names.append(name)
    return names


def variants():
    """(name, kwargs) for every non-default string option, and the costs."""
    out = []
    for name in operator_names():
        base = default_kwargs(name)
        args = dict(get_constructor_arguments(getattr(s, name)) or [])
        for param, values in CHOICES.items():
            if args.get(param) == "str":
                out += [(name, {**base, param: v}) for v in values]
        if args.get("output") == "str":
            out += [(name, {**base, "output": v}) for v in OUTPUT_CHOICES[name]]
        costs = {k: v for k, v in COSTS.items() if args.get(k) == "float"}
        if costs:
            out.append((name, {**base, **costs}))
    return [(n, kw) for n, kw in out if constructs(n, kw)]


def constructs(name, kwargs):
    """Whether the operator accepts these arguments (RollingHurst has one
    ``method``, RollingRSI two)."""
    try:
        getattr(s, name)(**kwargs)
        return True
    except Exception:
        return False


def series(seed, level, layout=0, t=T):
    """``MAX_INPUTS`` related random walks, each with a few NaN, in one of
    ``LAYOUTS``. The walks are an open ``o``, a close ``c``, a high above both,
    a low below both and a positive volume ``v``, then more walks."""
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.standard_normal(t)) + level
    xs = [base + rng.random(t) for _ in range(MAX_INPUTS)]
    o, c = xs[0], xs[3]
    xs[1] = np.maximum(np.maximum(o, c), xs[1]) + 1.0
    xs[2] = np.minimum(np.minimum(o, c), xs[2]) - 1.0
    xs[4] = np.abs(xs[4]) + 1.0
    # NaN in the second half only: operators that propagate a NaN through
    # their window still give a clean first half.
    for x in xs:
        x[t // 2 + rng.choice(t - t // 2, 3, replace=False)] = np.nan
    return [xs[i] for i in LAYOUTS[layout]]


# Input orders: (o, h, l, c, v, ...) as in devtools/bench_all_ops.py,
# (h, l, c, v, ...) for the HLC and HLCV operators, and (c, o, v, ...) for
# (target, price, size).
_REST = tuple(range(5, MAX_INPUTS))
LAYOUTS = ((0, 1, 2, 3, 4) + _REST, (1, 2, 3, 4, 0) + _REST, (3, 0, 4, 1, 2) + _REST)


def is_array_output(out):
    if isinstance(out, np.ndarray):
        return True
    return isinstance(out, (tuple, list)) and bool(out) and all(
        isinstance(o, np.ndarray) for o in out)


def informative(out):
    out = np.asarray(out, np.float64)
    return int(np.sum(np.isfinite(out) & (out != 0)))


def inputs_for(make):
    """The smallest number of inputs the operator accepts, and the layout
    that gives the most finite nonzero outputs, or None."""
    data = series(0, 1e3)
    for n in range(1, MAX_INPUTS + 1):
        try:
            out = make()(*data[:n])
        except Exception:
            continue
        if not is_array_output(out):
            return None
        scores = []
        for k in range(len(LAYOUTS)):
            try:
                scores.append(informative(make()(*series(0, 1e3, k)[:n])))
            except Exception:
                scores.append(-1)
        return n, scores.index(max(scores))
    return None


def event_values(out):
    return np.asarray(out, dtype=np.float64).ravel()


def max_ulp(a, b):
    """Largest distance in units in the last place over values finite in both."""
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return 0
    ia, ib = a[m].view(np.int64), b[m].view(np.int64)
    ia = np.where(ia < 0, np.int64(-2**63) - ia, ia)
    ib = np.where(ib < 0, np.int64(-2**63) - ib, ib)
    return int(max(abs(int(x) - int(y)) for x, y in zip(ia, ib)))


def check(batch, stream, where):
    batch, stream = np.asarray(batch, np.float64), np.asarray(stream, np.float64)
    try:
        assert_bitwise(batch, stream)
    except AssertionError:
        if batch.shape != stream.shape:
            pytest.fail(f"{where}: shape {batch.shape} != {stream.shape}")
        nan_diff = int(np.sum(np.isnan(batch) != np.isnan(stream)))
        pytest.fail(f"{where}: batch != per-event, max ulp {max_ulp(batch, stream)}, "
                    f"NaN position differences {nan_diff}")


def per_event(make, columns):
    """Fresh instance, one event per call; returns (T, M)."""
    op = make()
    return np.stack([event_values(op(*[float(c[t]) for c in columns]))
                     for t in range(len(columns[0]))])


def run_series_op(make):
    found = inputs_for(make)
    assert found is not None, "no array call with up to %d inputs" % MAX_INPUTS
    n, layout = found
    for seed in SEEDS:
        for level in LEVELS:
            xs = series(seed, level, layout)[:n]
            batch = np.asarray(make()(*xs), np.float64).reshape(T, -1)
            check(batch, per_event(make, xs), f"1-D seed={seed} level={level}")

            cols = [series(seed * K + k + 7, level, layout)[:n] for k in range(K)]
            X = [np.stack([c[i] for c in cols], axis=1) for i in range(n)]
            batch2 = np.asarray(make()(*X), np.float64).reshape(T, K, -1)
            for k in range(K):
                check(batch2[:, k], per_event(make, cols[k]),
                      f"2-D column {k} seed={seed} level={level}")


def run_row_op(make):
    probe = [np.stack(series(k + 50, 0.0), axis=1) for k in range(MAX_INPUTS)]
    n = None
    for m in range(1, MAX_INPUTS + 1):
        try:
            make()(*probe[:m])
        except Exception:
            continue
        n = m
        break
    assert n is not None
    for seed in SEEDS:
        for level in LEVELS:
            X = [np.stack(series(seed * 31 + i, level), axis=1) for i in range(n)]
            batch = np.asarray(make()(*X), np.float64).reshape(T, -1)
            op = make()
            stream = np.stack([event_values(op(*[x[t] for x in X])) for t in range(T)])
            check(batch, stream, f"rows seed={seed} level={level}")


def run_block_op(make):
    for seed in SEEDS:
        for level in LEVELS:
            # Per asset: equity, pnl, position, cost.
            X = np.stack([np.stack(series(seed * 7 + a, level)[:4], axis=1)
                          for a in range(3)], axis=1)
            batch = np.asarray(make()(X), np.float64).reshape(T, -1)
            op = make()
            stream = np.stack([event_values(op(X[t])) for t in range(T)])
            check(batch, stream, f"blocks seed={seed} level={level}")


def run(name, kwargs):
    def make():
        return getattr(s, name)(**kwargs)
    if name in BLOCK_OPS:
        run_block_op(make)
    elif name in ROW_OPS:
        run_row_op(make)
    else:
        run_series_op(make)


@pytest.mark.parametrize("name", operator_names())
def test_batch_equals_per_event_bitwise(name):
    run(name, default_kwargs(name))


@pytest.mark.parametrize("name, kwargs", variants(),
                         ids=lambda v: v if isinstance(v, str) else
                         ",".join(f"{k}={x}" for k, x in v.items()))
def test_batch_equals_per_event_bitwise_options(name, kwargs):
    run(name, kwargs)


def test_sweep_covers_the_registry():
    """Every public class is swept except the graph and stream nodes."""
    public = {n for n in dir(s) if not n.startswith("_") and inspect.isclass(getattr(s, n))}
    assert public - set(operator_names()) == EXCLUDED


def test_variants_cover_every_string_option():
    """Every string option of every operator is in CHOICES or OUTPUT_CHOICES."""
    for name in operator_names():
        for param, typ in get_constructor_arguments(getattr(s, name)) or []:
            if typ == "str":
                assert param in CHOICES or name in OUTPUT_CHOICES, (name, param)
