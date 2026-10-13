"""The Pipeline graphs of the serialization tests (G21, B8): every node kind,
every Resample agg (the functor reducers included), the three fill policies,
and settings that are not finite. Shared by tests/test_dag_io.py and
devtools/wasm/gen_pipeline_io_oracle.py, which writes them, with their feeds
and results, for the JavaScript suite.

``cases()`` gives {name: (pipeline, feeds)}; ``feeds`` are the call's
(values, index) pairs in input order.
"""
import math

import numpy as np

T = 40
_RNG = np.random.default_rng(21)
K = np.cumsum(_RNG.integers(1, 4, size=T)).astype(np.int64)
P = np.cumsum(_RNG.standard_normal(T)) + 100.0
V = np.abs(_RNG.standard_normal(T)) + 1.0
P[[5, 17]] = np.nan
WIDTH = {"ohlcv": 2, "ohlcv2": 2, "ohlc_bars": 4, "ohlcv_bars": 5}


def values(width):
    if width == 1:
        return P
    cols = [P, P + 1.0, P - 1.0, P, V][:width] if width >= 4 else [P, V - 1.5]
    return np.column_stack(cols)


def cases():
    import screamer as s
    from screamer import Input, Pipeline
    from screamer.streams import CombineLatest, Delay, Dropna, Filter, Resample, Select, _ALL_AGG_NAMES

    def one(node_fn, width=1):
        x = Input("x")
        return Pipeline([x], [node_fn(x)]), [(values(width), K)]

    def two(node_fn, align=True):
        x, y = Input("x"), Input("y")
        return Pipeline([x, y], node_fn(x, y), align_outputs=align), [(P, K), (V, K + 1)]

    reducers = {"RollingMean(2)": lambda: s.RollingMean(2),
                "BollingerBands(3)": lambda: s.BollingerBands(3),
                "EwMean(alpha=0.5)": lambda: s.EwMean(alpha=0.5),
                "ExpandingMax()": lambda: s.ExpandingMax()}
    out = {
        "functor": one(lambda x: s.RollingMean(3)(x)),
        "multi-output functor": one(lambda x: s.RollingMinMax(4)(x)),
        "CombineLatest": two(lambda x, y: [CombineLatest()(x, y)]),
        "CombineLatest when_all": two(lambda x, y: [CombineLatest(emit="when_all")(x, y)]),
        "CombineLatest on_any max_pending": two(
            lambda x, y: [CombineLatest(emit="on_any", max_pending=50)(x, y)]),
        "Dropna": one(lambda x: Dropna(how="all")(x), 2),
        "Select": one(lambda x: Select([1, 0])(x), 4),
        "Filter": two(lambda x, y: [Filter()(x, y)]),
        "Delay": one(lambda x: Delay(3)(x)),
        "forecast_pairs count": two(lambda x, y: [s.forecast_pairs(x, y, count=2)]),
        "forecast_pairs duration": two(lambda x, y: [s.forecast_pairs(x, y, duration=4)]),
        "two outputs unaligned": two(lambda x, y: [s.RollingMean(2)(x), s.EwMean(alpha=0.3)(y)], False),
        "two outputs aligned": two(lambda x, y: [s.RollingMean(2)(x), s.EwMean(alpha=0.3)(y)]),
        "Resample threshold": two(lambda x, y: [Resample(x, y, threshold=3.0, agg="sum")]),
        "Resample clock": two(lambda x, y: [Resample(x, y, clock=True, agg="last", fill="carry")]),
        "Resample clock max_age": two(
            lambda x, y: [Resample(x, y, clock=True, agg="last", fill="carry", max_age=3)]),
        "Resample max_fill": one(lambda x: Resample(freq=2, agg="mean", fill="nan", max_fill=50)(x)),
        "chain": two(lambda x, y: [s.RollingMean(3)(Dropna()(s.Sub()(CombineLatest()(x, y))))]),
        # Settings that are not finite: written as -Infinity and NaN.
        "Clip -inf": one(lambda x: s.Clip(lower=-math.inf, upper=101.0)(x)),
        "FillNa nan": one(lambda x: s.FillNa(math.nan)(x)),
        "positional settings": one(lambda x: s.BollingerBands(5, 1.5)(x)),
    }
    for agg in _ALL_AGG_NAMES:
        w = WIDTH.get(agg, 1)
        out[f"Resample freq agg={agg}"] = one(lambda x, a=agg: Resample(freq=5, agg=a)(x), w)
        out[f"Resample count agg={agg}"] = one(lambda x, a=agg: Resample(count=3, agg=a)(x), w)
    for fill in ("skip", "nan", "carry"):
        out[f"Resample fill={fill}"] = one(
            lambda x, f=fill: Resample(freq=2, agg="mean", fill=f, label="right")(x))
    for name, make in reducers.items():
        out[f"Resample freq agg={name}"] = one(lambda x, m=make: Resample(freq=5, agg=m())(x))
        out[f"Resample count agg={name}"] = one(lambda x, m=make: Resample(count=3, agg=m())(x))
        out[f"Resample freq agg={name} fill"] = one(
            lambda x, m=make: Resample(freq=5, agg=m(), fill="carry")(x))
    return out
