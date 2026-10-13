"""Compare the results of one matrix of input forms across two screamer builds.

The probe calls a fixed set of operators with every input form screamer
accepts (scalars, arrays of each rank, lists, tuples, iterators, pandas,
streams, microstructure keywords) and records one comparable line per call:
the container, shape and sum of a value, or the type and message of an error.

Record this build's results::

    .venv/bin/python devtools/compat_probe.py --json new.json

Print the rows whose result changed, tagged ``ERR->VALUE``, ``VALUE->ERR``,
``VALUE CHANGED``, ``ERR CHANGED`` or ``LAZY`` (the same values, now from a
lazy iterator); ``--all`` also prints ``same`` rows. Array digests round to
9 decimals, so differences below 1e-9 (last-bit changes) do not show.
The old side defaults to the committed 2.8.0 baseline
``devtools/compat_baseline_2.8.0.json``, and the new side to a run of this
build::

    .venv/bin/python devtools/compat_probe.py --diff                  # baseline vs this build
    .venv/bin/python devtools/compat_probe.py --diff new.json         # baseline vs new.json
    .venv/bin/python devtools/compat_probe.py --diff old.json new.json

Record a released wheel's results in a separate venv. Run the probe with
``-I`` from outside the repository so Python imports the installed wheel and
not the source tree. Regenerate the baseline this way whenever the matrix
changes::

    python -m venv /tmp/old && /tmp/old/bin/pip install screamer==2.8.0 pandas
    cd /tmp && /tmp/old/bin/python -I /path/to/devtools/compat_probe.py \
        --json /path/to/devtools/compat_baseline_2.8.0.json
"""
import argparse
import json
import os
import sys

BASELINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compat_baseline_2.8.0.json")


def cases():
    import numpy as np
    import pandas as pd
    import screamer as s

    rng = np.random.default_rng(0)
    T, K = 30, 3
    v = rng.normal(size=T).cumsum() + 100
    w = rng.normal(size=T).cumsum() + 100
    X = rng.normal(size=(T, K)).cumsum(axis=0) + 100
    idx = np.arange(T) * 10
    long = rng.normal(size=200).cumsum() + 100    # past the cycle operators' warm-up
    sign = np.sign(v - 100)
    ret = np.diff(v, prepend=v[0])
    gen = lambda a: (x for x in a)
    two_level = lambda: pd.DataFrame(
        np.arange(8.0).reshape(2, 4),
        columns=pd.MultiIndex.from_product([["a", "b"], ["o", "c"]]))

    table = {
        # one-input operators, every container
        "1in scalar": lambda: s.RollingMean(3)(1.5),
        "1in 1-D": lambda: s.RollingMean(3)(v),
        "1in (T,1)": lambda: s.RollingMean(3)(X[:, :1]),
        "1in (T,K)": lambda: s.RollingMean(3)(X),
        "1in (T,K,1)": lambda: s.RollingMean(3)(X[:, :, None]),
        "1in list of floats": lambda: s.RollingMean(3)(list(v[:5])),
        "1in tuple of floats": lambda: s.RollingMean(3)(tuple(v[:5])),
        "1in generator floats": lambda: list(s.RollingMean(3)(gen(v[:5]))),
        "1in (values,index) tuple": lambda: s.RollingMean(3)((v, idx)),
        "1in tuple of rows": lambda: s.RollingMean(3)((X[0], X[1])),
        "1in list of rows": lambda: s.RollingMean(3)([X[0], X[1]]),
        "1in generator of rows": lambda: list(s.RollingMean(3)(gen(X[:3]))),
        "1in list of length-1 arrays": lambda: s.RollingMean(2)([v[i:i + 1] for i in range(4)]),
        "1in list of Series": lambda: s.RollingMean(2)([pd.Series(X[i]) for i in range(4)]),
        "1in pandas Series": lambda: s.RollingMean(3)(pd.Series(v)),
        "1in pandas DataFrame": lambda: s.RollingMean(3)(pd.DataFrame(X)),
        "1in FunctorBase gen (IsNan)": lambda: s.IsNan()(gen(v[:3])),
        "1in FunctorBase (T,K,1)": lambda: s.IsNan()(X[:, :, None]),
        "1in FunctorBase tuple of floats": lambda: s.IsNan()(tuple(v[:3])),
        "1in FunctorBase (values,index)": lambda: s.IsNan()((v, idx)),
        "1-in M-out (T,K,1)": lambda: s.BollingerBands(3)(X[:, :, None]),
        "1-in M-out tuple of floats": lambda: s.BollingerBands(3)(tuple(v[:4])),
        "1-in M-out (values,index)": lambda: s.BollingerBands(3)((v, idx)),
        # row operators and PortfolioReport keep their 2.8.0 tuple behavior (D9)
        "row (values,index) tuple": lambda: s.CrossRank()((v, idx)),
        "row list of rows": lambda: s.CrossRank()([X[:, 0], X[:, 1]]),
        "row tuple of floats": lambda: s.CrossRank()(tuple(X[0])),
        "PortfolioReport tuple of arrays": lambda: s.PortfolioReport()((v, idx)),
        "row tuple of rows": lambda: s.CrossRank()((X[0], X[1])),
        "PortfolioReport tuple of (4,) rows": lambda: s.PortfolioReport()((X[0, :1].repeat(4), X[1, :1].repeat(4))),
        "PortfolioReport tuple of events": lambda: s.PortfolioReport()((np.arange(8.0).reshape(2, 4), np.arange(8.0, 16.0).reshape(2, 4))),
        "PortfolioReport list of events": lambda: s.PortfolioReport()([np.arange(8.0).reshape(2, 4), np.arange(8.0, 16.0).reshape(2, 4)]),
        # multi-input operators
        "2in scalars": lambda: s.Add()(1.0, 2.0),
        "2in tuple of two scalars": lambda: s.Add()((1.0, 2.0)),
        "2in two 1-D": lambda: s.Add()(v, w),
        "2in packed (T,2)": lambda: s.Add()(np.column_stack([v, w])),
        "2in tuple of two arrays": lambda: s.Add()((v, w)),
        "2in list of two arrays": lambda: s.Add()([v, w]),
        "2in list of pairs": lambda: s.Add()([(1.0, 2.0), (3.0, 4.0)]),
        "2in two lists": lambda: s.Add()([1.0, 2.0], [3.0, 4.0]),
        "2in two tuples": lambda: s.Add()((1.0, 2.0), (3.0, 4.0)),
        "2in two generators": lambda: list(s.Add()(gen([1.0, 2.0]), gen([3.0, 4.0]))),
        "2in generator of pairs": lambda: list(s.Add()(gen([(1.0, 2.0), (3.0, 4.0)]))),
        "2in (T,K)+(T,)": lambda: s.Add()(X, v),
        "2in (T,K)+(T,K)": lambda: s.Add()(X, X),
        "2in (T,K,1)+(T,K,1)": lambda: s.Add()(X[:, :, None], X[:, :, None]),
        # a list mixed with arrays, in either order (A8)
        "2in list+array": lambda: s.Add()(list(v), w),
        "2in array+list": lambda: s.Add()(v, list(w)),
        "2in (T,K)+list": lambda: s.Add()(X, list(v)),
        "2in M-out list+array": lambda: s.RollingLinearRegression(3)(list(v[:6]), w[:6]),
        "3in array+list+array": lambda: s.Where()(v, list(w), v),
        # streams and pandas
        "from_pandas flat": lambda: s.from_pandas(pd.DataFrame(X)),
        "from_pandas 2-level": lambda: s.from_pandas(two_level()),
        "from_pandas 2-level TAF": lambda: s.from_pandas(two_level(), layout="TAF"),
        "from_pandas ragged 2-level": lambda: s.from_pandas(pd.DataFrame(
            np.arange(6.0).reshape(2, 3),
            columns=pd.MultiIndex.from_tuples([("a", "o"), ("a", "c"), ("b", "o")]))),
        "from_pandas int": lambda: s.from_pandas(pd.DataFrame({"a": [1, 2]})),
        "from_pandas string": lambda: s.from_pandas(pd.DataFrame({"a": ["x", "y"]})),
        "from_pandas datetime": lambda: s.from_pandas(pd.DataFrame(
            {"a": pd.to_datetime(["2024-01-01", "2024-01-02"])})),
        "from_pandas Series": lambda: s.from_pandas(pd.Series(v, index=idx)),
        "Resample 1-D ohlc": lambda: s.Resample(count=5, agg="ohlc")(v),
        "Resample (values,index)": lambda: s.Resample(freq=50, agg="last")((v, idx)),
        "CombineLatest 1-D": lambda: s.CombineLatest()(v, w),
        # microstructure keywords (2.8.0 names) and positional forms
        "SignedVolume kw": lambda: s.SignedVolume()(sign=sign, volume=w),
        "SignedVolume positional": lambda: s.SignedVolume()(sign, w),
        "SignedVolume mixed": lambda: s.SignedVolume()(sign, volume=w),
        "RollingKyleLambda kw": lambda: s.RollingKyleLambda(5)(signed_flow=sign * w, return_=ret),
        "RollingKyleLambda positional": lambda: s.RollingKyleLambda(5)(sign * w, ret),
        "EwKyleLambda kw": lambda: s.EwKyleLambda(span=5.0)(signed_flow=sign * w, return_=ret),
        "EwKyleLambda positional": lambda: s.EwKyleLambda(span=5.0)(sign * w, ret),
        "QueueImbalance kw": lambda: s.QueueImbalance()(bid_size=v, ask_size=w),
        "QueueImbalance positional": lambda: s.QueueImbalance()(v, w),
        "RollingOrderImbalance kw": lambda: s.RollingOrderImbalance(5)(signed_flow=sign * w),
        "RollingOrderImbalance positional": lambda: s.RollingOrderImbalance(5)(sign * w),
        "SignedVolume missing input": lambda: s.SignedVolume()(sign),
        "QueueImbalance missing input": lambda: s.QueueImbalance()(v),
        "RollingKyleLambda missing input": lambda: s.RollingKyleLambda(5)(ret),
    }
    # The nine one-input, one-output operators that return a lazy iterator for
    # a generator (decision D4).
    for name in ["IsNan", "Not", "IsFinite", "TrendMode", "CyclePhase", "CycleAmplitude",
                 "CycleFrequency", "DominantCycle", "InstantaneousTrendline"]:
        table[f"lazy {name} generator"] = (lambda n: lambda: getattr(s, n)()(gen(long)))(name)
    # Microstructure keywords with scalars, lists and generators.
    micro = {
        "SignedVolume": (lambda: s.SignedVolume(), ("sign", "volume"), sign, w),
        "RollingKyleLambda": (lambda: s.RollingKyleLambda(5), ("signed_flow", "return_"), sign * w, ret),
        "EwKyleLambda": (lambda: s.EwKyleLambda(span=5.0), ("signed_flow", "return_"), sign * w, ret),
        "QueueImbalance": (lambda: s.QueueImbalance(), ("bid_size", "ask_size"), v, w),
    }
    for name, (make, (ka, kb), a, b) in micro.items():
        for form, conv in [("scalars", lambda x: float(x[3])), ("lists", list), ("generators", gen)]:
            def call(make=make, ka=ka, kb=kb, a=a, b=b, conv=conv):
                out = make()(**{ka: conv(a), kb: conv(b)})
                return out if isinstance(out, (float, list)) else list(out)
            table[f"{name} kw {form}"] = call
    # Stream state (decision D8): every call but op(number) is an independent
    # run that starts clean and resets the running series when it opens.
    def used(make, *first):
        op = make()
        op(*first)
        op(*first)
        return op

    def interleaved(make, a, b):
        op = make()
        i1, i2 = op(gen(a)), op(gen(b))
        return [(next(i1), next(i2)) for _ in range(len(a))]

    def live_after_open(make, x, src):
        op = used(make, x)
        it = op(src)
        return [op(x), list(it) if hasattr(it, "__next__") else it]

    async def adrain(it):
        return [e async for e in it]

    async def agen(a):
        for e in a:
            yield e

    import asyncio
    rm = lambda: s.RollingMean(3)                               # noqa: E731
    wts = np.sign(X - 100) / K                                 # target weights
    table.update({
        "D8 stream on a used operator": lambda: list(used(rm, 1.0)(gen(v[:6]))),
        "D8 asset stream on a used operator": lambda: list(used(rm, 1.0)(gen(X[:6]))),
        "D8 multi-input stream on a used operator":
            lambda: list(used(lambda: s.RollingCorr(3), 1.0, 2.0)(gen(v[:6]), gen(w[:6]))),
        "D8 async stream on a used operator":
            lambda: asyncio.run(adrain(used(rm, 1.0)(agen(v[:6])))),
        "D8 row stream on a used operator":
            lambda: list(used(lambda: s.BacktestPortfolioTarget(), wts[0], X[0])(
                gen(wts[:4]), gen(X[:4]))),
        "D8 interleaved streams": lambda: interleaved(rm, v[:6], w[:6]),
        "D8 interleaved asset streams": lambda: interleaved(rm, X[:6], X[6:12]),
        "D8 live call after opening a stream": lambda: live_after_open(rm, 5.0, gen(v[:6])),
        "D8 live call after a list (1-in)": lambda: live_after_open(rm, 5.0, list(v[:6])),
        "D8 list on a used operator (1-in M-out)":
            lambda: used(lambda: s.BollingerBands(3), 1.0)(list(v[:6])),
        "D8 list on a used operator (2-in)":
            lambda: used(lambda: s.RollingCorr(3), 1.0, 2.0)(list(v[:6]), list(w[:6])),
        "D8 list of pairs on a used operator (2-in)":
            lambda: used(lambda: s.RollingCorr(3), 1.0, 2.0)(list(zip(v[:6], w[:6]))),
    })
    for form, conv in [("scalar", lambda x: float(x[3])), ("list", list), ("generator", gen)]:
        table[f"RollingOrderImbalance kw {form}"] = (
            lambda conv=conv: (lambda out: out if isinstance(out, (float, list)) else list(out))(
                s.RollingOrderImbalance(5)(signed_flow=conv(sign * w))))
    # Microstructure operators in every input form: separate, packed, events,
    # iterators, pandas, streaming, and the keyword errors.
    A, B = np.sign(X - 100) * X, X[::-1] - 100
    for name, (make, (ka, kb), a, b) in micro.items():
        ev = lambda a=a, b=b: list(zip(a[:6], b[:6]))           # noqa: E731
        table.update({
            f"{name} scalars positional": lambda make=make, a=a, b=b: make()(float(a[3]), float(b[3])),
            f"{name} stream of scalars":
                lambda make=make, a=a, b=b: (lambda op: [op(float(x), float(y))
                                                         for x, y in zip(a[:8], b[:8])])(make()),
            f"{name} packed (T,2)": lambda make=make, a=a, b=b: make()(np.stack([a, b], axis=-1)),
            f"{name} packed (T,K,2)": lambda make=make: make()(np.stack([A, B], axis=-1)),
            f"{name} (T,K) separate": lambda make=make: make()(A, B),
            f"{name} (T,K)+(T,)": lambda make=make, b=b: make()(A, b),
            f"{name} kw (T,K)": lambda make=make, ka=ka, kb=kb: make()(**{ka: A, kb: B}),
            f"{name} list of events": lambda make=make, ev=ev: make()(ev()),
            f"{name} tuple of events": lambda make=make, ev=ev: make()(tuple(ev())),
            f"{name} generator of events": lambda make=make, ev=ev: list(make()(gen(ev()))),
            f"{name} async generator of events":
                lambda make=make, ev=ev: asyncio.run(adrain(make()(agen(ev())))),
            f"{name} two generators": lambda make=make, a=a, b=b: list(make()(gen(a[:6]), gen(b[:6]))),
            f"{name} two lists": lambda make=make, a=a, b=b: make()(list(a[:6]), list(b[:6])),
            f"{name} pandas Series": lambda make=make, a=a, b=b: make()(pd.Series(a), pd.Series(b)),
            f"{name} kw reversed order": lambda make=make, ka=ka, kb=kb, a=a, b=b: make()(**{kb: b, ka: a}),
            f"{name} kw unexpected": lambda make=make, a=a, b=b: make()(a, b, extra=a),
            f"{name} kw duplicate": lambda make=make, ka=ka, a=a, b=b: make()(a, **{ka: b}),
            f"{name} kw second only": lambda make=make, kb=kb, b=b: make()(**{kb: b}),
            f"{name} too many inputs": lambda make=make, a=a, b=b: make()(a, b, a),
            f"{name} no inputs": lambda make=make: make()(),
            f"{name} one scalar": lambda make=make, a=a: make()(float(a[3])),
        })
    roi = lambda: s.RollingOrderImbalance(5)                    # noqa: E731
    table.update({
        "RollingOrderImbalance (T,K)": lambda: roi()(A),
        "RollingOrderImbalance kw (T,K)": lambda: roi()(signed_flow=A),
        "RollingOrderImbalance two generators": lambda: list(roi()(gen(v[:6]), gen(w[:6]))),
        "RollingOrderImbalance kw unexpected": lambda: roi()(v, extra=v),
        "RollingOrderImbalance kw duplicate": lambda: roi()(v, signed_flow=v),
        "RollingOrderImbalance too many inputs": lambda: roi()(v, w),
        "RollingOrderImbalance no inputs": lambda: roi()(),
        "RollingOrderImbalance pandas Series": lambda: roi()(pd.Series(v)),
        "RollingOrderImbalance stream of scalars":
            lambda: (lambda op: [op(float(x)) for x in v[:8]])(roi()),
    })

    # backtest_report: every running column stacked, and the summary with a
    # checksum of the key order of both dicts appended
    def report(values, **kw):
        import zlib
        running, summary = s.backtest_report(values, **kw)
        keys = ",".join(list(running) + ["|"] + list(summary))
        return (np.column_stack([np.asarray(a, dtype=float) for a in running.values()]),
                np.append(np.array(list(summary.values())), float(zlib.crc32(keys.encode()))))
    engine = s.BacktestPriceTarget(spread=0.01)(np.sign(v - v.mean()), v)
    holes = engine.copy()
    holes[3, 1] = np.nan
    holes[10, 3] = np.inf
    holes[-2:, 0] = np.nan
    table.update({
        "backtest_report engine output": lambda: report(engine),
        "backtest_report holes, periods_per_year=252": lambda: report(holes, periods_per_year=252),
        "backtest_report index": lambda: report(engine, index=idx),
        "backtest_report no rows": lambda: report(engine[:0]),
        "backtest_report one row": lambda: report(engine[:1]),
        "backtest_report list with None": lambda: report([[1.0, None, 0, 0], [2, 1, 1, 0.1]]),
        "backtest_report Fortran order": lambda: report(np.asfortranarray(holes)),
        "backtest_report (T,3)": lambda: report(engine[:, :3]),
        "backtest_report periods_per_year=0": lambda: report(engine, periods_per_year=0),
    })

    # forecast_pairs: eager, lazy and graph, count and duration modes
    from screamer.supervised import forecast_pairs as fp
    yn = w.copy()
    yn[[4, 11]] = np.nan
    yidx = np.arange(0, T * 10, 15)[:T // 2] + 5             # y on its own clock
    vgen = lambda a, ix: ((float(x), int(i)) for x, i in zip(a, ix))   # noqa: E731

    def fp_graph(feeds, **kw):
        X_in, y_in = s.Input("X"), s.Input("y")
        return s.Pipeline([X_in, y_in], [fp(X_in, y_in, **kw)])(*feeds)

    table.update({
        "forecast_pairs count 1-D": lambda: fp(v, w, count=3),
        "forecast_pairs count dropna NaN target": lambda: fp(v, yn, count=2, dropna=True),
        "forecast_pairs count (T,K)": lambda: fp(X, w, count=2),
        "forecast_pairs count (T,K) dropna": lambda: fp(X, yn, count=2, dropna=True),
        "forecast_pairs count (T,1)": lambda: fp(X[:, :1], w, count=1),
        "forecast_pairs count lists with None": lambda: fp([1.0, None, 3.0, 4.0], [1.0, 2.0, None, 4.0], count=1),
        "forecast_pairs count float 1.5": lambda: fp(v, w, count=1.5),
        "forecast_pairs count True": lambda: fp(v, w, count=True),
        "forecast_pairs count str": lambda: fp(v, w, count="1"),
        "forecast_pairs count empty": lambda: fp(v[:0], w[:0], count=1),
        "forecast_pairs count 0": lambda: fp(v, w, count=0),
        "forecast_pairs count length mismatch": lambda: fp(v, w[:5], count=1),
        "forecast_pairs neither count nor duration": lambda: fp(v, w),
        "forecast_pairs count and duration": lambda: fp(v, w, count=1, duration=1),
        "forecast_pairs count lazy": lambda: fp(vgen(v, range(T)), vgen(w, range(T)), count=2),
        "forecast_pairs count lazy dropna": lambda: fp(vgen(v, range(T)), vgen(yn, range(T)), count=2, dropna=True),
        "forecast_pairs count lazy and array": lambda: fp(vgen(v, range(T)), w, count=2),
        "forecast_pairs count graph": lambda: fp_graph((v, w), count=2),
        "forecast_pairs count graph dropna": lambda: fp_graph((v, yn), count=2, dropna=True),
        "forecast_pairs duration": lambda: fp((v, idx), (w, idx), duration=20),
        "forecast_pairs duration dropna": lambda: fp((v, idx), (yn, idx), duration=20, dropna=True),
        "forecast_pairs duration y clock": lambda: fp((v, idx), (w[:len(yidx)], yidx), duration=10),
        "forecast_pairs duration 0": lambda: fp((v, idx), (w, idx), duration=0),
        "forecast_pairs duration (T,K)": lambda: fp((X, idx), (w, idx), duration=20),
        "forecast_pairs duration (T,1)": lambda: fp((X[:, :1], idx), (w, idx), duration=20),
        "forecast_pairs duration float 15.5": lambda: fp((v, idx), (w, idx), duration=15.5),
        "forecast_pairs duration datetime64": lambda: fp(
            (v, idx.astype("datetime64[s]")), (w, idx.astype("datetime64[s]")), duration=20),
        "forecast_pairs duration index None": lambda: fp((v, None), (w, None), duration=1),
        "forecast_pairs duration bare arrays": lambda: fp(v, w, duration=10),
        "forecast_pairs duration empty": lambda: fp(
            (v[:0], idx[:0]), (w[:0], idx[:0]), duration=10),
        "forecast_pairs count rank-3 X": lambda: fp(X.reshape(T, 1, K), w, count=1),
        "forecast_pairs duration rank-3 X": lambda: fp((X.reshape(T, 1, K), idx), (w, idx), duration=10),
        "forecast_pairs count (T,0) X": lambda: fp(X[:, :0], w, count=1),
        "forecast_pairs count (values,index) tuples": lambda: fp((v, idx), (w, idx), count=1),
        "forecast_pairs count 2-D y": lambda: fp(v, X[:, :2], count=1),
        "forecast_pairs duration 2-D y": lambda: fp((v, idx), (X[:, :2], idx), duration=10),
        "forecast_pairs duration negative": lambda: fp((v, idx), (w, idx), duration=-10),
        "forecast_pairs count 2**40": lambda: fp(v, w, count=2**40),
        "forecast_pairs count 2**70": lambda: fp(v, w, count=2**70),
        "forecast_pairs duration 2**70": lambda: fp((v, idx), (w, idx), duration=2**70),
        "forecast_pairs dropna str": lambda: fp(v, w, count=1, dropna="no"),
        "forecast_pairs count node (T,0) X": lambda: fp_graph((X[:, :0], w), count=1),
        "forecast_pairs duration node (T,0) X": lambda: fp_graph(
            ((X[:, :0], idx), (w, idx)), duration=10),
        "forecast_pairs count node rank-3 X": lambda: fp_graph((X.reshape(T, 1, K), w), count=1),
        "Pipeline rank-3 feed": lambda: (lambda x: s.Pipeline([x], [s.RollingMean(2)(x)]))(
            s.Input("x"))(X.reshape(T, 1, K)),
        "Pipeline (T,0) feed": lambda: (lambda x: s.Pipeline([x], [s.Dropna()(x)]))(
            s.Input("x"))(X[:, :0]),
        "forecast_pairs duration lazy": lambda: fp(vgen(v, idx), vgen(w[:len(yidx)], yidx), duration=10),
        "forecast_pairs duration graph": lambda: fp_graph(((v, idx), (yn, idx)), duration=20, dropna=True),
    })

    # split: the inverse of merge, every input form, setting and error
    tags = (np.arange(T) * 7 % 3)                           # sources 0, 1, 2
    from screamer.streams import merge
    mv, msrc, midx = merge((v, idx), (w, idx + 5))
    table.update({
        "split 1-D": lambda: s.split(v, tags),
        "split n larger": lambda: s.split(v, tags, n=5),
        "split n too small": lambda: s.split(v, tags, n=2),
        "split n 0 empty": lambda: s.split(v[:0], tags[:0], n=0),
        "split empty": lambda: s.split(v[:0], tags[:0]),
        "split int index": lambda: s.split(v, tags, index=idx),
        "split float index": lambda: s.split(v, tags, index=idx * 0.5),
        "split float index NaN": lambda: s.split(v[:3], [0, 1, 0], index=[np.nan, 1.0, 2.5]),
        "split list index": lambda: s.split(v[:4], [0, 1, 0, 1], index=[1, 2, 3, 4]),
        "split datetime64 index": lambda: s.split(v, tags, index=idx.astype("datetime64[s]")),
        "split str index": lambda: s.split(v[:2], [0, 1], index=["a", "b"]),
        "split (values,index) tuple": lambda: s.split((v, idx), tags),
        "split (values,None) tuple": lambda: s.split((v, None), tags),
        "split merge round trip": lambda: s.split((mv, midx), msrc),
        "split merge round trip n=3": lambda: s.split((mv, midx), msrc, n=3),
        "split (T,K) values": lambda: s.split(X, tags),
        "split scalar value": lambda: s.split(1.5, [0]),
        "split list with None": lambda: s.split([1.0, None, 3.0], [0, 1, 0]),
        "split str value": lambda: s.split(["a", 1.0], [0, 1]),
        "split list sources": lambda: s.split(v[:4], [0, 2, 0, 1]),
        "split uint8 sources": lambda: s.split(v, tags.astype(np.uint8)),
        "split float whole sources": lambda: s.split(v[:3], [0.0, 1.0, 0.0]),
        "split fractional sources": lambda: s.split(v[:3], [0.5, 1.7, 0.0]),
        "split NaN sources": lambda: s.split(v[:3], np.array([np.nan, 0.0, 1.0])),
        "split None source": lambda: s.split(v[:3], [None, 0, 1]),
        "split negative sources list": lambda: s.split(v[:3], [-1, 0, 1]),
        "split negative sources array": lambda: s.split(v[:3], np.array([-1, 0, 1])),
        "split sources 2**32": lambda: s.split(v[:2], [2**32, 0]),
        "split str sources": lambda: s.split(v[:2], ["0", "1"]),
        "split bool sources": lambda: s.split(v[:4], np.array([True, False, True, False])),
        "split (T,1) sources": lambda: s.split(v, tags[:, None]),
        "split sources length mismatch": lambda: s.split(v, tags[:5]),
        "split index length mismatch": lambda: s.split(v, tags, index=idx[:5]),
        "split n 3.0": lambda: s.split(v, tags, n=3.0),
        "split n str": lambda: s.split(v, tags, n="3"),
        "split n True": lambda: s.split(v[:2], [0, 0], n=True),
        "split n negative empty": lambda: s.split(v[:0], tags[:0], n=-1),
        "split n negative": lambda: s.split(v, tags, n=-1),
        "split n 2**40": lambda: s.split(v[:0], tags[:0], n=2**40),
    })
    return table


def describe(x):
    """A comparable record of a result that covers every element.

    Arrays record shape, dtype kind, the NaN-aware sum and a digest of all
    values rounded to 9 decimals, so a change in any element shows. Lists,
    tuples and lazy iterators record every element.
    """
    import hashlib
    import numpy as np
    if isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], np.ndarray):
        return ["pair", describe(x[0]), describe(x[1])]
    if isinstance(x, np.ndarray):
        if x.dtype.kind in "biuf":
            r = np.round(x.astype(np.float64), 9) + 0.0      # +0.0 folds -0.0 into 0.0
            r[np.isnan(r)] = np.nan                             # one NaN bit pattern
            total = round(float(np.nansum(x)), 6)
            digest = hashlib.sha1(np.ascontiguousarray(r).tobytes()).hexdigest()[:16]
            return ["ndarray", list(x.shape), x.dtype.kind, total, digest]
        return ["ndarray", list(x.shape), x.dtype.kind, [str(e) for e in x.ravel()]]
    if isinstance(x, list):
        return ["list", len(x), [describe(e) for e in x]]
    if isinstance(x, tuple):
        return ["tuple", len(x), [describe(e) for e in x]]
    if isinstance(x, float):
        return ["float", round(x, 9)]
    if hasattr(x, "__next__"):                                 # a lazy iterator
        return [type(x).__name__, describe(list(x))]
    return [type(x).__name__]


def run():
    out = {}
    for name, fn in cases().items():
        try:
            out[name] = describe(fn())
        except Exception as e:  # every error is a result to compare
            out[name] = ["ERROR", type(e).__name__, str(e)[:70]]
    return out


def tag(old, new):
    old_err = old is not None and old[0] == "ERROR"
    new_err = new is not None and new[0] == "ERROR"
    if json.dumps(old) == json.dumps(new):
        return "same"
    if old is None or new is None:
        return "ADDED" if old is None else "REMOVED"
    if old_err and new_err:
        return "ERR CHANGED"
    if old_err:
        return "ERR->VALUE"
    if (len(new) == 2 and new[0].endswith("Iterator")
            and json.dumps(new[1]) == json.dumps(old)):
        return "LAZY"                                  # same values, now lazy
    if new_err:
        return "VALUE->ERR"
    return "VALUE CHANGED"


def _load(path):
    with open(path) as f:
        return json.load(f)


def diff(old, new, show_all=False):
    changed = 0
    for name in list(old) + [n for n in new if n not in old]:
        t = tag(old.get(name), new.get(name))
        if t != "same":
            changed += 1
        if t != "same" or show_all:
            print(f"{t:13s} {name}")
            if t != "same":
                print(f"    old: {json.dumps(old.get(name))}")
                print(f"    new: {json.dumps(new.get(name))}")
    print(f"{changed} of {len(set(old) | set(new))} rows changed")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--json", metavar="OUT", help="write this build's results to OUT ('-' for stdout)")
    g.add_argument("--diff", nargs="*", metavar="FILE",
                   help="[[OLD] NEW]: print rows that changed; OLD defaults to the 2.8.0 "
                        "baseline and NEW to a run of this build")
    p.add_argument("--all", action="store_true", help="with --diff, also print unchanged rows")
    args = p.parse_args(argv)
    if args.diff is not None:
        if len(args.diff) > 2:
            p.error("--diff takes at most two files")
        files = [BASELINE] * (2 - len(args.diff)) + list(args.diff)
        old = _load(files[0])
        new = _load(files[1]) if len(args.diff) else run()
        diff(old, new, show_all=args.all)
        return
    out = run()
    if args.json == "-":
        json.dump(out, sys.stdout, indent=1)
    else:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
