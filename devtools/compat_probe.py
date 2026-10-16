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
        "Resample clock 1-D ohlc": lambda: s.Resample((v, idx), (np.zeros(5), np.arange(5) * 60 + 5), clock=True, agg="ohlc"),
        "Resample clock (T,K) ohlc": lambda: s.Resample((X, idx), (np.zeros(5), np.arange(5) * 60 + 5), clock=True, agg="ohlc"),
        "Resample clock (T,K) ohlc lazy": lambda: s.Resample(gen(zip(X, idx)), gen(zip(np.zeros(5), np.arange(5) * 60 + 5)), clock=True, agg="ohlc"),
        "Resample clock (T,K) last lazy": lambda: s.Resample(gen(zip(X, idx)), gen(zip(np.zeros(5), np.arange(5) * 60 + 5)), clock=True, agg="last"),
        "Resample count (T,K) ohlc lazy": lambda: s.Resample(count=5, agg="ohlc")(gen(zip(X, idx))),
        "Resample clock (T,2,3) last": lambda: s.Resample((X[:, [0, 1, 2, 0, 1, 2]].reshape(T, 2, 3), idx), (np.zeros(5), np.arange(5) * 60 + 5), clock=True, agg="last"),
        "Resample clock (T,K) last": lambda: s.Resample((X, idx), (np.zeros(5), np.arange(5) * 60 + 5), clock=True, agg="last"),
        "Resample clock (T,4) ohlc_bars": lambda: s.Resample((X[:, [0, 0, 0, 0]], idx), (np.zeros(5), np.arange(5) * 60 + 5), clock=True, agg="ohlc_bars"),
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
    # The Ehlers operators with two outputs, on an array past their warm-up
    # (Task 33 changed their values; MAMA with its default limits).
    for name in ["CycleSine", "MAMA", "HilbertPhasor"]:
        table[f"Ehlers {name} array"] = (lambda n: lambda: getattr(s, n)()(long))(name)
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
        "forecast_pairs count index target": lambda: fp(v, w, count=3, index="target"),
        "forecast_pairs duration index target": lambda: fp(
            (v, idx), (w[:len(yidx)], yidx), duration=10, index="target"),
        "forecast_pairs index bad": lambda: fp(v, w, count=1, index="entry"),
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
        "split (T,K) values with index": lambda: s.split(X, tags, index=idx, n=4),
        "split (T,1) values": lambda: s.split(X[:, :1], tags),
        "split (T,2,2) values": lambda: s.split(np.arange(T * 4.0).reshape(T, 2, 2), tags),
        "split (0,K) values": lambda: s.split(X[:0], tags[:0], n=2),
        "split list of rows": lambda: s.split([[1.0, 2.0], [3.0, None]], [1, 0]),
        "split bool list sources": lambda: s.split(v[:2], [True, False]),
        "split mixed bool list sources": lambda: s.split(v[:2], [True, 2]),
        "split np.int64 list sources": lambda: s.split(v[:4], list(tags[:4])),
        "split object sources": lambda: s.split(v[:3], np.array([0, 1, 0], dtype=object)),
        "split float32 sources": lambda: s.split(v[:3], np.array([0, 1, 0], dtype=np.float32)),
        "split source 2**31-1": lambda: s.split(v[:1], [2**31 - 1]),
        "split int64 source 2**40": lambda: s.split(v[:1], np.array([2**40])),
        "split uint64 source 2**40": lambda: s.split(v[:1], np.array([2**40], dtype=np.uint64)),
        "split 2-D index": lambda: s.split(v[:2], [0, 1], index=np.zeros((2, 1))),
        "split n 2.0": lambda: s.split(v, tags, n=2.0),
        "split n np.int64": lambda: s.split(v, tags, n=np.int64(4)),
    })

    # combine_latest: the layout join, every input form, batch and iterator
    CL = s.CombineLatest
    P = np.arange(T * 3.0).reshape(T, 3)                    # a (T, A) panel
    P3 = np.arange(T * 6.0).reshape(T, 3, 2)                # (T, A, F)
    lz = lambda r: list(r)                                  # noqa: E731
    rows = lambda a: (x for x in a)                         # noqa: E731
    table.update({
        "CombineLatest (T,K) + 1-D": lambda: CL()(X, v),
        "CombineLatest one input": lambda: CL()(v),
        "CombineLatest indexed": lambda: CL()((v, idx), (w[:10], idx[:10] * 3 + 5)),
        "CombineLatest indexed on_any": lambda: CL(emit="on_any")((v, idx + 5), (w, idx)),
        "CombineLatest indexed wide": lambda: CL()((X, idx), (w[:5], idx[:5] * 7)),
        "CombineLatest index= list": lambda: CL()(v, w, index=[idx, idx + 3]),
        "CombineLatest float index": lambda: CL()((v, idx * 0.5), (w, idx * 0.5 + 0.25)),
        "CombineLatest int and float index": lambda: CL()((v, idx), (w, idx + 0.5)),
        "CombineLatest bool index": lambda: CL()((v[:2], np.array([True, False])), (w[:2], idx[:2])),
        "CombineLatest datetime index": lambda: CL()(
            (v[:3], np.array(["2024-01-01", "2024-01-02", "2024-01-03"], dtype="M8[D]")),
            (w[:2], np.array(["2024-01-02", "2024-01-04"], dtype="M8[D]"))),
        "CombineLatest datetime and number": lambda: CL()(
            (v[:2], np.array(["2024-01-01", "2024-01-02"], dtype="M8[D]")), (w[:2], idx[:2])),
        "CombineLatest list None": lambda: CL()([1.0, None, 3.0], [4.0, 5.0, 6.0]),
        "CombineLatest list str": lambda: CL()([1.0, "x"], [4.0, 5.0]),
        "CombineLatest length mismatch": lambda: CL()(v, w[:5]),
        "CombineLatest positional and indexed": lambda: CL()(v, (w, idx)),
        "CombineLatest empty indexed on_any": lambda: CL(emit="on_any")(
            (v, idx), (np.array([]), np.array([], dtype=np.int64))),
        "CombineLatest 3-D": lambda: CL()(P3, v),
        "CombineLatest 3-D indexed": lambda: CL()((P3, idx), (w[:4], idx[:4] * 5 + 1)),
        "CombineLatest layout TA,T": lambda: CL(layout=("TA", "T"))(P, v),
        "CombineLatest layout TA": lambda: CL(layout="TA")(P, X),
        "CombineLatest layout list None": lambda: CL(layout=["TA", None])(P, X),
        "CombineLatest layout AT": lambda: CL(layout=("AT", "T"))(P.T.copy(), v),
        "CombineLatest layout AT indexed": lambda: CL(layout=("AT", "T"))((P.T.copy(), idx), (v, idx)),
        "CombineLatest lazy layout TFA": lambda: lz(CL(layout=("TFA", "T"))(
            rows(P3.transpose(0, 2, 1)[:3]), rows(v[:3]))),
        "CombineLatest layout TFA": lambda: CL(layout=("TFA", "T"))(P3.transpose(0, 2, 1).copy(), v),
        "CombineLatest layout TAF,TA": lambda: CL(layout=("TAF", "TA"))(P3, X),
        "CombineLatest layout ->": lambda: CL(layout=("TAF->TA", "T"))(P3[:, :, :1], v),
        "CombineLatest layout indexed": lambda: CL(layout=("TA", "T"))((P, idx), (w[:4], idx[:4] * 6)),
        "CombineLatest layout count": lambda: CL(layout=("TA",))(P, v),
        "CombineLatest layout bad letter": lambda: CL(layout=("TX", "T"))(P, v),
        "CombineLatest layout wrong rank": lambda: CL(layout=("TAF", "T"))(P, v),
        "CombineLatest assets do not broadcast": lambda: CL(layout="TA")(P, X[:, :2]),
        "CombineLatest layout on a node": lambda: CL(layout="TA")(s.Input("a"), s.Input("b")),
        "CombineLatest lazy positional": lambda: lz(CL()(rows(v[:5]), rows(w[:5]))),
        "CombineLatest lazy positional None": lambda: lz(CL()(iter([1.0, None]), iter([2.0, 3.0]))),
        "CombineLatest lazy positional str": lambda: lz(CL()(iter([1.0, "x"]), iter([2.0, 3.0]))),
        "CombineLatest lazy positional np scalars": lambda: lz(CL()(rows(v[:3]), rows(idx[:3]))),
        "CombineLatest lazy positional uneven": lambda: lz(CL()(rows(v[:3]), rows(w[:2]))),
        "CombineLatest lazy positional wide": lambda: lz(CL()(rows(X[:4]), rows(v[:4]))),
        "CombineLatest lazy positional list rows": lambda: lz(CL()(
            iter([[1.0, 2.0], [3.0, None]]), iter([5.0, 6.0]))),
        "CombineLatest lazy positional layout": lambda: lz(CL(layout=("TA", "T"))(rows(P[:4]), rows(v[:4]))),
        "CombineLatest lazy positional 3-D": lambda: lz(CL()(rows(P3[:3]), rows(v[:3]))),
        "CombineLatest lazy indexed": lambda: lz(CL()(vgen(v, idx), vgen(w[:6], idx[:6] * 4 + 3))),
        "CombineLatest lazy indexed on_any": lambda: lz(CL(emit="on_any")(vgen(v, idx + 5), vgen(w, idx))),
        "CombineLatest lazy indexed wide": lambda: lz(CL()(zip(X[:6], idx), vgen(w[:3], idx[:3] * 2))),
        "CombineLatest lazy indexed layout": lambda: lz(CL(layout=("TA", "T"))(
            zip(P[:6], idx), vgen(w[:3], idx[:3] * 2))),
        "CombineLatest lazy indexed 3-D": lambda: lz(CL()(zip(P3[:4], idx), vgen(w[:4], idx[:4]))),
        "CombineLatest lazy indexed float": lambda: lz(CL()(
            iter([(1.0, 0.5), (2.0, 1.5)]), iter([(3.0, 1.0)]))),
        "CombineLatest lazy mixed kinds": lambda: lz(CL()(vgen(v, idx), rows(w))),
        "CombineLatest lazy one empty": lambda: lz(CL(emit="on_any")(vgen(v[:3], idx), iter([]))),
        "CombineLatest lazy all empty": lambda: lz(CL()(iter([]), iter([]))),
        "CombineLatest lazy and array": lambda: CL()(rows(v), w),
        "CombineLatest index length mismatch": lambda: CL()((v, idx[:5]), (w, idx)),
        "CombineLatest layout AT,AT": lambda: CL(layout="AT")(P.T.copy(), P.T.copy()),
        "CombineLatest list of two rows": lambda: CL()([X[0, :2], X[1, :2]], [w[:2], v[:2]]),
        "CombineLatest lazy positional two-value lists": lambda: lz(CL()(
            iter([[1.0, 2.0], [3.0, 4.0]]), iter([[5.0, 6.0], [7.0, 8.0]]))),
        "CombineLatest tuple of lists": lambda: CL()(([1.0, 2.0], [0, 1]), ([3.0, 4.0], [0, 1])),
        "CombineLatest Series index": lambda: CL()((v, pd.Series(idx)), (w, pd.Index(idx + 5))),
        "CombineLatest tuple of 3 floats": lambda: CL()((1.0, 2.0, 3.0), [1.0, 2.0, 3.0]),
        "CombineLatest tuple of 2 floats": lambda: CL()((1.0, 2.0), [1.0, 2.0]),
        "CombineLatest range index": lambda: CL()((v[:3], range(3)), (w[:3], idx[:3])),
        "CombineLatest tuple index": lambda: CL()((v[:3], (0, 1, 2)), (w[:3], idx[:3])),
        "CombineLatest tuple of tuples": lambda: CL()(((1.0, 2.0, 3.0), (0, 1, 2)), (w[:3], idx[:3])),
        "merge range index": lambda: merge((v[:3], range(3)), (w[:3], idx[:3] + 10)),
        "split range index": lambda: s.split((v[:3], range(3)), [0, 1, 0]),
        "Pipeline range index": lambda: (lambda x: s.Pipeline([x], [s.RollingMean(1)(x)]))(
            s.Input("x"))((v[:3], range(3))),
        "Dropna tuple of lists": lambda: s.Dropna()(([1.0, np.nan], [0, 1])),
        "merge tuple of lists": lambda: merge(([1.0, 2.0], [0, 2]), ([3.0], [1])),
        "split tuple of lists": lambda: s.split(([1.0, 2.0], [5, 6]), [0, 1]),
        "Pipeline tuple of lists": lambda: (lambda x: s.Pipeline([x], [s.RollingMean(1)(x)]))(
            s.Input("x"))(([1.0, 2.0], [0, 1])),
        "forecast_pairs count tuple X": lambda: s.forecast_pairs((1.0, 2.0, 3.0), [1.0, 2.0, 3.0], count=1),
        "forecast_pairs duration lists": lambda: s.forecast_pairs(
            ([1.0, 2.0], [0, 10]), ([1.0, 2.0], [0, 10]), duration=5),
        "CombineLatest lazy indexed two-value tuples": lambda: lz(CL()(
            iter([(1.0, 2), (3.0, 4)]), iter([(5.0, 6), (7.0, 8)]))),
    })

    # A decreasing index on a stream (G26), the fill bars of one gap (G22), and
    # a clock resample whose values come through a Filter (G20).
    dv, dk = v[:4], np.array([0, 20, 10, 30])               # decreases at event 2

    def pipe1(feed, op=lambda x: s.RollingMean(1)(x)):
        x = s.Input("x")
        return s.Pipeline([x], [op(x)])(feed)

    def pipe2(fa, fb):
        a, b = s.Input("a"), s.Input("b")
        return s.Pipeline([a, b], [s.RollingMean(1)(a), s.RollingMean(1)(b)])(fa, fb)

    def live_push(ks):
        x = s.Input("x")
        sess = s.Pipeline([x], [s.RollingMean(1)(x)]).live()
        for val, k in zip(dv, ks):
            sess.push("x", int(k), float(val))
        sess.flush()
        return sess.result()

    def clock_via_filter():
        x, m, c = s.Input("x"), s.Input("m"), s.Input("c")
        node = s.Resample(s.Filter()(x, m), c, clock=True, agg="last")
        return s.Pipeline([x, m, c], [node])(
            (np.array([1.0]), np.array([5])),
            (np.ones(3), np.array([0, 5, 20])),
            (np.zeros(3), np.array([6, 9, 12])))

    gap = np.array([0, 2_000_001])                           # 2e6 empty buckets
    table.update({
        "merge decreasing index": lambda: merge((dv, dk), (w[:2], idx[:2])),
        "merge lazy decreasing index": lambda: lz(merge(vgen(dv, dk), vgen(w[:2], idx[:2]))),
        "merge equal indexes": lambda: merge((dv, np.array([0, 10, 10, 30])), (w[:2], idx[:2])),
        "CombineLatest decreasing index": lambda: CL()((dv, dk), (w[:2], idx[:2])),
        "CombineLatest lazy decreasing index": lambda: lz(CL()(vgen(dv, dk), vgen(w[:2], idx[:2]))),
        "Pipeline decreasing index": lambda: pipe1((dv, dk)),
        "Pipeline 2 inputs decreasing index": lambda: pipe2((dv, dk), (w[:2], idx[:2])),
        "Pipeline lazy decreasing index": lambda: lz(pipe1(vgen(dv, dk))),
        "Pipeline live decreasing index": lambda: live_push(dk),
        "Pipeline equal indexes": lambda: pipe1((dv, np.array([0, 10, 10, 30]))),
        "Resample decreasing index": lambda: s.Resample(freq=10, agg="last")((dv, dk)),
        "Resample count decreasing index": lambda: s.Resample(count=2, agg="last")((dv, dk)),
        "Delay decreasing index": lambda: s.Delay(5)(dv, dk),
        "Dropna decreasing index": lambda: s.Dropna()((dv, dk)),
        "split decreasing index": lambda: s.split((dv, dk), [0, 1, 0, 1]),
        "forecast_pairs decreasing index": lambda: s.forecast_pairs((dv, dk), (dv, dk), duration=5),
        "Resample fill nan 2e6-bucket gap": lambda: s.Resample(freq=1, fill="nan")((v[:2], gap)),
        "Resample fill carry 2e6-bucket gap": lambda: s.Resample(freq=1, fill="carry")((v[:2], gap)),
        "Resample fill nan small gap": lambda: s.Resample(freq=10, fill="nan")((v[:2], np.array([0, 95]))),
        "Resample fill max_fill=3e6": lambda: s.Resample(freq=1, fill="nan", max_fill=3_000_000)((v[:2], gap)),
        "Resample fill max_fill=0": lambda: s.Resample(freq=1, fill="nan", max_fill=0)((v[:2], gap)),
        "Resample clock value via Filter": clock_via_filter,
    })

    # Empty results keep their width (G18), fill="carry" is per column (G23),
    # a CombineLatest of zero-width feeds (G24), a bool as a number setting
    # (G25), to_json with a functor reducer (G21), and the field axis of a
    # lazy multi-output operator on wide events (A12).
    E3, e0 = np.zeros((0, K)), np.zeros(0, dtype=np.int64)
    Xn = X[:6, :2].copy()
    Xn[[1, 3], 0] = np.nan
    Xn[[2, 3], 1] = np.nan
    kn = np.array([0, 10, 25, 40, 41, 70])
    vn = np.array([v[0], np.nan, v[2]])

    def live_drain(align):
        a, b = s.Input("a"), s.Input("b")
        sess = s.Pipeline([a, b], [a, s.RollingMean(2)(b)], align_outputs=align).live()
        for t in range(2):
            sess.push("a", t, X[t].tolist())
            sess.push("b", t, float(v[t]))
        sess.result()
        return sess.result()

    def zero_width_combine():
        a, b = s.Input("a"), s.Input("b")
        p = s.Pipeline([a, b], [s.RollingMean(2)(s.CombineLatest()(a, b))])
        return p(np.zeros((3, 0)), np.zeros((3, 0)))

    def json_functor_agg():
        x = s.Input("x")
        p = s.Pipeline([x], [s.Resample(count=3, agg=s.RollingMean(2))(x)])
        return s.Pipeline.from_json(p.to_json())(v)

    clock = (np.zeros(4), np.array([5, 15, 30, 75]))
    table.update({
        "Pipeline empty (0,K) passthrough": lambda: pipe1(E3, op=lambda x: x),
        "Pipeline empty (0,K) resample ohlc": lambda: pipe1(E3, op=lambda x: s.Resample(count=2, agg="ohlc")(x)),
        "Pipeline live empty drain aligned": lambda: live_drain(True),
        "Pipeline live empty drain unaligned": lambda: live_drain(False),
        "Resample count empty (0,K)": lambda: s.Resample(count=2)(E3),
        "Resample freq ohlc empty (0,K)": lambda: s.Resample(freq=2, agg="ohlc")((E3, e0)),
        "Delay empty (0,K)": lambda: s.Delay(1)((E3, e0)),
        "Resample fill carry NaN cells (T,K)": lambda: s.Resample(freq=10, fill="carry")((Xn, kn)),
        "Resample fill carry NaN bar 1-D": lambda: s.Resample(freq=10, fill="carry")((vn, np.array([0, 10, 40]))),
        "Resample clock carry NaN cells (T,K)": lambda: s.Resample((Xn, kn), clock, clock=True, agg="last", fill="carry"),
        "Resample clock carry NaN bar 1-D": lambda: s.Resample((Xn[:, 0], kn), clock, clock=True, agg="last", fill="carry"),
        "CombineLatest of (T,0) feeds into an operator": zero_width_combine,
        "Pipeline to_json functor agg": json_functor_agg,
        "RollingMean(True)": lambda: s.RollingMean(True)(v),
        "RollingMean(np.True_)": lambda: s.RollingMean(np.True_)(v),
        "BollingerBands num_std=True": lambda: s.BollingerBands(3, True)(v),
        "MovingAverage bool taps": lambda: s.MovingAverage([True, 0.5])(v),
        "Resample count=True": lambda: s.Resample(count=True)(v),
        "Delay duration=True": lambda: s.Delay(True)(v, idx),
        "Pipeline max_pending=np.True_": lambda: (lambda x: s.Pipeline(
            [x], [s.RollingMean(1)(x)], max_pending=np.True_)(v))(s.Input("x")),
        "BollingerBands lazy wide events": lambda: lz(s.BollingerBands(3)(gen(X[:5]))),
        # Task 31 fix round 1: a 0-d bool array, a function's bool argument, a
        # finite setting given an infinity, and fill= under count=/threshold=.
        "RollingMean(np.array(True))": lambda: s.RollingMean(np.array(True))(v),
        "probabilistic_sharpe benchmark=True": lambda: s.probabilistic_sharpe(ret, True),
        "circular_shift n=True": lambda: s.circular_shift(sign, True),
        "RollingMean(2.5)": lambda: s.RollingMean(2.5)(v),
        "BollingerBands num_std=inf": lambda: s.BollingerBands(3, float("inf"))(v),
        "SchmittTrigger lower=-inf": lambda: s.SchmittTrigger(float("-inf"), 100.0)(v),
        "Resample count fill=nan": lambda: s.Resample(count=2, fill="nan")(v),
        "Resample threshold fill=carry": lambda: s.Resample(v, np.ones(T), threshold=2.0, fill="carry"),
    })

    # Task 34: Merge of wide streams (G27), a non-finite setting a page gives
    # no meaning (G31), an index of another length (G32), the packed
    # portfolio engines (G33), and the shared saved form (B8).
    # A short index that is a slice of a longer buffer: 2.8.0 read the
    # buffer's next values, which keeps its row reproducible.
    short = (np.arange(100) * 10)[:5]

    def short_index_pipeline():
        x = s.Input("x")
        return s.Pipeline([x], [s.Lag(1)(x)])((v[:10], short))

    def pipeline_json_inf():
        x = s.Input("x")
        p = s.Pipeline([x], [s.Clip(lower=float("-inf"), upper=101.0)(x)])
        return s.Pipeline.from_json(p.to_json())(v)

    prices = np.abs(X) + 1.0
    weights = np.full((T, K), 1.0 / K)
    table.update({
        "Merge wide batch": lambda: s.Merge()((X[:5], idx[:5]), (X[5:9] + 1, idx[:4] + 5)),
        "Merge wide positional": lambda: s.Merge()(X[:5], X[5:8]),
        "Merge wide mixed widths": lambda: s.Merge()((X[:5], idx[:5]), (X[:4, :2], idx[:4])),
        "Merge wide lazy indexed": lambda: list(s.Merge()(
            gen([(X[t], int(idx[t])) for t in range(4)]), gen([(X[t] + 1, int(idx[t]) + 5) for t in range(3)]))),
        "Merge wide lazy positional": lambda: list(s.Merge()(gen(X[:3]), gen(X[3:5]))),
        "Merge text index": lambda: s.Merge()((v[:2], np.array(["a", "b"])), (v[:2], np.arange(2))),
        "KeltnerChannels num_atr=inf": lambda: s.KeltnerChannels(5, float("inf")),
        "MAMA fast_limit=inf": lambda: s.MAMA(float("inf")),
        "RollingQuantile quantile=nan": lambda: s.RollingQuantile(5, float("nan"))(v),
        "BacktestL1Target taker_fee=inf": lambda: s.BacktestL1Target(taker_fee=float("inf")),
        "BacktestL1Target max_position=inf": lambda: s.BacktestL1Target(max_position=float("inf")),
        "Clip lower=-inf": lambda: s.Clip(lower=float("-inf"), upper=100.0)(v),
        "Pipeline short index": short_index_pipeline,
        "Merge short index": lambda: s.Merge()((v[:10], short), (v[:3], idx[:3])),
        "Resample short index": lambda: s.Resample(freq=10)((v[:10], short)),
        "split short index": lambda: s.split(v[:10], np.zeros(10, dtype=int), index=short),
        "BacktestPortfolioTarget packed": lambda: s.BacktestPortfolioTarget()(np.stack([weights, prices], -1)),
        "Pipeline to_json -inf setting": pipeline_json_inf,
        # Task 34 fix round 1.
        "Merge 2-D index": lambda: s.Merge()((v[:3], idx[:3, None]), (v[:3], idx[:3])),
        "CombineLatest 2-D index": lambda: s.CombineLatest()((v[:3], idx[:3]), (v[:3], idx[:3, None])),
        "Merge bare numbers": lambda: s.Merge()(5.0, 6.0),
        "RollingMean(-1)": lambda: s.RollingMean(-1),
        "WMA(0)": lambda: s.WMA(0),
        "Power p=inf": lambda: s.Power(float("inf"))(np.abs(v[:5]) / 200),
        "Linear scale=-inf": lambda: s.Linear(float("-inf"), 0.0)(v[:5]),
        "KalmanFilter initial_variance=inf": lambda: s.KalmanFilter(0.01, 1.0, 0.0, float("inf"))(v),
        "Hampel n_sigma=inf": lambda: s.Hampel(5, float("inf"))(v),
        # Task 34 fix round 2.
        "RollingMean(2**31)": lambda: s.RollingMean(2 ** 31),
        "RollingMean(-10**30)": lambda: s.RollingMean(-10 ** 30),
        "KalmanFilter finite overflow": lambda: s.KalmanFilter(1.7e308, 0.5, 0.0, 1.7e308)(v[:3]),
        "BacktestOHLCTarget delay=0 fill_at=open": lambda: s.BacktestOHLCTarget(delay=0),
    })

    # Task FA (final review, C++ core and engine).
    def live_after_flush():
        a, b = s.Input("a"), s.Input("b")
        sess = s.Pipeline([a, b], [s.CombineLatest()(a, b)]).live()
        sess.push("a", 1, 1.0)
        sess.push("b", 1, 2.0)
        sess.flush()
        for nm, k, val in [("a", 3, 1.0), ("b", 2, 2.0), ("a", 4, 3.0), ("b", 5, 4.0)]:
            sess.push(nm, k, val)
        sess.flush()
        return sess.result()

    def two_producers(op, fa, fb):
        a, b = s.Input("a"), s.Input("b")
        return s.Pipeline([a, b], [op(a, b)])(fa, fb)

    def shared_ops():
        x = s.Input("x")
        y = s.RollingSum(2)(x)
        s1, s2 = s.Pipeline([x], [y]).live(), s.Pipeline([x], [y]).live()
        for t in range(4):
            s1.push("x", t, 1.0)
            s2.push("x", t, 100.0)
        s1.flush()
        return s1.result()

    def two_sessions():
        x = s.Input("x")
        p = s.Pipeline([x], [s.RollingSum(2)(x)])
        s1, s2 = p.live(), p.live()
        for t in range(3):
            s1.push("x", t, 1.0)
            s2.push("x", t, 100.0)
        s1.flush()
        return s1.result()

    def from_json_big():
        x = s.Input("x")
        text = s.Pipeline([x], [s.RollingMean(3)(x)]).to_json()
        return s.Pipeline.from_json(text.replace('"window_size": 3', '"window_size": 18446744073709551616'))

    top = 2 ** 63 - 6
    table.update({
        "Pipeline two sessions from the same nodes": shared_ops,
        "Pipeline two sessions of one Pipeline": lambda: two_sessions(),
        "split n past 2**20": lambda: len(s.split([1.0], [0], n=2**20 + 1)),
        "Pipeline.from_json int past int64": from_json_big,
        "2in array beside 0-d array": lambda: s.Add()(v, np.array(1.0)),
        "2in 0-d array beside array": lambda: s.Add()(np.array(1.0), v),
        "2in array beside float": lambda: s.Add()(v, 1.0),
        "Pipeline operator two producers at one index": lambda: two_producers(
            lambda a, b: s.Sub()(b, a), (np.array([[1.0, 0.0]]), np.array([5])),
            (np.array([[9.0, 0.0]]), np.array([5]))),
        "Pipeline operator two producers delayed": lambda: two_producers(
            lambda a, b: s.Add()(s.Delay(10)(a), b),
            (np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0, 20])),
            (np.array([[5.0, 6.0], [7.0, 8.0]]), np.array([5, 15]))),
        "Pipeline live push after flush": live_after_flush,
        "Delay past int64": lambda: s.Delay(10)(np.ones(2), np.array([0, top])),
        "Resample right label past int64": lambda: s.Resample(freq=10, label="right", agg="last")(
            (np.ones(1), np.array([top]))),
        "Resample origin past int64": lambda: s.Resample(freq=10, origin=-10, agg="last")(
            (np.ones(1), np.array([top]))),
    })

    # Task FB (final review, Python): input counts, value kinds, wide
    # Dropna/Select with equal indexes, the combine_latest iterator index kinds.
    import array
    import ctypes
    import types
    dt = np.array(["2020-01-01", "2020-01-02"], "datetime64[ns]")
    def one_input_pipeline():
        x = s.Input("x")
        return s.Pipeline([x], [s.RollingSum(2)(x)])

    W = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [np.nan, 40.0], [5.0, 50.0]])
    WK = np.array([0, 1, 1, 1, 2])
    table.update({
        "2in one 1-D array": lambda: s.RollingCorr(3)(v),
        "2in three arrays": lambda: s.RollingCorr(3)(v, v, v),
        "2in one tuple of three": lambda: s.RollingCorr(3)((1.0, 2.0, 3.0)),
        "1in two numbers": lambda: s.RollingMean(3)(1.0, 2.0),
        "1in no argument": lambda: s.RollingMean(3)(),
        "2in unknown keyword": lambda: s.RollingCorr(3)(x=1.0),
        "row op one input of two": lambda: s.BacktestPortfolioTarget()(X),
        "micro missing input": lambda: s.SignedVolume()(v),
        "2in tuple beside array": lambda: s.RollingCorr(5)(tuple(v), w),
        "2in array beside tuple": lambda: s.RollingCorr(5)(w, tuple(v)),
        "2in empty list": lambda: s.Add()([]),
        "1in datetime64 array": lambda: s.RollingMean(1)(dt),
        "1in list of datetime64": lambda: s.RollingMean(1)(list(dt)),
        "merge datetime64 values": lambda: s.Merge()(dt, v[:2]),
        "Pipeline datetime64 values": lambda: one_input_pipeline()(dt),
        "Pipeline datetime64 index": lambda: one_input_pipeline()((v[:2], dt)),
        "1in ctypes double": lambda: s.RollingMean(1)(ctypes.c_double(1.5)),
        "merge ctypes array": lambda: s.Merge()((ctypes.c_double * 3)(1.0, 2.0, 3.0), v[:3]),
        "1in pandas Timestamp": lambda: s.RollingMean(1)(pd.Timestamp("2020-01-01")),
        "1in mappingproxy": lambda: s.RollingMean(1)(types.MappingProxyType({1.5: 2.0})),
        "1in numeric memoryview": lambda: s.RollingMean(2)(memoryview(array.array("d", v[:4]))),
        "1in memoryview of bytes": lambda: s.RollingMean(1)(memoryview(b"12")),
        "1in Decimal": lambda: s.RollingMean(1)(__import__("decimal").Decimal("1.5")),
        "Dropna two positional arrays": lambda: s.Dropna()(np.array([1.0, np.nan, 3.0]),
                                                          np.array([5.0, 6.0, 7.0])),
        "Dropna wide equal indexes": lambda: s.Dropna()(W, index=WK),
        "Dropna wide equal indexes lazy": lambda: list(s.Dropna()(gen(zip(W, WK)))),
        "Select wide equal indexes": lambda: s.Select([1])(W, index=WK),
        "Select wide equal indexes lazy": lambda: list(s.Select([1])(gen(zip(W, WK)))),
        "CombineLatest lazy fractional index": lambda: list(s.CombineLatest()(
            gen(zip(v[:3], [0.5, 1.5, 2.5])), gen(zip(w[:3], [0.0, 1.0, 2.0])))),
        "CombineLatest lazy indexed one-value rows": lambda: list(s.CombineLatest()(
            gen([(np.array([1.0]), 7)]))),
        "from_pandas list": lambda: s.from_pandas([1.0, 2.0]),
        "1in object DataFrame": lambda: s.RollingMean(2)(pd.DataFrame({"a": [object(), object()]})),
        # Task FB fix round 1
        "2in Decimal": lambda: s.Add()(__import__("decimal").Decimal("1.5"), 1.0),
        "1in list of Fraction": lambda: s.RollingMean(1)([__import__("fractions").Fraction(3, 2)]),
        "1in object array of datetime64": lambda: s.RollingMean(1)(np.array(list(dt), dtype=object)),
        "Select [0] 1-D": lambda: s.Select([0])(v[:4]),
        "Select [0, 0] 1-D": lambda: s.Select([0, 0])(v[:4]),
        "Select [0] lazy width 1": lambda: list(s.Select([0])(gen(zip(v[:3], range(3))))),
        "Merge NaN float index": lambda: s.Merge()((v[:3], np.array([0.0, np.nan, 1.0]))),
        "CombineLatest NaN float index": lambda: s.CombineLatest()((v[:3], np.array([0.0, np.nan, 1.0]))),
        "PortfolioReport no input": lambda: s.PortfolioReport()(),
        "PortfolioReport four numbers": lambda: s.PortfolioReport()(1.0, 2.0, 3.0, 4.0),
        "Dropna complex values": lambda: s.Dropna()(np.array([1 + 1j, 2.0])),
        "Dropna lazy bare values": lambda: list(s.Dropna()(gen([1.0, np.nan, 3.0]))),
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
