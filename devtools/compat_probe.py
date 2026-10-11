"""Compare the results of one matrix of input forms across two screamer builds.

The probe calls a fixed set of operators with every input form screamer
accepts (scalars, arrays of each rank, lists, tuples, iterators, pandas,
streams, microstructure keywords) and records one comparable line per call:
the container, shape and sum of a value, or the type and message of an error.

Record this build's results::

    .venv/bin/python devtools/compat_probe.py --json new.json

Record a released wheel's results in a separate venv. Run the probe with
``-I`` from outside the repository so Python imports the installed wheel and
not the source tree::

    python -m venv /tmp/old && /tmp/old/bin/pip install screamer==2.8.0 pandas
    cd /tmp && /tmp/old/bin/python -I /path/to/devtools/compat_probe.py --json old.json

Print the rows whose result changed, tagged ``ERR->VALUE``, ``VALUE->ERR``,
``VALUE CHANGED`` or ``ERR CHANGED`` (``--all`` also prints ``same`` rows)::

    .venv/bin/python devtools/compat_probe.py --diff old.json new.json
"""
import argparse
import json
import sys


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
    sign = np.sign(v - 100)
    ret = np.diff(v, prepend=v[0])
    gen = lambda a: (x for x in a)
    two_level = lambda: pd.DataFrame(
        np.arange(8.0).reshape(2, 4),
        columns=pd.MultiIndex.from_product([["a", "b"], ["o", "c"]]))

    return {
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
        "1in pandas Series": lambda: s.RollingMean(3)(pd.Series(v)),
        "1in pandas DataFrame": lambda: s.RollingMean(3)(pd.DataFrame(X)),
        "1in FunctorBase gen (IsNan)": lambda: s.IsNan()(gen(v[:3])),
        "1in FunctorBase (T,K,1)": lambda: s.IsNan()(X[:, :, None]),
        "1in FunctorBase tuple of floats": lambda: s.IsNan()(tuple(v[:3])),
        "1in FunctorBase (values,index)": lambda: s.IsNan()((v, idx)),
        "1-in M-out (T,K,1)": lambda: s.BollingerBands(3)(X[:, :, None]),
        "1-in M-out tuple of floats": lambda: s.BollingerBands(3)(tuple(v[:4])),
        "1-in M-out (values,index)": lambda: s.BollingerBands(3)((v, idx)),
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
    }


def describe(x):
    import numpy as np
    if isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], np.ndarray):
        return ["pair", describe(x[0]), describe(x[1])]
    if isinstance(x, np.ndarray):
        if x.dtype.kind in "biuf":
            total = round(float(np.nansum(x)), 6)
        else:
            total = [str(e) for e in x.ravel()[:4]]
        return ["ndarray", list(x.shape), x.dtype.kind, total]
    if isinstance(x, list):
        return ["list", len(x), describe(x[0]) if x else None]
    if isinstance(x, tuple):
        return ["tuple", len(x)]
    if isinstance(x, float):
        return ["float", round(x, 6)]
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
    if new_err:
        return "VALUE->ERR"
    return "VALUE CHANGED"


def diff(old_path, new_path, show_all=False):
    with open(old_path) as f:
        old = json.load(f)
    with open(new_path) as f:
        new = json.load(f)
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
    g.add_argument("--diff", nargs=2, metavar=("OLD", "NEW"), help="print rows that changed")
    p.add_argument("--all", action="store_true", help="with --diff, also print unchanged rows")
    args = p.parse_args(argv)
    if args.diff:
        diff(*args.diff, show_all=args.all)
        return
    out = run()
    if args.json == "-":
        json.dump(out, sys.stdout, indent=1)
    else:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
