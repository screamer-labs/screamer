#!/usr/bin/env python3
"""Write wasm/smoke/oracle_research.json: the operators and helpers of the
research toolkit, computed by the Python binding, for js/test/research.test.ts.

Each operator case names the operator, its settings in Python names, and one
of the input series below; js/test/research.test.ts builds the same operator
in JavaScript (settings in camelCase), runs it on the same series, and
compares the result, to 1e-9 relative unless the case gives a tolerance. The
helper cases hold the inputs and results of backtest_trades, backtest_report
with by=, walk_forward with carry= and portfolio engines, and
LiquidationDistance, RollingMultiRegression and EwMultiRegression on (T, K)
columns. A non-finite
number is written as null (NaN), "inf" or "-inf".

Modes:
    (no args)  write the fixture
    --stdout   print the fixture to stdout (freshness gate)
"""
import argparse
import json
import math
import os
import sys

import numpy as np

import screamer as s
from screamer.dag_io import _decode

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "wasm", "smoke", "oracle_research.json")


def _inputs():
    rng = np.random.default_rng(20261017)
    T = 600
    events = np.where(rng.random(T) < 0.15, np.round(rng.standard_normal(T), 3), 0.0)
    events[[40, 41, 300]] = np.nan
    scale = np.where((np.arange(T) // 150) % 2 == 0, 0.5, 2.0)
    returns = rng.standard_normal(T) * scale
    returns[[100, 450]] = np.nan
    return {"events": events, "returns": returns, "labels": _labels(T), "returns2": _returns2(T),
            **_liquidation_market(), **_ols_rows(), **_hlc_bars()}


def _labels(T):
    """Segment labels: runs of 1 to 40 rows over 4 values (a value returns),
    leading NaN labels and NaN labels inside segments."""
    rng = np.random.default_rng(20261022)
    labels = np.repeat(rng.integers(0, 4, T).astype(float), rng.integers(1, 41, T))[:T]
    labels[[0, 1, 60, 333]] = np.nan
    return labels


def _returns2(T):
    """A second return series, for two-input operators."""
    return np.random.default_rng(20261023).standard_normal(T)


def _hlc_bars(T=600):
    """Bars of trades at the bid or ask around a random-walk mid, with gaps
    between bars and NaN fields, for the high-low spread estimators."""
    rng = np.random.default_rng(20261022)
    mid = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, T)))
    close = mid * np.exp(rng.normal(0, 0.003, T))
    high = np.maximum(close, mid) * np.exp(np.abs(rng.normal(0, 0.006, T)))
    low = np.minimum(close, mid) * np.exp(-np.abs(rng.normal(0, 0.006, T)))
    high[[5, 100]] = np.nan
    low[201] = np.nan
    close[200] = np.nan
    return {"hlc_high": high, "hlc_low": low, "hlc_close": close}


def _liquidation_market(T=600, K=1):
    """A position with long, short, flat, flipping, and resized stretches and
    NaN holds, on volatile OHLC bars with NaN prices (one on an entry open),
    for LiquidationDistance. K > 1 gives (T, K) columns."""
    rng = np.random.default_rng(20261018 + K)
    steps = np.repeat(rng.choice([0.0, 1.0, 2.0, -1.0, -3.0], size=(T // 25 + 1, K)), 25, axis=0)[:T]
    position = steps.copy()
    position[rng.random((T, K)) < 0.03] = np.nan
    close = 100 * np.exp(np.cumsum(rng.standard_normal((T, K)) * 0.02, axis=0))
    open_ = np.vstack([close[:1], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.standard_normal((T, K))) * 0.01)
    low = np.minimum(open_, close) * (1 - np.abs(rng.standard_normal((T, K))) * 0.01)
    open_[[25, 26], 0] = np.nan                    # an entry bar and the bar after it
    high[[90, 200], 0] = np.nan
    low[[91, 201], 0] = np.nan
    cols = {"liq_position": position, "liq_open": open_, "liq_high": high, "liq_low": low}
    return {k: v[:, 0] for k, v in cols.items()} if K == 1 else cols


def _ols_rows(T=600, k=16, K=1, seed=20261018):
    """y and k regressors with a linear relation, a level of 1e3 on two of
    them, NaN rows, and a stretch where x1 is constant, for the multiple
    regression operators. K > 1 gives (T, K) columns."""
    rng = np.random.default_rng(seed + K)
    X = rng.standard_normal((T, k, K))
    X[:, 1] += 1e3
    X[:, 2] = 0.5 * X[:, 2] + 0.5 * X[:, 0]
    X[200:240, 0] = 1.5
    y = 0.3 + np.einsum("tjk,j->tk", X, rng.standard_normal(k) / k) + 0.2 * rng.standard_normal((T, K))
    y[[50, 290]] = np.nan
    X[[51, 120], 1] = np.nan
    cols = {"ols_y": y, **{f"ols_x{j + 1}": X[:, j] for j in range(k)}}
    return {n: v[:, 0] for n, v in cols.items()} if K == 1 else cols


# (operator, settings, input series or a list of them[, relative tolerance]).
# Each operator task adds its cases here.
_LIQ = ["liq_position", "liq_open", "liq_high", "liq_low"]
_OLS = ["ols_y"] + [f"ols_x{j}" for j in range(1, 17)]
_HLC = ["hlc_high", "hlc_low", "hlc_close"]
CASES = [
    ("Hold", {"n": 3}, "events"),
    ("Hold", {"n": 3, "cooldown": 4}, "events"),
    ("Hold", {"n": 1, "release": -1.0, "cooldown": 2}, "events"),
    ("CusumFilter", {"threshold": 1.0}, "returns"),
    ("CusumFilter", {"threshold": 2.5, "drift": 0.2}, "returns"),
    ("RollingSeasonalMean", {"period": 24, "window_size": 5}, "returns"),
    ("RollingSeasonalStd", {"period": 24, "window_size": 5, "start_policy": "expanding"}, "returns"),
    ("RollingSeasonalZscore", {"period": 7, "window_size": 6, "start_policy": "zero"}, "returns"),
    ("RollingUpsideDeviation", {"window_size": 20}, "returns"),
    ("RollingUpsideDeviation", {"window_size": 15, "mar": 0.2, "start_policy": "expanding"}, "returns"),
    ("RollingUpsideDeviation", {"window_size": 15, "start_policy": "zero"}, "returns"),
    ("RollingBipowerVar", {"window_size": 30}, "returns"),
    ("RollingJumpShare", {"window_size": 30, "start_policy": "expanding"}, "returns"),
    ("RollingJumpShare", {"window_size": 12, "start_policy": "zero"}, "returns"),
    ("RollingVarianceRatio", {"window_size": 60, "lag": 4}, "returns"),
    ("RollingVarianceRatio", {"window_size": 60, "lag": 4, "output": "zscore", "start_policy": "expanding"}, "returns"),
    ("RollingPermutationEntropy", {"window_size": 50, "order": 3}, "returns"),
    ("RollingPermutationEntropy", {"window_size": 40, "order": 5, "delay": 2, "start_policy": "expanding"}, "returns"),
    ("RollingPermutationEntropy", {"window_size": 20, "order": 4, "start_policy": "zero"}, "events"),
    ("BayesianChangepoint", {"expected_run": 100.0, "max_run": 300}, "returns", 1e-7),
    ("BayesianChangepoint", {"expected_run": 20.0, "max_run": 40, "prior_var": 4.0}, "returns", 1e-7),
    ("RollingHMM", {"window_size": 150, "refit_every": 40, "max_iter": 20, "tol": 0.0}, "returns", 1e-6),
    ("RollingHMM", {"window_size": 150, "refit_every": 40, "emission": "ar1", "max_iter": 20, "tol": 0.0},
     "returns", 1e-6),
    # Segment: an operator setting, written as {"functor", "params"}.
    ("Segment", {"op": {"functor": "CumSum", "params": {}}}, ["labels", "returns"]),
    ("Segment", {"op": {"functor": "RollingStd", "params": {"window_size": 10}}}, ["labels", "returns"]),
    ("Segment", {"op": {"functor": "EwMean", "params": {"span": 8.0}}}, ["labels", "returns"]),
    ("Segment", {"op": {"functor": "RollingCorr", "params": {"window_size": 12}}},
     ["labels", "returns", "returns2"]),
    ("Segment", {"op": {"functor": "BollingerBands", "params": {"window_size": 5}}}, ["labels", "returns"]),
    ("Segment", {"op": {"functor": "Segment", "params": {"op": {"functor": "CumSum", "params": {}}}}},
     ["labels", "events", "returns"]),
    ("LiquidationDistance", {"leverage": 10.0}, _LIQ),
    ("LiquidationDistance", {"leverage": 25.0, "maintenance": 0.01}, _LIQ),
    ("RollingMultiRegression", {"window_size": 40, "n_regressors": 3}, _OLS[:4]),
    ("RollingMultiRegression", {"window_size": 25, "n_regressors": 2, "intercept": False}, _OLS[:3]),
    ("RollingMultiRegression", {"window_size": 60, "n_regressors": 16}, _OLS),
    ("ExpandingMultiRegression", {"n_regressors": 4}, _OLS[:5]),
    ("ExpandingMultiRegression", {"n_regressors": 1, "intercept": False}, _OLS[:2]),
    ("EwMultiRegression", {"n_regressors": 3, "alpha": 0.05}, _OLS[:4]),
    ("EwMultiRegression", {"n_regressors": 2, "intercept": False, "halflife": 20.0}, _OLS[:3]),
    ("EwMultiRegression", {"n_regressors": 16, "span": 120.0}, _OLS),
    ("CorwinSchultzSpread", {"window_size": 20}, _HLC, 1e-12),
    ("CorwinSchultzSpread", {"window_size": 10, "adjust_overnight": False, "start_policy": "expanding"},
     _HLC, 1e-12),
    ("AbdiRanaldoSpread", {"window_size": 20}, _HLC, 1e-12),
    ("AbdiRanaldoSpread", {"window_size": 5, "start_policy": "zero"}, _HLC, 1e-12),
]


def _engine_output():
    """A BacktestPriceTarget output with long, short, flat, flipping, and
    resized stretches, two NaN rows, and a trade open on the last row."""
    rng = np.random.default_rng(7)
    target = np.repeat([0, 1, 1, -2, 0, 3, -1, 0, 2, 3, 0, -1], 10).astype(float)
    price = 100 + np.cumsum(rng.standard_normal(len(target)))
    out = np.asarray(s.BacktestPriceTarget(fee=1e-3)(target, price), dtype=float)
    out[[33, 34]] = np.nan
    return out


def _market(T):
    """The deterministic OHLC market of tests/test_walk_forward.py."""
    o, h, l, c = [], [], [], []
    p = 100.0
    for t in range(T):
        prev = p
        p *= np.exp(0.01 * np.sin(1.7 * t) * np.cos(0.31 * t * t))
        o.append(prev)
        c.append(p)
        h.append(max(prev, p) * 1.001)
        l.append(min(prev, p) * 0.999)
    return [np.array(x) for x in (o, h, l, c)]


def _candidates(T):
    t = np.arange(T)
    return np.column_stack([np.sign(np.sin(t / 6.0)), np.sign(np.cos(t / 17.0)),
                            ((t // 5) % 3) - 1.0]).astype(float)


def _helpers():
    """Inputs and results of the backtest helpers; Tasks 12 to 14 add entries."""
    out = _engine_output()
    trades = s.backtest_trades(out)
    labels = (np.arange(len(out)) // 30) * 10
    _, summary = s.backtest_report(out, periods_per_year=252.0, by=labels)
    by = {"labels": labels.tolist(), "periods_per_year": 252.0,
          "groups": [[int(k), {n: _num(v) for n, v in g.items()}]
                     for k, g in summary["by_group"].items()]}
    T = 260
    o, h, l, c = _market(T)
    X = _candidates(T)
    rate = 1e-3 * np.sin(np.arange(T) / 7.0)
    wf = s.walk_forward(s.BacktestOHLCTarget(taker_fee=2e-4), X, o, h, l, c, train=100, test=40,
                        carry=rate, carry_multiplier=2.0)
    walk_forward_carry = {
        "market": [_flat(x) for x in (o, h, l, c)], "targets": _flat(X), "K": X.shape[1],
        "carry": _flat(rate), "carry_multiplier": 2.0, "train": 100, "test": 40,
        "train_scores": _flat(wf.train_scores), "choice": wf.choice.tolist(),
        "output": _flat(wf.output), "stats": {k: _num(v) for k, v in wf.stats.items()}}
    Tp, Kp, Cp = 240, 3, 2
    rng = np.random.default_rng(11)
    prices = 100 * np.exp(np.cumsum(rng.standard_normal((Tp, Kp)) * 0.01, axis=0))
    t = np.arange(Tp)[:, None, None]
    Xp = 0.3 * np.sign(np.sin(t / (5.0 + 3.0 * np.arange(Cp))[None, None, :] + np.arange(Kp)[None, :, None]))
    ratep = 1e-3 * np.cos(np.arange(Tp)[:, None] / 5.0 + np.arange(Kp))
    wfp = s.walk_forward(s.BacktestPortfolioTarget(fee=1e-4), Xp, prices, train=80, test=40, carry=ratep)
    walk_forward_portfolio = {
        "T": Tp, "K": Kp, "C": Cp, "prices": _flat(prices), "targets": _flat(Xp), "carry": _flat(ratep),
        "train": 80, "test": 40, "train_scores": _flat(wfp.train_scores), "choice": wfp.choice.tolist(),
        "target": _flat(wfp.target), "output": _flat(wfp.output),
        "stats": {k: _num(v) for k, v in wfp.stats.items()}}
    # Sparse schedules (NaN holds between orders): the window-start restatement.
    Xh = np.full((Tp, Kp, Cp), np.nan)
    Xh[::7, :, 0] = Xp[::7, :, 0]
    Xh[3::11, :, 1] = Xp[3::11, :, 1]
    Xh[3::11, 2, 1] = np.nan
    wfh = s.walk_forward(s.BacktestPortfolioTarget(fee=1e-3), Xh, prices, train=60, test=30, carry=ratep)
    walk_forward_portfolio_hold = {
        "targets": _flat(Xh), "train": 60, "test": 30, "train_scores": _flat(wfh.train_scores),
        "choice": wfh.choice.tolist(), "target": _flat(wfh.target), "output": _flat(wfh.output),
        "stats": {k: _num(v) for k, v in wfh.stats.items()}}
    liq = _liquidation_market(T=300, K=3)
    liq_out = s.LiquidationDistance(leverage=8.0)(*(liq[k] for k in _LIQ))
    liquidation_2d = {"T": 300, "K": 3, "leverage": 8.0,
                      "inputs": [_flat(liq[k]) for k in _LIQ],
                      "shape": list(np.shape(liq_out)), "output": _flat(liq_out)}
    ols = _ols_rows(T=300, k=3, K=3)
    ols_out = s.RollingMultiRegression(30, 3)(*(ols[n] for n in _OLS[:4]))
    ols_2d = {"T": 300, "K": 3, "window_size": 30, "n_regressors": 3,
              "inputs": [_flat(ols[n]) for n in _OLS[:4]],
              "shape": list(np.shape(ols_out)), "output": _flat(ols_out)}
    ew_out = s.EwMultiRegression(3, span=30.0)(*(ols[n] for n in _OLS[:4]))
    ew_2d = {"T": 300, "K": 3, "span": 30.0, "n_regressors": 3,
             "inputs": ols_2d["inputs"], "shape": list(np.shape(ew_out)), "output": _flat(ew_out)}
    return {
        "liquidation_2d": liquidation_2d,
        "ols_2d": ols_2d,
        "ew_2d": ew_2d,
        "walk_forward_carry": walk_forward_carry,
        "walk_forward_portfolio": walk_forward_portfolio,
        "walk_forward_portfolio_hold": walk_forward_portfolio_hold,
        "by": by,
        "backtest_trades": {
            "values": _flat(out), "T": len(out),
            "result": {k: _flat(np.asarray(v, dtype=float)) for k, v in trades.items()},
        },
    }


def _num(v):
    v = float(v)
    if math.isnan(v):
        return None
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return v


def _flat(a):
    return [_num(v) for v in np.asarray(a, dtype=float).ravel()]


def build():
    inputs = _inputs()
    cases = []
    for case in CASES:
        name, settings, key = case[:3]
        args = [inputs[k] for k in key] if isinstance(key, list) else [inputs[key]]
        op = getattr(s, name)(**{k: _decode(v) for k, v in settings.items()})
        out = np.asarray(op(*args), dtype=float)
        entry = {"name": name, "settings": settings, "input": key,
                 "shape": list(out.shape), "out": _flat(out)}
        if len(case) > 3:
            entry["tol"] = case[3]
        cases.append(entry)
    return {"inputs": {k: _flat(v) for k, v in inputs.items()}, "cases": cases,
            "helpers": _helpers()}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stdout", action="store_true", help="print instead of writing")
    args = p.parse_args()
    text = json.dumps(build(), separators=(",", ":")) + "\n"
    if args.stdout:
        sys.stdout.write(text)
    else:
        with open(OUT, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
