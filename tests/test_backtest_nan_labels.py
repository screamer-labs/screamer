"""The `nan_policy` label of every Backtest* page must match what the engine does.

The label is read from the page frontmatter. Two behaviours are observed:

* `ignore`: the output row at the NaN step is all NaN, and the rows after it
  equal a run with that step removed (the step never touched the state).
* `nan-aware`: the output row at the NaN step is finite (the position is held
  and marked, or the NaN has a defined meaning such as a market order).

The label is checked against the NaN-target case, which is the input that
differs between engines. The NaN-price behaviour is stated in each test id and
asserted too. For an engine with no target (the order engines, the report
nodes), the "target" is its own bid price, or its first input.
"""
from pathlib import Path

import numpy as np
import pytest
import yaml

import screamer as s

DOCS = Path(__file__).resolve().parent.parent / "docs" / "functions_fin"
T, STEP = 40, 20

_rng = np.random.default_rng(1)
_c = 100 + np.cumsum(_rng.normal(0, 0.3, T))
_o = np.r_[_c[0], _c[:-1]] + _rng.normal(0, 0.05, T)
_h = np.maximum(_o, _c) + 0.2
_l = np.minimum(_o, _c) - 0.2
_v = np.full(T, 50.0)
_tgt = np.sign(np.sin(np.arange(T) / 3.0))
_tgt[_tgt == 0] = 1.0
_bid, _ask, _sz = _c - 0.05, _c + 0.05, np.full(T, 5.0)
_tp, _ts = _c + _rng.normal(0, 0.03, T), np.full(T, 3.0)
_ob, _oa, _osz = _c - 0.2, _c + 0.2, np.full(T, 2.0)
_eq, _pnl, _pos, _cost = np.asarray(s.BacktestPriceTarget()(_tgt, _c)).T
_pt = np.c_[_tgt, -_tgt] * 0.5
_two = lambda x: np.c_[x, x * 1.1]


def _arrays(name):
    return {
        "BacktestPriceTarget": [_tgt, _c],
        "BacktestOHLCTarget": [_tgt, _o, _h, _l, _c],
        "BacktestOHLCVTarget": [_tgt, _o, _h, _l, _c, _v],
        "BacktestTradesTarget": [_tgt, _tp, _ts],
        "BacktestL1Target": [_tgt, _bid, _ask, _sz, _sz],
        "BacktestOHLCOrders": [_ob, _osz, _oa, _osz, _o, _h, _l, _c],
        "BacktestTradesOrders": [_ob, _osz, _oa, _osz, _tp, _ts],
        "BacktestL1Orders": [_ob, _osz, _oa, _osz, _bid, _ask, _sz, _sz],
        "BacktestL1TradesOrders": [_ob, _osz, _oa, _osz, _bid, _ask, _sz, _sz, _tp, _ts],
        "BacktestPortfolioTarget": [_pt, _two(_c)],
        "BacktestPortfolioOHLCTarget": [_pt, _two(_o), _two(_h), _two(_l), _two(_c)],
        "BacktestReport": [_eq, _pnl, _pos, _cost],
        "BacktestStats": [_eq, _pnl, _pos, _cost],
        "BacktestCarry": [_eq, _pnl, _pos, _cost, _c, np.full(T, 1e-4)],
    }[name]


# name -> (target input, price inputs, observed NaN-price behaviour, inputs that
# give a NaN a defined meaning and so behave as nan-aware: own ask price and
# sizes, and the trade of BacktestL1TradesOrders, a quote-only update)
# The behaviour is what the engine does on a NaN price: "ignore" is an all-NaN
# row with the state untouched, "nan-aware" is a finite row (the asset is
# skipped and its equity repeated).
CASES = {
    "BacktestPriceTarget": (0, [1], "ignore", []),
    "BacktestOHLCTarget": (0, [1, 2, 3, 4], "ignore", []),
    "BacktestOHLCVTarget": (0, [1, 2, 3, 4], "ignore", []),
    "BacktestTradesTarget": (0, [1, 2], "ignore", []),
    "BacktestL1Target": (0, [1, 2], "ignore", []),
    "BacktestOHLCOrders": (0, [4, 5, 6, 7], "ignore", [1, 2, 3]),
    "BacktestTradesOrders": (0, [4, 5], "ignore", [1, 2, 3]),
    "BacktestL1Orders": (0, [4, 5, 6, 7], "ignore", [1, 2, 3]),
    "BacktestL1TradesOrders": (0, [4, 5, 6, 7], "ignore", [1, 2, 3, 8, 9]),
    "BacktestPortfolioTarget": (0, [1], "nan-aware", []),
    "BacktestPortfolioOHLCTarget": (0, [1, 2, 3, 4], "nan-aware", []),
    "BacktestReport": (0, [1, 2, 3], "ignore", []),
    "BacktestStats": (0, [1, 2, 3], "ignore", []),
    "BacktestCarry": (0, [1, 2, 3, 4, 5], "ignore", []),
}


def _label(name):
    text = (DOCS / f"{name}.md").read_text()
    front = yaml.safe_load(text.split("---")[1])
    return front["nan_policy"]


def _run(name, arrays):
    return np.asarray(getattr(s, name)()(*arrays))


def _observe(name, idx):
    """Return 'ignore' or 'nan-aware' for a NaN at step STEP of input idx."""
    arrays = _arrays(name)
    holed = [a.copy() for a in arrays]
    holed[idx][STEP] = np.nan            # first asset column for 2-D inputs
    out = _run(name, holed)
    row = out[STEP]
    if np.isfinite(row).all():
        return "nan-aware"
    assert np.isnan(row).all(), f"{name}: row {STEP} is partly NaN"
    removed = _run(name, [np.delete(a, STEP, axis=0) for a in arrays])
    np.testing.assert_allclose(out[STEP + 1:], removed[STEP:], equal_nan=True,
                               err_msg=f"{name}: an ignored step changed later rows")
    return "ignore"


def test_every_backtest_page_is_covered():
    # startswith: Windows globs ignore case, and backtest_report.md is a guide page.
    pages = {p.stem for p in DOCS.glob("*.md") if p.stem.startswith("Backtest")}
    assert pages == set(CASES)


@pytest.mark.parametrize(
    "name", [pytest.param(n, id=f"{n}-nan_price_{CASES[n][2]}") for n in sorted(CASES)]
)
def test_label_matches_nan_target_and_price_behavior(name):
    target, prices, price_behavior, own = CASES[name]
    label = _label(name)
    assert label in ("ignore", "nan-aware")
    assert _observe(name, target) == label, (
        f"{name} declares {label} but behaves differently on a NaN target")
    for idx in prices:
        assert _observe(name, idx) == price_behavior, (
            f"{name}: NaN in input {idx} no longer behaves as '{price_behavior}'")
    for idx in own:
        assert _observe(name, idx) == "nan-aware", (
            f"{name}: NaN in input {idx} is expected to give a finite row")
