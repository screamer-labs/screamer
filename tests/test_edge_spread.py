"""EdgeSpread: the EDGE estimator of Ardia, Guidotti and Kroencke (2024).

The reference values come from the authors' implementation
(github.com/eguidotti/bidask, MIT License): tests/data/edge_bidask_reference.csv
holds the first 300 bars of their test data, with and without missing values,
and the output of their edge() function on each trailing window.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from devtools.baselines.EdgeSpread import EdgeSpread_numpy
from screamer import CorwinSchultzSpread, EdgeSpread
from tests.regime_helpers import as_batch, as_scalar_loop, assert_bitwise

_REF = pd.read_csv(Path(__file__).parent / "data" / "edge_bidask_reference.csv", comment="#")


def _ohlc(tag):
    return [_REF[f"{tag}_{k}"].to_numpy() for k in ("open", "high", "low", "close")]


@pytest.mark.parametrize("tag", ["full", "miss"])
@pytest.mark.parametrize("column,kwargs", [
    ("edge21", dict(window_size=21)),
    ("edge21_signed", dict(window_size=21, sign=True)),
    ("edge_expanding_signed", dict(window_size=1000, sign=True, start_policy="expanding")),
])
def test_matches_the_authors_reference(tag, column, kwargs):
    got = np.asarray(EdgeSpread(**kwargs)(*_ohlc(tag)))
    want = _REF[f"{tag}_{column}"].to_numpy()
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-13, equal_nan=True)


def test_authors_published_signed_value():
    # bidask's test_edge: the signed estimate on the first 10 bars.
    out = EdgeSpread(10, sign=True)(*(a[:10] for a in _ohlc("full")))
    assert out[9] == pytest.approx(-0.016889917516422, rel=1e-12)
    assert np.isnan(out[:9]).all()


def test_flat_bars_give_nan():
    # bidask's test_edge: fewer than two bars with tau = 1.
    out = EdgeSpread(3)([18.21, 17.61, 17.61], [18.21, 17.61, 17.61],
                        [17.61, 17.61, 17.61], [17.61, 17.61, 17.61])
    assert np.isnan(out).all()


def test_unsigned_is_the_absolute_signed_value():
    bars = _ohlc("full")
    signed = np.asarray(EdgeSpread(21, sign=True)(*bars))
    plain = np.asarray(EdgeSpread(21)(*bars))
    assert (signed[21:] < 0).any()
    assert_bitwise(plain, np.abs(signed))


def _bars(n=400, seed=2, nan=True):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = close * np.exp(rng.normal(0, 0.004, n))
    wick = np.abs(rng.normal(0, 0.006, n))
    high = np.maximum(open_, close) * np.exp(wick)
    low = np.minimum(open_, close) * np.exp(-wick)
    if nan:
        open_[[7, 150]] = np.nan
        high[40] = np.nan
        low[41] = np.nan
        close[[90, 300]] = np.nan
        for a in (open_, high, low, close):
            a[200] = np.nan
    return [open_, high, low, close]


@pytest.mark.parametrize("start_policy", ["strict", "expanding"])
@pytest.mark.parametrize("sign", [False, True])
@pytest.mark.parametrize("window", [3, 4, 25])
def test_matches_the_pseudocode_baseline(window, sign, start_policy):
    bars = _bars()
    got = EdgeSpread(window, sign, start_policy)(*bars)
    ref = EdgeSpread_numpy(window, sign, start_policy)(*bars)
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-13, equal_nan=True)


def test_missing_values_follow_the_reference():
    # A missing price removes only the terms that use it, and a bar with all
    # prices missing still takes its place in the window: the estimate is
    # edge() on the last window_size bars, as in the authors' code.
    bars = _bars(80, nan=False)
    gapped = [a.copy() for a in bars]
    gapped[0][30] = np.nan                  # open missing: the bar still counts
    for a in gapped:
        a[50] = np.nan                      # a whole bar missing
    got = np.asarray(EdgeSpread(10)(*gapped))
    ref = EdgeSpread_numpy(10)(*gapped)
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-13, equal_nan=True)
    assert np.isfinite(got[30]) and np.isfinite(got[50])
    clean = np.asarray(EdgeSpread(10)(*bars))
    assert got[31] != clean[31]
    np.testing.assert_allclose(got[60:], clean[60:], rtol=1e-12)   # both gaps have left


def test_nonpositive_price_is_missing():
    bars = _bars(60, nan=False)
    zero = [a.copy() for a in bars]
    zero[2][20] = 0.0
    nan = [a.copy() for a in bars]
    nan[2][20] = np.nan
    assert_bitwise(EdgeSpread(10)(*zero), EdgeSpread(10)(*nan))


@pytest.mark.parametrize("make", [
    lambda: EdgeSpread(21),
    lambda: EdgeSpread(5, sign=True, start_policy="expanding"),
    lambda: EdgeSpread(400, start_policy="expanding"),
])
def test_batch_equals_stream(make):
    bars = _bars(1000)
    assert_bitwise(as_batch(make, *bars), as_scalar_loop(make, *bars))


def test_reset_restores_the_initial_state():
    bars = _bars(400)
    op = EdgeSpread(21, sign=True)
    first = np.asarray(op(*bars))
    op.reset()
    assert_bitwise(np.asarray(op(*bars)), first)


def test_long_stream_does_not_drift():
    # The window sums are updated by adding and subtracting pairs; a long
    # stream must still match a fresh computation on the last window.
    bars = _bars(20000, nan=False)
    got = np.asarray(EdgeSpread(50)(*bars))[-1]
    ref = EdgeSpread_numpy(50)(*(a[-50:] for a in bars))[-1]
    assert got == pytest.approx(ref, rel=1e-10)


def test_one_pair_with_x_has_zero_variance():
    # Only the first pair has all of o, h, l, h1, l1, c1, so x1 and x2 have
    # one value each and their variances are 0, as in the authors' code;
    # rounding left in the expanded sums must not set the weights of e1 and
    # e2. The value is the authors' reference edge() (github.com/eguidotti/bidask,
    # python/bidask/edge.py) on these four bars.
    o = [100.0, 101.0, np.nan, np.nan]
    h = [102.0, 103.0, 104.0, 102.5]
    l = [99.0, 100.0, 101.0, 100.5]
    c = [101.0, 102.0, 103.0, 101.0]
    want = 0.0057568346747761505
    assert np.asarray(EdgeSpread(4)(o, h, l, c))[-1] == pytest.approx(want, rel=1e-12)
    assert np.asarray(EdgeSpread(4, sign=True)(o, h, l, c))[-1] == pytest.approx(-want, rel=1e-12)


@pytest.mark.parametrize("field", [0, 3])
def test_outlier_leaving_the_window_leaves_no_rounding(field):
    # One bar's open or close 10000 times the rest: once it has left the
    # window, the estimate matches a fresh computation, without waiting for
    # the periodic recompute.
    bars = _bars(60, nan=False)
    bars[field][8] *= 1e4
    bars[1][8] = max(bars[1][8], bars[field][8])
    got = np.asarray(EdgeSpread(5, sign=True)(*bars))
    ref = EdgeSpread_numpy(5, sign=True)(*bars)
    np.testing.assert_allclose(got, ref, rtol=1e-11, equal_nan=True)


def test_constructor_validation():
    with pytest.raises(ValueError, match="window_size must be at least 3"):
        EdgeSpread(2)
    with pytest.raises(ValueError, match="start_policy"):
        EdgeSpread(20, start_policy="zero")


def _simulate(rng, spread, sigma, trades, bars):
    """Bars of trades at the bid or ask around a random-walk mid: the
    simulation of Ardia, Guidotti and Kroencke (2024) with a trade at every
    step."""
    mid = np.cumsum(rng.normal(0, sigma / math.sqrt(trades), bars * trades))
    side = rng.choice([-1.0, 1.0], bars * trades)
    price = np.exp(mid + side * spread / 2).reshape(bars, trades)
    return price[:, 0], price.max(axis=1), price.min(axis=1), price[:, -1]


def test_less_biased_than_corwin_schultz_when_the_spread_is_small():
    # A 20 bp spread on bars with 2% volatility: the floored Corwin-Schultz
    # estimate is several times the spread, EDGE is close to it.
    rng = np.random.default_rng(1)
    spread, edge, cs = 0.002, [], []
    for _ in range(20):
        o, h, l, c = _simulate(rng, spread, 0.02, 200, 1000)
        edge.append(np.asarray(EdgeSpread(1000)(o, h, l, c))[-1])
        cs.append(np.asarray(CorwinSchultzSpread(999, adjust_overnight=False)(h, l, c))[-1])
    assert abs(np.mean(edge) - spread) < 0.25 * spread
    assert np.mean(cs) > 3 * spread
