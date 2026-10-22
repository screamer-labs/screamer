import math

import numpy as np
import pytest

from devtools.baselines.AbdiRanaldoSpread import AbdiRanaldoSpread_numpy
from devtools.baselines.CorwinSchultzSpread import CorwinSchultzSpread_numpy
from screamer import AbdiRanaldoSpread, CorwinSchultzSpread
from tests.regime_helpers import as_batch, as_scalar_loop, assert_bitwise


def _bars(n=500, seed=1, nan=True):
    """Bars of trades at the bid or ask around a random-walk mid, with gaps."""
    rng = np.random.default_rng(seed)
    mid = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    close = mid * np.exp(rng.normal(0, 0.003, n))
    high = np.maximum(close, mid) * np.exp(np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(close, mid) * np.exp(-np.abs(rng.normal(0, 0.006, n)))
    if nan:
        high[[5, 100]] = np.nan
        close[200] = np.nan
        low[201] = np.nan
    return high, low, close


def _quoted(spread, n=12, seed=0):
    """A constant mid of 100: every high is the ask, every low the bid, and
    each close one of the two."""
    ask, bid = 100 * (1 + spread / 2), 100 * (1 - spread / 2)
    side = np.random.default_rng(seed).random(n) < 0.5
    return np.full(n, ask), np.full(n, bid), np.where(side, ask, bid)


def test_corwin_schultz_recovers_a_quoted_spread():
    # beta = 2 r^2 and gamma = r^2 with r = ln(ask / bid), so alpha = r and
    # S = 2 (ask/bid - 1) / (1 + ask/bid) = (ask - bid) / 100 exactly.
    out = CorwinSchultzSpread(3)(*_quoted(0.01))
    assert np.isnan(out[:3]).all()
    np.testing.assert_allclose(out[3:], 0.01, rtol=1e-12)


def test_abdi_ranaldo_recovers_a_quoted_spread():
    # c - eta = +-r/2 and c_{t-1} - eta_t = c_{t-1} - eta_{t-1}, so each
    # product is r^2 and S = r = ln(ask / bid).
    out = AbdiRanaldoSpread(3)(*_quoted(0.01))
    assert np.isnan(out[:3]).all()
    np.testing.assert_allclose(out[3:], math.log(100.5 / 99.5), rtol=1e-12)


def _cs_pair(h0, l0, h1, l1):
    """The paper's two-bar formula, written out for one pair."""
    k = 3 - 2 * math.sqrt(2)
    beta = math.log(h0 / l0) ** 2 + math.log(h1 / l1) ** 2
    gamma = math.log(max(h0, h1) / min(l0, l1)) ** 2
    alpha = (math.sqrt(2 * beta) - math.sqrt(beta)) / k - math.sqrt(gamma / k)
    return max(0.0, 2 * (math.exp(alpha) - 1) / (1 + math.exp(alpha)))


@pytest.mark.parametrize("bar1,shifted", [
    ((104.0, 102.0, 103.0), (102.5, 100.5)),     # low above the close: shift down 1.5
    ((99.0, 97.0, 98.0), (100.5, 98.5)),         # high below the close: shift up 1.5
    ((101.5, 99.5, 100.0), (101.5, 99.5)),       # inside the close: unchanged
])
def test_corwin_schultz_overnight_adjustment(bar1, shifted):
    high = np.array([101.0, bar1[0]])
    low = np.array([99.0, bar1[1]])
    close = np.array([100.5, bar1[2]])
    adj = CorwinSchultzSpread(1)(high, low, close)
    raw = CorwinSchultzSpread(1, adjust_overnight=False)(high, low, close)
    np.testing.assert_allclose(adj[1], _cs_pair(101.0, 99.0, *shifted), rtol=1e-12)
    np.testing.assert_allclose(raw[1], _cs_pair(101.0, 99.0, bar1[0], bar1[1]), rtol=1e-12)


def test_corwin_schultz_floors_each_pair_at_zero():
    # The second bar's range is far from the first: gamma > beta, so the
    # two-bar estimate is negative and enters the window as 0.
    high = np.array([101.0, 103.0, 101.0])
    low = np.array([100.0, 102.0, 100.0])
    close = np.array([100.5, 102.5, 100.5])
    out = CorwinSchultzSpread(2, adjust_overnight=False)(high, low, close)
    assert np.isnan(out[:2]).all()
    assert out[2] == 0.0


def test_abdi_ranaldo_floors_the_mean_at_zero():
    # Bar 0 closes at its high and bar 1's midpoint is above that close, so
    # the product is negative; one product in the window gives 0.
    high = np.array([101.0, 103.0])
    low = np.array([99.0, 101.0])
    close = np.array([101.0, 102.0])
    out = AbdiRanaldoSpread(1)(high, low, close)
    assert np.isnan(out[0]) and out[1] == 0.0


def test_abdi_ranaldo_averages_before_flooring():
    # The first product is negative, the second positive and larger: the
    # window of two averages them and only then takes the square root.
    high = np.array([101.0, 103.0, 104.0])
    low = np.array([99.0, 101.0, 102.0])
    close = np.array([101.0, 101.0, 103.0])
    c, eta = np.log(close), np.log(high * low) / 2
    x = 4 * (c[:-1] - eta[:-1]) * (c[:-1] - eta[1:])
    assert x[0] < 0 < x[1] and x.mean() > 0
    out = AbdiRanaldoSpread(2)(high, low, close)
    np.testing.assert_allclose(out[2], math.sqrt(x.mean()), rtol=1e-12)


@pytest.mark.parametrize("start_policy", ["strict", "expanding", "zero"])
@pytest.mark.parametrize("adjust", [True, False])
def test_corwin_schultz_matches_the_paper_formula(start_policy, adjust):
    bars = _bars()
    got = CorwinSchultzSpread(20, adjust, start_policy)(*bars)
    ref = CorwinSchultzSpread_numpy(20, adjust, start_policy)(*bars)
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-16, equal_nan=True)


@pytest.mark.parametrize("start_policy", ["strict", "expanding", "zero"])
def test_abdi_ranaldo_matches_the_paper_formula(start_policy):
    bars = _bars()
    got = AbdiRanaldoSpread(20, start_policy=start_policy)(*bars)
    ref = AbdiRanaldoSpread_numpy(20, start_policy)(*bars)
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-14, equal_nan=True)


def test_nan_bar_is_skipped_whole():
    high, low, close = _bars(60, nan=False)
    gap_h, gap_l, gap_c = (np.insert(a, 30, v) for a, v in
                           ((high, np.nan), (low, 1.0), (close, 1.0)))
    for op in (lambda: CorwinSchultzSpread(10), lambda: AbdiRanaldoSpread(10)):
        clean = as_batch(op, high, low, close)
        gapped = as_batch(op, gap_h, gap_l, gap_c)
        assert np.isnan(gapped[30])
        assert_bitwise(np.delete(gapped, 30), clean)


@pytest.mark.parametrize("make", [
    lambda: CorwinSchultzSpread(15),
    lambda: CorwinSchultzSpread(15, adjust_overnight=False, start_policy="expanding"),
    lambda: AbdiRanaldoSpread(15, start_policy="zero"),
])
def test_batch_equals_stream(make):
    bars = _bars(300)
    assert_bitwise(as_batch(make, *bars), as_scalar_loop(make, *bars))


def test_reset_restores_the_initial_state():
    bars = _bars(120, nan=False)
    for op in (CorwinSchultzSpread(10), AbdiRanaldoSpread(10)):
        first = np.asarray(op(*bars))
        op.reset()
        assert_bitwise(np.asarray(op(*bars)), first)


@pytest.mark.parametrize("cls", [CorwinSchultzSpread, AbdiRanaldoSpread])
def test_window_must_be_positive(cls):
    with pytest.raises(ValueError, match="window_size must be at least 1"):
        cls(0)
