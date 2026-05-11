"""
Tests for range-based volatility estimators:

  Parkinson      -- H, L                   (zero drift, no gaps)
  Garman-Klass   -- O, H, L, C             (zero drift, no gaps)
  Rogers-Satchell -- O, H, L, C            (drift-robust)

Each estimator has four classes: Rolling*Var, Rolling*Vol, Ew*Var,
Ew*Vol. *Vol is exactly sqrt of *Var.
"""
import numpy as np
import pandas as pd
import pytest

from screamer import (
    RollingParkinsonVar, RollingParkinsonVol,
    EwParkinsonVar, EwParkinsonVol,
    RollingGarmanKlassVar, RollingGarmanKlassVol,
    EwGarmanKlassVar, EwGarmanKlassVol,
    RollingRogersSatchellVar, RollingRogersSatchellVol,
    EwRogersSatchellVar, EwRogersSatchellVol,
)


def _ohlc(n, seed=0):
    rng = np.random.default_rng(seed)
    open_ = 100 + np.cumsum(rng.normal(0, 1, n))
    close = open_ + rng.normal(0, 0.5, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.3, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.3, n))
    return open_, high, low, close


# ---------------------------------------------------------------------------
# Per-bar formula references
# ---------------------------------------------------------------------------

def parkinson_perbar(high, low):
    return np.log(high / low) ** 2 / (4 * np.log(2))


def garman_klass_perbar(open_, high, low, close):
    return 0.5 * np.log(high / low) ** 2 \
         - (2 * np.log(2) - 1) * np.log(close / open_) ** 2


def rogers_satchell_perbar(open_, high, low, close):
    return np.log(high / close) * np.log(high / open_) \
         + np.log(low  / close) * np.log(low  / open_)


# ---------------------------------------------------------------------------
# Rolling variance parity
# ---------------------------------------------------------------------------

class TestRollingVar:

    @pytest.mark.parametrize("w", [5, 10, 20])
    def test_parkinson(self, w):
        _, high, low, _ = _ohlc(150, seed=w)
        ours = RollingParkinsonVar(w)(high, low)
        ref = pd.Series(parkinson_perbar(high, low)).rolling(w).mean().to_numpy()
        np.testing.assert_allclose(ours, ref, equal_nan=True, atol=1e-15)

    @pytest.mark.parametrize("w", [5, 10, 20])
    def test_garman_klass(self, w):
        o, h, l, c = _ohlc(150, seed=w + 100)
        ours = RollingGarmanKlassVar(w)(o, h, l, c)
        ref = pd.Series(garman_klass_perbar(o, h, l, c)).rolling(w).mean().to_numpy()
        np.testing.assert_allclose(ours, ref, equal_nan=True, atol=1e-15)

    @pytest.mark.parametrize("w", [5, 10, 20])
    def test_rogers_satchell(self, w):
        o, h, l, c = _ohlc(150, seed=w + 200)
        ours = RollingRogersSatchellVar(w)(o, h, l, c)
        ref = pd.Series(rogers_satchell_perbar(o, h, l, c)).rolling(w).mean().to_numpy()
        np.testing.assert_allclose(ours, ref, equal_nan=True, atol=1e-15)


# ---------------------------------------------------------------------------
# EW variance parity (vs pandas ewm)
# ---------------------------------------------------------------------------

class TestEwVar:

    def test_parkinson(self):
        _, high, low, _ = _ohlc(150, seed=0)
        ours = EwParkinsonVar(span=20)(high, low)
        ref = pd.Series(parkinson_perbar(high, low)).ewm(span=20).mean().to_numpy()
        np.testing.assert_allclose(ours, ref, atol=1e-15)

    def test_garman_klass(self):
        o, h, l, c = _ohlc(150, seed=1)
        ours = EwGarmanKlassVar(span=20)(o, h, l, c)
        ref = pd.Series(garman_klass_perbar(o, h, l, c)).ewm(span=20).mean().to_numpy()
        np.testing.assert_allclose(ours, ref, atol=1e-15)

    def test_rogers_satchell(self):
        o, h, l, c = _ohlc(150, seed=2)
        ours = EwRogersSatchellVar(span=20)(o, h, l, c)
        ref = pd.Series(rogers_satchell_perbar(o, h, l, c)).ewm(span=20).mean().to_numpy()
        np.testing.assert_allclose(ours, ref, atol=1e-15)


# ---------------------------------------------------------------------------
# Vol = sqrt(Var) identity
# ---------------------------------------------------------------------------

class TestVolEqualsSqrtVar:

    def test_rolling_parkinson(self):
        _, h, l, _ = _ohlc(100, seed=10)
        var = RollingParkinsonVar(14)(h, l)
        vol = RollingParkinsonVol(14)(h, l)
        # std::sqrt and numpy.sqrt can differ by one ULP on some
        # platforms (Linux x86_64 vs. macOS arm64); tolerate that.
        np.testing.assert_allclose(vol, np.sqrt(var), equal_nan=True, atol=1e-15)

    def test_rolling_garman_klass(self):
        o, h, l, c = _ohlc(100, seed=11)
        var = RollingGarmanKlassVar(14)(o, h, l, c)
        vol = RollingGarmanKlassVol(14)(o, h, l, c)
        # std::sqrt and numpy.sqrt can differ by one ULP on some
        # platforms (Linux x86_64 vs. macOS arm64); tolerate that.
        np.testing.assert_allclose(vol, np.sqrt(var), equal_nan=True, atol=1e-15)

    def test_rolling_rogers_satchell(self):
        o, h, l, c = _ohlc(100, seed=12)
        var = RollingRogersSatchellVar(14)(o, h, l, c)
        vol = RollingRogersSatchellVol(14)(o, h, l, c)
        # std::sqrt and numpy.sqrt can differ by one ULP on some
        # platforms (Linux x86_64 vs. macOS arm64); tolerate that.
        np.testing.assert_allclose(vol, np.sqrt(var), equal_nan=True, atol=1e-15)

    def test_ew_parkinson(self):
        _, h, l, _ = _ohlc(100, seed=13)
        var = EwParkinsonVar(span=20)(h, l)
        vol = EwParkinsonVol(span=20)(h, l)
        # std::sqrt and numpy.sqrt can differ by one ULP on some
        # platforms (Linux x86_64 vs. macOS arm64); tolerate that.
        np.testing.assert_allclose(vol, np.sqrt(var), equal_nan=True, atol=1e-15)

    def test_ew_garman_klass(self):
        o, h, l, c = _ohlc(100, seed=14)
        var = EwGarmanKlassVar(span=20)(o, h, l, c)
        vol = EwGarmanKlassVol(span=20)(o, h, l, c)
        # std::sqrt and numpy.sqrt can differ by one ULP on some
        # platforms (Linux x86_64 vs. macOS arm64); tolerate that.
        np.testing.assert_allclose(vol, np.sqrt(var), equal_nan=True, atol=1e-15)

    def test_ew_rogers_satchell(self):
        o, h, l, c = _ohlc(100, seed=15)
        var = EwRogersSatchellVar(span=20)(o, h, l, c)
        vol = EwRogersSatchellVol(span=20)(o, h, l, c)
        # std::sqrt and numpy.sqrt can differ by one ULP on some
        # platforms (Linux x86_64 vs. macOS arm64); tolerate that.
        np.testing.assert_allclose(vol, np.sqrt(var), equal_nan=True, atol=1e-15)


# ---------------------------------------------------------------------------
# Warmup and edge cases
# ---------------------------------------------------------------------------

class TestWarmup:

    def test_rolling_warmup_is_window_minus_one(self):
        _, h, l, _ = _ohlc(30, seed=0)
        out = RollingParkinsonVar(10)(h, l)
        assert np.all(np.isnan(out[:9]))
        assert np.all(np.isfinite(out[9:]))

    def test_ew_no_nan_warmup(self):
        """EwMean is well-defined from t=0, so EW vol/var classes
        return a value from sample 0."""
        _, h, l, _ = _ohlc(10, seed=0)
        out = EwParkinsonVar(span=10)(h, l)
        assert np.all(np.isfinite(out))


class TestEdgeCases:

    def test_flat_bar_parkinson_zero(self):
        """high == low => ln(1) == 0 => sigma2 == 0."""
        out = RollingParkinsonVar(5)(np.full(20, 100.0), np.full(20, 100.0))
        np.testing.assert_array_equal(out[4:], 0.0)

    def test_rogers_satchell_drift_immune(self):
        """RS should give roughly the same volatility for a trending
        market as for a flat one (its key advantage over GK/Parkinson).
        We construct two paths -- one drifting up, one flat in mean --
        with identical intraday H-L moves, and check the RS estimates
        differ by less than GK's."""
        n = 100
        rng = np.random.default_rng(1)
        intraday_range = np.abs(rng.normal(0, 1, n))
        # Path 1: heavy upward drift
        open1 = 100 + np.arange(n) * 1.0
        close1 = open1 + 0.5 * intraday_range
        high1 = np.maximum(open1, close1) + 0.1 * intraday_range
        low1 = np.minimum(open1, close1) - 0.1 * intraday_range
        # Path 2: no drift, same intraday structure
        open2 = np.full(n, 100.0)
        close2 = open2 + 0.5 * intraday_range
        high2 = np.maximum(open2, close2) + 0.1 * intraday_range
        low2 = np.minimum(open2, close2) - 0.1 * intraday_range

        rs1 = RollingRogersSatchellVar(20)(open1, high1, low1, close1)
        rs2 = RollingRogersSatchellVar(20)(open2, high2, low2, close2)
        gk1 = RollingGarmanKlassVar(20)(open1, high1, low1, close1)
        gk2 = RollingGarmanKlassVar(20)(open2, high2, low2, close2)
        rs_gap = np.nanmean(np.abs(rs1 - rs2))
        gk_gap = np.nanmean(np.abs(gk1 - gk2))
        # RS should be less sensitive to the drift than GK.
        assert rs_gap < gk_gap


# ---------------------------------------------------------------------------
# Reset / scalar-loop parity (one block per class via parametrize)
# ---------------------------------------------------------------------------

ALL_CLASSES = [
    ("RollingParkinsonVar",   RollingParkinsonVar,        20,    2),
    ("RollingParkinsonVol",   RollingParkinsonVol,        20,    2),
    ("EwParkinsonVar",        lambda: EwParkinsonVar(span=20),     None, 2),
    ("EwParkinsonVol",        lambda: EwParkinsonVol(span=20),     None, 2),
    ("RollingGarmanKlassVar", RollingGarmanKlassVar,      20,    4),
    ("RollingGarmanKlassVol", RollingGarmanKlassVol,      20,    4),
    ("EwGarmanKlassVar",      lambda: EwGarmanKlassVar(span=20),   None, 4),
    ("EwGarmanKlassVol",      lambda: EwGarmanKlassVol(span=20),   None, 4),
    ("RollingRogersSatchellVar", RollingRogersSatchellVar, 20,   4),
    ("RollingRogersSatchellVol", RollingRogersSatchellVol, 20,   4),
    ("EwRogersSatchellVar",   lambda: EwRogersSatchellVar(span=20), None, 4),
    ("EwRogersSatchellVol",   lambda: EwRogersSatchellVol(span=20), None, 4),
]


def _construct(cls, arg):
    if arg is None:
        return cls()        # lambda factory
    return cls(arg)


@pytest.mark.parametrize("name,cls,arg,n_inputs", ALL_CLASSES,
                         ids=[t[0] for t in ALL_CLASSES])
def test_reset_clears_history(name, cls, arg, n_inputs):
    o, h, l, c = _ohlc(40, seed=hash(name) & 0xFFFF)
    obj = _construct(cls, arg)
    args = (h, l) if n_inputs == 2 else (o, h, l, c)
    first = obj(*args)
    obj.reset()
    second = obj(*args)
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("name,cls,arg,n_inputs", ALL_CLASSES,
                         ids=[t[0] for t in ALL_CLASSES])
def test_scalar_loop_matches_array(name, cls, arg, n_inputs):
    o, h, l, c = _ohlc(30, seed=hash(name + "loop") & 0xFFFF)
    args = (h, l) if n_inputs == 2 else (o, h, l, c)
    obj = _construct(cls, arg)
    streamed = np.array([obj(*tup) for tup in zip(*args)])
    arr_result = _construct(cls, arg)(*args)
    np.testing.assert_allclose(streamed, arr_result, equal_nan=True, atol=1e-12)
