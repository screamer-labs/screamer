"""
Cross-validation against independent third-party libraries.

Why this file exists: pandas alone has the same EMA / variance / sum
conventions as screamer, so a bug shared with pandas would not be
caught by tests that only compare against pandas. TA-Lib (the
canonical C-based industry library) and pandas-ta-classic implement
the same indicators independently, with different code, and often
with different conventions. Cross-validating against them detects
both genuine bugs and convention-mismatch surprises that real users
will hit.

The full set of agreements and documented divergences is in
docs/conventions.md. This test file is the ground-truth check that
those claims still hold.

Both libraries are imported via pytest.importorskip, so this whole
file skips cleanly when they are not installed (TA-Lib in particular
requires its C library to be installed first via the OS package
manager).
"""
import numpy as np
import pandas as pd
import pytest

talib = pytest.importorskip("talib", reason="TA-Lib python wrapper not installed")
pta = pytest.importorskip(
    "pandas_ta_classic", reason="pandas-ta-classic not installed"
)

import screamer as sc


@pytest.fixture(scope="module")
def random_series():
    rng = np.random.default_rng(0)
    return rng.standard_normal(200)


# ---------------------------------------------------------------------------
# Indicators that align with TA-Lib / pandas-ta exactly (post-warmup)
# ---------------------------------------------------------------------------

class TestExactAlignment:
    """Each method validates one indicator against an independent
    reference. Tolerance is ~1e-12 -- these are bit-identical
    implementations modulo IEEE-754 rounding."""

    def test_wma_matches_talib(self, random_series):
        n = 14
        ours = sc.WMA(n)(random_series)
        ref = talib.WMA(random_series, timeperiod=n)
        np.testing.assert_allclose(ours[n - 1:], ref[n - 1:], atol=1e-12)

    def test_wma_matches_pandas_ta(self, random_series):
        n = 14
        ours = sc.WMA(n)(random_series)
        ref = np.asarray(pta.wma(pd.Series(random_series), length=n))
        np.testing.assert_allclose(ours[n - 1:], ref[n - 1:], atol=1e-12)

    def test_trima_matches_talib(self, random_series):
        n = 10
        ours = sc.TRIMA(n)(random_series)
        ref = talib.TRIMA(random_series, timeperiod=n)
        np.testing.assert_allclose(ours[n - 1:], ref[n - 1:], atol=1e-12)

    def test_trima_matches_pandas_ta(self, random_series):
        n = 10
        ours = sc.TRIMA(n)(random_series)
        ref = np.asarray(pta.trima(pd.Series(random_series), length=n))
        np.testing.assert_allclose(ours[n - 1:], ref[n - 1:], atol=1e-12)

    def test_rolling_mean_matches_talib_sma(self, random_series):
        n = 14
        ours = sc.RollingMean(n)(random_series)
        ref = talib.SMA(random_series, timeperiod=n)
        np.testing.assert_allclose(ours[n - 1:], ref[n - 1:], atol=1e-12)

    def test_rolling_min_matches_talib(self, random_series):
        n = 12
        ours = sc.RollingMin(n)(random_series)
        ref = talib.MIN(random_series, timeperiod=n)
        # No NaN warmup on our side; TA-Lib has NaN for t < n-1.
        np.testing.assert_array_equal(ours[n - 1:], ref[n - 1:])

    def test_rolling_max_matches_talib(self, random_series):
        n = 12
        ours = sc.RollingMax(n)(random_series)
        ref = talib.MAX(random_series, timeperiod=n)
        np.testing.assert_array_equal(ours[n - 1:], ref[n - 1:])

    def test_rolling_argmin_matches_talib_minindex(self, random_series):
        """TA-Lib MININDEX returns the absolute sample index of the min;
        we return the offset within the current window. Convert one
        into the other and compare."""
        n = 12
        ours_offset = sc.RollingArgmin(n)(random_series)
        talib_abs = talib.MININDEX(random_series, timeperiod=n)
        ts = np.arange(len(random_series))
        # talib_abs = our_offset + (t - n + 1)  =>  our_offset = talib_abs - (t - n + 1)
        talib_offset = talib_abs - (ts - n + 1)
        np.testing.assert_array_equal(ours_offset[n - 1:], talib_offset[n - 1:])

    def test_rolling_argmax_matches_talib_maxindex(self, random_series):
        n = 12
        ours_offset = sc.RollingArgmax(n)(random_series)
        talib_abs = talib.MAXINDEX(random_series, timeperiod=n)
        ts = np.arange(len(random_series))
        talib_offset = talib_abs - (ts - n + 1)
        np.testing.assert_array_equal(ours_offset[n - 1:], talib_offset[n - 1:])

    def test_rolling_median_matches_pandas_ta(self, random_series):
        n = 11
        ours = sc.RollingMedian(n)(random_series)
        ref = np.asarray(pta.median(pd.Series(random_series), length=n))
        np.testing.assert_allclose(ours[n - 1:], ref[n - 1:], atol=1e-12)

    def test_hull_ma_matches_pandas_ta(self, random_series):
        """TA-Lib does not have Hull MA; pandas-ta-classic is the only
        independent reference."""
        n = 16
        ours = sc.HullMA(n)(random_series)
        ref = np.asarray(pta.hma(pd.Series(random_series), length=n))
        n_sqrt = int(np.sqrt(n))
        warmup = n + n_sqrt - 1
        np.testing.assert_allclose(ours[warmup - 1:], ref[warmup - 1:], atol=1e-12)

    def test_bollinger_middle_band_matches_talib(self, random_series):
        """The middle band is an SMA; matches TA-Lib exactly. The upper
        and lower bands have a documented divergence (different ddof)
        -- see TestDocumentedDivergences below."""
        n = 14
        bb = sc.BollingerBands(n, num_std=2.0)(random_series)
        _, mid_ref, _ = talib.BBANDS(random_series, timeperiod=n,
                                      nbdevup=2, nbdevdn=2, matype=0)
        np.testing.assert_allclose(bb[n - 1:, 1], mid_ref[n - 1:], atol=1e-12)

    def test_rolling_rsi_default_matches_talib_wilder(self, random_series):
        """Default RollingRSI uses Wilder's smoothing, matching TA-Lib's
        RSI and pandas-ta-classic's rsi (both use Wilder)."""
        n = 14
        ours = sc.RollingRSI(n)(random_series)
        ref = talib.RSI(random_series, timeperiod=n)
        mask = ~(np.isnan(ours) | np.isnan(ref))
        np.testing.assert_allclose(ours[mask], ref[mask], atol=1e-10)

    def test_rolling_rsi_default_matches_pandas_ta(self, random_series):
        n = 14
        ours = sc.RollingRSI(n)(random_series)
        ref = np.asarray(pta.rsi(pd.Series(random_series), length=n))
        mask = ~(np.isnan(ours) | np.isnan(ref))
        np.testing.assert_allclose(ours[mask], ref[mask], atol=1e-10)

    def test_kama_matches_talib(self, random_series):
        """KAMA's smoothing-constant formula and warmup are unambiguous
        across the major implementations -- this should match TA-Lib
        bit-for-bit."""
        n = 30
        x = np.cumsum(random_series)  # KAMA is meaningful on walks
        ours = sc.KAMA(n)(x)
        ref = talib.KAMA(x, timeperiod=n)
        mask = ~(np.isnan(ours) | np.isnan(ref))
        np.testing.assert_allclose(ours[mask], ref[mask], atol=1e-10)

    def test_kama_matches_pandas_ta(self, random_series):
        """pandas-ta-classic emits one sample earlier than TA-Lib
        (it seeds differently at the boundary), but post both-valid
        the values are bit-equivalent."""
        n = 30
        x = np.cumsum(random_series)
        ours = sc.KAMA(n)(x)
        ref = np.asarray(pta.kama(pd.Series(x), length=n))
        mask = ~(np.isnan(ours) | np.isnan(ref))
        np.testing.assert_allclose(ours[mask], ref[mask], atol=1e-10)

    def test_williams_r_matches_talib(self, random_series):
        """WilliamsR has no smoothing / seeding subtleties -- bit-exact
        match to TA-Lib's WILLR post-warmup."""
        rng = np.random.default_rng(0)
        n_samples = len(random_series)
        close = 100 + np.cumsum(random_series)
        high = close + np.abs(rng.normal(0, 0.5, n_samples))
        low = close - np.abs(rng.normal(0, 0.5, n_samples))
        period = 14
        ours = sc.WilliamsR(period)(high, low, close)
        ref = talib.WILLR(high, low, close, timeperiod=period)
        mask = ~(np.isnan(ours) | np.isnan(ref))
        np.testing.assert_allclose(ours[mask], ref[mask], atol=1e-10)


# ---------------------------------------------------------------------------
# Documented divergences (asserts the divergence is in the EXPECTED
# direction so future drift -- in either screamer or the third-party
# library -- trips the test).
# ---------------------------------------------------------------------------

class TestDocumentedDivergences:

    def test_dema_matches_pandas_adjust_true_not_talib(self, random_series):
        """screamer.DEMA = 2*EwMean - EwMean(EwMean), and EwMean is
        pandas's adjust=True form (bias-corrected). TA-Lib's DEMA is
        built on its recursive SMA-seeded EMA, so post-warmup the
        two converge but never become bit-equal during the input.
        Document both: we MATCH pandas adjust=True, and we DIFFER
        from TA-Lib by a measurable but bounded amount."""
        n = 10
        x = random_series
        s = pd.Series(x)
        ours = sc.DEMA(span=n)(x)

        # We match pandas adjust=True manual composition exactly.
        e1 = s.ewm(span=n, adjust=True).mean()
        e2 = e1.ewm(span=n, adjust=True).mean()
        ref_ours = (2 * e1 - e2).to_numpy()
        np.testing.assert_allclose(ours, ref_ours, atol=1e-12)

        # We deliberately DIFFER from TA-Lib. Assert the gap is non-trivial
        # near the warmup edge but bounded -- if either implementation
        # changes convention this test will fail loudly.
        ref_talib = talib.DEMA(x, timeperiod=n)
        mask = ~np.isnan(ref_talib)
        diff = np.abs(ours[mask] - ref_talib[mask])
        assert diff.max() > 1e-3, (
            "DEMA vs TA-Lib divergence is unexpectedly small -- "
            "did one of the libraries change EMA convention?"
        )
        assert diff.max() < 1.0, (
            "DEMA vs TA-Lib divergence is unexpectedly large -- "
            "investigate."
        )

    def test_macd_matches_pandas_adjust_true_not_talib(self, random_series):
        """screamer.MACD is composed from three EwMean = pandas adjust=True
        EMAs. We pin the pandas-composition reference bit-exact and assert
        the TA-Lib gap is in the expected ballpark (TA-Lib uses adjust=False
        with an SMA seed)."""
        x = random_series
        s = pd.Series(x)
        fast, slow, signal = 12, 26, 9
        out = sc.MACD(fast, slow, signal)(x)

        # We match pandas adjust=True composition exactly.
        ema_fast = s.ewm(span=fast, adjust=True).mean()
        ema_slow = s.ewm(span=slow, adjust=True).mean()
        macd_ref = (ema_fast - ema_slow).to_numpy()
        signal_ref = (ema_fast - ema_slow).ewm(span=signal, adjust=True).mean().to_numpy()
        np.testing.assert_allclose(out[:, 0], macd_ref, atol=1e-12)
        np.testing.assert_allclose(out[:, 1], signal_ref, atol=1e-12)

        # We deliberately differ from TA-Lib (same EMA-convention reason
        # as DEMA / TEMA). Assert the gap is bounded.
        macd_t, sig_t, _ = talib.MACD(x, fastperiod=fast, slowperiod=slow,
                                       signalperiod=signal)
        mask = ~(np.isnan(macd_t) | np.isnan(sig_t))
        macd_diff = np.abs(out[mask, 0] - macd_t[mask])
        sig_diff = np.abs(out[mask, 1] - sig_t[mask])
        assert macd_diff.max() > 1e-3, (
            "MACD vs TA-Lib unexpectedly close -- did one side change "
            "its EMA convention?"
        )
        assert macd_diff.max() < 5.0
        assert sig_diff.max() < 5.0

    def test_tema_matches_pandas_adjust_true_not_talib(self, random_series):
        n = 10
        x = random_series
        s = pd.Series(x)
        ours = sc.TEMA(span=n)(x)

        e1 = s.ewm(span=n, adjust=True).mean()
        e2 = e1.ewm(span=n, adjust=True).mean()
        e3 = e2.ewm(span=n, adjust=True).mean()
        ref_ours = (3 * e1 - 3 * e2 + e3).to_numpy()
        np.testing.assert_allclose(ours, ref_ours, atol=1e-12)

        ref_talib = talib.TEMA(x, timeperiod=n)
        mask = ~np.isnan(ref_talib)
        diff = np.abs(ours[mask] - ref_talib[mask])
        assert diff.max() > 1e-3
        assert diff.max() < 1.0

    def test_rolling_std_matches_pandas_ddof1_not_pta_ddof0(self, random_series):
        """We follow pandas's ddof=1 (sample std). pandas-ta-classic
        uses ddof=0 (population std). The two are related by a scale
        factor of sqrt(n / (n-1))."""
        n = 14
        x = random_series
        s = pd.Series(x)
        ours = sc.RollingStd(n)(x)

        # Match pandas ddof=1 exactly.
        ref_pandas = s.rolling(n).std(ddof=1).to_numpy()
        np.testing.assert_allclose(ours[n - 1:], ref_pandas[n - 1:], atol=1e-12)

        # And pandas-ta-classic stdev, by virtue of using ddof=0,
        # disagrees by exactly the bias-correction scale factor.
        ref_pta = np.asarray(pta.stdev(s, length=n))
        scale = np.sqrt(n / (n - 1))
        np.testing.assert_allclose(
            ours[n - 1:], ref_pta[n - 1:] * scale,
            atol=1e-12,
        )

    def test_bollinger_band_width_uses_ddof1(self, random_series):
        """Bollinger upper/lower differ from TA-Lib by exactly the same
        ddof factor as RollingStd. The middle band (SMA) matches; the
        bandwidth carries the std convention."""
        n = 14
        x = random_series
        bb = sc.BollingerBands(n, num_std=2.0)(x)
        upper_t, _, lower_t = talib.BBANDS(x, timeperiod=n,
                                            nbdevup=2, nbdevdn=2, matype=0)
        # Half-width difference equals 2 * (sigma_ddof1 - sigma_ddof0)
        # (with num_std=2). Check the ratio is constant after warmup.
        half_ours = (bb[n - 1:, 2] - bb[n - 1:, 0]) / 2.0
        half_talib = (upper_t[n - 1:] - lower_t[n - 1:]) / 2.0
        scale = np.sqrt(n / (n - 1))
        np.testing.assert_allclose(half_ours, half_talib * scale, atol=1e-12)

    def test_rolling_rsi_cutler_mode_diverges_from_wilder_as_expected(
        self, random_series
    ):
        """The opt-in Cutler RSI (method="cutler") uses SMA smoothing of
        gains and losses. TA-Lib uses Wilder's smoothing (the default
        for screamer too). The two are both correct and differ by a
        few RSI points; this test pins the divergence in the expected
        ballpark so a regression in either side fires."""
        n = 14
        x = random_series
        cutler = sc.RollingRSI(n, method="cutler")(x)
        wilder_ref = talib.RSI(x, timeperiod=n)
        mask = ~(np.isnan(cutler) | np.isnan(wilder_ref))
        diff = np.abs(cutler[mask] - wilder_ref[mask])
        assert diff.max() > 0.5, (
            "Cutler vs Wilder divergence is unexpectedly small -- "
            "did one of the smoothing conventions change?"
        )
        assert diff.max() < 30.0


# ---------------------------------------------------------------------------
# A summary smoke test: print which alignments hold in the current build.
# Useful for regenerating docs/conventions.md when something changes.
# ---------------------------------------------------------------------------

def test_summary_print(random_series, capsys):
    """Not really a test: prints the alignment table for inspection.
    Always passes; rerun manually with -s to see the numbers."""
    x = random_series
    s = pd.Series(x)
    n = 14

    pairs = [
        ("WMA vs TA-Lib WMA",            sc.WMA(n)(x),           talib.WMA(x, n)),
        ("TRIMA vs TA-Lib TRIMA",         sc.TRIMA(n)(x),         talib.TRIMA(x, n)),
        ("RollingMean vs TA-Lib SMA",     sc.RollingMean(n)(x),   talib.SMA(x, n)),
        ("HullMA vs pta.hma",             sc.HullMA(16)(x),       np.asarray(pta.hma(s, length=16))),
        ("DEMA vs TA-Lib (divergent)",    sc.DEMA(span=n)(x),     talib.DEMA(x, n)),
        ("TEMA vs TA-Lib (divergent)",    sc.TEMA(span=n)(x),     talib.TEMA(x, n)),
        ("RollingStd vs pta.stdev (ddof)",sc.RollingStd(n)(x),    np.asarray(pta.stdev(s, length=n))),
        ("RollingRSI default (Wilder) vs TA-Lib", sc.RollingRSI(n)(x), talib.RSI(x, n)),
        ("RollingRSI cutler mode vs TA-Lib (divergent)", sc.RollingRSI(n, method="cutler")(x), talib.RSI(x, n)),
        ("KAMA vs TA-Lib",                   sc.KAMA(30)(np.cumsum(x)), talib.KAMA(np.cumsum(x), 30)),
        ("MACD macd  vs TA-Lib (divergent)", sc.MACD()(x)[:, 0], talib.MACD(x, 12, 26, 9)[0]),
        ("MACD signal vs TA-Lib (divergent)", sc.MACD()(x)[:, 1], talib.MACD(x, 12, 26, 9)[1]),
    ]
    # WilliamsR needs HLC inputs; build a self-consistent triple from x.
    rng_hlc = np.random.default_rng(99)
    close_hlc = 100 + np.cumsum(x)
    high_hlc = close_hlc + np.abs(rng_hlc.normal(0, 0.5, len(x)))
    low_hlc = close_hlc - np.abs(rng_hlc.normal(0, 0.5, len(x)))
    pairs.append((
        "WilliamsR vs TA-Lib WILLR",
        sc.WilliamsR(14)(high_hlc, low_hlc, close_hlc),
        talib.WILLR(high_hlc, low_hlc, close_hlc, timeperiod=14),
    ))
    print()
    print(f"{'comparison':45s}  max_abs_diff (post-warmup)")
    print("-" * 72)
    for name, a, b in pairs:
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        mask = ~(np.isnan(a) | np.isnan(b))
        diff = np.max(np.abs(a[mask] - b[mask])) if mask.any() else float("nan")
        print(f"{name:45s}  {diff:.3e}")
