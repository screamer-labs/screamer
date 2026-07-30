"""The range-based volatility estimators recover a known sigma.

The baseline comparison in `tests/test_baselines.py` checks these four against a
reference, and for Parkinson and Garman-Klass that reference is QuantLib, an
independent implementation of the formula. Rogers-Satchell and Yang-Zhang are
not in QuantLib, so their references are transcriptions of the documented
formula: if a documented constant were wrong, operator and reference would be
wrong together and the comparison would still pass. That is how `EwSkew` and
`EwKurt` stayed broken.

This file anchors all four against ground truth instead. It simulates a
geometric Brownian motion at fine intra-bar resolution, aggregates it into OHLC
bars, and requires each estimator to recover the sigma that generated the path,
in the regime its page claims to handle:

    | estimator       | no drift | with drift | overnight gaps |
    |-----------------|----------|------------|----------------|
    | Parkinson       | yes      | no         | no             |
    | Garman-Klass    | yes      | no         | no             |
    | Rogers-Satchell | yes      | yes        | no             |
    | Yang-Zhang      | yes      | yes        | yes            |

The negative entries are asserted too. An estimator that silently became
drift-robust, or lost it, fails here.

Sampling a continuous path at finite resolution misses the true extremes, so
every estimate runs a few percent low. The tolerances below allow for that and
are still far tighter than any of the failure modes they guard against.
"""
import numpy as np
import pytest

from screamer import (
    RollingGarmanKlassVar,
    RollingParkinsonVar,
    RollingRogersSatchellVar,
    RollingYangZhangVar,
)

WINDOW = 20
SIGMA = 0.02          # per-bar intraday volatility of the simulated path
N_BARS = 1500
STEPS = 500           # intra-bar resolution


def simulate_bars(sigma=SIGMA, drift=0.0, gap_sigma=0.0, seed=1):
    """OHLC bars from a GBM path sampled `STEPS` times per bar.

    `drift` is the log drift per bar, `gap_sigma` the standard deviation of an
    overnight jump applied between bars.
    """
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift / STEPS, sigma / np.sqrt(STEPS), (N_BARS, STEPS))
    if gap_sigma:
        steps[:, 0] += rng.normal(0.0, gap_sigma, N_BARS)

    path = np.cumsum(steps.ravel()).reshape(N_BARS, STEPS)
    log_open, log_close = path[:, 0], path[:, -1]
    log_high, log_low = path.max(axis=1), path.min(axis=1)
    return (np.exp(log_open), np.exp(log_high), np.exp(log_low), np.exp(log_close))


def _sigma_of(variance_series):
    return float(np.sqrt(np.nanmean(variance_series)))


def estimates(o, h, l, c):
    return {
        "parkinson": _sigma_of(RollingParkinsonVar(WINDOW)(h, l)),
        "garman_klass": _sigma_of(RollingGarmanKlassVar(WINDOW)(o, h, l, c)),
        "rogers_satchell": _sigma_of(RollingRogersSatchellVar(WINDOW)(o, h, l, c)),
        "yang_zhang": _sigma_of(RollingYangZhangVar(WINDOW)(o, h, l, c)),
    }


@pytest.mark.parametrize(
    "name", ["parkinson", "garman_klass", "rogers_satchell", "yang_zhang"]
)
def test_recovers_sigma_without_drift_or_gaps(name):
    """The regime every one of them is designed for."""
    got = estimates(*simulate_bars())[name]
    assert got == pytest.approx(SIGMA, rel=0.10), (
        f"{name} returned {got:.5f} for a path generated with sigma {SIGMA}"
    )


def test_rogers_satchell_is_drift_robust_and_the_others_are_not():
    """Drift inflates a range, and only Rogers-Satchell corrects for it.

    At a drift of 5x sigma per bar the difference is not subtle: Parkinson
    reads about 3x the true value and Garman-Klass about 2x, while
    Rogers-Satchell stays near sigma. This is the property that page claims.
    """
    got = estimates(*simulate_bars(drift=5 * SIGMA))

    assert got["rogers_satchell"] == pytest.approx(SIGMA, rel=0.20), (
        f"Rogers-Satchell is meant to be drift-robust but returned "
        f"{got['rogers_satchell']:.5f} against a true {SIGMA}"
    )
    for biased in ("parkinson", "garman_klass"):
        assert got[biased] > 1.5 * SIGMA, (
            f"{biased} returned {got[biased]:.5f} under heavy drift. It is not "
            "drift-robust, so it should be inflated; if it no longer is, the "
            "estimator changed."
        )


def test_yang_zhang_captures_overnight_gaps_and_the_others_do_not():
    """Yang-Zhang measures total volatility, gap included.

    With an overnight jump of the same size as the intraday move, the total
    per-bar volatility is sqrt(2) * sigma. Yang-Zhang should report that; the
    other three see only the intraday part and therefore understate it.
    """
    o, h, l, c = simulate_bars(gap_sigma=SIGMA)
    got = estimates(o, h, l, c)
    total = SIGMA * np.sqrt(2.0)

    assert got["yang_zhang"] == pytest.approx(total, rel=0.10), (
        f"Yang-Zhang returned {got['yang_zhang']:.5f}, expected about "
        f"{total:.5f} for intraday and overnight volatility of {SIGMA} each"
    )
    for intraday_only in ("parkinson", "garman_klass", "rogers_satchell"):
        assert got[intraday_only] < 0.8 * total, (
            f"{intraday_only} returned {got[intraday_only]:.5f}, close to the "
            f"gap-inclusive {total:.5f}. It cannot see overnight gaps, so it "
            "should report only the intraday component."
        )


def test_vol_is_the_square_root_of_var():
    """The Vol and Var forms are the same estimator."""
    from screamer import (
        RollingGarmanKlassVol, RollingParkinsonVol,
        RollingRogersSatchellVol, RollingYangZhangVol,
    )

    o, h, l, c = simulate_bars()
    pairs = [
        (RollingParkinsonVar(WINDOW)(h, l), RollingParkinsonVol(WINDOW)(h, l)),
        (RollingGarmanKlassVar(WINDOW)(o, h, l, c), RollingGarmanKlassVol(WINDOW)(o, h, l, c)),
        (RollingRogersSatchellVar(WINDOW)(o, h, l, c), RollingRogersSatchellVol(WINDOW)(o, h, l, c)),
        (RollingYangZhangVar(WINDOW)(o, h, l, c), RollingYangZhangVol(WINDOW)(o, h, l, c)),
    ]
    for var, vol in pairs:
        np.testing.assert_allclose(np.sqrt(var), vol, equal_nan=True, rtol=1e-12)


# ---------------------------------------------------------------------------
# The exponentially weighted twins
# ---------------------------------------------------------------------------
# Same per-bar kernels, an EW mean in place of a rolling one, so they must
# recover the same sigma in the same regimes. Their references share the
# QuantLib kernel (Parkinson, Garman-Klass) or are transcriptions
# (Rogers-Satchell), so the transcribed one needs this anchor for the same
# reason the rolling form does.

SPAN = 50


def ew_estimates(o, h, l, c):
    from screamer import (
        EwGarmanKlassVar, EwParkinsonVar, EwRogersSatchellVar,
    )
    return {
        "ew_parkinson": _sigma_of(EwParkinsonVar(span=SPAN)(h, l)),
        "ew_garman_klass": _sigma_of(EwGarmanKlassVar(span=SPAN)(o, h, l, c)),
        "ew_rogers_satchell": _sigma_of(EwRogersSatchellVar(span=SPAN)(o, h, l, c)),
    }


@pytest.mark.parametrize(
    "name", ["ew_parkinson", "ew_garman_klass", "ew_rogers_satchell"]
)
def test_ew_forms_recover_sigma(name):
    got = ew_estimates(*simulate_bars())[name]
    assert got == pytest.approx(SIGMA, rel=0.10), (
        f"{name} returned {got:.5f} for a path generated with sigma {SIGMA}"
    )


def test_ew_rogers_satchell_is_drift_robust():
    """The EW form inherits the drift robustness of the per-bar kernel."""
    got = ew_estimates(*simulate_bars(drift=5 * SIGMA))
    assert got["ew_rogers_satchell"] == pytest.approx(SIGMA, rel=0.20), (
        f"EwRogersSatchell returned {got['ew_rogers_satchell']:.5f} under heavy "
        f"drift, against a true {SIGMA}"
    )
    for biased in ("ew_parkinson", "ew_garman_klass"):
        assert got[biased] > 1.5 * SIGMA, (
            f"{biased} returned {got[biased]:.5f} under heavy drift; it is not "
            "drift-robust and should be inflated"
        )


def test_ew_vol_is_the_square_root_of_ew_var():
    from screamer import (
        EwGarmanKlassVar, EwGarmanKlassVol, EwParkinsonVar, EwParkinsonVol,
        EwRogersSatchellVar, EwRogersSatchellVol,
    )
    o, h, l, c = simulate_bars()
    pairs = [
        (EwParkinsonVar(span=SPAN)(h, l), EwParkinsonVol(span=SPAN)(h, l)),
        (EwGarmanKlassVar(span=SPAN)(o, h, l, c), EwGarmanKlassVol(span=SPAN)(o, h, l, c)),
        (EwRogersSatchellVar(span=SPAN)(o, h, l, c), EwRogersSatchellVol(span=SPAN)(o, h, l, c)),
    ]
    for var, vol in pairs:
        np.testing.assert_allclose(np.sqrt(var), vol, equal_nan=True, rtol=1e-12)
