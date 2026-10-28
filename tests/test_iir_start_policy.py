"""start_policy on the IIR filters: "steady" (default) and "zero".

"zero" starts from an all-zero state and equals scipy.signal.lfilter.
"steady" sets the state, on the first non-NaN input x0, to the steady state
for a constant input x0, which is lfilter(b, a, x, zi=lfilter_zi(b, a) * x0).
"""
import numpy as np
import pytest
from scipy.signal import butter, lfilter, lfilter_zi

import screamer
from tests.regime_helpers import assert_regimes_agree
from screamer import (
    Butter, ButterBandpass, ButterBandstop, ButterHighpass,
    Decycler, RoofingFilter, SuperSmoother,
)


def _supersmoother_ba(period):
    a1 = np.exp(-np.sqrt(2) * np.pi / period)
    b1 = 2 * a1 * np.cos(np.sqrt(2) * np.pi / period)
    c2, c3 = b1, -a1 * a1
    c1 = 1 - c2 - c3
    return np.array([c1 / 2, c1 / 2, 0.0]), np.array([1.0, -c2, -c3])


def _decycler_ba(period):
    w = 2 * np.pi / period
    alpha = (np.cos(w) + np.sin(w) - 1) / np.cos(w)
    return np.array([alpha / 2, alpha / 2]), np.array([1.0, -(1 - alpha)])


def _roofing_ba(hp_period, lp_period):
    w = 0.707 * 2 * np.pi / hp_period
    alpha = (np.cos(w) + np.sin(w) - 1) / np.cos(w)
    g, om = 1 - alpha / 2, 1 - alpha
    b_hp = np.array([g * g, -2 * g * g, g * g])
    a_hp = np.array([1.0, -2 * om, om * om])
    b_lp, a_lp = _supersmoother_ba(lp_period)
    # The cascade is one linear filter; its steady state for a constant input
    # is the cascade of both stages' steady states.
    return np.convolve(b_hp, b_lp), np.convolve(a_hp, a_lp)


# name, factory(**kw), (b, a), kind ("low" passes a constant, "high" removes it)
CASES = [
    ("Butter", lambda **kw: Butter(3, 0.1, **kw), butter(3, 0.1, btype="low"), "low"),
    ("ButterHighpass", lambda **kw: ButterHighpass(2, 0.1, **kw),
     butter(2, 0.1, btype="high"), "high"),
    ("ButterBandpass", lambda **kw: ButterBandpass(2, 0.05, 0.2, **kw),
     butter(2, [0.05, 0.2], btype="band"), "high"),
    ("ButterBandstop", lambda **kw: ButterBandstop(2, 0.05, 0.2, **kw),
     butter(2, [0.05, 0.2], btype="bandstop"), "low"),
    ("SuperSmoother", lambda **kw: SuperSmoother(period=10.0, **kw),
     _supersmoother_ba(10.0), "low"),
    ("Decycler", lambda **kw: Decycler(period=20.0, **kw), _decycler_ba(20.0), "low"),
    ("RoofingFilter", lambda **kw: RoofingFilter(hp_period=48.0, lp_period=10.0, **kw),
     _roofing_ba(48.0, 10.0), "high"),
]
IDS = [c[0] for c in CASES]


def _series(n=300, seed=0, level=100.0):
    rng = np.random.default_rng(seed)
    return level + np.cumsum(rng.standard_normal(n))


def _steady_ref(b, a, x):
    """lfilter from the steady state at the first non-NaN value, NaN skipped."""
    x = np.asarray(x, float)
    out = np.full_like(x, np.nan)
    ok = ~np.isnan(x)
    xf = x[ok]
    out[ok] = lfilter(b, a, xf, zi=lfilter_zi(b, a) * xf[0])[0]
    return out


def _zero_ref(b, a, x):
    x = np.asarray(x, float)
    out = np.full_like(x, np.nan)
    ok = ~np.isnan(x)
    out[ok] = lfilter(b, a, x[ok])
    return out


@pytest.mark.parametrize("name, make, ba, kind", CASES, ids=IDS)
class TestStartPolicy:
    def test_zero_matches_lfilter(self, name, make, ba, kind):
        x = _series()
        b, a = ba
        np.testing.assert_allclose(np.asarray(make(start_policy="zero")(x)),
                                   lfilter(b, a, x), rtol=1e-9, atol=1e-9)

    def test_steady_matches_lfilter_zi(self, name, make, ba, kind):
        x = _series()
        b, a = ba
        np.testing.assert_allclose(np.asarray(make(start_policy="steady")(x)),
                                   _steady_ref(b, a, x), rtol=1e-9, atol=1e-9)

    def test_default_is_steady(self, name, make, ba, kind):
        x = _series(seed=3)
        np.testing.assert_array_equal(np.asarray(make()(x)),
                                      np.asarray(make(start_policy="steady")(x)))

    def test_constant_input(self, name, make, ba, kind):
        x = np.full(50, 42.5)
        y = np.asarray(make()(x))
        expected = x if kind == "low" else np.zeros_like(x)
        np.testing.assert_allclose(y, expected, rtol=1e-9, atol=1e-9)

    def test_leading_nans(self, name, make, ba, kind):
        x = _series(seed=5)
        x[:4] = np.nan
        x[20] = np.nan
        b, a = ba
        y = np.asarray(make()(x))
        assert np.isnan(y[:4]).all() and np.isnan(y[20])
        np.testing.assert_allclose(y, _steady_ref(b, a, x), rtol=1e-9, atol=1e-9,
                                   equal_nan=True)
        z = np.asarray(make(start_policy="zero")(x))
        np.testing.assert_allclose(z, _zero_ref(b, a, x), rtol=1e-9, atol=1e-9,
                                   equal_nan=True)

    @pytest.mark.parametrize("policy", ["steady", "zero"])
    def test_batch_equals_stream(self, name, make, ba, kind, policy):
        x = _series(seed=7)
        x[:3] = np.nan
        x[50] = np.nan
        batch = np.asarray(make(start_policy=policy)(x))
        op = make(start_policy=policy)
        stream = np.array([op(float(v)) for v in x])
        np.testing.assert_array_equal(batch, stream)
        # Eager, per-event, Pipeline and lazy runs agree bit for bit.
        assert_regimes_agree(lambda: make(start_policy=policy), x)

    def test_reset_restarts(self, name, make, ba, kind):
        x = _series(seed=9)
        op = make()
        first = np.asarray(op(x))
        op(_series(seed=11, level=-50.0))
        op.reset()
        np.testing.assert_array_equal(np.asarray(op(x)), first)

    def test_unknown_policy_raises(self, name, make, ba, kind):
        with pytest.raises(ValueError, match=name):
            make(start_policy="bogus")


@pytest.mark.parametrize("make, ba", [
    # Order 9: ten coefficients, the general (non-unrolled) path.
    (lambda: Butter(9, 0.1), butter(9, 0.1, btype="low")),
    # Order 4 band-pass: nine coefficients, the largest unrolled path.
    (lambda: ButterBandpass(4, 0.1, 0.3), butter(4, [0.1, 0.3], btype="band")),
], ids=["Butter9", "ButterBandpass4"])
def test_steady_high_order(make, ba):
    x = _series(seed=13)
    x[:2] = np.nan
    b, a = ba
    # A high-order transfer function is badly conditioned, so the tolerance
    # is looser than for the low-order cases.
    np.testing.assert_allclose(np.asarray(make()(x)), _steady_ref(b, a, x),
                               rtol=1e-7, atol=1e-7, equal_nan=True)


def test_two_dimensional_columns_start_independently():
    x = np.column_stack([_series(seed=1, level=10.0), _series(seed=2, level=-30.0)])
    x[:5, 1] = np.nan
    y = np.asarray(Butter(2, 0.1)(x))
    b, a = butter(2, 0.1)
    for k in range(2):
        np.testing.assert_allclose(y[:, k], _steady_ref(b, a, x[:, k]),
                                   rtol=1e-9, atol=1e-9, equal_nan=True)


def test_start_policy_in_pipeline():
    x = _series(seed=4)
    node = screamer.Input("x")
    p = screamer.Pipeline(inputs=[node], outputs=[SuperSmoother(period=10.0)(node)])
    values, _index = p(x)
    np.testing.assert_array_equal(np.asarray(values),
                                  np.asarray(SuperSmoother(period=10.0)(x)))
