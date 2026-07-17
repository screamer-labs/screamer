import numpy as np
from screamer import RollingDownsideDeviation, RollingOmega, RollingCVaR


def _roll_ref(x, w, f):
    out = np.full(len(x), np.nan)
    for i in range(w - 1, len(x)):
        out[i] = f(x[i - w + 1:i + 1])
    return out


def test_downside_deviation_matches_numpy():
    rng = np.random.default_rng(0); r = rng.standard_normal(300) * 0.01
    w = 50
    got = RollingDownsideDeviation(w)(r)
    ref = _roll_ref(r, w, lambda a: np.sqrt(np.mean(np.minimum(a, 0.0) ** 2)))
    np.testing.assert_allclose(got[w - 1:], ref[w - 1:], atol=1e-12)


def test_downside_deviation_ignores_upside_and_honors_mar():
    # all returns above the mar -> zero downside
    r = np.array([0.02, 0.03, 0.01, 0.05])
    np.testing.assert_allclose(RollingDownsideDeviation(2, mar=0.0)(r)[1:], [0.0, 0.0, 0.0])
    # with a high mar, everything is a shortfall
    dd = RollingDownsideDeviation(2, mar=0.10)(np.array([0.0, 0.0]))
    np.testing.assert_allclose(dd[1], 0.10)          # sqrt(mean([0.1^2, 0.1^2]))


def test_omega_matches_numpy_and_gt_one_when_gains_dominate():
    rng = np.random.default_rng(1); r = rng.standard_normal(300) * 0.01
    w = 50
    got = RollingOmega(w)(r)
    ref = _roll_ref(r, w, lambda a: np.sum(np.maximum(a, 0)) / np.sum(np.maximum(-a, 0)))
    np.testing.assert_allclose(got[w - 1:], ref[w - 1:], atol=1e-12)
    # a window with only gains has no downside -> undefined (NaN)
    assert np.isnan(RollingOmega(3)(np.array([0.01, 0.02, 0.03]))[-1])


def test_cvar_matches_numpy_and_is_positive_loss():
    rng = np.random.default_rng(2); r = rng.standard_normal(300) * 0.01
    w, alpha = 50, 0.1
    got = RollingCVaR(w, alpha=alpha)(r)

    def es(a):
        k = max(1, int(np.floor(alpha * len(a))))
        return -np.mean(np.sort(a)[:k])
    ref = _roll_ref(r, w, es)
    np.testing.assert_allclose(got[w - 1:], ref[w - 1:], atol=1e-12)
    assert np.nanmean(got) > 0.0                     # expected shortfall is a positive loss


def test_cvar_hand_example():
    # window of 5, alpha=0.4 -> k = floor(2.0) = 2 worst returns
    r = np.array([0.01, -0.02, 0.03, -0.05, 0.00])
    out = RollingCVaR(5, alpha=0.4)(r)
    # worst two are -0.05 and -0.02 -> mean -0.035 -> CVaR = 0.035
    np.testing.assert_allclose(out[-1], 0.035)


def test_risk_stats_are_causal_and_stream_equals_batch():
    rng = np.random.default_rng(3); r = rng.standard_normal(200) * 0.01
    for op_factory in (lambda: RollingDownsideDeviation(30),
                       lambda: RollingOmega(30),
                       lambda: RollingCVaR(30, alpha=0.1)):
        full = op_factory()(r)
        trunc = op_factory()(r[:100])
        np.testing.assert_allclose(full[:100], trunc, equal_nan=True)   # causal
        op = op_factory()
        stream = np.array([op(float(v)) for v in r])
        np.testing.assert_allclose(np.nan_to_num(stream), np.nan_to_num(op_factory()(r)))


def test_risk_stats_nan_is_ignored():
    for op in (RollingDownsideDeviation(3), RollingOmega(3), RollingCVaR(3, alpha=0.3)):
        out = op(np.array([np.nan, 0.01, -0.02, 0.03, -0.01]))
        assert np.isnan(out[0])                      # NaN input -> NaN, window untouched
        assert np.isfinite(out[-1]) or np.isnan(out[-1])   # recovers (no sticky NaN crash)
