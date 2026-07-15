import numpy as np
from screamer.microstructure import OFI


def test_ofi_normalized_imbalance():
    buy = np.array([3.0, 0.0, 5.0, 0.0])
    sell = np.array([1.0, 4.0, 5.0, 0.0])
    out = OFI()(buy, sell)
    # (3-1)/4=0.5 ; (0-4)/4=-1 ; (5-5)/10=0 ; empty bucket -> 0
    np.testing.assert_allclose(out, [0.5, -1.0, 0.0, 0.0])


def test_ofi_is_causal():
    buy = np.array([3.0, 2.0, 5.0]); sell = np.array([1.0, 2.0, 1.0])
    full = OFI()(buy, sell)
    trunc = OFI()(buy[:2], sell[:2])
    np.testing.assert_allclose(full[:2], trunc)   # future rows do not change past values


def test_ofi_propagates_nan_but_zeros_empty_bucket():
    buy = np.array([np.nan, 0.0, 2.0])
    sell = np.array([1.0, 0.0, 2.0])
    out = OFI()(buy, sell)
    assert np.isnan(out[0])          # NaN input -> NaN output (nan_policy: ignore)
    assert out[1] == 0.0             # empty bucket -> 0.0, not NaN
    assert out[2] == 0.0


def test_signed_volume_elementwise():
    from screamer.microstructure import SignedVolume
    sign = np.array([1.0, -1.0, 1.0]); vol = np.array([2.0, 5.0, 3.0])
    np.testing.assert_allclose(SignedVolume()(sign, vol), [2.0, -5.0, 3.0])


def test_tick_rule_sign_carries_on_unchanged():
    from screamer.microstructure import TickRuleSign
    price = np.array([100.0, 101.0, 101.0, 100.5, 100.5])
    # up -> +1 ; unchanged -> carry +1 ; down -> -1 ; unchanged -> carry -1
    out = TickRuleSign()(price)
    assert np.isnan(out[0])   # warmup: no prior tick (Diff), nan_policy: ignore
    np.testing.assert_allclose(out[1:], [1.0, 1.0, -1.0, -1.0])


def test_tick_rule_sign_is_causal():
    from screamer.microstructure import TickRuleSign
    price = np.array([100.0, 101.0, 101.0, 99.0])
    full = TickRuleSign()(price)
    trunc = TickRuleSign()(price[:3])
    np.testing.assert_allclose(full[:3], trunc)


def test_tick_rule_sign_nan_price_is_nan_not_carried():
    from screamer.microstructure import TickRuleSign
    price = np.array([100.0, 101.0, np.nan, 100.5])
    out = TickRuleSign()(price)
    assert out[1] == 1.0            # up-tick
    assert np.isnan(out[2])         # missing price -> NaN, NOT the carried prior sign
    # out[3]: Diff(price[3], price[2]) = 100.5 - NaN = NaN, so Ffill carries +1 from index 1
    assert out[3] == 1.0


def test_signed_volume_propagates_nan():
    from screamer.microstructure import SignedVolume
    out = SignedVolume()(np.array([1.0, np.nan]), np.array([np.nan, 3.0]))
    assert np.isnan(out).all()


def test_kyle_lambda_equals_rolling_beta():
    from screamer import RollingBeta
    from screamer.microstructure import RollingKyleLambda
    rng = np.random.default_rng(0)
    flow = rng.normal(size=200)
    ret = 2.5 * flow + rng.normal(scale=0.1, size=200)   # true impact slope 2.5
    lam = RollingKyleLambda(window_size=50)(flow, ret)
    ref = RollingBeta(50)(ret, flow)                     # slope of ret on flow
    np.testing.assert_allclose(lam, ref, equal_nan=True)
    # recovers the true slope once the window is full
    assert abs(np.nanmedian(lam) - 2.5) < 0.2


def test_ew_kyle_lambda_equals_ew_beta():
    from screamer import EwBeta
    from screamer.microstructure import EwKyleLambda
    rng = np.random.default_rng(1)
    flow = rng.normal(size=200); ret = 1.5 * flow + rng.normal(scale=0.1, size=200)
    np.testing.assert_allclose(EwKyleLambda(span=30.0)(flow, ret),
                               EwBeta(span=30.0)(ret, flow), equal_nan=True)


def test_amihud_matches_rolling_mean_of_ratio():
    from screamer import RollingMean
    from screamer.microstructure import AmihudIlliquidity
    ret = np.array([0.01, -0.02, 0.015, -0.005, 0.02])
    notional = np.array([1e6, 2e6, 5e5, 1e6, 4e6])
    out = AmihudIlliquidity(window_size=3)(ret, notional)
    ref = RollingMean(3)(np.abs(ret) / notional)
    np.testing.assert_allclose(out, ref, equal_nan=True)


def test_operators_expose_reset():
    from screamer.microstructure import (OFI, SignedVolume, TickRuleSign,
        RollingKyleLambda, EwKyleLambda, AmihudIlliquidity)
    for op in [OFI(), SignedVolume(), TickRuleSign(), RollingKyleLambda(),
               EwKyleLambda(), AmihudIlliquidity()]:
        op.reset()   # must exist and be callable


def test_reset_restarts_streaming_state():
    # scalar-stream, reset, scalar-stream again -> identical to a fresh instance
    from screamer.microstructure import RollingKyleLambda
    flow = [0.5, -0.3, 0.8, -0.1, 0.4, 0.2]; ret = [1.0, -0.6, 1.7, -0.2, 0.9, 0.5]
    op = RollingKyleLambda(window_size=3)
    a = [op(float(f), float(r)) for f, r in zip(flow, ret)]
    op.reset()
    b = [op(float(f), float(r)) for f, r in zip(flow, ret)]
    np.testing.assert_allclose(np.asarray(a, float), np.asarray(b, float), equal_nan=True)


def test_amihud_zero_notional_is_nan_not_inf():
    from screamer.microstructure import AmihudIlliquidity
    ret = np.array([0.01, 0.02, 0.03, 0.01]); notional = np.array([1e6, 0.0, 1e6, 1e6])
    out = AmihudIlliquidity(window_size=2)(ret, notional)
    assert not np.isinf(out).any()   # zero notional must not produce inf


def test_rolling_order_imbalance_equals_rolling_sum():
    from screamer import RollingSum
    from screamer.microstructure import RollingOrderImbalance
    flow = np.array([1.0, -2.0, 3.0, -1.0, 2.0])
    np.testing.assert_allclose(RollingOrderImbalance(window_size=3)(flow),
                               RollingSum(3)(flow), equal_nan=True)


def test_lee_ready_sign_uses_mid_then_tick_fallback():
    from screamer.microstructure import LeeReadySign
    price = np.array([100.0, 101.0, 101.0, 100.0])
    mid   = np.array([100.5, 100.5, 101.0, 100.5])
    # p<mid -> -1 ; p>mid -> +1 ; p==mid -> tick rule (101 vs prev 101 = unchanged -> carry +1) ; p<mid -> -1
    out = LeeReadySign()(price, mid)
    np.testing.assert_allclose(out, [-1.0, 1.0, 1.0, -1.0])


def test_lee_ready_sign_is_causal():
    from screamer.microstructure import LeeReadySign
    price = np.array([100.0, 101.0, 100.5, 101.0]); mid = np.array([100.0, 100.0, 100.0, 100.0])
    full = LeeReadySign()(price, mid)
    trunc = LeeReadySign()(price[:3], mid[:3])
    np.testing.assert_allclose(full[:3], trunc, equal_nan=True)


def test_bvc_is_normal_cdf_of_standardized_return():
    from screamer import RollingStd, Erf
    from screamer.microstructure import BulkVolumeClassifier
    rng = np.random.default_rng(0)
    ret = rng.normal(scale=0.01, size=200)
    out = BulkVolumeClassifier(window_size=50)(ret)
    sigma = np.asarray(RollingStd(50)(ret))
    z = ret / sigma
    ref = 0.5 * (1.0 + np.asarray(Erf()(z / np.sqrt(2.0))))
    np.testing.assert_allclose(out, ref, equal_nan=True)
    assert np.nanmin(out) >= 0.0 and np.nanmax(out) <= 1.0   # a fraction
