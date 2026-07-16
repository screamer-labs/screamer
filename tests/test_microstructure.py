import numpy as np
from screamer import OFI


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


def test_ofi_nan_and_empty_bucket():
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
    from screamer import TickRuleSign
    price = np.array([100.0, 101.0, 101.0, 100.5, 100.5])
    # up -> +1 ; unchanged -> carry +1 ; down -> -1 ; unchanged -> carry -1
    out = TickRuleSign()(price)
    assert np.isnan(out[0])   # warmup: no prior tick (Diff), nan_policy: ignore
    np.testing.assert_allclose(out[1:], [1.0, 1.0, -1.0, -1.0])


def test_tick_rule_sign_is_causal():
    from screamer import TickRuleSign
    price = np.array([100.0, 101.0, 101.0, 99.0])
    full = TickRuleSign()(price)
    trunc = TickRuleSign()(price[:3])
    np.testing.assert_allclose(full[:3], trunc)


def test_tick_rule_sign_nan_price_is_nan_not_carried():
    from screamer import TickRuleSign
    price = np.array([100.0, 101.0, np.nan, 100.5])
    out = TickRuleSign()(price)
    assert out[1] == 1.0            # up-tick
    assert np.isnan(out[2])         # missing price -> NaN, NOT the carried prior sign
    # nan_policy ignore: the NaN tick is skipped and state is left untouched, so
    # 100.5 is a down-tick from the last real price (101), giving -1.
    assert out[3] == -1.0


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
    from screamer import AmihudIlliquidity
    ret = np.array([0.01, -0.02, 0.015, -0.005, 0.02])
    notional = np.array([1e6, 2e6, 5e5, 1e6, 4e6])
    out = AmihudIlliquidity(window_size=3)(ret, notional)
    ref = RollingMean(3)(np.abs(ret) / notional)
    np.testing.assert_allclose(out, ref, equal_nan=True)


def test_operators_expose_reset():
    from screamer import OFI, TickRuleSign, AmihudIlliquidity
    from screamer.microstructure import SignedVolume, RollingKyleLambda, EwKyleLambda
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
    from screamer import AmihudIlliquidity
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
    from screamer import LeeReadySign
    price = np.array([100.0, 101.0, 101.0, 100.0])
    mid   = np.array([100.5, 100.5, 101.0, 100.5])
    # p<mid -> -1 ; p>mid -> +1 ; p==mid -> tick rule (101 vs prev 101 = unchanged -> carry +1) ; p<mid -> -1
    out = LeeReadySign()(price, mid)
    np.testing.assert_allclose(out, [-1.0, 1.0, 1.0, -1.0])


def test_lee_ready_sign_is_causal():
    from screamer import LeeReadySign
    price = np.array([100.0, 101.0, 100.5, 101.0]); mid = np.array([100.0, 100.0, 100.0, 100.0])
    full = LeeReadySign()(price, mid)
    trunc = LeeReadySign()(price[:3], mid[:3])
    np.testing.assert_allclose(full[:3], trunc, equal_nan=True)


def test_roll_spread_recovers_bounce_and_is_nan_when_undefined():
    from screamer import RollSpread
    # a clean +/-0.1 bid-ask bounce around 100 -> serial cov of price changes is
    # negative -> Roll spread is defined and positive
    price = 100.0 + 0.1 * np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    out = RollSpread(window_size=6)(price)
    assert np.isfinite(out[-1]) and out[-1] > 0.0
    # a monotonic ramp -> serial cov >= 0 -> undefined -> NaN
    ramp = np.arange(12.0)
    assert np.isnan(RollSpread(window_size=6)(ramp)[-1])


def test_bvc_is_normal_cdf_of_standardized_return():
    from screamer import RollingStd, Erf
    from screamer import BulkVolumeClassifier
    rng = np.random.default_rng(0)
    ret = rng.normal(scale=0.01, size=200)
    out = BulkVolumeClassifier(window_size=50)(ret)
    sigma = np.asarray(RollingStd(50)(ret))
    z = ret / sigma
    ref = 0.5 * (1.0 + np.asarray(Erf()(z / np.sqrt(2.0))))
    np.testing.assert_allclose(out, ref, equal_nan=True)
    assert np.nanmin(out) >= 0.0 and np.nanmax(out) <= 1.0   # a fraction


def test_hawkes_intensity_hand_recursion_and_stream_equals_batch():
    from screamer import HawkesIntensity
    x = np.array([1.0, 0.0, 0.0, 2.0, 0.0])
    # lam0=0 ; lam1=0.9*(0+1)=0.9 ; lam2=0.9*0.9=0.81 ; lam3=0.9*0.81=0.729 ;
    # lam4=0.9*(0.729+2)=2.4561
    batch = HawkesIntensity(decay=0.9, alpha=1.0, mu=0.0)(x)
    np.testing.assert_allclose(batch, [0.0, 0.9, 0.81, 0.729, 2.4561], atol=1e-9)
    op = HawkesIntensity(decay=0.9, alpha=1.0, mu=0.0)
    stream = np.array([op(float(v)) for v in x])   # one sample at a time
    np.testing.assert_allclose(batch, stream, equal_nan=True)


def test_hawkes_nan_does_not_poison_state():
    from screamer import HawkesIntensity
    x = np.array([1.0, np.nan, 1.0])
    out = HawkesIntensity(decay=0.5, alpha=1.0, mu=0.0)(x)
    assert np.isnan(out[1])           # NaN input -> NaN output
    assert np.isfinite(out[2])        # state recovered (not poisoned)


def test_hawkes_reset_restarts_state():
    from screamer import HawkesIntensity
    x = [1.0, 0.5, 2.0]
    op = HawkesIntensity(decay=0.8)
    a = [op(v) for v in x]; op.reset(); b = [op(v) for v in x]
    np.testing.assert_allclose(a, b)


def test_propagator_kernel_convolution_and_warmup():
    from screamer import Propagator
    flow = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    out = Propagator(window=3, g0=1.0, gamma=0.5)(flow)
    # G = [1, 2^-0.5, 3^-0.5] = [1, 0.70711, 0.57735]
    # warmup: t=0,1 -> NaN ; t=2: G0*f2+G1*f1+G2*f0 = 0.57735 ;
    # t=3: G0*f3+G1*f2+G2*f1 = 1.0 ; t=4: G2*f2? = G0*f4+G1*f3+G2*f2 = 0.70711 ;
    # t=5: G0*f5+G1*f4+G2*f3 = 0.57735
    assert np.isnan(out[0]) and np.isnan(out[1])
    np.testing.assert_allclose(out[2:], [0.57735, 1.0, 0.70711, 0.57735], atol=1e-4)


def test_propagator_stream_equals_batch():
    from screamer import Propagator
    rng = np.random.default_rng(3); flow = rng.normal(size=60)
    batch = Propagator(window=5)(flow)
    op = Propagator(window=5); stream = np.array([op(float(v)) for v in flow])
    np.testing.assert_allclose(batch, stream, equal_nan=True)


def test_propagator_reset_clears_buffer():
    from screamer import Propagator
    flow = [1.0, 2.0, 3.0, 4.0]
    op = Propagator(window=2)
    a = [op(v) for v in flow]; op.reset(); b = [op(v) for v in flow]
    np.testing.assert_allclose(a, b, equal_nan=True)


# --- VPIN (order-flow toxicity) ------------------------------------------------

def test_vpin_toxicity_warmup_and_transition():
    from screamer import VPIN
    # bucket_volume=10 so each trade fills exactly one bucket; n_buckets=3.
    buy = np.array([10.0, 10.0, 10.0, 5.0, 5.0, 5.0])
    sell = np.array([0.0, 0.0, 0.0, 5.0, 5.0, 5.0])
    out = VPIN(bucket_volume=10.0, n_buckets=3)(buy, sell)
    assert np.isnan(out[0]) and np.isnan(out[1])          # warmup: < 3 buckets closed
    # all-buy buckets -> imbalance 10/10 = 1.0, then balanced buckets pull it down
    np.testing.assert_allclose(out[2:], [1.0, 2 / 3, 1 / 3, 0.0])


def test_vpin_splits_trade_across_buckets():
    from screamer import VPIN
    # One trade of 30 volume with a fixed 5:1 buy:sell ratio, bucket_volume=10,
    # n_buckets=3: it fills exactly 3 buckets, each with imbalance |8.333-1.667|.
    out = VPIN(bucket_volume=10.0, n_buckets=3)(np.array([25.0]), np.array([5.0]))
    np.testing.assert_allclose(out, [(25 - 5) / 30])       # = 2/3, normalized imbalance


def test_vpin_is_in_unit_range_and_causal():
    from screamer import VPIN
    rng = np.random.default_rng(0)
    buy, sell = rng.exponential(1.0, 300), rng.exponential(1.0, 300)
    full = VPIN(bucket_volume=15.0, n_buckets=10)(buy, sell)
    finite = full[~np.isnan(full)]
    assert np.all((finite >= 0.0) & (finite <= 1.0))
    trunc = VPIN(bucket_volume=15.0, n_buckets=10)(buy[:150], sell[:150])
    np.testing.assert_allclose(full[:150], trunc, equal_nan=True)


def test_vpin_nan_input_leaves_state_untouched():
    from screamer import VPIN
    op = VPIN(bucket_volume=10.0, n_buckets=2)
    a = op(10.0, 0.0)              # closes bucket 1
    b = op(np.nan, 5.0)           # NaN -> NaN, no bucket change
    c = op(10.0, 0.0)             # closes bucket 2 -> first finite VPIN
    assert np.isnan(a) and np.isnan(b) and c == 1.0


def test_vpin_stream_equals_batch():
    from screamer import VPIN
    rng = np.random.default_rng(1)
    buy, sell = rng.exponential(1.0, 200), rng.exponential(1.0, 200)
    op = VPIN(bucket_volume=12.0, n_buckets=8)
    stream = np.array([op(float(bi), float(si)) for bi, si in zip(buy, sell)])
    batch = VPIN(bucket_volume=12.0, n_buckets=8)(buy, sell)
    np.testing.assert_allclose(np.nan_to_num(stream), np.nan_to_num(batch))


def test_vpin_reset_restarts():
    from screamer import VPIN
    buy, sell = np.array([10.0, 10.0, 10.0]), np.array([0.0, 0.0, 0.0])
    op = VPIN(bucket_volume=10.0, n_buckets=2)
    a = [op(float(b), float(s)) for b, s in zip(buy, sell)]
    op.reset()
    b = [op(float(b_), float(s_)) for b_, s_ in zip(buy, sell)]
    np.testing.assert_allclose(np.nan_to_num(a), np.nan_to_num(b))


# --- MicroPrice (imbalance-weighted mid) ---------------------------------------

def test_microprice_leans_toward_heavier_queue():
    from screamer import MicroPrice
    bid = np.array([100.0, 100.0, 100.0, 100.0])
    ask = np.array([101.0, 101.0, 101.0, 101.0])
    bid_size = np.array([9.0, 1.0, 5.0, 0.0])
    ask_size = np.array([1.0, 9.0, 5.0, 0.0])
    out = MicroPrice()(bid, ask, bid_size, ask_size)
    # heavy bid -> toward ask (100.9); heavy ask -> toward bid (100.1);
    # balanced and empty book -> plain mid (100.5)
    np.testing.assert_allclose(out, [100.9, 100.1, 100.5, 100.5])


def test_microprice_nan_input_is_nan():
    from screamer import MicroPrice
    out = MicroPrice()(np.array([100.0, np.nan]), np.array([101.0, 101.0]),
                       np.array([1.0, 1.0]), np.array([1.0, 1.0]))
    assert out[0] == 100.5 and np.isnan(out[1])


# --- QueueImbalance (L1 book imbalance = OFI synonym) --------------------------

def test_queue_imbalance_matches_ofi():
    from screamer.microstructure import QueueImbalance
    bid_size = np.array([9.0, 1.0, 5.0, 0.0])
    ask_size = np.array([1.0, 9.0, 5.0, 0.0])
    np.testing.assert_allclose(QueueImbalance()(bid_size, ask_size),
                               OFI()(bid_size, ask_size))
    np.testing.assert_allclose(QueueImbalance()(bid_size, ask_size),
                               [0.8, -0.8, 0.0, 0.0])
