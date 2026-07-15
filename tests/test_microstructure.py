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
    np.testing.assert_allclose(out[1:], [1.0, 1.0, -1.0, -1.0])


def test_tick_rule_sign_is_causal():
    from screamer.microstructure import TickRuleSign
    price = np.array([100.0, 101.0, 101.0, 99.0])
    full = TickRuleSign()(price)
    trunc = TickRuleSign()(price[:3])
    np.testing.assert_allclose(full[:3], trunc)
