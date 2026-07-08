import numpy as np
from screamer import ExpandingSum, ExpandingSkew
from screamer.streams import resample


def test_functor_sum_equals_builtin_sum():
    x = np.arange(20.0); idx = np.arange(20, dtype=np.int64)
    a, ia = resample(x, idx, every=5, agg="sum")
    b, ib = resample(x, idx, every=5, agg=ExpandingSum())
    np.testing.assert_array_equal(np.asarray(ia), np.asarray(ib))
    np.testing.assert_allclose(np.asarray(a), np.asarray(b))


def test_functor_reducer_resets_each_bar():
    # skew over each bar independently; a functor reset per bar must not leak across bars
    bar1 = np.zeros(5)
    bar2 = np.arange(5.0)
    x = np.concatenate([bar1, bar2]); idx = np.arange(10, dtype=np.int64)
    vals, _ = resample(x, idx, every=5, agg=ExpandingSkew())
    assert vals.shape[0] == 2  # two bars, independent
    # Anchor per-bar values against ExpandingSkew run fresh on each bar in isolation.
    # Without reset(), bar2's reducer would see all 10 samples and yield ~1.258,
    # not the correct 0.0 for the symmetric [0,1,2,3,4] distribution.
    bar1_expected = np.asarray(ExpandingSkew()(bar1))[-1]  # NaN (zero-variance)
    bar2_expected = np.asarray(ExpandingSkew()(bar2))[-1]  # 0.0 (symmetric)
    np.testing.assert_array_equal(vals[0], bar1_expected)
    np.testing.assert_array_equal(vals[1], bar2_expected)
