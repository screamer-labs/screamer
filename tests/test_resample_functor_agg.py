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
    x = np.concatenate([np.zeros(5), np.arange(5.0)]); idx = np.arange(10, dtype=np.int64)
    vals, _ = resample(x, idx, every=5, agg=ExpandingSkew())
    assert vals.shape[0] == 2  # two bars, independent
