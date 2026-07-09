"""
Tests for the 1-input / M>1-output rolling indicators.

Two indicators exercise the FunctorBase<_, 1, M> dispatch path:

  RollingMinMax  (M=2)  - rolling minimum and maximum together
  BollingerBands (M=3)  - rolling (lower, mid, upper) bands

The contract the dispatcher promises (see docs/polymorphic_api.md):

  * scalar input -> Python tuple of M floats
  * 1-D array input of shape (T,) -> numpy array of shape (T, M)
  * 2-D array input of shape (T, K) -> numpy array of shape (T, K, M)
  * iterable -> list of M-tuples (eager)
  * the (T, K, M) result with column k bit-exactly equals the 1-D
    (T, M) result on a single column

Plus the algorithm-specific correctness checks (pandas parity for
BollingerBands, brute-force window check for RollingMinMax).
"""
import numpy as np
import pandas as pd
import pytest

from screamer import RollingMinMax, BollingerBands


# ---------------------------------------------------------------------------
# RollingMinMax (M = 2)
# ---------------------------------------------------------------------------

class TestRollingMinMax:

    @pytest.mark.parametrize("window", [3, 5, 10])
    def test_matches_brute_force_min_max(self, window):
        rng = np.random.default_rng(window)
        x = rng.standard_normal(60)
        out = RollingMinMax(window)(x)
        assert out.shape == (60, 2)

        # Brute-force reference: for each i, min/max of x[max(0, i-w+1):i+1]
        for i in range(len(x)):
            lo = max(0, i - window + 1)
            ref_min = x[lo : i + 1].min()
            ref_max = x[lo : i + 1].max()
            assert out[i, 0] == ref_min, f"min at i={i}"
            assert out[i, 1] == ref_max, f"max at i={i}"

    def test_scalar_input_returns_tuple(self):
        mm = RollingMinMax(window_size=3)
        r = mm(5.0)
        assert isinstance(r, tuple) and len(r) == 2
        assert r == (5.0, 5.0)

        r = mm(2.0)
        assert r == (2.0, 5.0)

        r = mm(7.0)
        assert r == (2.0, 7.0)

    def test_1d_output_shape(self):
        x = np.arange(20, dtype=float)
        out = RollingMinMax(5)(x)
        assert out.shape == (20, 2)

    def test_2d_pairs_columnwise(self):
        rng = np.random.default_rng(0)
        T, K, w = 40, 3, 5
        X = rng.standard_normal((T, K))
        out_2d = RollingMinMax(w)(X)
        assert out_2d.shape == (T, K, 2)
        for k in range(K):
            ref = RollingMinMax(w)(X[:, k].copy())
            np.testing.assert_array_equal(out_2d[:, k, :], ref)

    def test_2d_strided_view_matches_contig(self):
        rng = np.random.default_rng(1)
        big = rng.standard_normal((50, 9))
        view = big[::2, ::3]
        rc = RollingMinMax(window_size=5)
        np.testing.assert_array_equal(rc(view), rc(view.copy()))

    def test_iterable_returns_lazy_iterator_of_tuples(self):
        out = RollingMinMax(3)(iter([3.0, 1.0, 4.0, 1.0, 5.0]))
        assert hasattr(out, "__next__") and not isinstance(out, list)
        rows = list(out)
        assert len(rows) == 5
        assert all(isinstance(p, tuple) and len(p) == 2 for p in rows)

    def test_window_size_must_be_positive(self):
        with pytest.raises(ValueError):
            RollingMinMax(window_size=0)


# ---------------------------------------------------------------------------
# BollingerBands (M = 3)
# ---------------------------------------------------------------------------

class TestBollingerBands:

    @pytest.mark.parametrize("window,num_std", [
        (10, 1.0), (20, 2.0), (30, 2.5), (5, 0.5),
    ])
    def test_matches_pandas(self, window, num_std):
        rng = np.random.default_rng(window + int(num_std * 10))
        prices = np.cumsum(rng.standard_normal(150)) + 100.0
        out = BollingerBands(window, num_std=num_std)(prices)
        assert out.shape == (150, 3)

        s = pd.Series(prices)
        mid = s.rolling(window).mean().to_numpy()
        std = s.rolling(window).std().to_numpy()  # ddof=1 by default
        lower_ref = mid - num_std * std
        upper_ref = mid + num_std * std

        np.testing.assert_allclose(out[:, 0], lower_ref, equal_nan=True, atol=1e-10)
        np.testing.assert_allclose(out[:, 1], mid,       equal_nan=True, atol=1e-10)
        np.testing.assert_allclose(out[:, 2], upper_ref, equal_nan=True, atol=1e-10)

    def test_scalar_input_returns_tuple(self):
        bb = BollingerBands(window_size=3, num_std=2.0)
        r1 = bb(1.0)
        assert isinstance(r1, tuple) and len(r1) == 3
        assert all(np.isnan(v) for v in r1)  # warmup
        bb(2.0)
        r3 = bb(3.0)
        # After 3 samples the bands are well defined.
        lower, mid, upper = r3
        assert mid == pytest.approx(2.0)        # mean of 1,2,3
        # std (ddof=1) of [1,2,3] = 1.0; bands at 2 +/- 2.
        assert lower == pytest.approx(0.0)
        assert upper == pytest.approx(4.0)

    def test_2d_output_shape(self):
        X = np.random.default_rng(0).standard_normal((50, 4))
        out = BollingerBands(window_size=10)(X)
        assert out.shape == (50, 4, 3)

    def test_2d_pairs_columnwise(self):
        rng = np.random.default_rng(0)
        T, K = 40, 3
        X = rng.standard_normal((T, K)) + 100.0
        out_2d = BollingerBands(10, num_std=2.0)(X)
        for k in range(K):
            ref = BollingerBands(10, num_std=2.0)(X[:, k].copy())
            np.testing.assert_array_equal(out_2d[:, k, :], ref)

    def test_iterable_returns_lazy_iterator_of_tuples(self):
        out = BollingerBands(3, num_std=1.0)(iter([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert hasattr(out, "__next__") and not isinstance(out, list)
        rows = list(out)
        assert len(rows) == 5
        assert all(isinstance(t, tuple) and len(t) == 3 for t in rows)

    def test_window_size_must_be_at_least_two(self):
        with pytest.raises(ValueError):
            BollingerBands(window_size=1)
        BollingerBands(window_size=2)

    def test_num_std_validation(self):
        with pytest.raises(ValueError):
            BollingerBands(window_size=10, num_std=-1.0)
        with pytest.raises(ValueError):
            BollingerBands(window_size=10, num_std=float("nan"))
        # zero is allowed (degenerate but valid: lower == mid == upper)
        BollingerBands(window_size=10, num_std=0.0)


# ---------------------------------------------------------------------------
# Cross-class invariants
# ---------------------------------------------------------------------------

class TestDispatchInvariants:

    @pytest.mark.parametrize("cls,args", [
        (RollingMinMax, (5,)),
        (BollingerBands, (5,)),
    ])
    def test_streaming_matches_array_path(self, cls, args):
        """Scalar loop produces the same per-step values as the array path."""
        rng = np.random.default_rng(2)
        x = rng.standard_normal(40)
        expected = cls(*args)(x)         # shape (40, M)

        streaming = cls(*args)
        out = np.array([streaming(v) for v in x])  # shape (40, M)
        np.testing.assert_array_equal(out, expected)

    @pytest.mark.parametrize("cls,args,M", [
        (RollingMinMax, (5,), 2),
        (BollingerBands, (5,), 3),
    ])
    def test_3d_input_appends_M_axis(self, cls, args, M):
        rng = np.random.default_rng(3)
        X = rng.standard_normal((20, 3, 2))
        out = cls(*args)(X)
        assert out.shape == (20, 3, 2, M)
