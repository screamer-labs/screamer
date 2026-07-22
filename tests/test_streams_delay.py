import numpy as np
import pytest
from screamer import Delay


def test_delay_shifts_index_leaves_values():
    vals = np.array([1.0, 2.0, 3.0])
    idx = np.array([0, 7, 14], dtype=np.int64)
    v, i = Delay(5)(vals, idx)
    np.testing.assert_array_equal(v, vals)           # values unchanged
    np.testing.assert_array_equal(i, [5, 12, 19])    # index + duration


def test_delay_requires_explicit_index():
    with pytest.raises(TypeError):
        Delay(5)(np.array([1.0, 2.0, 3.0]))          # no index -> error


def test_delay_on_regular_grid_matches_index_shift():
    vals = np.arange(10.0)
    idx = np.arange(10, dtype=np.int64) * 100         # 100-unit grid
    v, i = Delay(300)(vals, idx)                       # 3-step delay
    np.testing.assert_array_equal(i, idx + 300)
    np.testing.assert_array_equal(v, vals)
