"""An operator reads a float64, C-contiguous, aligned array argument in place
and copies any other one. Both readings give the same values."""
import numpy as np
import pytest

import screamer as s


def misaligned(values):
    """A float64 view of `values` at an odd byte offset of a byte buffer."""
    values = np.asarray(values, dtype=np.float64)
    buf = np.zeros(values.size * 8 + 1, dtype=np.uint8)
    view = np.frombuffer(buf, dtype=np.float64, count=values.size, offset=1).reshape(values.shape)
    assert view.ctypes.data % 8 != 0 and view.flags["C_CONTIGUOUS"]
    buf[1:] = np.frombuffer(values.tobytes(), dtype=np.uint8)
    return view


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


rng = np.random.default_rng(11)
X = np.cumsum(rng.standard_normal(997)) + 100.0
X[rng.random(X.size) < 0.03] = np.nan
H = X + rng.random(X.size)
L = X - rng.random(X.size)


@pytest.mark.parametrize("op", [s.EwMean(span=5), s.Abs(), s.RollingMean(7), s.RollingMax(9),
                                s.BollingerBands(10)])
def test_one_input_misaligned_reads_like_aligned(op):
    assert same(op(misaligned(X)), op(X.copy()))
    W = X[:990].reshape(99, 10)
    assert same(op(misaligned(W)), op(W.copy()))


@pytest.mark.parametrize("op, fields", [(s.WilliamsR(20), (H, L, X)), (s.Sub(), (H, L)),
                                        (s.AmihudIlliquidity(5), (X - 100, H)),
                                        (s.RollingCorr(15), (H, L))])
def test_multi_input_misaligned_reads_like_aligned(op, fields):
    ref = op(*[f.copy() for f in fields])
    assert same(op(*[misaligned(f) for f in fields]), ref)
    assert same(op(misaligned(np.stack(fields, -1))), ref)


@pytest.mark.parametrize("window", [2, 5, 14, 20, 50, 200])
def test_williams_r_packed_and_strided_fields_read_like_separate(window):
    op = s.WilliamsR(window)
    clean = [np.nan_to_num(f, nan=100.0) for f in (H, L, X)]
    for fields in (clean, (H, L, X)):
        ref = op(*fields)
        assert same(op(np.stack(fields, -1)), ref)                          # (T, 3), stride 3
        assert same(op(np.stack(fields + fields[:1], -1)[:, :3]), ref)       # stride 4 view
        assert same(op(np.asfortranarray(np.stack(fields, -1))), ref)
        W = [f[:990].reshape(99, 10) for f in fields]
        assert same(op(np.stack(W, -1)), op(*W))                            # (T, K, 3)
