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


def misaligned_any(values, dtype):
    """A `dtype` view of `values` at an odd byte offset of a byte buffer."""
    values = np.asarray(values).astype(dtype)
    size = np.dtype(dtype).itemsize
    buf = np.zeros(values.size * size + 1, dtype=np.uint8)
    view = np.frombuffer(buf, dtype=dtype, count=values.size, offset=1).reshape(values.shape)
    buf[1:] = np.frombuffer(values.tobytes(), dtype=np.uint8)
    return view


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.float16, np.int64, np.int32,
                                   np.int16, np.uint64, np.uint32, np.uint16])
@pytest.mark.parametrize("op", [s.EwMean(span=5), s.Abs(), s.RollingMean(7), s.RollingMax(9),
                                s.BollingerBands(10)])
def test_misaligned_strided_view_reads_like_its_copy(op, dtype):
    """A non-contiguous view at an odd address is copied element by element
    without a typed load at that address (G33 b)."""
    base = np.nan_to_num(X, nan=3.0) if np.dtype(dtype).kind in "iu" else X
    view = misaligned_any(base, dtype)[::2]
    assert view.ctypes.data % view.itemsize != 0 or view.itemsize == 1
    assert not view.flags["C_CONTIGUOUS"]
    assert same(op(view), op(np.ascontiguousarray(view)))
    W = misaligned_any(base[:990].reshape(99, 10), dtype)[:, ::3]
    assert same(op(W), op(np.ascontiguousarray(W)))


@pytest.mark.parametrize("op, fields", [(s.WilliamsR(20), (H, L, X)), (s.Sub(), (H, L)),
                                        (s.RollingCorr(15), (H, L))])
def test_multi_input_misaligned_strided_reads_like_its_copy(op, fields):
    views = [misaligned(f)[::3] for f in fields]
    ref = op(*[np.ascontiguousarray(v) for v in views])
    assert same(op(*views), ref)
    packed = misaligned(np.stack(fields, -1))[::3]
    assert same(op(packed), ref)
