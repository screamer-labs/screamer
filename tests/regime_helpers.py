import numpy as np


def as_batch(op_factory, *arrays):
    """Run an op on concrete arrays (eager regime). op_factory() -> a fresh op."""
    return np.asarray(op_factory()(*arrays))


def as_scalar_loop(op_factory, *arrays):
    """Feed the op one row at a time (the functor scalar-loop / lazy analogue)."""
    op = op_factory()
    n = len(arrays[0])
    out = [op(*[np.asarray(a, float)[i] for a in arrays]) for i in range(n)]
    return np.asarray(out, float)


def assert_batch_equals_scalar(op_factory, *arrays):
    """Crown-jewel: array call == row-by-row call, NaN-aware."""
    b = as_batch(op_factory, *arrays)
    s = as_scalar_loop(op_factory, *arrays)
    np.testing.assert_allclose(b, s, equal_nan=True)


def assert_bitwise(a, b):
    """Equal bit for bit, NaN positions equal (NaN payloads may differ)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape
    na, nb = np.isnan(a), np.isnan(b)
    assert np.array_equal(na, nb)
    assert np.array_equal(a[~na].view(np.uint64), b[~nb].view(np.uint64))
