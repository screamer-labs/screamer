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
