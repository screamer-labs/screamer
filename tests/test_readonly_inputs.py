"""Read-only numpy arrays are valid inputs.

pandas returns read-only views from ``Series.to_numpy()`` under copy-on-write,
so a read-only array is the common case for pandas users, not an edge case.
"""
import inspect

import numpy as np
import pytest

import screamer


def _readonly(a):
    a = np.array(a, dtype=float)
    a.flags.writeable = False
    return a


def _ops():
    for name in sorted(dir(screamer)):
        obj = getattr(screamer, name)
        if inspect.isclass(obj):
            try:
                op = obj()
            except Exception:
                continue
            if callable(op):
                yield name


@pytest.mark.parametrize("name", list(_ops()))
def test_readonly_input_matches_writable(name):
    rng = np.random.default_rng(0)
    base = 100 + np.cumsum(rng.standard_normal(64))
    for k in range(1, 11):                      # find the op's arity
        op = getattr(screamer, name)()
        try:
            expected = op(*[base.copy() for _ in range(k)])
        except Exception:
            continue
        op = getattr(screamer, name)()
        got = op(*[_readonly(base) for _ in range(k)])
        if isinstance(expected, tuple):
            for e, g in zip(expected, got):
                np.testing.assert_array_equal(np.asarray(e), np.asarray(g))
        else:
            np.testing.assert_array_equal(np.asarray(expected), np.asarray(got))
        return
    pytest.skip("no array arity accepted")


def test_pandas_to_numpy_input():
    pd = pytest.importorskip("pandas")
    s = pd.Series(np.arange(10.0))
    out = screamer.RollingMean(3)(s.to_numpy())
    np.testing.assert_allclose(out[-1], 8.0)
