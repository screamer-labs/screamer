import types
import numpy as np
import pytest
from screamer import CumSum


def test_lazy_iterator_is_lazy_and_correct():
    # A generator input must yield a lazy iterator, not a list, and match batch.
    def gen():
        for x in [1.0, 2.0, 3.0, 4.0]:
            yield x

    out = CumSum()(gen())
    assert not isinstance(out, list)
    assert hasattr(out, "__next__")          # it is an iterator
    assert list(out) == [1.0, 3.0, 6.0, 10.0]

    # Laziness: the input generator is consumed one item at a time.
    pulled = []
    def spy():
        for x in [1.0, 2.0, 3.0]:
            pulled.append(x)
            yield x
    it = CumSum()(spy())
    assert pulled == []                      # nothing consumed yet
    next(it)
    assert pulled == [1.0]                    # exactly one pulled


def test_1i1o_batch_equals_lazy():
    from screamer import CumSum
    xs = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0]
    batch = CumSum()(np.array(xs))                 # array in, array out
    lazy = list(CumSum()(x for x in xs))           # generator in, lazy iterator out
    np.testing.assert_allclose(np.asarray(lazy), batch)


def test_empty_input_yields_empty():
    from screamer import CumSum
    assert list(CumSum()(x for x in [])) == []
