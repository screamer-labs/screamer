"""G32: every (values, index) entry point checks that the index has one value
per row before it reads either array: ValueError naming both lengths.

2.8.0 read past the end of a shorter index (garbage indexes or a false
"index decreases" error) and ignored the extra values of a longer one.
"""
import numpy as np
import pytest

import screamer as s

T = 10
V = np.arange(T, dtype=float)
W = np.arange(2 * T, dtype=float).reshape(T, 2)
SHORT, LONG = np.arange(5), np.arange(15)


def _lag():
    x = s.Input("x")
    return s.Pipeline([x], [s.Lag(1)(x)])


def _two():
    x, y = s.Input("x"), s.Input("y")
    return s.Pipeline([x, y], [s.CombineLatest()(x, y)])


CASES = {
    "Pipeline": ("Pipeline: input 0", lambda i: _lag()((V, i))),
    "Pipeline wide": ("Pipeline: input 0", lambda i: _lag()((W, i))),
    "Pipeline second input": ("Pipeline: input 1", lambda i: _two()((V, np.arange(T)), (V, i))),
    "Merge": ("merge: input 0", lambda i: s.Merge()((V, i), (V, np.arange(T)))),
    "Merge wide": ("merge: input 1", lambda i: s.Merge()((W, np.arange(T)), (W, i))),
    "CombineLatest": ("combine_latest: input 0", lambda i: s.CombineLatest()((V, i), (V, np.arange(T)))),
    "Resample freq": ("resample: input 0", lambda i: s.Resample(freq=2)((V, i))),
    "Resample count": ("resample: input 0", lambda i: s.Resample(count=2)((V, i))),
    "Resample clock": ("resample: input 0",
                       lambda i: s.Resample((V, i), (np.ones(3), np.arange(3)), clock=True)),
    "Delay": ("delay: input 0", lambda i: s.Delay(2)((V, i))),
    "Dropna": ("dropna: input 0", lambda i: s.Dropna()((V, i))),
    "Select": ("select: input 0", lambda i: s.Select([1])((W, i))),
    "split": ("split: the stream", lambda i: s.split(V, np.zeros(T, int), index=i)),
    "forecast_pairs X": ("forecast_pairs: input 0",
                         lambda i: s.forecast_pairs((V, i), (V, np.arange(T)), duration=2)),
    "forecast_pairs y": ("forecast_pairs: input 1",
                         lambda i: s.forecast_pairs((V, np.arange(T)), (V, i), duration=2)),
}


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("index", [SHORT, LONG], ids=["shorter", "longer"])
def test_an_index_of_another_length_raises_naming_both(case, index):
    who, call = CASES[case]
    with pytest.raises(ValueError) as e:
        call(index)
    assert str(e.value) == f"{who} has {T} rows of values but {len(index)} index values"


def test_a_slice_of_a_longer_buffer_is_its_own_length():
    big = np.arange(100)
    with pytest.raises(ValueError, match="has 10 rows of values but 5 index values"):
        _lag()((V, big[:5]))
    values, idx = _lag()((V, big[:10]))
    np.testing.assert_array_equal(idx, np.arange(10))
