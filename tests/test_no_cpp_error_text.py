"""No raw C++ exception text (std::bad_cast, "vector", nanobind's conversion
errors) reaches the user from a stream entry point: a malformed index or
value is a ValueError or TypeError that names the problem (G27 review)."""
import numpy as np
import pytest

import screamer as s

V = np.arange(3.0)
W = np.zeros((3, 2))
I = np.arange(3)
BAD_INDEXES = {
    "2-D": np.array([[0], [1], [2]]),
    "3-D": np.zeros((3, 1, 1), dtype=np.int64),
    "0-d": np.array(5),
    "text": np.array(["a", "b", "c"]),
    "object": np.array([0, "b", 2], dtype=object),
    "complex": np.array([0, 1, 2], dtype=complex),
}


def _pipe():
    x = s.Input("x")
    return s.Pipeline([x], [s.Lag(1)(x)])


ENTRY = {
    "Merge": lambda i: s.Merge()((V, I), (V, i)),
    "Merge wide": lambda i: s.Merge()((W, i), (W, I)),
    "CombineLatest": lambda i: s.CombineLatest()((V, I), (V, i)),
    "CombineLatest wide": lambda i: s.CombineLatest()((W, i), (W, I)),
    "Pipeline": lambda i: _pipe()((V, i)),
    "Resample": lambda i: s.Resample(freq=2)((V, i)),
    "Delay": lambda i: s.Delay(1)((V, i)),
    "Dropna": lambda i: s.Dropna()((V, i)),
    "Select": lambda i: s.Select([0])((W, i)),
    "split": lambda i: s.split(V, np.zeros(3, dtype=int), index=i),
    "forecast_pairs": lambda i: s.forecast_pairs((V, i), (V, I), duration=1),
}


@pytest.mark.parametrize("bad", sorted(BAD_INDEXES))
@pytest.mark.parametrize("entry", sorted(ENTRY))
def test_a_malformed_index_raises_a_clean_error(entry, bad):
    with pytest.raises((ValueError, TypeError)) as e:
        ENTRY[entry](BAD_INDEXES[bad])
    text = str(e.value)
    assert "std::" not in text and "nanobind" not in text and text != "vector", (entry, bad, text)


@pytest.mark.parametrize("call", [
    lambda: s.Merge()((np.array(["a"]), [0]), (V, I)),
    lambda: s.Merge()(5.0, 6.0),
    lambda: list(s.Merge()(iter([(np.zeros((2, 2)), [0, 1])]), iter([(1.0, 1)]))),
    lambda: s.CombineLatest()((V, I), {"a": 1}),
    lambda: _pipe()(("a", I)),
])
def test_a_malformed_value_raises_a_clean_error(call):
    with pytest.raises((ValueError, TypeError)) as e:
        call()
    assert "std::" not in str(e.value) and "nanobind" not in str(e.value)
