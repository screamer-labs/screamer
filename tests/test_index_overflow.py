"""Index arithmetic outside the int64 range raises ValueError naming the setting.

A Delay adds its duration to every index, and a freq= (every=) Resample
measures an index from its origin and labels each bar with its start or end.
Near the ends of the int64 range these sums can leave it; they raise
ValueError (RangeError in JavaScript) instead of wrapping around. A watermark
(advance(now)) is a bound, not a row: it never raises, and a bar it opens
raises only once a row of it would be emitted.
"""
import numpy as np
import pytest

import screamer as sc
from screamer import Input, Pipeline

MAX = 2**63 - 1
MIN = -2**63


def run(op, values, index):
    x = Input("x")
    return Pipeline([x], [op(x)])((np.asarray(values, float), np.asarray(index, np.int64)))


def test_delay_past_the_int64_range_raises():
    with pytest.raises(ValueError, match=r"a delay by duration=10 moves index 9223372036854775803 outside the int64 range"):
        run(sc.Delay(duration=10), [1, 2], [0, MAX - 4])
    with pytest.raises(ValueError, match=r"duration=-10 moves index -9223372036854775807"):
        run(sc.Delay(duration=-10), [1, 2], [MIN + 1, 0])
    # The last index that fits passes.
    v, k = run(sc.Delay(duration=10), [1, 2], [0, MAX - 10])
    assert list(k) == [10, MAX]


def test_delay_overflow_raises_in_a_live_session():
    x = Input("x")
    s = Pipeline([x], [sc.Delay(duration=10)(x)]).live()
    s.push("x", 0, 1.0)
    with pytest.raises(ValueError, match="duration=10"):
        s.push("x", MAX - 4, 2.0)


def test_forecast_pairs_duration_past_the_int64_range_raises():
    with pytest.raises(ValueError, match=r"duration=9223372036854775807 moves index 5"):
        sc.forecast_pairs((np.ones(2), np.array([0, 5])), (np.ones(2), np.array([0, 5])),
                          duration=MAX)


@pytest.mark.parametrize("reducer", [False, True])
def test_resample_label_past_the_int64_range_raises(reducer):
    agg = sc.Last() if reducer else "last"
    with pytest.raises(ValueError, match=r'right label \(label="right", every=10, origin=0\) is outside the int64 range'):
        run(sc.Resample(every=10, label="right", agg=agg), [1], [MAX - 5])
    # The left label of the same bar fits.
    v, k = run(sc.Resample(every=10, agg=agg), [1], [MAX - 5])
    assert list(k) == [MAX - 7]
    with pytest.raises(ValueError, match=r"left label .* is outside the int64 range"):
        run(sc.Resample(every=10, origin=-5, agg=agg), [1], [MIN + 1])


@pytest.mark.parametrize("reducer", [False, True])
def test_resample_index_far_from_origin_raises(reducer):
    agg = sc.Last() if reducer else "last"
    with pytest.raises(ValueError, match=r"index 9223372036854775802 is outside the int64 range when measured from origin=-10"):
        run(sc.Resample(every=10, origin=-10, agg=agg), [1], [MAX - 5])


@pytest.mark.parametrize("reducer", [False, True])
def test_resample_advance_near_the_top_does_not_raise(reducer):
    agg = sc.Last() if reducer else "last"
    x = Input("x")
    p = Pipeline([x], [sc.Resample(every=10, label="right", agg=agg)(x)])
    s = p.live()
    s.push("x", 0, 1.0)
    s.advance(MAX - 2)           # closes the bar of 0; the open bar's label does not fit
    v, k = s.result()
    assert list(k) == [10]
    s.flush()                    # the open bar is empty: nothing to label
    v, k = s.result()
    assert len(k) == 0
    s = p.live()
    s.advance(MAX - 2)
    with pytest.raises(ValueError, match="right label"):
        s.push("x", MAX - 1, 1.0)
