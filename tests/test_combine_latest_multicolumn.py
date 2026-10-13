"""CombineLatest concatenates fields and broadcasts assets (shape contract rule 5)."""
import numpy as np
import pytest

import screamer as s

T = 12
bars = np.arange(T * 4.0).reshape(T, 4)
sig = np.arange(T) * 10.0


def test_eager_concatenates_fields():
    values, index = s.CombineLatest()(bars, sig)
    assert index is None
    np.testing.assert_array_equal(values, np.column_stack([bars, sig]))


def test_eager_indexed_wide_as_of():
    v, i = s.CombineLatest()((bars, np.arange(T) * 2), (sig[:3], np.array([1, 5, 9])))
    assert v.shape[1] == 5
    row = list(i).index(10)
    np.testing.assert_array_equal(v[row], [*bars[5], sig[2]])


def test_eager_on_any_wide_nan_until_seen():
    v, _ = s.CombineLatest(emit="on_any")((bars, np.arange(T) + 5), (sig, np.arange(T)))
    assert v.shape == (T + 5, 5)
    assert np.isnan(v[0, :4]).all() and v[0, 4] == sig[0]


def test_layout_broadcasts_assets():
    panel = np.arange(T * 3.0).reshape(T, 3)
    v, _ = s.CombineLatest(layout=("TA", "T"))(panel, sig)
    assert v.shape == (T, 3, 2)
    np.testing.assert_array_equal(v[..., 0], panel)
    np.testing.assert_array_equal(v[..., 1], np.repeat(sig[:, None], 3, axis=1))


def test_layout_string_applies_to_every_input():
    a, b = np.zeros((T, 3)), np.ones((T, 3))
    v, _ = s.CombineLatest(layout="TA")(a, b)
    assert v.shape == (T, 3, 2)


def test_three_d_inputs_read_as_assets_and_fields():
    v, _ = s.CombineLatest()(np.zeros((T, 3, 4)), sig)
    assert v.shape == (T, 3, 5)


def test_layout_count_must_match():
    with pytest.raises(ValueError, match="one layout per input"):
        s.CombineLatest(layout=("TA",))(bars, sig)


def test_lazy_positional_wide():
    out = list(s.CombineLatest()(iter(bars), iter(sig)))
    np.testing.assert_array_equal(np.array([r for r, _ in out]), np.column_stack([bars, sig]))


def test_lazy_indexed_wide_matches_batch():
    idx_a, idx_b = np.arange(T) * 2, np.array([1, 5, 9])
    batch, bidx = s.CombineLatest()((bars, idx_a), (sig[:3], idx_b))
    lazy = list(s.CombineLatest()(iter(zip(bars, idx_a)), iter(zip(sig[:3], idx_b))))
    np.testing.assert_array_equal(np.array([r for r, _ in lazy]), batch)
    assert [i for _, i in lazy] == list(bidx)


def test_pipeline_joins_a_bar_node_and_a_signal():
    x, y = s.Input("x"), s.Input("y")
    p = s.Pipeline([x, y], [s.CombineLatest()(s.Resample(count=2, agg="ohlc")(x), y)])
    values, _ = p(np.arange(20.0), np.arange(20.0))
    assert values.shape[1] == 5
    np.testing.assert_array_equal(values[2, :4], [2.0, 3.0, 2.0, 3.0])


def test_pipeline_wide_input_when_all():
    x, y = s.Input("x"), s.Input("y")
    p = s.Pipeline([x, y], [s.CombineLatest()(x, y)])
    values, _ = p(bars, sig)
    np.testing.assert_array_equal(values, np.column_stack([bars, sig]))


def test_pipeline_on_any_wide_input_matches_eager():
    # The node holds rows until every port has reported, so a wide Input that
    # starts late still fixes its width before the first row is due.
    x, y = s.Input("x"), s.Input("y")
    p = s.Pipeline([x, y], [s.CombineLatest(emit="on_any")(x, y)])
    feeds = ((bars, np.arange(T) + 5), (sig, np.arange(T)))
    values, index = p(*feeds)
    eager, eager_index = s.CombineLatest(emit="on_any")(*feeds)
    np.testing.assert_array_equal(values, eager)
    np.testing.assert_array_equal(index, eager_index)


def test_pipeline_on_any_wide_input_raises_clearly():
    # Dropna advances its port's watermark past dropped rows, so a row is due
    # before the first 4-column event reaches the node.
    gappy = bars.copy()
    gappy[:4] = np.nan
    x, y = s.Input("x"), s.Input("y")
    p = s.Pipeline([x, y], [s.CombineLatest(emit="on_any")(s.Dropna()(x), y)])
    with pytest.raises(ValueError, match="when_all"):
        p((gappy, np.arange(T)), (sig, np.arange(T)))


def test_pipeline_rejects_layout():
    x, y = s.Input("x"), s.Input("y")
    with pytest.raises(ValueError, match="applies to arrays and iterators"):
        s.CombineLatest(layout=("TA", "T"))(x, y)


def _empty_indexed():
    return np.array([]), np.array([], dtype=np.int64)


def test_eager_on_any_one_empty_indexed_input_pads_nan():
    v, i = s.CombineLatest(emit="on_any")((np.arange(5.0), np.arange(5)), _empty_indexed())
    assert v.shape == (5, 2)
    np.testing.assert_array_equal(v[:, 0], np.arange(5.0))
    assert np.isnan(v[:, 1]).all()
    np.testing.assert_array_equal(i, np.arange(5))


def test_eager_on_any_empty_input_beside_a_wide_one():
    v, _ = s.CombineLatest(emit="on_any")((bars, np.arange(T)), _empty_indexed())
    assert v.shape == (T, 5)
    np.testing.assert_array_equal(v[:, :4], bars)
    assert np.isnan(v[:, 4]).all()


def test_lazy_positional_one_empty_source_raises():
    with pytest.raises(ValueError, match="stay in step"):
        list(s.CombineLatest()(iter([1.0, 2.0]), iter([])))


def test_lazy_indexed_on_any_one_empty_source_matches_eager():
    eager, eager_index = s.CombineLatest(emit="on_any")((bars, np.arange(T)), _empty_indexed())
    lazy = list(s.CombineLatest(emit="on_any")(iter(zip(bars, np.arange(T))), iter([])))
    np.testing.assert_array_equal(np.array([r for r, _ in lazy]), eager)
    assert [i for _, i in lazy] == list(eager_index)


def test_lazy_all_empty_sources_give_nothing():
    assert list(s.CombineLatest()(iter([]), iter([]))) == []
    assert list(s.CombineLatest(emit="on_any")(iter([]), iter([]))) == []


# The join plan is C++ (include/screamer/streams/join_plan.h); these call it
# through the binding, then check that every iterator form reads events as the
# batch join does.
from screamer import screamer_bindings as _b


def test_join_plan_reads_layouts_and_broadcasts_assets():
    plan = _b._JoinPlan([(1, 3), (1,)], ("TA", None))
    assert plan.wide and plan.assets
    np.testing.assert_array_equal(plan.row(np.array([1.0, 2.0, 3.0, 9.0])),
                                  [[1.0, 9.0], [2.0, 9.0], [3.0, 9.0]])
    narrow = _b._JoinPlan([(1,), (1,)], None)
    assert not narrow.wide and not narrow.assets


def test_join_plan_gathers_a_reordered_event():
    # an event of shape (F=2, A=3) read as "TFA" becomes (A, F), flat
    plan = _b._JoinPlan([(1, 2, 3)], "TFA")
    assert not plan.in_order(0)
    np.testing.assert_array_equal(plan.flat(0, np.arange(6.0).reshape(2, 3)), [0, 3, 1, 4, 2, 5])
    with pytest.raises(ValueError, match="has 4 values, but its first event had 6"):
        plan.flat(0, np.zeros(4))


def test_join_plan_errors():
    with pytest.raises(ValueError, match="one layout per input"):
        _b._JoinPlan([(1,), (1,)], ["T"])
    with pytest.raises(ValueError, match="do not broadcast"):
        _b._JoinPlan([(1, 3), (1, 2)], "TA")
    with pytest.raises(ValueError, match="must start with T"):
        _b._JoinPlan([(1, 3)], "AT")
    with pytest.raises(TypeError, match="a layout is a string or None"):
        _b._JoinPlan([(1, 3)], [3])
    # a source with no events: size 1 on every axis its layout names
    empty = _b._JoinPlan([None, (1,)], ("TA", None))
    assert empty.wide and empty.assets
    np.testing.assert_array_equal(empty.row(np.array([np.nan, 5.0])), [[np.nan, 5.0]])


def test_lazy_reordering_layout_matches_batch():
    cube = np.arange(T * 6.0).reshape(T, 2, 3)          # (T, F, A)
    batch, _ = s.CombineLatest(layout=("TFA", "T"))(cube, sig)
    pos = list(s.CombineLatest(layout=("TFA", "T"))(iter(cube), iter(sig)))
    np.testing.assert_array_equal(np.array([r for r, _ in pos]), batch)
    idx = list(s.CombineLatest(layout=("TFA", "T"))(zip(cube, range(T)), zip(sig, range(T))))
    np.testing.assert_array_equal(np.array([r for r, _ in idx]), batch)


def test_lazy_one_field_rows_with_an_asset_axis_match_batch():
    col = np.arange(T * 1.0)[:, None]                    # (T, 1) read as one asset
    batch, bidx = s.CombineLatest(layout="TA")((col, np.arange(T)))
    lazy = list(s.CombineLatest(layout="TA")(zip(col, range(T))))
    np.testing.assert_array_equal(np.array([r for r, _ in lazy]), batch)


def test_lazy_positional_rows_of_one_value_match_batch():
    col = np.arange(T * 1.0)[:, None]                    # events of shape (1,)
    batch, _ = s.CombineLatest()(col, sig)
    lazy = list(s.CombineLatest()(iter(col), iter(sig)))
    np.testing.assert_array_equal(np.array([r for r, _ in lazy]), batch)


def test_lazy_positional_event_of_another_size_raises():
    with pytest.raises(ValueError, match="has 3 values, but its first event had 2"):
        list(s.CombineLatest()(iter([[1.0, 2.0], [1.0, 2.0, 3.0]]), iter([1.0, 2.0])))


def test_positional_batch_counts_rows_along_the_layout_t_axis():
    at = np.arange(3.0 * T).reshape(3, T)                # (A, T)
    v, i = s.CombineLatest(layout=("AT", "T"))(at, sig)
    assert i is None and v.shape == (T, 3, 2)
    np.testing.assert_array_equal(v[..., 0], at.T)
    np.testing.assert_array_equal(v[..., 1], np.repeat(sig[:, None], 3, axis=1))
    both, _ = s.CombineLatest(layout="AT")(at, at)
    assert both.shape == (T, 3, 2)
    tfa = np.arange(T * 6.0).reshape(T, 2, 3)            # (T, F, A)
    w, _ = s.CombineLatest(layout=("TFA", "T"))(tfa, sig)
    np.testing.assert_array_equal(w[..., :2], tfa.transpose(0, 2, 1))
    with pytest.raises(ValueError, match="lengths must match"):
        s.CombineLatest(layout=("AT", "T"))(at, sig[:5])


def test_index_length_differs_from_values_raises_value_error():
    with pytest.raises(ValueError, match="input 0 has 12 rows of values but 4 index values"):
        s.CombineLatest()((sig, np.arange(4)), (sig, np.arange(T)))


def test_two_value_lists_are_rows_and_tuples_are_value_index():
    # A list is always values; the tuple marks (value, index).
    rows = list(s.CombineLatest()(iter([[1.0, 2.0], [3.0, 4.0]]), iter([[5.0, 6.0], [7.0, 8.0]])))
    np.testing.assert_array_equal(np.array([r for r, _ in rows]), [[1, 2, 5, 6], [3, 4, 7, 8]])
    assert [i for _, i in rows] == [None, None]
    pairs = list(s.CombineLatest()(iter([(1.0, 2), (3.0, 4)]), iter([(5.0, 6), (7.0, 8)])))
    assert pairs == [((3.0, 5.0), 6), ((3.0, 7.0), 8)]
    batch, idx = s.CombineLatest()([np.array([1.0, 2.0]), np.array([3.0, 4.0])], [[5.0, 6.0], [7.0, 8.0]])
    assert idx is None
    np.testing.assert_array_equal(batch, [[1, 2, 5, 6], [3, 4, 7, 8]])


def test_a_tuple_marks_values_and_index_wherever_a_stream_is_taken():
    import pandas as pd
    from screamer.streams import merge
    v, i = s.CombineLatest()(([1.0, 2.0], [0, 1]), ([3.0, 4.0], [0, 1]))
    np.testing.assert_array_equal(v, [[1, 3], [2, 4]])
    np.testing.assert_array_equal(i, [0, 1])
    v2, i2 = s.CombineLatest()((np.array([1.0, 2.0]), pd.Series([0, 1])), ([3.0, 4.0], pd.Index([0, 1])))
    np.testing.assert_array_equal(v2, v)
    np.testing.assert_array_equal(i2, i)
    mv, src, mi = merge(([1.0, 2.0], [0, 2]), ([3.0], [1]))
    np.testing.assert_array_equal(mv, [1, 3, 2])
    np.testing.assert_array_equal(mi, [0, 1, 2])
    (a, ka), (b, kb) = s.split(([1.0, 2.0], [5, 6]), [0, 1])
    np.testing.assert_array_equal(ka, [5])
    x = s.Input("x")
    out, idx = s.Pipeline([x], [s.RollingMean(1)(x)])(([1.0, 2.0], [7, 9]))
    np.testing.assert_array_equal(idx, [7, 9])


def test_a_tuple_that_is_not_values_and_index_raises():
    for bad in [(1.0, 2.0, 3.0), (1.0, 2.0), ([1.0, 2.0], "ab"), ([1.0], [0], [1])]:
        with pytest.raises(TypeError, match=r"a tuple marks a \(values, index\) stream"):
            s.CombineLatest()(bad, [1.0, 2.0])
    with pytest.raises(TypeError, match=r"a tuple marks a \(values, index\) stream"):
        s.split((1.0, 2.0), [0, 1])


def test_a_range_or_tuple_index_and_a_tuple_of_tuples_are_pairs():
    from screamer.streams import merge
    v, i = s.CombineLatest()((np.array([1.0, 2.0, 3.0]), range(3)), (np.array([4.0, 5.0]), (0, 2)))
    np.testing.assert_array_equal(i, [0, 1, 2])
    np.testing.assert_array_equal(v, [[1, 4], [2, 4], [3, 5]])
    w, j = s.CombineLatest()(((1.0, 2.0, 3.0), (0, 1, 2)), ((4.0, 5.0), (0, 2)))
    np.testing.assert_array_equal(w, v)
    np.testing.assert_array_equal(j, i)
    mv, _, mi = merge(([1.0, 2.0], range(0, 4, 2)), ((3.0,), (1,)))
    np.testing.assert_array_equal(mi, [0, 1, 2])
    (a, ka), _ = s.split(((1.0, 2.0), range(5, 7)), [0, 1])
    np.testing.assert_array_equal(ka, [5])
    x = s.Input("x")
    _, idx = s.Pipeline([x], [s.RollingMean(1)(x)])(([1.0, 2.0], range(7, 9)))
    np.testing.assert_array_equal(idx, [7, 8])
    d, di = s.Dropna()(([1.0, np.nan], [0, 1]))
    np.testing.assert_array_equal(d, [1.0])
    np.testing.assert_array_equal(di, [0])
    with pytest.raises(TypeError):                     # a string item, by the value rules
        s.CombineLatest()((("a", "b"), (0, 1)), ([1.0, 2.0], [0, 1]))
