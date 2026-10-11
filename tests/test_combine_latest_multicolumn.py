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
