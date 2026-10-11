"""Compatibility decisions D1 to D3: code that worked in 2.8.0 keeps its result."""
import numpy as np
import pandas as pd
import pytest

import screamer as s

T = 12
rng = np.random.default_rng(7)
v = rng.normal(size=T).cumsum() + 100
w = rng.normal(size=T).cumsum() + 100
idx = np.arange(T) * 10


# ---------------------------------------------------------------- D1. Tuples

@pytest.mark.parametrize("op", [s.RollingMean(2), s.BollingerBands(3), s.IsNan()])
def test_values_index_tuple_into_one_input_operator_raises(op):
    with pytest.raises(TypeError, match=r"\(values, index\)"):
        op((v, idx))


def test_values_index_tuple_message_points_to_stream_operators():
    with pytest.raises(TypeError) as err:
        s.RollingMean(2)((v, idx))
    msg = str(err.value)
    assert "Pipeline" in msg and "values" in msg


def test_tuple_of_numbers_stays_a_sequence_of_scalar_events():
    out = s.RollingMean(3)((1.0, 2.0, 3.0))
    assert isinstance(out, list) and len(out) == 3
    assert out[2] == pytest.approx(2.0)
    bb = s.BollingerBands(3)((1.0, 2.0, 3.0, 4.0))
    assert isinstance(bb, list) and len(bb) == 4


def test_tuple_of_arrays_into_n_input_operator_stays_the_n_inputs():
    np.testing.assert_array_equal(s.Add()((v, w)), v + w)


def test_list_of_arrays_is_a_list_of_events():
    out = s.RollingMean(2)([np.ones(3), np.ones(3)])
    assert isinstance(out, list) and len(out) == 2
    assert all(isinstance(o, np.ndarray) and o.shape == (3,) for o in out)


# ---------------------------------------------------------- D2. from_pandas

def _two_level():
    cols = pd.MultiIndex.from_product([["a", "b"], ["o", "c"]])
    return pd.DataFrame(np.arange(8.0).reshape(2, 4), columns=cols, index=[10, 20])


def test_from_pandas_flat_frame_as_in_2_8_0():
    values, index = s.from_pandas(pd.DataFrame({"x": [1.5, 2.5], "y": [3.5, 4.5]}))
    np.testing.assert_array_equal(values, np.array([[1.5, 3.5], [2.5, 4.5]]))
    np.testing.assert_array_equal(index, np.array([0, 1]))


def test_from_pandas_int_frame_as_in_2_8_0():
    values, _ = s.from_pandas(pd.DataFrame({"n": [1, 2]}))
    np.testing.assert_array_equal(values, np.array([[1], [2]]))
    assert values.dtype.kind == "i"


def test_from_pandas_string_frame_as_in_2_8_0():
    values, _ = s.from_pandas(pd.DataFrame({"name": ["x", "y"]}))
    assert values.dtype == object and values.tolist() == [["x"], ["y"]]


def test_from_pandas_datetime_frame_as_in_2_8_0():
    when = pd.to_datetime(["2024-01-01", "2024-01-02"])
    values, _ = s.from_pandas(pd.DataFrame({"when": when}))
    np.testing.assert_array_equal(
        values, np.array([["2024-01-01"], ["2024-01-02"]], dtype="datetime64[ns]"))


def test_from_pandas_ragged_two_level_frame_as_in_2_8_0():
    ragged = pd.MultiIndex.from_tuples([("BTC", "open"), ("BTC", "close"), ("ETH", "open")])
    values, _ = s.from_pandas(pd.DataFrame(np.arange(6.0).reshape(2, 3), columns=ragged))
    np.testing.assert_array_equal(values, np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]))


def test_from_pandas_complete_two_level_frame_is_flat_by_default():
    values, index = s.from_pandas(_two_level())
    np.testing.assert_array_equal(
        values, np.array([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]))
    np.testing.assert_array_equal(index, np.array([10, 20]))


def test_from_pandas_two_level_frame_with_taf_layout():
    values, _ = s.from_pandas(_two_level(), layout="TAF")
    assert values.shape == (2, 2, 2)
    assert values[1, 1, 0] == 6.0    # asset b, field o, at t=1


# ------------------------------------------------ D3. Microstructure keywords

sign = np.sign(v - v.mean())
flow = sign * w


def _forms(cls, names, a, b, ctor=()):
    """Results of one class for keyword, positional, packed and iterator input."""
    kw = cls(*ctor)(**{names[0]: a, names[1]: b})
    pos = cls(*ctor)(a, b)
    packed = cls(*ctor)(np.column_stack([a, b]))
    it = np.array(list(cls(*ctor)(iter(list(zip(a, b))))))
    mixed = cls(*ctor)(a, **{names[1]: b})
    return kw, pos, packed, it, mixed


@pytest.mark.parametrize("cls, names, ctor", [
    (s.SignedVolume, ("sign", "volume"), ()),
    (s.RollingKyleLambda, ("signed_flow", "return_"), (5,)),
    (s.EwKyleLambda, ("signed_flow", "return_"), (None, 5.0)),
    (s.QueueImbalance, ("bid_size", "ask_size"), ()),
])
def test_two_input_microstructure_keywords(cls, names, ctor):
    a, b = (sign, w) if cls is s.SignedVolume else (flow, np.diff(v, prepend=v[0]))
    if cls is s.QueueImbalance:
        a, b = np.abs(v), np.abs(w)
    kw, pos, packed, it, mixed = _forms(cls, names, a, b, ctor)
    np.testing.assert_array_equal(kw, pos)
    np.testing.assert_array_equal(packed, pos)
    np.testing.assert_array_equal(it, pos)
    np.testing.assert_array_equal(mixed, pos)


@pytest.mark.parametrize("cls, names", [
    (s.SignedVolume, ("sign", "volume")),
    (s.RollingKyleLambda, ("signed_flow", "return_")),
    (s.EwKyleLambda, ("signed_flow", "return_")),
    (s.QueueImbalance, ("bid_size", "ask_size")),
])
def test_two_input_microstructure_keyword_and_positional_for_one_field_raises(cls, names):
    op = cls() if cls is not s.EwKyleLambda else cls(span=5.0)
    with pytest.raises(TypeError):
        op(v, w, **{names[0]: v})
    with pytest.raises(TypeError):
        op(v, **{names[0]: w})
    with pytest.raises(TypeError):
        op(v, w, bogus=v)


def test_rolling_order_imbalance_keyword():
    pos = s.RollingOrderImbalance(4)(flow)
    np.testing.assert_array_equal(s.RollingOrderImbalance(4)(signed_flow=flow), pos)
    np.testing.assert_array_equal(np.array(list(s.RollingOrderImbalance(4)(iter(flow)))), pos)
    with pytest.raises(TypeError):
        s.RollingOrderImbalance(4)(flow, signed_flow=flow)


@pytest.mark.parametrize("make, missing", [
    (lambda: s.SignedVolume(), "volume"),
    (lambda: s.RollingKyleLambda(5), "return_"),
    (lambda: s.EwKyleLambda(span=5.0), "return_"),
    (lambda: s.QueueImbalance(), "ask_size"),
])
@pytest.mark.parametrize("one", [v, 1.5, list(v), v[:2]])
def test_two_input_microstructure_missing_input_names_it(make, missing, one):
    with pytest.raises(TypeError, match=f"missing.*'{missing}'"):
        make()(one)


def test_two_input_microstructure_missing_input_with_keyword():
    with pytest.raises(TypeError, match="missing.*'volume'"):
        s.SignedVolume()(sign=v)
