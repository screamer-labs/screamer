"""The C++ shape module: layout strings and the asset broadcast."""
import pytest

from screamer import screamer_bindings as _b


@pytest.mark.parametrize("layout, shape, out, result", [
    ("TA", [100, 3], "TAF", [100, 3, 1]),
    ("TAF->TA", [100, 3, 1], "TA", [100, 3]),
    ("TF->TAF", [100, 4], "TAF", [100, 1, 4]),
    ("AT->TA", [3, 100], "TA", [100, 3]),
    ("TFA", [100, 4, 3], "TAF", [100, 3, 4]),
    ("T", [100], "TF", [100, 1]),
    ("TAAF", [100, 2, 3, 4], "TAAF", [100, 2, 3, 4]),
    (" T A ", [100, 3], "TAF", [100, 3, 1]),
])
def test_layout_result(layout, shape, out, result):
    got_out, _, _, got_shape = _b._layout_plan(layout, shape)
    assert got_out == out
    assert list(got_shape) == result


def test_layout_source_and_dropped():
    out, source, dropped, shape = _b._layout_plan("TFA->TA", [100, 1, 3])
    assert out == "TA" and list(source) == [0, 2] and list(dropped) == [1]


@pytest.mark.parametrize("layout, shape, message", [
    ("TA", [100, 3, 1], "names 2 axes"),
    ("TX", [100, 3], "unknown axis letter"),
    ("AA", [3, 3], "exactly one T"),
    ("TFF", [100, 2, 2], "at most one F"),
    ("TAF->TA", [100, 3, 4], "only a size-1 axis can be dropped"),
    ("TA->AT", [100, 3], "must start with T"),
    ("TA->TA->TA", [100, 3], "at most one '->'"),
])
def test_layout_errors(layout, shape, message):
    with pytest.raises(ValueError, match=message):
        _b._layout_plan(layout, shape)


def test_layout_error_names_layout_and_shape():
    with pytest.raises(ValueError, match=r"layout 'TAF->TA'.*\(100, 3, 4\)"):
        _b._layout_plan("TAF->TA", [100, 3, 4])


@pytest.mark.parametrize("shapes, result", [
    ([[10], [10, 3]], [10, 3]),
    ([[10, 1], [10, 3]], [10, 3]),
    ([[10, 3], [10, 2, 3]], [10, 2, 3]),
    ([[10, 2, 1], [10, 1, 3]], [10, 2, 3]),
    ([[10], [10]], [10]),
    ([[0, 3], [0]], [0, 3]),
    ([[10, 0], [10]], [10, 0]),
])
def test_broadcast_shape(shapes, result):
    assert list(_b._broadcast_shape(shapes)) == result


def test_broadcast_rejects_mismatched_assets():
    with pytest.raises(ValueError, match=r"\(10, 3\).*\(10, 4\).*do not broadcast"):
        _b._broadcast_shape([[10, 3], [10, 4]])


def test_broadcast_rejects_mismatched_time():
    with pytest.raises(ValueError, match="same length along time"):
        _b._broadcast_shape([[10, 3], [9, 3]])


@pytest.mark.parametrize("shapes, n_in, result", [
    ([[10, 3, 1]], 1, [10, 3]),          # rule 3
    ([[10, 1]], 1, [10, 1]),             # (T, 1) is one asset
    ([[10, 2]], 2, [10]),                # packed (T, N)
    ([[10, 3, 2]], 2, [10, 3]),          # packed (T, K, N)
    ([[10, 3, 1], [10]], 2, [10, 3]),    # rule 3 on a separate argument
])
def test_input_shape(shapes, n_in, result):
    assert list(_b._input_shape(shapes, n_in)) == result


def test_input_shape_rejects_wrong_packed_width():
    with pytest.raises(ValueError, match="expects 2 inputs"):
        _b._input_shape([[10, 3]], 2)
