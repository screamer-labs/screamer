"""The shape contract, asserted against every operator in the registry.

An array is (T, *A, F?): K assets give K independent series, a size-1 field
axis is consumed, separate inputs broadcast over assets, a packed array splits
along its last axis, and an iterator of events gives the batch result. See
docs/polymorphic_api.md.
"""
import numpy as np
import pytest

from tests.test_contract_compliance import FUNCTORS, _instantiate, _inputs

K = 3
NAMES = sorted(FUNCTORS)


def _panel(n_in, seed=7):
    cols = [_inputs(n_in, seed=seed + k) for k in range(K)]          # K lists of n_in (T,)
    return [np.stack([cols[k][i] for k in range(K)], axis=1) for i in range(n_in)]


def _out(name, arrays):
    return np.asarray(_instantiate(FUNCTORS[name])(*arrays), dtype=float)


@pytest.mark.parametrize("name", NAMES)
def test_assets_are_independent_series(name):
    n_in = int(FUNCTORS[name].get("inputs", 1))
    panel = _panel(n_in)
    got = _out(name, panel)
    for k in range(K):
        np.testing.assert_array_equal(got[:, k], _out(name, [p[:, k] for p in panel]))


@pytest.mark.parametrize("name", NAMES)
def test_unit_field_is_consumed(name):
    n_in = int(FUNCTORS[name].get("inputs", 1))
    panel = _panel(n_in)
    np.testing.assert_array_equal(_out(name, [p[:, :, None] for p in panel]), _out(name, panel))


@pytest.mark.parametrize("name", [n for n in NAMES if int(FUNCTORS[n].get("inputs", 1)) > 1])
def test_one_d_inputs_broadcast(name):
    n_in = int(FUNCTORS[name]["inputs"])
    panel = _panel(n_in)
    shared = [p[:, 0] for p in panel[1:]]
    want = _out(name, [panel[0]] + [np.repeat(s[:, None], K, axis=1) for s in shared])
    np.testing.assert_array_equal(_out(name, [panel[0]] + shared), want)
    np.testing.assert_array_equal(_out(name, [panel[0]] + [s[:, None] for s in shared]), want)


@pytest.mark.parametrize("name", [n for n in NAMES if int(FUNCTORS[n].get("inputs", 1)) > 1])
def test_incompatible_assets_raise(name):
    n_in = int(FUNCTORS[name]["inputs"])
    arrays = [np.ones((50, 3))] + [np.ones((50, 4))] * (n_in - 1)
    with pytest.raises(ValueError, match="do not broadcast"):
        _instantiate(FUNCTORS[name])(*arrays)


@pytest.mark.parametrize("name", [n for n in NAMES if int(FUNCTORS[n].get("inputs", 1)) > 1])
def test_packed_array_splits(name):
    panel = _panel(int(FUNCTORS[name]["inputs"]))
    packed = np.stack(panel, axis=-1)
    np.testing.assert_array_equal(np.asarray(_instantiate(FUNCTORS[name])(packed)), _out(name, panel))


@pytest.mark.parametrize("name", NAMES)
def test_iterator_of_events_equals_batch(name):
    n_in = int(FUNCTORS[name].get("inputs", 1))
    panel = _panel(n_in)
    op = _instantiate(FUNCTORS[name])
    events = list(op(*[iter(p) for p in panel]))
    np.testing.assert_array_equal(np.array(events, dtype=float), _out(name, panel))
