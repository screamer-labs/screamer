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


# ---- Row operators, lists of events, and other containers -----------------

from tests.test_contract_compliance import HELP, _is_row_op  # noqa: E402

ROW_NAMES = sorted(n for n in HELP if _is_row_op(n))


def _row_inputs(n_in, T=12, k=4, seed=3):
    rng = np.random.default_rng(seed)
    return [rng.uniform(0.5, 1.5, size=(T, k)) for _ in range(n_in)]


@pytest.mark.parametrize("name", ROW_NAMES)
def test_row_op_unit_field_is_consumed(name):
    xs = _row_inputs(int(HELP[name].get("inputs", 1)))
    want = np.asarray(_instantiate(HELP[name])(*xs))
    got = np.asarray(_instantiate(HELP[name])(*[x[:, :, None] for x in xs]))
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("name", [n for n in ROW_NAMES if int(HELP[n].get("inputs", 1)) == 1])
def test_row_op_iterator_of_unit_field_events(name):
    (x,) = _row_inputs(1)
    want = np.asarray(_instantiate(HELP[name])(x))
    got = list(_instantiate(HELP[name])(iter(x[:, :, None])))
    np.testing.assert_array_equal(np.array(got), want)


def test_row_op_on_reshaped_inputs():
    import screamer as s
    w, p = _row_inputs(2)
    np.testing.assert_array_equal(s.CrossZscore()(s.reshape(w, "TA")), s.CrossZscore()(w))
    np.testing.assert_array_equal(
        s.BacktestPortfolioTarget()(s.reshape(w, "TA"), s.reshape(p, "TA")),
        s.BacktestPortfolioTarget()(w, p))


@pytest.mark.parametrize("name", NAMES)
def test_list_of_events_equals_batch(name):
    n_in = int(FUNCTORS[name].get("inputs", 1))
    panel = _panel(n_in)
    got = _instantiate(FUNCTORS[name])(*[list(p) for p in panel])
    assert isinstance(got, list)
    np.testing.assert_array_equal(np.array(got, dtype=float), _out(name, panel))


@pytest.mark.parametrize("name", [n for n in NAMES if int(FUNCTORS[n].get("inputs", 1)) > 1])
def test_packed_events_equal_batch(name):
    panel = _panel(int(FUNCTORS[name]["inputs"]))
    packed = np.stack(panel, axis=-1)                       # (T, K, n_in)
    op = _instantiate(FUNCTORS[name])
    np.testing.assert_array_equal(np.array(list(op(iter(packed)))), _out(name, panel))
    got = _instantiate(FUNCTORS[name])(list(packed))
    np.testing.assert_array_equal(np.array(got), _out(name, panel))


@pytest.mark.parametrize("name", [n for n in NAMES if int(FUNCTORS[n].get("inputs", 1)) > 1])
def test_scalar_iterator_broadcasts_over_asset_events(name):
    n_in = int(FUNCTORS[name]["inputs"])
    panel = _panel(n_in)
    market = panel[0][:, 0]                                 # (T,), one value per step
    arrays = [market] + panel[1:]
    sources = [iter(market)] + [iter(p) for p in panel[1:]]
    got = list(_instantiate(FUNCTORS[name])(*sources))
    np.testing.assert_array_equal(np.array(got, dtype=float), _out(name, arrays))


def test_list_of_asset_events_examples():
    import screamer as s
    x = np.arange(1.0, 7.0).reshape(3, 2)
    got = s.RollingMean(2)(list(x))
    assert isinstance(got, list) and len(got) == 3
    np.testing.assert_array_equal(np.array(got), s.RollingMean(2)(x))
    packed = np.arange(1.0, 13.0).reshape(3, 2, 2)          # 3 events of 2 assets x 2 fields
    np.testing.assert_array_equal(np.array(s.Add()(list(packed))), packed[..., 0] + packed[..., 1])
    np.testing.assert_array_equal(np.array(s.Add()(list(x), list(x))), 2 * x)


def test_combine_latest_zero_width_port():
    import screamer as s
    x, y = s.Input("x"), s.Input("y")
    pipe = s.Pipeline([x, y], [s.CombineLatest()(x, y)])
    values, index = pipe(np.zeros((3, 0)), np.arange(3.0))
    np.testing.assert_array_equal(values, np.arange(3.0))
    np.testing.assert_array_equal(index, np.arange(3))


def test_combine_latest_rejects_layout_on_nodes():
    import screamer as s
    x, y = s.Input("x"), s.Input("y")
    with pytest.raises(ValueError, match="layout"):
        s.CombineLatest(layout="TF")(x, y)


def test_lazy_wide_resample_with_functor_agg_raises_like_eager():
    import screamer as s
    from screamer.streams import resample
    x = np.random.default_rng(0).uniform(size=(10, 3))
    idx = np.arange(10)
    with pytest.raises(ValueError, match="a functor agg aggregates one column"):
        resample((x, idx), count=3, agg=s.ExpandingSkew())
    with pytest.raises(ValueError, match="a functor agg aggregates one column"):
        list(resample(((r, i) for r, i in zip(x, idx)), count=3, agg=s.ExpandingSkew()))


def test_failed_copy_leaves_the_iterator_retryable():
    import screamer as s
    x = np.arange(1.0, 13.0).reshape(3, 4)
    op = s.RollingMean(2)
    calls = []

    def failing():
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("no copy")
        return type(op)(2)

    op._fresh = failing
    it = op(iter(x))
    with pytest.raises(RuntimeError, match="no copy"):
        next(it)
    want = s.RollingMean(2)(x[1:])                        # starts again from this event
    np.testing.assert_array_equal(np.array([next(it), next(it)]), want)
