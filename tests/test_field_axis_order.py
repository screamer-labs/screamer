"""A12: every multi-output operator puts the field axis last (rule 1).

On 1-D inputs an operator with F outputs gives (T, F); on (T, K) inputs it
gives (T, K, F), and column k is the 1-D result of column k. A stream of (K,)
events gives one (K, F) array per event, the rows of the batch result, and a
stream of numbers gives one F-tuple per event. The operators come from the
registry of the bitwise sweep (tests/test_batch_stream_bitwise.py), so a new
multi-output operator is checked without a change here. The JavaScript side
reads wasm/smoke/field_axis_oracle.json (js/test/field_axis.test.ts).
"""
import pathlib
import subprocess
import sys

from tests._oracle_fresh import assert_oracle_fresh

import numpy as np
import pytest

import screamer as s
from tests.param_cases import ROW_OPS
from tests.test_batch_stream_bitwise import (
    BLOCK_OPS, default_kwargs, inputs_for, operator_names, series)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
GEN = REPO_ROOT / "devtools" / "wasm" / "gen_field_axis_oracle.py"
FIXTURE = REPO_ROOT / "wasm" / "smoke" / "field_axis_oracle.json"

T, K = 300, 5


def _multi_output_series_ops():
    out = []
    for name in operator_names():
        if name in ROW_OPS or name in BLOCK_OPS:
            continue
        kw = default_kwargs(name)
        make = lambda name=name, kw=kw: getattr(s, name)(**kw)  # noqa: E731
        found = inputs_for(make)
        if found is None:
            continue
        n, layout = found
        if np.ndim(make()(*series(0, 1e3, layout)[:n])) == 2:
            out.append((name, n, layout))
    return out


SERIES_OPS = _multi_output_series_ops()


def test_the_registry_finds_the_multi_output_operators():
    names = {n for n, _, _ in SERIES_OPS}
    assert {"BollingerBands", "MACD", "ADX", "Stoch", "BacktestPriceTarget"} <= names
    assert len(names) >= 25


@pytest.mark.parametrize("name, n, layout", SERIES_OPS, ids=[c[0] for c in SERIES_OPS])
def test_field_axis_is_last_in_batch_and_streams(name, n, layout):
    make = lambda: getattr(s, name)(**default_kwargs(name))  # noqa: E731
    cols = [series(k + 7, 1e3, layout)[:n] for k in range(K)]
    X = [np.stack([c[i] for c in cols], axis=1) for i in range(n)]

    one = make()(*cols[0])
    F = one.shape[1]
    assert one.shape == (T, F) and F > 1
    wide = make()(*X)
    assert wide.shape == (T, K, F)
    np.testing.assert_array_equal(wide[:, 0], one)

    rows = list(make()(*[iter(list(x)) for x in X]))
    assert len(rows) == T and all(r.shape == (K, F) for r in rows)
    np.testing.assert_array_equal(np.stack(rows), wide)

    tuples = list(make()(*[iter(c.tolist()) for c in cols[0]]))
    assert all(isinstance(r, tuple) and len(r) == F for r in tuples)
    np.testing.assert_array_equal(np.array(tuples), one)


def test_row_operators_put_the_field_axis_last():
    for name in ("BacktestPortfolioTarget", "BacktestPortfolioOHLCTarget"):
        make = lambda: getattr(s, name)(**default_kwargs(name))  # noqa: E731
        probe = [np.stack(series(k + 50, 0.0)[:K], axis=1) for k in range(10)]
        n = next(m for m in range(1, 11) if _builds(make, probe[:m]))
        wide = make()(*probe[:n])
        assert wide.ndim == 3 and wide.shape[:2] == (T, K)
        rows = list(make()(*[iter(list(x)) for x in probe[:n]]))
        np.testing.assert_array_equal(np.stack(rows), wide)


def _builds(make, xs):
    try:
        make()(*xs)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not GEN.exists(), reason="needs the source tree")
def test_field_axis_oracle_is_fresh():
    assert_oracle_fresh(GEN, FIXTURE, REPO_ROOT)


@pytest.mark.parametrize("name, n, layout", SERIES_OPS, ids=[c[0] for c in SERIES_OPS])
def test_field_axis_is_last_on_two_asset_axes_and_async(name, n, layout):
    """(T, A, K) gives (T, A, K, F), and an async stream of (K,) events gives
    the rows of the (T, K) batch."""
    import asyncio
    make = lambda: getattr(s, name)(**default_kwargs(name))  # noqa: E731
    cols = [series(k + 7, 1e3, layout)[:n] for k in range(2 * K)]
    X = [np.stack([c[i] for c in cols], axis=1) for i in range(n)]      # (T, 2K)
    X3 = [x.reshape(T, 2, K) for x in X]
    wide = make()(*X)
    deep = make()(*X3)
    assert deep.shape == (T, 2, K, wide.shape[-1])
    np.testing.assert_array_equal(deep.reshape(wide.shape), wide)

    async def events(x):
        for row in x[:, :K]:
            yield row

    async def collect():
        return [r async for r in make()(*[events(x) for x in X])]

    rows = asyncio.run(collect())
    np.testing.assert_array_equal(np.stack(rows), wide[:, :K])
