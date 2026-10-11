"""Cross-sectional (row) operators: references, NaN handling, and regimes.

Each event is one row of K assets. The references are pandas and numpy applied
row by row; the regimes are a (T, K) batch, one row at a time, and a stream of
rows, which must agree exactly.
"""
import numpy as np
import pandas as pd
import pytest

import screamer as s

rng = np.random.default_rng(11)
X = rng.standard_normal((40, 7))
X[3, 2] = np.nan
X[5] = np.nan                       # a row with no assets
X[8, :6] = np.nan                   # a row with one asset
X[10, :] = 1.5                      # all equal
X[12, [1, 4]] = X[12, 0]            # ties
X[:, 6] = np.where(np.arange(40) < 15, np.nan, X[:, 6])   # listed later


def _rowwise(f, x):
    return np.array([f(r) for r in x])


def _rank_ref(r, signed=True):
    s_ = pd.Series(r)
    n = s_.notna().sum()
    if n == 0:
        return np.full(len(r), np.nan)
    u = (s_.rank(method="average") - 1) / (n - 1) if n > 1 else s_.where(s_.isna(), 0.5)
    u = u.to_numpy(dtype=float)
    return 2 * u - 1 if signed else u


def _zscore_ref(r):
    v = r[~np.isnan(r)]
    out = np.full(len(r), np.nan)
    if v.size < 2:
        return out
    sd = v.std(ddof=1)
    out[~np.isnan(r)] = (v - v.mean()) / sd if sd > 0 else 0.0
    return out


def _winsor_ref(r, lo, hi):
    v = r[~np.isnan(r)]
    if v.size == 0:
        return np.full(len(r), np.nan)
    return np.clip(r, np.quantile(v, lo), np.quantile(v, hi))


def _normalize_ref(r, gross, net):
    ok = ~np.isnan(r)
    out = np.full(len(r), np.nan)
    v = r[ok]
    if net is None:
        a = np.abs(v).sum()
        out[ok] = v * gross / a if a > 0 else 0.0
        return out
    d = v - v.mean() if v.size else v
    L, S = d[d > 0].sum(), -d[d < 0].sum()
    if L > 0 and S > 0:
        out[ok] = np.where(d > 0, d * 0.5 * (gross + net) / L, d * 0.5 * (gross - net) / S)
    else:
        out[ok] = 0.0
    return out


def _topbottom_ref(r, count):
    ok = np.flatnonzero(~np.isnan(r))
    out = np.full(len(r), np.nan)
    out[ok] = 0.0
    c = min(count, ok.size // 2)
    if c == 0:
        return out
    order = ok[np.argsort(r[ok], kind="stable")]
    if r[order[0]] == r[order[-1]]:
        return out
    out[order[:c]] = -1.0
    out[order[-c:]] = 1.0
    return out


CASES = [
    (lambda: s.CrossRank(), lambda r: _rank_ref(r, True)),
    (lambda: s.CrossRank("unit"), lambda r: _rank_ref(r, False)),
    (lambda: s.CrossZscore(), _zscore_ref),
    (lambda: s.CrossDemean(), lambda r: r - np.nanmean(r) if np.isfinite(r).any() else r),
    (lambda: s.CrossWinsorize(0.1, 0.8), lambda r: _winsor_ref(r, 0.1, 0.8)),
    (lambda: s.CrossNormalize(), lambda r: _normalize_ref(r, 1.0, None)),
    (lambda: s.CrossNormalize(gross=2.0, net=0.0), lambda r: _normalize_ref(r, 2.0, 0.0)),
    (lambda: s.CrossNormalize(gross=1.5, net=-0.5), lambda r: _normalize_ref(r, 1.5, -0.5)),
    (lambda: s.CrossTopBottom(2), lambda r: _topbottom_ref(r, 2)),
]
IDS = ["rank", "rank-unit", "zscore", "demean", "winsorize", "normalize",
       "normalize-neutral", "normalize-net", "topbottom"]


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("make,ref", CASES, ids=IDS)
def test_matches_reference(make, ref):
    np.testing.assert_allclose(make()(X), _rowwise(ref, X), rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("make,ref", CASES, ids=IDS)
def test_regimes_agree(make, ref):
    batch = make()(X)
    op = make()
    rows = np.array([op(r) for r in X])
    streamed = np.array(list(make()(iter(X))))
    lists = make()(X.tolist())
    for other in (rows, streamed, lists):
        np.testing.assert_array_equal(other, batch)


@pytest.mark.parametrize("make,ref", CASES, ids=IDS)
def test_nan_assets_are_left_out(make, ref):
    out = make()(X)
    assert np.all(np.isnan(out[np.isnan(X)]))
    assert np.all(np.isnan(out[5]))
    # A NaN asset changes nothing for the others: the same row with NaN
    # columns inserted gives the same values for the original assets.
    row = X[0, :6]
    padded = np.insert(row, [0, 3, 6], np.nan)
    np.testing.assert_array_equal(make()(padded)[~np.isnan(padded)], make()(row))


def test_portfolio_properties():
    w = s.CrossNormalize(gross=2.0, net=0.5)(X)
    rows = [t for t in range(40) if np.sum(~np.isnan(X[t])) > 1 and np.nanstd(X[t]) > 0]
    np.testing.assert_allclose(np.nansum(np.abs(w[rows]), axis=1), 2.0)
    np.testing.assert_allclose(np.nansum(w[rows], axis=1), 0.5)
    tb = s.CrossTopBottom(2)(X)
    assert np.all(np.nansum(tb, axis=1) == 0)
    r = s.CrossRank()(X)
    assert np.nanmin(r) == -1 and np.nanmax(r) == 1


def test_width_is_fixed_until_reset():
    op = s.CrossRank()
    op([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        op([1.0, 2.0])
    op.reset()
    assert op([2.0, 1.0]).tolist() == [1.0, -1.0]
    # A batch call resets before and after, so it never sees the old width.
    assert s.CrossRank()(np.ones((2, 4))).shape == (2, 4)


def test_argument_checks():
    with pytest.raises(ValueError):
        s.CrossRank("percent")
    with pytest.raises(ValueError):
        s.CrossWinsorize(0.9, 0.1)
    with pytest.raises(ValueError):
        s.CrossNormalize(gross=1.0, net=2.0)
    with pytest.raises(ValueError):
        s.CrossTopBottom(0)
    with pytest.raises(ValueError):
        s.CrossZscore()(np.ones((2, 3, 4)))
    with pytest.raises(TypeError):
        s.CrossZscore()(s.Pipeline.Input("x") if hasattr(s.Pipeline, "Input") else object())


def test_list_of_rows_returns_list_of_rows():
    r1, r2 = np.array([1., 3., 2.]), np.array([3., 1., 2.])
    out = s.CrossRank()([r1, r2])
    assert isinstance(out, list) and len(out) == 2
    assert all(isinstance(r, np.ndarray) and r.shape == (3,) for r in out)
    np.testing.assert_array_equal(np.vstack(out), s.CrossRank()(np.vstack([r1, r2])))
    out2 = s.CrossRank()([[1, 3, 2], [3, 1, 2]])
    assert isinstance(out2, list)
    np.testing.assert_array_equal(np.vstack(out2), np.vstack(out))


def test_tuple_of_rows_keeps_2_8_0_ndarray():
    r1, r2 = np.array([1., 3., 2.]), np.array([3., 1., 2.])
    assert isinstance(s.CrossRank()((r1, r2)), np.ndarray)
    assert isinstance(s.CrossRank()(np.vstack([r1, r2])), np.ndarray)
    assert isinstance(s.CrossRank()(r1), np.ndarray)
    assert isinstance(s.CrossRank()([1., 3., 2.]), np.ndarray)


def test_list_of_rows_into_two_input_row_op():
    t = [[.5, .5], [.2, .8], [.4, .6]]
    p = [[1., 2.], [1.1, 2.1], [1.2, 2.0]]
    ref = s.BacktestPortfolioTarget()(np.array(t), np.array(p))
    out = s.BacktestPortfolioTarget()(t, p)
    assert isinstance(out, list) and len(out) == 3
    np.testing.assert_array_equal(np.array(out), ref)
    assert isinstance(s.BacktestPortfolioTarget()(tuple(t), tuple(p)), np.ndarray)
