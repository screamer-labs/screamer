"""Portfolio backtest engines: one capital account for K assets."""
import numpy as np
import pytest

import screamer as s

rng = np.random.default_rng(21)
T, K = 400, 4
PRICE = 100 * np.exp(np.cumsum(rng.standard_normal((T, K)) * 0.01, axis=0))
CLOSE = PRICE
OPEN = np.vstack([CLOSE[:1], CLOSE[:-1]]) * np.exp(rng.standard_normal((T, K)) * 1e-3)
HIGH = np.maximum(OPEN, CLOSE) * 1.002
LOW = np.minimum(OPEN, CLOSE) * 0.998


def _equity(out, capital):
    return capital + out[:, :, 0].sum(axis=1)


def _weights(out, price, capital):
    return out[:, :, 2] * price / _equity(out, capital)[:, None]


# --- one asset reproduces the single-asset engines ---------------------------

@pytest.mark.parametrize("fee,slip,unit,tick", [(0.0, 0.0, "fraction", 0.0),
                                                (1e-3, 5e-4, "fraction", 0.0),
                                                (2e-4, 2.0, "ticks", 0.01)])
def test_one_asset_matches_price_target(fee, slip, unit, tick):
    tgt = np.round(rng.standard_normal(T) * 2)
    a = s.BacktestPortfolioTarget(fee=fee, slippage=slip, slippage_unit=unit, tick_size=tick,
                                  size="units", initial_capital=0.0)(tgt[:, None], PRICE[:, :1])
    b = s.BacktestPriceTarget(fee=fee, slippage=slip, slippage_unit=unit,
                              tick_size=tick)(tgt, PRICE[:, 0])
    np.testing.assert_allclose(a[:, 0, :], b, rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize("fill_at,delay", [("open", 1), ("close", 0), ("typical", 2), ("ohlc4", 1)])
def test_one_asset_matches_ohlc_target(fill_at, delay):
    tgt = np.round(rng.standard_normal(T) * 2)
    cols = [x[:, :1] for x in (OPEN, HIGH, LOW, CLOSE)]
    a = s.BacktestPortfolioOHLCTarget(fee=3e-4, slippage=1e-4, size="units", initial_capital=0.0,
                                      fill_at=fill_at, delay=delay)(tgt[:, None], *cols)
    b = s.BacktestOHLCTarget(taker_fee=3e-4, slippage=1e-4, fill_at=fill_at,
                             delay=delay)(tgt, *(c[:, 0] for c in cols))
    np.testing.assert_allclose(a[:, 0, :], b, rtol=1e-12, atol=1e-9)


def test_one_asset_weight_matches_ohlc_target_weight():
    # The single-asset engine re-sizes when the target value changes; passing
    # the target only on its changes gives the portfolio engine the same orders.
    w = np.repeat(rng.choice([-0.5, 0.0, 0.8, 1.0], T // 20), 20)
    changes = np.where(np.r_[True, w[1:] != w[:-1]], w, np.nan)
    cols = [x[:, :1] for x in (OPEN, HIGH, LOW, CLOSE)]
    a = s.BacktestPortfolioOHLCTarget(fee=5e-4, size="weight", initial_capital=1e4)(
        changes[:, None], *cols)
    b = s.BacktestOHLCTarget(taker_fee=5e-4, size="weight", initial_capital=1e4)(
        w, *(c[:, 0] for c in cols))
    np.testing.assert_allclose(a[:, 0, :], b, rtol=1e-10, atol=1e-8)


# --- sizing and rebalancing ---------------------------------------------------

def test_weights_are_fractions_of_portfolio_equity():
    w = np.full((T, K), np.nan)
    w[::25] = [0.4, 0.3, 0.2, 0.1]
    out = s.BacktestPortfolioTarget(size="weight", initial_capital=1e6)(w, PRICE)
    held = _weights(out, PRICE, 1e6)
    np.testing.assert_allclose(held[::25], np.tile([0.4, 0.3, 0.2, 0.1], (len(held[::25]), 1)),
                               rtol=1e-12)
    # Between rebalances the units do not change.
    assert np.all(out[1:25, :, 2] == out[0, :, 2])


def test_notional_and_units_sizing():
    n = s.BacktestPortfolioTarget(size="notional", initial_capital=0.0)(
        np.full((T, K), 1000.0), PRICE)
    np.testing.assert_allclose(n[:, :, 2] * PRICE, 1000.0)
    u = s.BacktestPortfolioTarget(size="units", initial_capital=0.0)(np.full((T, K), 3.0), PRICE)
    assert np.all(u[:, :, 2] == 3.0)


def test_rebalance_band_drifted_and_all():
    w = np.full((T, K), np.nan)
    w[0] = 0.25
    hold = s.BacktestPortfolioTarget(size="weight")(w, PRICE)
    band = s.BacktestPortfolioTarget(size="weight", rebalance_band=0.02)(w, PRICE)
    every = s.BacktestPortfolioTarget(size="weight", rebalance_band=0.02, rebalance="all")(w, PRICE)
    drift_hold = np.abs(_weights(hold, PRICE, 1.0) - 0.25).max()
    drift_band = np.abs(_weights(band, PRICE, 1.0) - 0.25).max()
    assert drift_hold > 0.05
    assert drift_band < drift_hold
    # "drifted" trades one asset at a time; "all" trades every asset together.
    traded_d = np.diff(band[:, :, 2], axis=0) != 0
    traded_a = np.diff(every[:, :, 2], axis=0) != 0
    rows_a = traded_a.any(axis=1)
    assert np.all(traded_a[rows_a].all(axis=1))
    assert np.any(traded_d.any(axis=1) & ~traded_d.all(axis=1))


def test_exposure_limits():
    out = s.BacktestPortfolioTarget(size="weight", max_gross=1.5)(np.full((T, K), 0.6), PRICE)
    np.testing.assert_allclose(np.abs(_weights(out, PRICE, 1.0)).sum(axis=1), 1.5, rtol=1e-12)
    w = np.tile([0.5, 0.4, -0.2, 0.1], (T, 1))
    out = s.BacktestPortfolioTarget(size="weight", max_net=0.3)(w, PRICE)
    np.testing.assert_allclose(_weights(out, PRICE, 1.0).sum(axis=1), 0.3, rtol=1e-12)
    # Within the limits nothing changes.
    a = s.BacktestPortfolioTarget(size="weight", max_gross=5, max_net=5)(w, PRICE)
    b = s.BacktestPortfolioTarget(size="weight")(w, PRICE)
    np.testing.assert_array_equal(a, b)


# --- costs, contracts, and missing prices -------------------------------------

def test_per_asset_costs_and_multipliers():
    w = np.full((T, 2), 0.3)
    p = PRICE[:, :2]
    both = s.BacktestPortfolioTarget(size="weight", fee=[1e-3, 4e-3], multiplier=[1.0, 50.0])(w, p)
    one = s.BacktestPortfolioTarget(size="weight", fee=4e-3, multiplier=50.0)(w[:, 1:], p[:, 1:])
    assert both[:, 1, 3].sum() > both[:, 0, 3].sum() > 0
    # Units account for the multiplier: notional = units * price * multiplier.
    # Weights are sized on equity before the step's fees, so the weight on
    # equity after them is a little higher.
    held = both[:, 1, 2] * p[:, 1] * 50 / _equity(both, 1.0)
    np.testing.assert_allclose(held, 0.3, rtol=2e-3)
    assert one.shape == (T, 1, 4)
    with pytest.raises(ValueError):
        s.BacktestPortfolioTarget(fee=[1e-3, 2e-3, 3e-3])(w, p)


def test_unquoted_assets_hold_and_keep_the_sum_defined():
    p = PRICE.copy()
    p[:50, 3] = np.nan                     # listed at step 50
    p[200:210, 1] = np.nan                 # halted
    out = s.BacktestPortfolioTarget(size="weight")(np.full((T, K), 0.25), p)
    assert not np.isnan(out).any()
    assert np.all(out[:50, 3, :] == 0.0)
    halted = out[200:210, 1]
    assert np.all(halted[:, 1] == 0.0) and np.all(halted[:, 2] == out[199, 1, 2])


def test_delay_and_fill_at():
    w = np.full((T, K), np.nan)
    w[10] = 0.25
    out = s.BacktestPortfolioOHLCTarget(size="weight", delay=3)(w, OPEN, HIGH, LOW, CLOSE)
    assert np.all(out[:13, :, 2] == 0.0) and np.all(out[13, :, 2] != 0.0)
    with pytest.raises(ValueError):
        s.BacktestPortfolioOHLCTarget(delay=0, fill_at="open")


# --- regimes and the portfolio report -----------------------------------------

def test_event_by_event_equals_batch():
    w = np.where(rng.random((T, K)) < 0.1, rng.uniform(-0.3, 0.3, (T, K)), np.nan)
    make = lambda: s.BacktestPortfolioOHLCTarget(fee=2e-4, slippage=1e-4, size="weight",
                                                 rebalance_band=0.03, max_gross=1.2)
    batch = make()(w, OPEN, HIGH, LOW, CLOSE)
    op = make()
    events = np.array([op(*row) for row in zip(w, OPEN, HIGH, LOW, CLOSE)])
    np.testing.assert_array_equal(events, batch)
    # A batch call resets, so a second batch is identical.
    np.testing.assert_array_equal(make()(w, OPEN, HIGH, LOW, CLOSE), batch)


def test_portfolio_report_and_capital():
    out = s.BacktestPortfolioTarget(fee=1e-3, size="weight", initial_capital=1e6)(
        np.full((T, K), 0.25), PRICE)
    report = s.PortfolioReport()(out)
    np.testing.assert_allclose(report[-1, 1], out[:, :, 3].sum())     # cumulative cost
    assert report.shape == (T, 6)


def test_argument_checks():
    with pytest.raises(ValueError):
        s.BacktestPortfolioTarget(size="shares")
    with pytest.raises(ValueError):
        s.BacktestPortfolioTarget(size="weight", initial_capital=0.0)
    with pytest.raises(ValueError):
        s.BacktestPortfolioTarget(size="units", initial_capital=0.0, rebalance_band=0.1)
    with pytest.raises(ValueError):
        s.BacktestPortfolioTarget(rebalance="never")
    with pytest.raises(ValueError):
        s.BacktestPortfolioTarget(slippage=1.0, slippage_unit="ticks")
    with pytest.raises(ValueError):
        s.BacktestPortfolioTarget(multiplier=0.0)
    with pytest.raises(TypeError):
        s.BacktestPortfolioTarget()(np.ones((5, 2)))
    with pytest.raises(ValueError):
        s.BacktestPortfolioTarget()(np.ones((5, 2)), np.ones((5, 3)))
