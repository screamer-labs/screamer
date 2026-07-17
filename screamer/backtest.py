"""Backtest reporting helper.

A thin convenience layer over the backtest engines (`BacktestSignal` and, later,
the market-data engines). It only aggregates an engine's already-computed
`[equity, pnl, position, cost]` output into the common running statistics and a
summary; it adds no operator logic of its own, so it works for every engine in
the family.
"""
import numpy as np
import pandas as pd

__all__ = ["backtest_report"]


def backtest_report(values, index=None):
    """Running statistics and a summary for a backtest engine's output.

    ``values`` is the ``(T, 4)`` array a backtest engine emits, with columns
    ``[equity, pnl, position, cost]``. Returns ``(running, summary)``:

    - ``running``: a pandas ``DataFrame`` with one row per bar, carrying the
      engine columns plus the running ``drawdown`` (dollar), ``cum_cost``,
      ``turnover`` (units traded), and ``trades`` (count). Each is a causal
      series whose last value is the summary.
    - ``summary``: a pandas ``Series`` of the final statistics: ``total_pnl``,
      ``max_drawdown``, ``total_cost``, ``turnover``, ``num_trades``, ``sharpe``.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(
            "values must be a (T, 4) array of [equity, pnl, position, cost]")
    idx = pd.RangeIndex(len(values)) if index is None else pd.Index(np.asarray(index))
    equity = pd.Series(values[:, 0], index=idx)
    pnl = pd.Series(values[:, 1], index=idx)
    position = pd.Series(values[:, 2], index=idx)
    cost = pd.Series(values[:, 3], index=idx)

    # Units traded per bar, including the initial move from flat.
    traded = position.diff()
    if len(traded):
        traded.iloc[0] = position.iloc[0]
    traded = traded.abs()

    eq = equity.ffill().fillna(0.0)                 # a skipped (NaN) bar holds the last equity
    running = pd.DataFrame({
        "equity": equity,
        "pnl": pnl,
        "position": position,
        "cost": cost,
        "drawdown": eq - eq.cummax(),               # dollar drawdown, <= 0
        "cum_cost": cost.fillna(0.0).cumsum(),
        "turnover": traded.fillna(0.0).cumsum(),
        "trades": (traded.fillna(0.0) > 0).cumsum().astype(float),
    })
    pnl_valid = pnl.dropna()
    summary = pd.Series({
        "total_pnl": eq.iloc[-1] if len(eq) else np.nan,
        "max_drawdown": running["drawdown"].min() if len(running) else np.nan,
        "total_cost": running["cum_cost"].iloc[-1] if len(running) else np.nan,
        "turnover": running["turnover"].iloc[-1] if len(running) else np.nan,
        "num_trades": running["trades"].iloc[-1] if len(running) else np.nan,
        "sharpe": (pnl_valid.mean() / pnl_valid.std()
                   if len(pnl_valid) > 1 and pnl_valid.std() > 0 else np.nan),
    })
    return running, summary
