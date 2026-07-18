"""Backtest reporting helper.

A thin wrapper over the C++ ``BacktestReport`` node. That node does all the
aggregation (drawdown, cumulative cost, turnover, trade count, running Sharpe);
this helper only labels its columns and reads the last row for the summary, so
pure-C++ users get the same functionality by calling ``BacktestReport`` directly.
No pandas: ``running`` is a dict of numpy arrays and ``summary`` a dict of floats.
Wrap ``running`` in ``pandas.DataFrame`` yourself if you want a frame.
"""
import numpy as np

from .screamer_bindings import BacktestReport

__all__ = ["backtest_report"]

# Column order emitted by the C++ BacktestReport node.
_REPORT_COLUMNS = ("drawdown", "cum_cost", "turnover", "trades", "max_drawdown", "sharpe")


def backtest_report(values, index=None):
    """Running report columns and a summary for a backtest engine's output.

    ``values`` is the ``(T, 4)`` array a backtest engine emits, with columns
    ``[equity, pnl, position, cost]``. Returns ``(running, summary)``:

    - ``running``: a dict of numpy arrays, the four engine columns plus the
      ``BacktestReport`` node's ``drawdown`` (dollar), ``cum_cost``, ``turnover``
      (units traded), ``trades`` (count), ``max_drawdown`` (running worst), and
      ``sharpe`` (running). Each is a causal series whose last finite value is the
      summary.
    - ``summary``: a dict of the final statistics: ``total_pnl``, ``max_drawdown``,
      ``total_cost``, ``turnover``, ``num_trades``, ``sharpe``.

    The aggregation lives in the C++ ``BacktestReport`` node; this wrapper only
    labels its columns and reads the last finite row. No pandas dependency.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(
            "values must be a (T, 4) array of [equity, pnl, position, cost]")
    equity, pnl, position, cost = (values[:, i] for i in range(4))

    report = np.asarray(BacktestReport()(equity, pnl, position, cost), dtype=float)
    running = {"equity": equity, "pnl": pnl, "position": position, "cost": cost}
    for i, name in enumerate(_REPORT_COLUMNS):
        running[name] = report[:, i]
    if index is not None:
        running["index"] = np.asarray(index)

    def _last_finite(a):
        finite = a[np.isfinite(a)]
        return float(finite[-1]) if finite.size else float("nan")

    summary = {
        "total_pnl": _last_finite(equity),
        "max_drawdown": _last_finite(running["max_drawdown"]),
        "total_cost": _last_finite(running["cum_cost"]),
        "turnover": _last_finite(running["turnover"]),
        "num_trades": _last_finite(running["trades"]),
        "sharpe": _last_finite(running["sharpe"]),
    }
    return running, summary
