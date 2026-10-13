"""Backtest reporting helper.

A thin wrapper over the C++ report (include/screamer/backtest_summary.h), which
runs the ``BacktestReport`` and ``BacktestStats`` nodes and reads the summary;
this helper only labels the columns and the summary with the names C++ gives,
as the JavaScript ``backtestReport`` does.
No pandas: ``running`` is a dict of numpy arrays and ``summary`` a dict of floats.
Wrap ``running`` in ``pandas.DataFrame`` yourself if you want a frame.
"""
import math
import numpy as np

from .dag import _as_float64

from .screamer_bindings import _backtest_report, _backtest_report_names

#: Market-order price sentinel for the backtest engines. A quote or limit price of
#: ``MARKET`` (or any non-finite price) is a market order in the aggressive
#: direction; ``NaN`` works as a side-agnostic shorthand.
MARKET = math.inf

__all__ = ["backtest_report", "MARKET"]

# Names from C++: the engine columns, the BacktestReport and BacktestStats
# columns, and the summary.
_ENGINE_COLUMNS, _REPORT_COLUMNS, _STATS_COLUMNS, _SUMMARY = _backtest_report_names()


def backtest_report(values, index=None, periods_per_year=1.0):
    """Running report columns and a summary for a backtest engine's output.

    ``values`` is the ``(T, 4)`` array a backtest engine emits, with columns
    ``[equity, pnl, position, cost]``. Returns ``(running, summary)``:

    - ``running``: a dict of numpy arrays, the four engine columns plus the
      ``BacktestReport`` node's ``drawdown`` (dollar), ``cum_cost``, ``turnover``
      (units traded), ``trades`` (count), ``max_drawdown`` (running worst), and
      ``sharpe`` (running), and the ``BacktestStats`` node's ``exposure``,
      ``round_trips``, ``win_rate``, ``profit_factor``, ``avg_win``,
      ``avg_loss``, ``avg_holding``, ``sortino``, and ``calmar``. Each is a
      causal series whose last finite value is the summary.
    - ``summary``: a dict of the final statistics: ``total_pnl``, ``max_drawdown``,
      ``total_cost``, ``turnover``, ``num_trades``, ``sharpe``, and the nine
      ``BacktestStats`` values.

    ``periods_per_year`` annualizes ``sharpe``, ``sortino``, and ``calmar``
    (for example 252 for daily bars). The default 1 gives per-step ratios.

    The report and the summary come from C++; this wrapper only labels them.
    No pandas dependency.
    """
    values = _as_float64(values)
    report, stats, summary = _backtest_report(values, periods_per_year)
    running = {name: values[:, i] for i, name in enumerate(_ENGINE_COLUMNS)}
    for i, name in enumerate(_REPORT_COLUMNS):
        running[name] = report[:, i]
    for i, name in enumerate(_STATS_COLUMNS):
        running[name] = stats[:, i]
    if index is not None:
        running["index"] = np.asarray(index)
    return running, dict(zip(_SUMMARY, summary))
