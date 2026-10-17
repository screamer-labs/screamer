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

from .screamer_bindings import _backtest_groups, _backtest_report, _backtest_report_names, _backtest_trades, _check_feed_length

#: Market-order price sentinel for the backtest engines. A quote or limit price of
#: ``MARKET`` (or any non-finite price) is a market order in the aggressive
#: direction; ``NaN`` works as a side-agnostic shorthand.
MARKET = math.inf

__all__ = ["backtest_report", "backtest_trades", "MARKET"]

# Names from C++: the engine columns, the BacktestReport and BacktestStats
# columns, and the summary.
_ENGINE_COLUMNS, _REPORT_COLUMNS, _STATS_COLUMNS, _SUMMARY, _GROUP_STATS = _backtest_report_names()


def _group_codes(by):
    """The labels of ``by`` as int64 codes numbered in order of first
    appearance, and the labels in that order."""
    by = np.asarray(by)
    if by.ndim != 1:
        raise ValueError("backtest_report: by must be a 1-D array of labels, one per row")
    labels, first, inverse = np.unique(by, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order), dtype=np.int64)
    return np.ascontiguousarray(rank[inverse.reshape(-1)]), [labels[j] for j in order]


def backtest_report(values, index=None, periods_per_year=1.0, by=None):
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

    ``by`` is one label per row (any NumPy dtype: ``datetime64[M]`` months,
    integer years, strings), as pandas ``groupby(by=)``. With it, ``summary``
    gains ``by_group``: a dict from label, in order of first appearance, to
    ``total_pnl``, ``total_cost``, ``turnover``, ``sharpe`` (annualized with
    ``periods_per_year``), ``max_drawdown`` (of the cumulative pnl over the
    group's rows), and ``steps`` (its rows without NaN). Trade statistics are
    not grouped, because a trade can span groups; group the rows of
    ``backtest_trades`` by their entry instead.

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
        index = np.asarray(index)
        # One index value per row (checked_index.h), as a Pipeline feed.
        _check_feed_length("backtest_report", -1, len(values),
                           index.shape[0] if index.ndim else 1)
        running["index"] = index
    summary = dict(zip(_SUMMARY, summary))
    if by is not None:
        codes, labels = _group_codes(by)
        table = _backtest_groups(values, codes, len(labels), periods_per_year)
        summary["by_group"] = {
            label: {**dict(zip(_GROUP_STATS, map(float, row))), "steps": int(row[-1])}
            for label, row in zip(labels, table)}
    return running, summary


def backtest_trades(values, index=None):
    """One row per round trip of a backtest engine's output.

    ``values`` is the ``(T, 4)`` array a backtest engine emits, with columns
    ``[equity, pnl, position, cost]``. Returns a dict of equal-length numpy
    arrays, one entry per trade:

    - ``entry``: the row of the trade's first nonzero position (the index
      value when ``index`` is given);
    - ``exit``: the row where the position returns to zero or flips sign, or
      the last row for a trade still open (the index value when given);
    - ``side``: +1 long, -1 short;
    - ``bars``: ``exit - entry`` in rows;
    - ``pnl``: the pnl over the trade's rows (the pnl column is after cost);
    - ``cost``: the cost over the trade's rows;
    - ``closed``: False for a trade still open on the last row.

    Trades are counted as ``BacktestStats`` counts them: flat to flat, a sign
    flip closes one trade and opens the next, and the flip row's pnl goes to
    the closing trade. Rows with a NaN field are skipped. A portfolio output
    ``(T, K, 4)`` raises; call this once per asset. The trades come from C++;
    this wrapper only labels them.
    """
    values = _as_float64(values)
    entry, exit_, side, bars, pnl, cost, closed = _backtest_trades(values)
    if index is not None:
        index = np.asarray(index)
        _check_feed_length("backtest_trades", -1, len(values), index.shape[0] if index.ndim else 1)
        entry, exit_ = index[entry], index[exit_]
    return {"entry": entry, "exit": exit_, "side": side, "bars": bars, "pnl": pnl,
            "cost": cost, "closed": closed.astype(bool)}
