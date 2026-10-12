"""Randomization tests for backtest significance.

A backtest result can come from skill or from luck. Two tests measure how
often luck alone does as well:

- ``position_test`` keeps the market data and the strategy's trading profile,
  randomizes when the strategy holds its positions, and replays each copy
  through the same backtest engine.
- ``strategy_test`` keeps the strategy and randomizes its data: it permutes
  the market (``"bars"`` or ``"returns"``) or shifts chosen feature inputs
  (``"shift"``), and re-runs the whole strategy, model included, on each copy.

These functions are batch-only. They read whole series to build surrogates, so
they are validation tools, not streaming operators. The surrogates are built
in C++ (``circular_shift``, ``segment_shuffle``), and the backtests and the
report are the ordinary C++ engines run on ``(T, n)`` columns. This module
only chains them and counts.

``walk_forward`` is the out-of-sample counterpart: it picks the best of
several strategy variants on each training window and trades the pick on the
window that follows.
"""
from typing import Callable, NamedTuple, Optional, Union

import numpy as np

from .screamer_bindings import (
    BacktestReport,
    BacktestStats,
    PortfolioReport,
    RowOp,
    _circular_shift,
    _default_min_shift,
    _deflated_sharpe,
    _probabilistic_sharpe,
    _permute_bars,
    _permute_returns,
    _segment_shuffle,
)

__all__ = ["position_test", "strategy_test", "circular_shift", "segment_shuffle",
           "permute_bars", "permute_returns", "probabilistic_sharpe", "deflated_sharpe",
           "walk_forward"]

# The statistics computed for every run, with the same names as the summary of
# backtest_report, and their direction as a test metric (None: no natural
# direction, so larger_is_better must be given).
# Engine output columns: [equity, pnl, position, cost].
# BacktestReport columns: [drawdown, cum_cost, turnover, trades, max_drawdown, sharpe].
# BacktestStats columns: [exposure, round_trips, win_rate, profit_factor, avg_win,
#                         avg_loss, avg_holding, sortino, calmar].
_METRICS = {
    # name: (source, column, larger_is_better)
    "total_pnl": ("engine", 0, True),
    "max_drawdown": ("report", 4, True),   # <= 0, so closer to 0 is better
    "total_cost": ("report", 1, False),
    "turnover": ("report", 2, None),
    "num_trades": ("report", 3, None),
    "sharpe": ("report", 5, True),
    "exposure": ("stats", 0, None),
    "round_trips": ("stats", 1, None),
    "win_rate": ("stats", 2, True),
    "profit_factor": ("stats", 3, True),
    "avg_win": ("stats", 4, True),
    "avg_loss": ("stats", 5, True),        # <= 0, so closer to 0 is better
    "avg_holding": ("stats", 6, None),
    "sortino": ("stats", 7, True),
    "calmar": ("stats", 8, True),
}

# Doubles per chunk, about 256 MB, for the (T, chunk) working arrays.
_CHUNK_DOUBLES = 32_000_000


def _seed(seed):
    # Seeds are limited to [0, 2**53) so they are exact JavaScript numbers too:
    # a seed gives the same surrogates in Python and in JS/WASM.
    seed = int(seed)
    if not 0 <= seed < 2**53:
        raise ValueError("seed must be an integer in [0, 2**53)")
    return seed


def _as_series(x, name):
    x = np.ascontiguousarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"{name} must be a 1-D series")
    return x


def _as_block(x, name):
    """A float64 input of shape (T,) or (T, K), for strategy_test."""
    if x is None or isinstance(x, (str, bytes, dict)):
        raise TypeError(f"{name} must be a numeric array, not {type(x).__name__}")
    try:
        x = np.asarray(x, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric array: {exc}") from None
    if x.ndim not in (1, 2):
        raise ValueError(f"{name} must be 1-D (T,) or 2-D (T, K), got {x.ndim}-D")
    return np.ascontiguousarray(x)


def circular_shift(x, n, min_shift=None, seed=0):
    """``n`` circularly shifted copies of a series, as a ``(T, n)`` array.

    Each column is ``x`` rotated by a random offset in
    ``[min_shift, L - min_shift]``, where ``L`` is the length after the leading
    NaN warmup, which stays in place. A rotation keeps every property of the
    series (trade count, holding periods, long/short balance, autocorrelation)
    and changes only its alignment in time. ``min_shift=None`` uses twice the
    mean run length, capped at a quarter of ``L``. The same ``seed`` gives the
    same columns on every platform.
    """
    x = _as_series(x, "x")
    m = -1 if min_shift is None else int(min_shift)
    if min_shift is not None and m < 1:
        raise ValueError("min_shift must be at least 1")
    return _circular_shift(x, m, _seed(seed), 0, int(n))


def segment_shuffle(x, n, seed=0):
    """``n`` copies of a series with its runs in random order, as ``(T, n)``.

    The series after its leading NaN warmup is split into runs of equal value
    (each trade, and each flat or hold stretch, is one run), and the runs are
    written back in a random order. Every copy holds exactly the same runs:
    the same trades with the same sides, sizes, and holding periods, at random
    times. The same ``seed`` gives the same columns on every platform.
    """
    return _segment_shuffle(_as_series(x, "x"), _seed(seed), 0, int(n))


class PositionTestResult(NamedTuple):
    """The outcome of ``position_test``.

    ``observed`` is the metric of the real backtest and ``null`` the metric of
    each randomized backtest. ``p_value`` is the share of the null that does
    at least as well as ``observed``, counting the observed result itself, so
    it is never zero. Randomized runs whose metric is NaN are left out of
    ``null``.

    For ``K`` strategies (a ``(T, K)`` target), ``observed`` and ``p_value``
    have one entry per strategy, ``null`` has shape ``(n, K)`` (NaN where a
    run's metric is NaN), and ``family_p_value`` compares each strategy with
    the best of all ``K`` in every randomized round. That corrects for having
    picked the best of several variants. For a single strategy it is None.

    ``observed_stats`` and ``null_stats`` hold every backtest statistic, not
    only the tested metric, under the names of the ``backtest_report``
    summary: ``observed_stats[name]`` is the real run's value and
    ``null_stats[name]`` an array with one entry per randomized run, in run
    order and with NaN kept, so two statistics of the same run line up for a
    scatter plot. For ``K`` strategies they have shapes ``(K,)`` and
    ``(n, K)``.
    """
    observed: Union[float, np.ndarray]
    null: np.ndarray
    p_value: Union[float, np.ndarray]
    metric: str
    method: str
    min_shift: Optional[int]
    family_p_value: Optional[np.ndarray] = None
    observed_stats: Optional[dict] = None
    null_stats: Optional[dict] = None


def _last_finite(a):
    """Last finite value along axis 0 for each column of a (T, k) array."""
    finite = np.isfinite(a)
    has = finite.any(axis=0)
    last = a.shape[0] - 1 - np.argmax(finite[::-1], axis=0)
    out = a[last, np.arange(a.shape[1])]
    out[~has] = np.nan
    return out


def _run_stats(engine_out, periods_per_year):
    """Every statistic per column of an engine's (T, k, 4) output."""
    cols = [np.ascontiguousarray(engine_out[:, :, i]) for i in range(4)]
    src = {"engine": engine_out,
           "report": np.asarray(BacktestReport(periods_per_year)(*cols), dtype=float),
           "stats": np.asarray(BacktestStats(periods_per_year)(*cols), dtype=float)}
    return {name: _last_finite(src[s][:, :, c]) for name, (s, c, _) in _METRICS.items()}


def _book_stats(out, periods_per_year):
    """Every statistic of one portfolio run, from its (T, K, 4) output.

    The book is the sum over assets: equity, pnl, and cost add up, and its
    position is the gross position, so the trade statistics count a round trip
    from a flat book to a flat book. Turnover and the trade count come from
    PortfolioReport, which counts each asset's position changes.
    """
    out = np.ascontiguousarray(out, dtype=float)
    book = [out[:, :, 0].sum(axis=1), out[:, :, 1].sum(axis=1),
            np.abs(out[:, :, 2]).sum(axis=1), out[:, :, 3].sum(axis=1)]
    report = np.asarray(BacktestReport(periods_per_year)(*book), dtype=float)
    per_asset = np.asarray(PortfolioReport()(out), dtype=float)
    report[:, 2:4] = per_asset[:, 2:4]                 # turnover, trades
    src = {"engine": np.column_stack(book)[:, None, :],
           "report": report[:, None, :],
           "stats": np.asarray(BacktestStats(periods_per_year)(*book), dtype=float)[:, None, :]}
    return {name: _last_finite(src[s][:, :, c]) for name, (s, c, _) in _METRICS.items()}


def _score(engine_out, metric, stats):
    """Metric per column of an engine's (T, k, 4) output."""
    if callable(metric):
        return np.asarray(metric(engine_out), dtype=float).reshape(-1)
    return stats[metric]


def _metric_name(metric, larger_is_better):
    """The metric's name and its direction, checked."""
    if isinstance(metric, str):
        if metric not in _METRICS:
            raise ValueError(f"metric must be one of {sorted(_METRICS)} or a callable")
        if larger_is_better is None:
            larger_is_better = _METRICS[metric][2]
        if larger_is_better is None:
            raise ValueError(f'metric "{metric}" has no natural direction; '
                             "pass larger_is_better")
        return metric, larger_is_better
    if callable(metric):
        name = getattr(metric, "__name__", "custom")
        return name, True if larger_is_better is None else larger_is_better
    raise TypeError("metric must be a string or a callable")


def position_test(
    engine,
    target,
    *market,
    n: int = 1000,
    method: str = "shift",
    metric: Union[str, Callable] = "sharpe",
    larger_is_better: Optional[bool] = None,
    min_shift: Optional[int] = None,
    seed: int = 0,
    periods_per_year: float = 1.0,
) -> PositionTestResult:
    """Test whether a strategy's timing beats randomly timed copies of itself.

    ``engine`` is a target-position backtest engine, such as
    ``BacktestOHLCTarget(taker_fee=5e-4)``. ``target`` is the strategy's target
    series and ``market`` the engine's market inputs, exactly as you would call
    ``engine(target, *market)``.

    The test builds ``n`` surrogate target series that keep the strategy's
    trading profile but break its link to the market, runs each through a
    fresh copy of the engine, and reports where the real metric falls:

    - ``method="shift"``: rotate the whole target series by a random offset.
      Every property of the series is kept; only the timing changes.
    - ``method="segments"``: shuffle the order of the runs (trades and flat
      stretches). The same trades occur, at random times.

    The null hypothesis is that the timing carries no information about future
    returns, given this trading profile. A long-biased strategy in a rising
    market therefore gets a long-biased null, so the test measures timing
    beyond exposure.

    ``metric`` is the name of a ``backtest_report`` summary statistic
    (``"sharpe"``, ``"total_pnl"``, ``"max_drawdown"``, ``"sortino"``,
    ``"win_rate"``, ...) or a callable that maps an engine output of shape
    ``(T, k, 4)`` to ``k`` values. For a callable, ``larger_is_better``
    defaults to True; statistics with no natural direction, such as
    ``"turnover"``, need it. ``periods_per_year`` annualizes the ratios, as in
    ``backtest_report``.

    Several strategies: pass a ``(T, K)`` target, one column per variant (with
    ``method="shift"``). Every randomized round shifts all ``K`` columns by the
    same offset, and the result adds ``family_p_value``: each strategy's metric
    against the best of the ``K`` randomized metrics in each round (the max
    statistic of White's Reality Check). Use it when the strategy you report
    is the best of several you tried.

    Order-posting engines (``*Orders``) are not supported: their limit prices
    are tied to the market at each moment, so moving them in time is not
    meaningful.

    Returns a ``PositionTestResult`` with ``observed``, ``null``, ``p_value``,
    ``metric``, ``method``, ``min_shift``, ``family_p_value``, and every
    statistic of the real and the randomized runs in ``observed_stats`` and
    ``null_stats``.
    """
    if method not in ("shift", "segments"):
        raise ValueError('method must be "shift" or "segments"')
    name, larger_is_better = _metric_name(metric, larger_is_better)
    if n < 1:
        raise ValueError("n must be at least 1")
    seed = _seed(seed)

    target = np.asarray(target, dtype=float)
    if isinstance(engine, RowOp):
        return _position_test_portfolio(engine, target, market, n, method, metric, name,
                                        larger_is_better, min_shift, seed, periods_per_year)
    if target.ndim == 2:
        return _position_test_family(engine, target, market, n, method, metric, name,
                                     larger_is_better, min_shift, seed, periods_per_year)
    target = _as_series(target, "target")
    market = [_as_series(m, "market input") for m in market]
    T = target.shape[0]
    if any(m.shape[0] != T for m in market):
        raise ValueError("target and market inputs must have the same length")

    engine.reset()
    observed_out = np.asarray(engine(target, *market), dtype=float)[:, None, :]
    stats = _run_stats(observed_out, periods_per_year)
    observed = float(_score(observed_out, metric, stats)[0])
    observed_stats = {k: float(v[0]) for k, v in stats.items()}

    shift = None
    if method == "shift":
        shift = int(_default_min_shift(target)) if min_shift is None else int(min_shift)
        if shift < 1:
            raise ValueError("min_shift must be at least 1")

    width = len(market) + 1 + 4 + 6 + 9      # inputs, engine output, report, stats
    chunk = max(1, min(n, _CHUNK_DOUBLES // max(1, T * width)))
    null = np.empty(n)
    null_stats = {key: np.empty(n) for key in _METRICS}
    for col0 in range(0, n, chunk):
        k = min(chunk, n - col0)
        if method == "shift":
            surr = _circular_shift(target, shift, seed, col0, k)
        else:
            surr = _segment_shuffle(target, seed, col0, k)
        engine.reset()
        out = np.asarray(engine(surr, *market), dtype=float)
        stats = _run_stats(out, periods_per_year)
        for key, v in stats.items():
            null_stats[key][col0:col0 + k] = v
        null[col0:col0 + k] = _score(out, metric, stats)
    engine.reset()

    null = null[np.isfinite(null)]
    if larger_is_better:
        hits = np.count_nonzero(null >= observed)
    else:
        hits = np.count_nonzero(null <= observed)
    p_value = (1.0 + hits) / (1.0 + null.size) if np.isfinite(observed) else float("nan")
    return PositionTestResult(observed, null, float(p_value), name, method, shift, None,
                              observed_stats, null_stats)


def _position_test_family(engine, target, market, n, method, metric, name,
                          larger_is_better, min_shift, seed, periods_per_year):
    """position_test for a (T, K) target: a joint shift and a max statistic."""
    if method != "shift":
        raise ValueError('several strategies need method="shift", which shifts them jointly')
    T, K = target.shape
    market = [_as_series(m, "market input") for m in market]
    if any(m.shape[0] != T for m in market):
        raise ValueError("target and market inputs must have the same length")
    cols = [np.ascontiguousarray(target[:, k]) for k in range(K)]
    start, shift = _joint_shift_setup(cols, T, min_shift)    # one offset for all K

    engine.reset()
    observed_out = np.asarray(engine(target, *market), dtype=float)
    observed_stats = _run_stats(observed_out, periods_per_year)
    observed = _score(observed_out, metric, observed_stats)
    width = (len(market) + 1 + 4 + 6 + 9) * K
    chunk = max(1, min(n, _CHUNK_DOUBLES // max(1, T * width)))
    null = np.empty((n, K))
    null_stats = {key: np.empty((n, K)) for key in _METRICS}
    for col0 in range(0, n, chunk):
        m = min(chunk, n - col0)
        # (T, m * K): replica-major blocks, all K strategies share each offset.
        surr = np.hstack([_circular_shift(x, shift, seed, col0, m, start) for x in cols])
        engine.reset()
        out = np.asarray(engine(surr, *market), dtype=float)
        stats = _run_stats(out, periods_per_year)
        for key, v in stats.items():
            null_stats[key][col0:col0 + m] = v.reshape(K, m).T
        null[col0:col0 + m] = _score(out, metric, stats).reshape(K, m).T
    engine.reset()

    sign = 1.0 if larger_is_better else -1.0
    p_value = np.empty(K)
    for k in range(K):
        col = null[:, k][np.isfinite(null[:, k])]
        hits = np.count_nonzero(sign * col >= sign * observed[k])
        p_value[k] = (1.0 + hits) / (1.0 + col.size) if np.isfinite(observed[k]) else np.nan
    best = np.nanmax(sign * null, axis=1)          # the best strategy in each round
    best = best[np.isfinite(best)]
    family = np.array([(1.0 + np.count_nonzero(best >= sign * o)) / (1.0 + best.size)
                       if np.isfinite(o) else np.nan for o in observed])
    return PositionTestResult(observed, null, p_value, name, method, shift, family,
                              observed_stats, null_stats)


def _joint_shift_setup(cols, T, min_shift):
    """Common warmup start and minimum shift for columns shifted together."""
    def warmup(x):
        finite = np.flatnonzero(~np.isnan(x))
        return int(finite[0]) if finite.size else T
    start = max(warmup(x) for x in cols)
    if min_shift is None:
        shift = max(int(_default_min_shift(x)) for x in cols)
        shift = max(1, min(shift, (T - start) // 4))
    else:
        shift = int(min_shift)
    if shift < 1:
        raise ValueError("min_shift must be at least 1")
    return start, shift


def _position_test_portfolio(engine, target, market, n, method, metric, name,
                             larger_is_better, min_shift, seed, periods_per_year):
    """position_test for a portfolio engine: one book of K assets, shifted together."""
    if method != "shift":
        raise ValueError('a portfolio is tested with method="shift", which moves all assets together')
    if target.ndim != 2:
        raise ValueError("a portfolio engine needs a (T, K) target")
    T, K = target.shape
    market = [np.ascontiguousarray(m, dtype=float) for m in market]
    if any(m.shape != target.shape for m in market):
        raise ValueError("target and market inputs must all have shape (T, K)")
    cols = [np.ascontiguousarray(target[:, k]) for k in range(K)]
    start, shift = _joint_shift_setup(cols, T, min_shift)

    def run(tgt):
        engine.reset()
        out = np.asarray(engine(tgt, *market), dtype=float)
        stats = _book_stats(out, periods_per_year)
        value = (float(np.asarray(metric(out), dtype=float).reshape(-1)[0]) if callable(metric)
                 else float(stats[metric][0]))
        return value, {k: float(v[0]) for k, v in stats.items()}

    observed, observed_stats = run(target)
    null = np.empty(n)
    null_stats = {key: np.empty(n) for key in _METRICS}
    chunk = max(1, min(n, 256))
    for col0 in range(0, n, chunk):
        m = min(chunk, n - col0)
        blocks = [_circular_shift(x, shift, seed, col0, m, start) for x in cols]   # K x (T, m)
        for j in range(m):
            tgt = np.column_stack([b[:, j] for b in blocks])
            null[col0 + j], stats = run(tgt)
            for key, v in stats.items():
                null_stats[key][col0 + j] = v
    engine.reset()
    finite = null[np.isfinite(null)]
    hits = np.count_nonzero(finite >= observed if larger_is_better else finite <= observed)
    p_value = (1.0 + hits) / (1.0 + finite.size) if np.isfinite(observed) else float("nan")
    return PositionTestResult(observed, finite, float(p_value), name, method, shift, None,
                              observed_stats, null_stats)


_EMPTY = np.empty(0)
_MODES = ("log", "diff", "level")
_VOLUME_MODES = ("keep", "notional")


def _check_modes(mode, volume_mode):
    if mode not in _MODES:
        raise ValueError('mode must be "log", "diff", or "level"')
    if volume_mode not in _VOLUME_MODES:
        raise ValueError('volume_mode must be "keep" or "notional"')


def permute_bars(open, high, low, close, n, mode="log", seed=0, volume=None,
                 volume_mode="keep"):
    """``n`` permuted copies of an OHLC series, as four ``(T, n)`` arrays.

    ``mode`` sets what is shuffled:

    - ``"log"`` (default): each bar after the first splits into a gap (its open
      over the previous close) and an intrabar shape (high, low, and close over
      its open), as log ratios. The gaps and the shapes are shuffled
      independently and the path is rebuilt from the first bar (Masters 2020).
      Each copy keeps the first bar, the last close, the distributions of gaps
      and shapes, and valid bars, but none of the serial structure. Prices must
      be positive.
    - ``"diff"``: the same with differences instead of log ratios, for prices
      that can be zero or negative (spreads, rates).
    - ``"level"``: whole bars are shuffled. Use it for series that are
      stationary in level, where a rebuilt path would be meaningless.

    With ``volume``, a fifth ``(T, n)`` array holds the volume of the bar each
    row's shape came from. ``volume_mode="keep"`` keeps it as it is;
    ``"notional"`` rescales it so the traded amount, volume times the typical
    price (high + low + close) / 3, stays that of its source bar: where a copy
    trades at twice the price, the volume halves. The same ``seed`` gives the
    same copies on every platform.
    """
    _check_modes(mode, volume_mode)
    o, h, l, c = (_as_series(a, name) for a, name in
                  ((open, "open"), (high, "high"), (low, "low"), (close, "close")))
    v = _EMPTY if volume is None else _as_series(volume, "volume")
    out = _permute_bars(o, h, l, c, mode, _seed(seed), 0, int(n), v,
                        volume_mode == "notional")
    return out[:4] if volume is None else (*out[:4], out[5])


def permute_returns(x, n, mode="log", seed=0, volume=None, volume_mode="keep"):
    """``n`` permuted copies of a price series, as a ``(T, n)`` array.

    ``mode`` sets what is shuffled: ``"log"`` (default) shuffles the log
    returns and ``"diff"`` the differences, and both rebuild the path from the
    first valid value, so each copy keeps the first and last values and the
    distribution of changes but not their order. ``"level"`` shuffles the
    values themselves, for series that are stationary in level. The NaN warmup
    prefix stays in place. With ``"log"`` the values must be positive.

    With ``volume``, returns ``(prices, volume)``: the volume of the row each
    step came from, kept as it is (``volume_mode="keep"``) or rescaled so that
    volume times price stays that of its source row (``"notional"``). The same
    ``seed`` gives the same copies on every platform.
    """
    _check_modes(mode, volume_mode)
    v = _EMPTY if volume is None else _as_series(volume, "volume")
    out = _permute_returns(_as_series(x, "x"), mode, _seed(seed), 0, int(n), v,
                           volume_mode == "notional")
    return out[0] if volume is None else (out[0], out[2])


class StrategyTestResult(NamedTuple):
    """The outcome of ``strategy_test``; the fields mean the same as in
    ``PositionTestResult``."""
    observed: float
    null: np.ndarray
    p_value: float
    metric: str
    method: str
    observed_stats: Optional[dict] = None
    null_stats: Optional[dict] = None


def _engine_output(out):
    """The engine output from a strategy's return value: (T, 4), or (T, K, 4)
    for a portfolio."""
    if isinstance(out, tuple):            # a Pipeline returns (values, index)
        out = out[0]
    out = np.asarray(out, dtype=float)
    if not ((out.ndim == 2 and out.shape[1] == 4) or (out.ndim == 3 and out.shape[2] == 4)):
        raise ValueError(
            "strategy must return a backtest engine output of shape (T, 4) "
            "[equity, pnl, position, cost], a portfolio output of shape (T, K, 4), "
            "or a (values, index) pair holding one")
    return out


def strategy_test(
    strategy: Callable,
    *inputs,
    n: int = 500,
    method: str = "bars",
    ohlc=(0, 1, 2, 3),
    price: int = 0,
    mode: str = "log",
    volume: Optional[int] = None,
    volume_mode: str = "keep",
    columns=None,
    fixed=(),
    metric: Union[str, Callable] = "sharpe",
    larger_is_better: Optional[bool] = None,
    min_shift: Optional[int] = None,
    seed: int = 0,
    periods_per_year: float = 1.0,
) -> StrategyTestResult:
    """Test a whole strategy against the same strategy run on randomized data.

    ``strategy`` is any callable that takes ``inputs`` positionally and returns
    a backtest engine's ``(T, 4)`` output: a function built from screamer ops,
    a compiled ``Pipeline`` (its ``(values, index)`` result is accepted), or
    your own code. The test runs it once on the real inputs and ``n`` times on
    randomized copies, and reports where the real metric falls.

    ``method`` chooses what is randomized:

    - ``"bars"``: the OHLC inputs at positions ``ohlc`` are replaced by
      permuted copies (see ``permute_bars``). The null hypothesis is that the
      market has no serial structure the strategy can use. Because the whole
      strategy runs again, anything it fits or tunes on the data is re-fitted
      on every copy, so the test also accounts for that fitting.
    - ``"returns"``: the single price input at position ``price`` is replaced
      by permuted copies (see ``permute_returns``).

    For both, ``mode`` is ``"log"`` (shuffle log returns, the default),
    ``"diff"`` (shuffle differences, for prices that can be zero or negative),
    or ``"level"`` (shuffle the values or whole bars, for series that are
    stationary in level). ``volume`` is the position of a volume input:
    ``volume_mode="keep"`` moves it with its bar unchanged, and ``"notional"``
    also rescales it so the traded amount stays that of its source bar.
    - ``"shift"``: the inputs at positions ``columns`` (features) are rotated
      in time by one random offset, the same for all of them, while prices
      stay as they are. This asks whether those features carry information,
      with their own autocorrelation and their relations to each other kept.
      ``min_shift`` defaults to a tenth of the series.

    With ``"bars"`` and ``"returns"``, every other input moves with its bar,
    so per-bar data such as volume keeps its relation to the bar it belongs
    to. List inputs that must not move, such as timestamps, in ``fixed``.

    Each input is a series of shape ``(T,)`` or a panel of shape ``(T, K)``
    with one column per asset, and series and panels can be mixed. For a
    panel, each randomized copy applies one time permutation (or one shift) to
    every asset, so the correlation between assets is kept and only the
    structure over time is destroyed. A ``"notional"`` volume must have the
    shape of the prices.

    ``metric`` is the name of a ``backtest_report`` summary statistic or a
    callable that maps an output of shape ``(T, k, 4)`` to ``k`` values, as in
    ``position_test``. The p-value counts the observed result itself, so it is
    never zero. ``observed_stats`` and ``null_stats`` in the result hold every
    statistic of the real and the randomized runs.
    """
    if method not in ("bars", "returns", "shift"):
        raise ValueError('method must be "bars", "returns", or "shift"')
    name, larger_is_better = _metric_name(metric, larger_is_better)
    if n < 1:
        raise ValueError("n must be at least 1")
    seed = _seed(seed)
    data = [_as_block(x, "input") for x in inputs]
    if not data:
        raise ValueError("strategy_test needs at least one input")
    T = data[0].shape[0]
    if any(x.shape[0] != T for x in data):
        raise ValueError("all inputs must have the same length")
    k = len(data)

    def check_positions(pos, what):
        pos = [int(i) for i in pos]
        if any(i < 0 or i >= k for i in pos):
            raise ValueError(f"{what} refers to an input that does not exist")
        return pos

    fixed = set(check_positions(fixed, "fixed"))
    _check_modes(mode, volume_mode)
    vol_col = None if volume is None else check_positions([volume], "volume")[0]
    if vol_col is not None and method == "shift":
        raise ValueError('volume applies to method="bars" or "returns"')
    if method == "bars":
        price_cols = check_positions(ohlc, "ohlc")
        if len(price_cols) != 4:
            raise ValueError("ohlc must list the positions of open, high, low, close")
        if any(data[i].shape != data[price_cols[0]].shape for i in price_cols):
            raise ValueError("open, high, low, and close must have the same shape")
    elif method == "returns":
        price_cols = check_positions([price], "price")
    else:
        if not columns:
            raise ValueError('method="shift" needs the feature positions in columns')
        price_cols = []
        shifted = check_positions(columns, "columns")
        shift = max(1, T // 10) if min_shift is None else int(min_shift)
        if shift < 1 or 2 * shift > T:
            raise ValueError("min_shift must be in [1, T / 2]")
        positions = np.arange(T, dtype=float)

    def run_stats(out):
        out = _engine_output(out)
        if out.ndim == 3:                  # a portfolio: score the whole book
            stats = _book_stats(out, periods_per_year)
        else:
            out = out[:, None, :]
            stats = _run_stats(out, periods_per_year)
        return (float(_score(out, metric, stats)[0]),
                {k: float(v[0]) for k, v in stats.items()})

    # A volume with the shape of the prices goes to the permutation kernel.
    # Otherwise, such as one (T,) volume next to (T, K) prices, "keep" moves it
    # with its bar like any other input, and "notional" cannot rescale it.
    notional = volume_mode == "notional"
    if vol_col is not None and data[vol_col].shape != data[price_cols[0]].shape:
        if notional:
            raise ValueError('volume_mode="notional" needs a volume input with the '
                             "shape of the prices")
        vol_col = None
    vol_block = _EMPTY if vol_col is None else data[vol_col]

    observed, observed_stats = run_stats(strategy(*data))

    null = np.empty(n)
    null_stats = {key: np.empty(n) for key in _METRICS}
    # Copies per chunk: up to 64, fewer for a large panel (six (T, m, K) blocks).
    width = max(x.size for x in data)
    chunk = max(1, min(n, 64, _CHUNK_DOUBLES // (6 * width)))
    for col0 in range(0, n, chunk):
        m = min(chunk, n - col0)
        if method == "bars":
            out = _permute_bars(*(data[i] for i in price_cols), mode, seed, col0, m,
                                vol_block, notional)
            src = out[4]
            replaced = dict(zip(price_cols, out[:4]))
            if vol_col is not None:
                replaced[vol_col] = out[5]
        elif method == "returns":
            out = _permute_returns(data[price_cols[0]], mode, seed, col0, m, vol_block,
                                   notional)
            src = out[1]
            replaced = {price_cols[0]: out[0]}
            if vol_col is not None:
                replaced[vol_col] = out[2]
        else:
            src = _circular_shift(positions, shift, seed, col0, m)
            replaced = {}
        src = src.astype(np.intp)
        for j in range(m):
            run = []
            for i, x in enumerate(data):
                if i in replaced:
                    run.append(np.ascontiguousarray(replaced[i][:, j]))
                elif i in fixed or (method == "shift" and i not in shifted):
                    run.append(x)
                else:
                    run.append(x[src[:, j]])
            null[col0 + j], stats = run_stats(strategy(*run))
            for key, v in stats.items():
                null_stats[key][col0 + j] = v

    null = null[np.isfinite(null)]
    if larger_is_better:
        hits = np.count_nonzero(null >= observed)
    else:
        hits = np.count_nonzero(null <= observed)
    p_value = (1.0 + hits) / (1.0 + null.size) if np.isfinite(observed) else float("nan")
    return StrategyTestResult(observed, null, float(p_value), name, method,
                              observed_stats, null_stats)


def _pnl_of(values):
    """Per-step pnl from a pnl series or a backtest engine's (T, 4) output."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 2 and values.shape[1] == 4:
        values = values[:, 1]
    return _as_series(values, "pnl")


def probabilistic_sharpe(pnl, benchmark=0.0):
    """Probability that the true Sharpe ratio is above ``benchmark``.

    ``pnl`` is a per-step pnl series, or a backtest engine's ``(T, 4)`` output
    whose pnl column is used. The observed per-step Sharpe ratio is adjusted
    for the sample length and for the skewness and kurtosis of the pnl
    (Bailey and Lopez de Prado 2012), so a short or fat-tailed record counts
    for less. ``benchmark`` is a per-step Sharpe ratio. NaN values are skipped.
    """
    return float(_probabilistic_sharpe(_pnl_of(pnl), float(benchmark)))


def deflated_sharpe(pnl, trial_sharpes):
    """Probabilistic Sharpe ratio against the best of the strategies you tried.

    ``trial_sharpes`` holds the per-step Sharpe ratio of every strategy variant
    you tried, this one included. Their spread and number set the Sharpe ratio
    that the best of them would reach by luck alone, and that becomes the
    benchmark (Bailey and Lopez de Prado 2014). Returns
    ``(probability, benchmark)``: the probability that this strategy's true
    Sharpe ratio is above that benchmark, and the benchmark itself.
    """
    trials = np.ascontiguousarray(trial_sharpes, dtype=float).reshape(-1)
    p, bench = _deflated_sharpe(_pnl_of(pnl), trials)
    return float(p), float(bench)


class WalkForwardResult(NamedTuple):
    """The outcome of ``walk_forward``.

    ``target`` is the stitched out-of-sample target, ``choice`` the column
    chosen for each fold (-1 when no candidate had a score), ``fold_start``
    the first step of each test window, and ``train_scores`` the metric of
    every candidate on every training window, shape ``(n_folds, K)``.
    ``output`` is the engine output on ``target`` and ``stats`` every
    statistic of that output over the out-of-sample span, from the first
    test window to the end.
    """
    target: np.ndarray
    choice: np.ndarray
    fold_start: np.ndarray
    train_scores: np.ndarray
    output: np.ndarray
    stats: dict
    metric: str


def _window(out, a, b):
    """Rows [a, b) of a (T, k, 4) engine output, scored as a backtest of its own.

    The equity column is shifted so the window starts from the equity the run
    started from: its total_pnl, drawdowns, and calmar ratio then cover the
    window only. A position held at the window start counts as entered there.
    """
    w = np.array(out[a:b], dtype=float)
    if a > 0:
        eq, pnl = out[:, :, 0], out[:, :, 1]
        prior = _last_finite(eq[:a])                  # equity before the window
        finite = ~np.isnan(eq)
        first = np.argmax(finite, axis=0)
        cols = np.arange(eq.shape[1])
        start = eq[first, cols] - pnl[first, cols]    # equity the run started from
        shift = np.where(np.isnan(prior), 0.0, prior - start)
        w[:, :, 0] -= shift
    return w


def _pick(scores, larger_is_better):
    """The best column of one fold's scores; NaN loses, ties go to the lowest
    index, and -1 means every score is NaN."""
    valid = ~np.isnan(scores)
    if not valid.any():
        return -1
    signed = np.where(valid, scores if larger_is_better else -scores, -np.inf)
    best = signed[valid].max()
    return int(np.flatnonzero(valid & (signed == best))[0])


def walk_forward(
    engine,
    targets,
    *prices,
    train: int,
    test: int,
    metric: Union[str, Callable] = "sharpe",
    anchored: bool = False,
    periods_per_year: float = 1.0,
    larger_is_better: Optional[bool] = None,
) -> WalkForwardResult:
    """Walk-forward selection: pick the best candidate on each training window
    and trade it on the next test window.

    ``targets`` is a ``(T, K)`` array with one target series per candidate
    (parameter variant) of one strategy, and ``prices`` the engine's market
    inputs, as in ``engine(target, *prices)``. A ``(T,)`` target is one
    candidate.

    The folds start at ``t0 = train, train + test, train + 2 * test, ...``
    while ``t0 < T``. Fold ``i`` tests on ``[t0, min(t0 + test, T))`` and
    trains on ``[t0 - train, t0)``, or on ``[0, t0)`` with ``anchored=True``.
    Every candidate is backtested once over the whole history, and on each
    training window the candidate with the best ``metric`` is chosen. A
    window is scored as a backtest of its own: its equity starts where the run
    started and a position held at its start counts as entered there. NaN
    scores lose, ties go to the lowest column, and a fold where every score is
    NaN stays flat with choice -1.

    The out-of-sample target holds the chosen candidate's target in each test
    window and 0 (flat) before the first one. It runs through the engine once,
    so positions and costs carry across fold boundaries, and ``stats`` holds
    every ``backtest_report`` statistic of that run from the first test window
    on.

    ``metric``, ``larger_is_better``, and ``periods_per_year`` mean the same as
    in ``position_test``. Portfolio engines are not supported.
    """
    if isinstance(engine, RowOp):
        raise ValueError("walk_forward needs a single-asset engine; portfolio engines "
                         "are not supported")
    name, larger_is_better = _metric_name(metric, larger_is_better)
    targets = np.asarray(targets, dtype=float)
    if targets.ndim == 1:
        targets = targets[:, None]
    if targets.ndim != 2:
        raise ValueError("targets must be a (T,) or (T, K) array")
    targets = np.ascontiguousarray(targets)
    T, K = targets.shape
    prices = [_as_series(p, "price input") for p in prices]
    if any(p.shape[0] != T for p in prices):
        raise ValueError("targets and price inputs must have the same length")
    if int(train) != train or train < 2:
        raise ValueError("train must be an integer of at least 2")
    if int(test) != test or test < 1:
        raise ValueError("test must be an integer of at least 1")
    train, test = int(train), int(test)
    if train >= T:
        raise ValueError("train must be shorter than the series")

    fold_start = np.arange(train, T, test)
    engine.reset()
    full = np.asarray(engine(targets, *prices), dtype=float).reshape(T, K, 4)

    scores = np.empty((fold_start.size, K))
    choice = np.empty(fold_start.size, dtype=int)
    target = np.zeros(T)
    for i, t0 in enumerate(fold_start):
        w = _window(full, 0 if anchored else t0 - train, t0)
        stats = None if callable(metric) else _run_stats(w, periods_per_year)
        scores[i] = _score(w, metric, stats)
        choice[i] = _pick(scores[i], larger_is_better)
        if choice[i] >= 0:
            target[t0:t0 + test] = targets[t0:t0 + test, choice[i]]

    engine.reset()
    output = np.asarray(engine(target, *prices), dtype=float)
    engine.reset()
    oos = _run_stats(_window(output[:, None, :], int(fold_start[0]), T), periods_per_year)
    return WalkForwardResult(target, choice, fold_start, scores, output,
                             {k: float(v[0]) for k, v in oos.items()}, name)
