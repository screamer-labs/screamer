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
they are validation tools, not streaming operators. Each test runs in a C++
driver it shares with JavaScript (``include/screamer/validation/``:
``position_test.h`` for ``position_test`` and ``walk_forward``,
``strategy_test.h`` for ``strategy_test``), which calls the engine, the
strategy, and a custom metric back; here they only convert the arguments and
the result.

``walk_forward`` is the out-of-sample counterpart: it picks the best of
several strategy variants on each training window and trades the pick on the
window that follows.
"""
import operator
from typing import Callable, NamedTuple, Optional, Union

import numpy as np

from .dag import _as_float64

from .screamer_bindings import (
    RowOp,
    _circular_shift,
    _deflated_sharpe,
    _permute_bars,
    _permute_returns,
    _position_test,
    _probabilistic_sharpe,
    _segment_shuffle,
    _strategy_test,
    _walk_forward,
)

__all__ = ["position_test", "strategy_test", "circular_shift", "segment_shuffle",
           "permute_bars", "permute_returns", "probabilistic_sharpe", "deflated_sharpe",
           "walk_forward"]

_INT64 = (-2**63, 2**63 - 1)


def _no_bool(x, name, integer):
    """A number argument is not a bool (a NumPy bool or a 0-d bool array
    either): TypeError "<function>: <argument> must be an integer, got bool"."""
    if isinstance(x, (bool, np.bool_)) or (
            isinstance(x, np.ndarray) and x.ndim == 0 and x.dtype.kind == "b"):
        raise TypeError(f"{name} must be {'an integer' if integer else 'a number'}, got bool")
    return x


def _int64(x, name):
    """An integer argument for C++: values outside int64 are clamped, so they
    fail the range check of their argument with its own message."""
    return min(max(int(_no_bool(x, name, True)), _INT64[0]), _INT64[1])


def _int(x, name):
    """An integer argument, not a bool."""
    return int(_no_bool(x, name, True))


def _float(x, name):
    """A float argument, not a bool."""
    return float(_no_bool(x, name, False))


def _count(x, name):
    """A count such as n: an integer, not a bool (TypeError otherwise)."""
    return _int64(operator.index(_no_bool(x, name, True)), name)


def _number(x, name):
    """A size argument for C++ as a float, which checks it is an integer; a
    string becomes NaN, which fails that check. A bool is a TypeError."""
    return float("nan") if isinstance(x, (str, bytes)) else float(_no_bool(x, name, True))


def _flag(x):
    """An optional flag such as larger_is_better: None, or its truth value."""
    return None if x is None else bool(x)


def _metric_label(metric):
    """The metric's name for the result, and whether it is a callable. C++
    checks a statistic name and its direction."""
    if isinstance(metric, str):
        return metric, False
    if callable(metric):
        return getattr(metric, "__name__", "custom"), True
    raise TypeError("metric must be a string or a callable")


def _engine_calls(engine, market, metric):
    """The engine and the metric as the C++ drivers call them: run(target)
    runs a fresh engine and returns a float64 output, and score(out) gives
    the metric of a (T, k, 4) output (a (T, 4) output is read as (T, 1, 4))
    as a 1-D array."""
    def run(target):
        engine.reset()
        return np.ascontiguousarray(engine(target, *market), dtype=float)

    return run, engine.reset, _metric_call(metric)


# The kinds C++ checks (MetricKind in include/screamer/validation/scoring.h).
_NUMBER, _NUMBERS, _OTHER = 0, 1, 2
_NO_VALUES = np.empty(0)


def _is_number(v):
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


def _metric_result(value):
    """(kind, values) of a custom metric's result, as JavaScript classifies
    its own: a number; a 1-D array of numbers (a numeric 1-D ndarray, or a
    flat list or tuple of numbers); anything else, such as a bool, a string,
    a nested list, or a 2-D array. C++ checks the kind and the count."""
    if _is_number(value):
        return _NUMBER, np.array([value], dtype=float)
    if isinstance(value, np.ndarray):
        if value.ndim <= 1 and value.dtype.kind in "iuf":
            return _NUMBERS, np.ascontiguousarray(value, dtype=float).reshape(-1)
        return _OTHER, _NO_VALUES
    if isinstance(value, (list, tuple)) and all(_is_number(v) for v in value):
        return _NUMBERS, np.array(value, dtype=float).reshape(-1)
    return _OTHER, _NO_VALUES


def _metric_call(metric):
    """score(out): a callable metric of a (T, k, 4) output (a (T, 4) output
    is read as (T, 1, 4)), as (kind, values) for C++ to check; None for a
    statistic name."""
    if not callable(metric):
        return None

    def score(out):
        return _metric_result(metric(out if out.ndim == 3 else out[:, None, :]))
    return score


def _as_series(x, name):
    x = _as_float64(x, contiguous=True)
    if x.ndim != 1:
        raise ValueError(f"{name} must be a 1-D series")
    return x


def _as_block(x, name):
    """A float64 input of shape (T,) or (T, K), for strategy_test."""
    if x is None or isinstance(x, (str, bytes, dict)):
        raise TypeError(f"{name} must be a numeric array, not {type(x).__name__}")
    try:
        x = _as_float64(x)
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
    m = -1 if min_shift is None else _int(min_shift, "circular_shift: min_shift")
    if min_shift is not None and m < 1:
        raise ValueError("min_shift must be at least 1")
    return _circular_shift(x, m, _int64(seed, "circular_shift: seed"), 0, _int(n, "circular_shift: n"))


def segment_shuffle(x, n, seed=0):
    """``n`` copies of a series with its runs in random order, as ``(T, n)``.

    The series after its leading NaN warmup is split into runs of equal value
    (each trade, and each flat or hold stretch, is one run), and the runs are
    written back in a random order in which no two neighbouring runs have the
    same value, so no two runs merge. Every copy holds exactly the same runs:
    the same trades with the same sides, sizes, and holding periods, at random
    times, with as many position changes as the original. The same ``seed``
    gives the same columns on every platform.
    """
    return _segment_shuffle(_as_series(x, "x"), _int64(seed, "segment_shuffle: seed"), 0,
                            _int(n, "segment_shuffle: n"))


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
    name, custom = _metric_label(metric)
    target = _as_float64(target)
    data = np.ascontiguousarray(target)
    market = [_as_float64(m, contiguous=True) for m in market]
    run, reset, score = _engine_calls(engine, market, metric)
    observed, null, p_value, family, observed_stats, null_stats, shift = _position_test(
        run, reset, score, target if target.ndim == 2 else data, data, market,
        _count(n, "position_test: n"), method, name, custom, _flag(larger_is_better),
        None if min_shift is None else _int64(min_shift, "position_test: min_shift"),
        _int64(seed, "position_test: seed"),
        _float(periods_per_year, "position_test: periods_per_year"),
        isinstance(engine, RowOp))
    return PositionTestResult(observed, null, p_value, name, method, shift, family,
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
    out = _permute_bars(o, h, l, c, mode, _int64(seed, "permute_bars: seed"), 0,
                        _int(n, "permute_bars: n"), v,
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
    out = _permute_returns(_as_series(x, "x"), mode, _int64(seed, "permute_returns: seed"), 0,
                           _int(n, "permute_returns: n"), v,
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
    structure over time is destroyed. With ``"returns"``, the rows before
    every asset has its first value stay in place in every copy, so an asset
    listed late freezes the rows before its listing; ``"bars"`` needs finite
    prices everywhere. A ``"notional"`` volume must have the shape of the
    prices.

    ``metric`` is the name of a ``backtest_report`` summary statistic or a
    callable that maps an output of shape ``(T, k, 4)`` to ``k`` values, as in
    ``position_test``. The p-value counts the observed result itself, so it is
    never zero. ``observed_stats`` and ``null_stats`` in the result hold every
    statistic of the real and the randomized runs.
    """
    name, custom = _metric_label(metric)
    data = [_as_block(x, "input") for x in inputs]

    def run(*args):
        out = strategy(*args)
        if isinstance(out, tuple):            # a Pipeline returns (values, index)
            out = out[0]
        return np.ascontiguousarray(out, dtype=float)

    score = _metric_call(metric)

    positions = lambda pos: [_int64(i, "strategy_test: ohlc") for i in pos]  # noqa: E731
    observed, null, p_value, observed_stats, null_stats = _strategy_test(
        run, score, data, _count(n, "strategy_test: n"), method, positions(ohlc),
        _int64(price, "strategy_test: price"), mode,
        None if volume is None else _int64(volume, "strategy_test: volume"), volume_mode,
        [] if columns is None else positions(columns), positions(fixed),
        name, custom, _flag(larger_is_better),
        None if min_shift is None else _int64(min_shift, "strategy_test: min_shift"),
        _int64(seed, "strategy_test: seed"), _float(periods_per_year, "strategy_test: periods_per_year"))
    return StrategyTestResult(observed, null, p_value, name, method, observed_stats, null_stats)


def _pnl_of(values):
    """Per-step pnl from a pnl series or a backtest engine's (T, 4) output."""
    values = _as_float64(values)
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
    return float(_probabilistic_sharpe(_pnl_of(pnl),
                                       _float(benchmark, "probabilistic_sharpe: benchmark")))


def deflated_sharpe(pnl, trial_sharpes):
    """Probabilistic Sharpe ratio against the best of the strategies you tried.

    ``trial_sharpes`` holds the per-step Sharpe ratio of every strategy variant
    you tried, this one included. Their spread and number set the Sharpe ratio
    that the best of them would reach by luck alone, and that becomes the
    benchmark (Bailey and Lopez de Prado 2014). Returns
    ``(probability, benchmark)``: the probability that this strategy's true
    Sharpe ratio is above that benchmark, and the benchmark itself.
    """
    trials = _as_float64(trial_sharpes, contiguous=True).reshape(-1)
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
    name, custom = _metric_label(metric)
    targets = _as_float64(targets)
    if targets.ndim == 1:
        targets = targets[:, None]
    targets = np.ascontiguousarray(targets)
    prices = [_as_float64(p, contiguous=True) for p in prices]
    run, reset, score = _engine_calls(engine, prices, metric)
    target, choice, fold_start, scores, output, stats = _walk_forward(
        run, reset, score, targets, targets, prices, _number(train, "walk_forward: train"),
        _number(test, "walk_forward: test"), name, custom,
        _flag(larger_is_better), bool(anchored),
        _float(periods_per_year, "walk_forward: periods_per_year"), isinstance(engine, RowOp))
    return WalkForwardResult(target, choice, fold_start, scores, output, stats, name)
