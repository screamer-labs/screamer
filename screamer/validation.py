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
"""
from typing import Callable, NamedTuple, Optional, Union

import numpy as np

from .screamer_bindings import (
    BacktestReport,
    _circular_shift,
    _default_min_shift,
    _permute_bars,
    _permute_returns,
    _segment_shuffle,
)

__all__ = ["position_test", "strategy_test", "circular_shift", "segment_shuffle",
           "permute_bars", "permute_returns"]

# Engine output columns: [equity, pnl, position, cost].
# BacktestReport columns: [drawdown, cum_cost, turnover, trades, max_drawdown, sharpe].
_METRICS = {
    # name: (source, column, larger_is_better)
    "total_pnl": ("engine", 0, True),
    "sharpe": ("report", 5, True),
    "max_drawdown": ("report", 4, True),   # <= 0, so closer to 0 is better
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
    """
    observed: float
    null: np.ndarray
    p_value: float
    metric: str
    method: str
    min_shift: Optional[int]


def _last_finite(a):
    """Last finite value along axis 0 for each column of a (T, k) array."""
    finite = np.isfinite(a)
    has = finite.any(axis=0)
    last = a.shape[0] - 1 - np.argmax(finite[::-1], axis=0)
    out = a[last, np.arange(a.shape[1])]
    out[~has] = np.nan
    return out


def _score(engine_out, metric):
    """Metric per column of an engine's (T, k, 4) output."""
    if callable(metric):
        return np.asarray(metric(engine_out), dtype=float).reshape(-1)
    source, col, _ = _METRICS[metric]
    if source == "engine":
        return _last_finite(engine_out[:, :, col])
    cols = [np.ascontiguousarray(engine_out[:, :, i]) for i in range(4)]
    report = np.asarray(BacktestReport()(*cols), dtype=float)
    return _last_finite(report[:, :, col])


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

    ``metric`` is ``"sharpe"`` (per step), ``"total_pnl"``, ``"max_drawdown"``,
    or a callable that maps an engine output of shape ``(T, k, 4)`` to ``k``
    values. For a callable, ``larger_is_better`` defaults to True.

    Order-posting engines (``*Orders``) are not supported: their limit prices
    are tied to the market at each moment, so moving them in time is not
    meaningful.

    Returns a ``PositionTestResult`` with ``observed``, ``null``, ``p_value``,
    ``metric``, ``method``, and ``min_shift``.
    """
    if method not in ("shift", "segments"):
        raise ValueError('method must be "shift" or "segments"')
    if isinstance(metric, str):
        if metric not in _METRICS:
            raise ValueError(f"metric must be one of {sorted(_METRICS)} or a callable")
        name = metric
        if larger_is_better is None:
            larger_is_better = _METRICS[metric][2]
    elif callable(metric):
        name = getattr(metric, "__name__", "custom")
        if larger_is_better is None:
            larger_is_better = True
    else:
        raise TypeError("metric must be a string or a callable")
    if n < 1:
        raise ValueError("n must be at least 1")
    seed = _seed(seed)

    target = _as_series(target, "target")
    market = [_as_series(m, "market input") for m in market]
    T = target.shape[0]
    if any(m.shape[0] != T for m in market):
        raise ValueError("target and market inputs must have the same length")

    engine.reset()
    observed_out = np.asarray(engine(target, *market), dtype=float)[:, None, :]
    observed = float(_score(observed_out, metric)[0])

    shift = None
    if method == "shift":
        shift = int(_default_min_shift(target)) if min_shift is None else int(min_shift)
        if shift < 1:
            raise ValueError("min_shift must be at least 1")

    width = len(market) + 1 + 4 + 6          # inputs, engine output, report
    chunk = max(1, min(n, _CHUNK_DOUBLES // max(1, T * width)))
    null = np.empty(n)
    for col0 in range(0, n, chunk):
        k = min(chunk, n - col0)
        if method == "shift":
            surr = _circular_shift(target, shift, seed, col0, k)
        else:
            surr = _segment_shuffle(target, seed, col0, k)
        cols = [np.broadcast_to(m[:, None], (T, k)) for m in market]
        engine.reset()
        out = np.asarray(engine(surr, *cols), dtype=float)
        null[col0:col0 + k] = _score(out, metric)
    engine.reset()

    null = null[np.isfinite(null)]
    if larger_is_better:
        hits = np.count_nonzero(null >= observed)
    else:
        hits = np.count_nonzero(null <= observed)
    p_value = (1.0 + hits) / (1.0 + null.size) if np.isfinite(observed) else float("nan")
    return PositionTestResult(observed, null, float(p_value), name, method, shift)


def permute_bars(open, high, low, close, n, seed=0):
    """``n`` bar-permuted copies of an OHLC series, as four ``(T, n)`` arrays.

    In log prices each bar after the first splits into a gap (its open over the
    previous close) and an intrabar shape (high, low, and close relative to its
    open). The gaps and the shapes are shuffled independently and the path is
    rebuilt from the first bar (Masters 2020). Each copy keeps the first bar,
    the last close, the distribution of gaps and intrabar shapes, and valid
    bars, but none of the serial structure. Prices must be finite and positive.
    The same ``seed`` gives the same copies on every platform.
    """
    o, h, l, c = (_as_series(a, name) for a, name in
                  ((open, "open"), (high, "high"), (low, "low"), (close, "close")))
    return _permute_bars(o, h, l, c, _seed(seed), 0, int(n))[:4]


def permute_returns(x, n, log=True, seed=0):
    """``n`` copies of a price series with its returns in random order, ``(T, n)``.

    The steps of the series (log returns with ``log=True``, else differences)
    are shuffled and the path is rebuilt from the first valid value. Each copy
    keeps the first and last values and the distribution of returns, but none
    of their order. The NaN warmup prefix stays in place. The same ``seed``
    gives the same copies on every platform.
    """
    return _permute_returns(_as_series(x, "x"), bool(log), _seed(seed), 0, int(n))[0]


class StrategyTestResult(NamedTuple):
    """The outcome of ``strategy_test``; the fields mean the same as in
    ``PositionTestResult``."""
    observed: float
    null: np.ndarray
    p_value: float
    metric: str
    method: str


def _engine_output(out):
    """The (T, 4) engine output from a strategy's return value."""
    if isinstance(out, tuple):            # a Pipeline returns (values, index)
        out = out[0]
    out = np.asarray(out, dtype=float)
    if out.ndim != 2 or out.shape[1] != 4:
        raise ValueError(
            "strategy must return a backtest engine output of shape (T, 4) "
            "[equity, pnl, position, cost], or a (values, index) pair holding one")
    return out


def strategy_test(
    strategy: Callable,
    *inputs,
    n: int = 500,
    method: str = "bars",
    ohlc=(0, 1, 2, 3),
    price: int = 0,
    log: Optional[bool] = None,
    columns=None,
    fixed=(),
    metric: Union[str, Callable] = "sharpe",
    larger_is_better: Optional[bool] = None,
    min_shift: Optional[int] = None,
    seed: int = 0,
) -> StrategyTestResult:
    """Test a whole strategy against the same strategy run on randomized data.

    ``strategy`` is any callable that takes ``inputs`` positionally and returns
    a backtest engine's ``(T, 4)`` output: a function built from screamer ops,
    a compiled ``Pipeline`` (its ``(values, index)`` result is accepted), or
    your own code. The test runs it once on the real inputs and ``n`` times on
    randomized copies, and reports where the real metric falls.

    ``method`` chooses what is randomized:

    - ``"bars"``: the OHLC inputs at positions ``ohlc`` are replaced by
      bar-permuted copies (see ``permute_bars``). The null hypothesis is that
      the market has no serial structure the strategy can use. Because the
      whole strategy runs again, anything it fits or tunes on the data is
      re-fitted on every copy, so the test also accounts for that fitting.
    - ``"returns"``: the single price input at position ``price`` is replaced
      by return-permuted copies (see ``permute_returns``). ``log`` chooses log
      returns or differences; the default uses log returns when every price is
      positive.
    - ``"shift"``: the inputs at positions ``columns`` (features) are rotated
      in time by one random offset, the same for all of them, while prices
      stay as they are. This asks whether those features carry information,
      with their own autocorrelation and their relations to each other kept.
      ``min_shift`` defaults to a tenth of the series.

    With ``"bars"`` and ``"returns"``, every other input moves with its bar,
    so per-bar data such as volume keeps its relation to the bar it belongs
    to. List inputs that must not move, such as timestamps, in ``fixed``.

    ``metric`` is ``"sharpe"`` (per step), ``"total_pnl"``, ``"max_drawdown"``,
    or a callable that maps an output of shape ``(T, k, 4)`` to ``k`` values.
    The p-value counts the observed result itself, so it is never zero.
    """
    if method not in ("bars", "returns", "shift"):
        raise ValueError('method must be "bars", "returns", or "shift"')
    if isinstance(metric, str):
        if metric not in _METRICS:
            raise ValueError(f"metric must be one of {sorted(_METRICS)} or a callable")
        name = metric
        if larger_is_better is None:
            larger_is_better = _METRICS[metric][2]
    elif callable(metric):
        name = getattr(metric, "__name__", "custom")
        if larger_is_better is None:
            larger_is_better = True
    else:
        raise TypeError("metric must be a string or a callable")
    if n < 1:
        raise ValueError("n must be at least 1")
    seed = _seed(seed)
    data = [_as_series(x, "input") for x in inputs]
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
    if method == "bars":
        price_cols = check_positions(ohlc, "ohlc")
        if len(price_cols) != 4:
            raise ValueError("ohlc must list the positions of open, high, low, close")
    elif method == "returns":
        price_cols = check_positions([price], "price")
        if log is None:
            px = data[price_cols[0]]
            log = bool(np.all(px[np.isfinite(px)] > 0))
    else:
        if not columns:
            raise ValueError('method="shift" needs the feature positions in columns')
        price_cols = []
        shifted = check_positions(columns, "columns")
        shift = max(1, T // 10) if min_shift is None else int(min_shift)
        if shift < 1 or 2 * shift > T:
            raise ValueError("min_shift must be in [1, T / 2]")
        positions = np.arange(T, dtype=float)

    observed = float(_score(_engine_output(strategy(*data))[:, None, :], metric)[0])

    null = np.empty(n)
    chunk = max(1, min(n, 64))
    for col0 in range(0, n, chunk):
        m = min(chunk, n - col0)
        if method == "bars":
            o, h, l, c, src = _permute_bars(*(data[i] for i in price_cols), seed, col0, m)
            replaced = dict(zip(price_cols, (o, h, l, c)))
        elif method == "returns":
            p, src = _permute_returns(data[price_cols[0]], bool(log), seed, col0, m)
            replaced = {price_cols[0]: p}
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
            out = _engine_output(strategy(*run))
            null[col0 + j] = _score(out[:, None, :], metric)[0]

    null = null[np.isfinite(null)]
    if larger_is_better:
        hits = np.count_nonzero(null >= observed)
    else:
        hits = np.count_nonzero(null <= observed)
    p_value = (1.0 + hits) / (1.0 + null.size) if np.isfinite(observed) else float("nan")
    return StrategyTestResult(observed, null, float(p_value), name, method)
