# Backtesting best practices

A backtest estimates how a strategy would have performed. Most backtests that
look good and then fail live have the same few faults: information from the
future, fills that were never available, missing costs, and results that are
luck or the best of many tries. This page goes through those faults and, for
each one, the screamer tool that guards against it.

| Fault | What goes wrong | What screamer provides |
|:------|:----------------|:-----------------------|
| Look-ahead | A decision uses data from after the moment it is made | Causal operators; engines that execute after the decision |
| Unrealistic fills | Trades at prices that were not available | `fill_at`, `delay`, `slippage`, fill rules for limit orders |
| Missing costs | Fees, spread, slippage, and funding left out | Fee, slippage, and carry parameters; a `cost` column |
| Fragile timing | The edge disappears with a small execution delay | `delay` and `fill_at` to test sensitivity; feature shifting |
| Luck | The result is within what chance produces | `position_test`, `strategy_test`, `probabilistic_sharpe` |
| Selection | The reported strategy is the best of many tries | Family-wise `position_test`, `deflated_sharpe` |
| Backtest and live differ | The deployed code is not the tested code | One implementation for batch and live |

## Look-ahead

Look-ahead bias is any use of information that was not available when the
decision was made, such as a signal computed from a bar's close and then traded
at that bar's open.

Every screamer operator is causal: its output at step `t` depends only on inputs
up to step `t`, whether it runs on a stored array or on a live feed. Signals
built from screamer operators therefore cannot see the future by construction.

The remaining risk is in the timing between the signal and the fill. The
backtest engines make that timing explicit:

- [`BacktestOHLCTarget`](functions_fin/BacktestOHLCTarget.md) treats a target
  as decided at a bar's close and executes it `delay` bars later (default 1) at
  the price chosen by `fill_at` (default `"open"`). `delay=0` is allowed only
  with `fill_at="close"`, because every other price of the decision bar was
  already in the past when the signal was known.
- [`BacktestPriceTarget`](functions_fin/BacktestPriceTarget.md) fills at the
  same step's price. That assumes you can trade at the price you just observed;
  if the signal comes from that price, lag it or use `BacktestOHLCTarget`.
- For machine learning, [`forecast_pairs`](functions_supervised/forecast_pairs.md)
  pairs the features at each step with the target over the horizon after it,
  so a model is trained only on the features it would have had. See
  [Labels and training sets](labels.md).

Data you supply can still carry look-ahead, for example an index whose members
were chosen with hindsight, or values revised after the fact. Use data as it was
known at each point in time.

## Realistic fills

A fill price should be one you could have traded at. Choose the engine by the
data you have (see [Choosing a backtest engine](functions_fin/choosing_a_backtest_engine.md)):
bars, the trade tape, or top-of-book quotes. Finer data allows more realistic
fills, especially for limit orders.

For orders that take liquidity, every engine accepts:

- `fill_at` and `delay` (OHLC target engine): when and at which bar price the
  order executes.
- `slippage` with `slippage_unit`: an adverse price offset as a fraction of the
  price (`"fraction"`, so 5 bps is `5e-4`), in ticks of `tick_size`
  (`"ticks"`), or in price units (`"price"`).
- `taker_fee` (or `fee`) and `fee_per_contract`: proportional and fixed fees.

Large orders need one more check: the market may not have traded enough to
fill them at one price. [`BacktestOHLCVTarget`](functions_fin/BacktestOHLCVTarget.md)
fills at most `max_participation` of each bar's volume and carries the rest to
later bars, and `BacktestTradesTarget` does the same per print. To trade a
fraction of equity rather than a fixed number of units, set `size="weight"` and
`initial_capital` on any target engine.

A strategy that trades several assets from one account belongs on the
portfolio engines, which size every asset from the same equity and can cap the
book's gross and net exposure; see [Portfolios and pairs](portfolio_backtesting.md).

For resting limit orders, the `*Orders` engines fill only when the market
reaches the order price. `fill="breach"` requires the market to trade through
it, which is conservative. `participation_ratio` scales the filled size down to
model your place in the queue: on the trade and quote engines it is a share of
the size traded or displayed at your price, and on bars, which carry no volume,
a share of your order.

## Costs and carry

Fees, the spread, and slippage are usually the difference between a strategy
that works on paper and one that works live. Every engine reports them in its
`cost` column: the difference between the fill price and the fair price at
execution, plus fees. The move from the fill price to the next mark is profit
or loss, not cost, so `cost` measures what trading cost you.

Holding a position can also cost money: perpetual futures pay funding, short
positions pay a borrow fee, and leverage pays interest.
[`BacktestCarry`](functions_fin/BacktestCarry.md) adds this to any engine's
output from a per-step rate series.
[`walk_forward`](functions_fin/walk_forward.md) takes the same rate as
`carry=`, so every candidate is chosen and scored after its carry.

Run every strategy with and without costs. A large gap means the edge is small
relative to the cost of trading it.

## Fragile timing

An edge that disappears when execution is one bar later will not survive live
trading, where orders take time to send and fill. Testing timing sensitivity is
cheap: run the same strategy with a few values of `delay`, `fill_at`, and
`slippage` and compare the results.

The example below uses hourly bars whose returns have mild momentum, and a
trend-following signal.

```{eval-rst}
.. exec_code::

    import numpy as np
    from screamer import (BacktestOHLCTarget, EwMean, LogReturn, Sign, FillNa,
                          backtest_report)

    rng = np.random.default_rng(7)
    n = 4000
    r = np.zeros(n)
    e = rng.standard_normal(n) * 0.004
    for t in range(1, n):
        r[t] = 0.12 * r[t - 1] + e[t]
    close = 100 * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]] * np.exp(rng.standard_normal(n) * 0.0005)
    high, low = np.maximum(open_, close) * 1.001, np.minimum(open_, close) * 0.999
    signal = Sign()(FillNa(0.0)(EwMean(span=12)(LogReturn()(close))))

    print("delay  slippage  annual Sharpe")
    for delay in (1, 2, 3):
        for slippage in (0.0, 2e-4, 5e-4):
            engine = BacktestOHLCTarget(taker_fee=2e-4, slippage=slippage, delay=delay)
            out = engine(signal, open_, high, low, close)
            sharpe = backtest_report(out, periods_per_year=24 * 365)[1]["sharpe"]
            print(f"{delay:5d}  {slippage:8.4f}  {sharpe:13.2f}")
```

The edge exists only when the order executes on the next bar. One more bar of
delay removes it, because the momentum in these returns lasts about one bar. A
strategy like this needs fast execution, and its live results will depend on
achieving it.

Timing also matters for the inputs. `strategy_test` with `method="shift"`
rotates chosen feature inputs in time, keeping their own autocorrelation, and
re-runs the strategy. If the result with the features misaligned is as good as
the original, the features carry no timing information about the market.

## Luck

A positive result can come from chance alone. A randomization test measures how
often chance does as well: it reruns the backtest many times on data where the
link between the strategy and the market has been broken, and reports the share
of those runs that do at least as well as the real one, the p-value.

- [`position_test`](functions_fin/position_test.md) keeps the market and the
  strategy's trades and randomizes when the trades happen. It asks whether the
  timing beats random timing with the same exposure, and it needs no rerun of
  the model.
- [`strategy_test`](functions_fin/strategy_test.md) keeps the strategy and
  randomizes the market, by permuting the bars or the returns (see
  [`permute_bars`](functions_fin/permute_bars.md)), and reruns the whole
  strategy, model included, on each copy. Anything the strategy fits or tunes on
  the data is fitted again on every copy, so the test includes the effect of
  that fitting.
- [`probabilistic_sharpe`](functions_fin/probabilistic_sharpe.md) is an
  analytic check with no resampling: the probability that the true Sharpe ratio
  is above a benchmark, given the length of the record and the skewness and
  kurtosis of the pnl.

A p-value covers one statistic. Both randomization tests also return every
`backtest_report` statistic of the real run and of each randomized run, in
`observed_stats` and `null_stats`, so you can see whether the real result
stands out on drawdown, win rate, or turnover as well (see the scatter plots in
[`position_test`](functions_fin/position_test.md#distributions-of-every-statistic)).

[Test whether a strategy's timing beats chance](guide/not-luck/timing-test.md)
runs `position_test` on a daily trend rule and computes `probabilistic_sharpe`
for the result.

Which test answers which question:

| Question | Test | Randomizes | Reruns the strategy | Example |
|---|---|---|---|---|
| Does the timing beat random timing with the same trades? | `position_test` | when the trades happen | no | [Is my strategy's timing real?](notebooks/20-is-the-timing-real.ipynb) |
| Does the best of several variants survive the choice? | `position_test` with a `(T, K)` target, `family_p_value` | all variants' timing together | no | [Check that a result holds across nearby parameters](guide/not-luck/parameter-grid.md) |
| Does a tuned or fitted strategy beat the same procedure on random markets? | `strategy_test`, `method="bars"` or `"returns"` | the market's bars or returns | yes | [I tuned my parameters. Is the result real?](notebooks/21-is-the-tuned-result-real.ipynb) |
| Does a feature carry information? | `strategy_test`, `method="shift"` | the features' alignment with the market | yes | [Does my feature carry information?](notebooks/22-does-the-feature-carry-information.ipynb) |
| Is the Sharpe ratio above zero, or above the best of the variants tried? | `probabilistic_sharpe`, `deflated_sharpe` | nothing (analytic) | no | [Test whether a strategy's timing beats chance](guide/not-luck/timing-test.md), [Check that a result holds across nearby parameters](guide/not-luck/parameter-grid.md) |

`position_test` keeps the strategy's exposure in every randomized run, so a
long-biased strategy in a rising market is tested on its timing only, not on
the trend it rides.

## Selection

If you try many variants (parameters, signals, filters) and report the best,
its result is inflated even when none of them has an edge, because the best of
many random results is high by chance. Keep a record of everything you tried
and correct for it:

- `position_test` with a `(T, K)` target, one column per variant, shifts all
  variants together and returns `family_p_value`. That compares each variant
  with the best of the `K` randomized results in every round (the max statistic
  of White's Reality Check).
- [`deflated_sharpe`](functions_fin/deflated_sharpe.md) raises the benchmark of
  `probabilistic_sharpe` to the Sharpe ratio that the best of your tried
  variants would reach by luck, given how many there were and how much their
  results varied.
- [`walk_forward`](functions_fin/walk_forward.md) chooses a variant on each
  training window and trades it on the next, so the reported result never
  uses the data the choice was made on. With a portfolio engine the
  candidates are whole books, `(T, K, C)`.

[Check that a result holds across nearby parameters](guide/not-luck/parameter-grid.md)
uses the first two, and [Choose parameters with walk-forward
testing](guide/not-luck/walk-forward.md) the third.

Reserve a final stretch of data that no variant was tuned on, and confirm the
chosen variant on it once.

## Measuring results

[`backtest_report`](functions_fin/backtest_report.md) turns an engine's output
into running statistics and a summary: profit, drawdown, cost, turnover, trade
count, Sharpe, and, from [`BacktestStats`](functions_fin/BacktestStats.md),
exposure, win rate, profit factor, average win and loss, holding period,
Sortino, and Calmar. Pass `periods_per_year` (for example 252 for daily bars)
to annualize the ratios; the default 1 gives per-step values.

[`backtest_trades`](functions_fin/backtest_trades.md) lists the round trips
behind those statistics, one row per trade with its entry, exit, side,
length, pnl, and cost. `backtest_report(..., by=labels)` adds the totals,
Sharpe ratio, and drawdown of each group of rows, for example per month with
`by=dates.astype("datetime64[M]")`. A result that comes from one month, or
from a few trades, shows there before it shows in the summary.

Read the statistics together. A high Sharpe ratio with a few trades, a large
share of the profit from one trade, or a profit factor near 1 with high
turnover all point to a result that is less reliable than its headline number.

## From backtest to live

Each screamer operator and engine has one implementation that runs on stored
arrays and on a live feed one event at a time, and gives the same results on
both. A strategy built from screamer operators, or wrapped in a
[`Pipeline`](pipelines.md), is therefore the same code in the backtest and in
production, which removes a common source of differences between the two.
