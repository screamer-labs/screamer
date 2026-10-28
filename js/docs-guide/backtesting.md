---
title: Backtesting and significance tests
---

# Backtesting and significance tests

A backtest engine turns a target position and market data into a costed equity curve.
`backtestReport` summarizes that curve, and `positionTest` and `strategyTest` measure whether
the result could come from chance. This page shows the JavaScript calls. The Python guides
[Backtesting best practices](https://screamer.readthedocs.io/en/latest/backtesting_best_practices.html)
and [Portfolios and pairs](https://screamer.readthedocs.io/en/latest/portfolio_backtesting.html)
cover the reasoning in depth, and every name here has the same meaning there.

## A target-position backtest

A target-position engine takes the position you want to hold at each step and trades toward
it. `BacktestOHLCTarget` reads OHLC bars: a target decided at the close of bar `t` executes at
the open of bar `t + 1` (`delay` and `fillAt` change this). The engine does the lag, so pass
the signal as it is known at each bar and do not shift it yourself.

```js
import {
  ready, BacktestOHLCTarget, EwMean, LogReturn, Sign, FillNa, backtestReport,
} from "@screamer-labs/screamer";

await ready();

// Synthetic hourly bars whose returns have mild momentum (a seeded AR(1) walk).
let seed = 7;
const uniform = () => (seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5;
const T = 2000;
const open = new Float64Array(T), high = new Float64Array(T);
const low = new Float64Array(T), close = new Float64Array(T);
for (let t = 0, r = 0, p = 100; t < T; t++) {
  r = 0.15 * r + 0.01 * uniform();
  open[t] = p;
  p *= Math.exp(r);
  close[t] = p;
  high[t] = Math.max(open[t], close[t]) * 1.001;
  low[t] = Math.min(open[t], close[t]) * 0.999;
}

// Long when the smoothed return is positive, short when it is negative.
const target = Sign()(FillNa(0)(EwMean({ span: 12 })(LogReturn()(close))));

const engine = BacktestOHLCTarget({ takerFee: 2e-4 });
const out = engine(target, open, high, low, close); // NdArray, shape [2000, 4]
```

The output has one row per bar with the columns `[equity, pnl, position, cost]`. Fees,
slippage, position limits, stops, and sizing by notional or weight are options of the engine.
Other engines read a price series (`BacktestPriceTarget`), the trade tape
(`BacktestTradesTarget`), top-of-book quotes (`BacktestL1Target`), or post limit orders
(`BacktestOHLCOrders` and others); see
[Choosing a backtest engine](https://screamer.readthedocs.io/en/latest/functions_fin/choosing_a_backtest_engine.html).
`BacktestPriceTarget` fills at the same step's price, so a signal computed from that price
needs a lag there.

## The report

`backtestReport(out)` returns `running`, an object of report columns (drawdown, cost, turnover,
trades, Sharpe ratio, win rate, and more), and `summary`, their final values.
`periodsPerYear` annualizes the ratios; the default 1 gives per-step values.

```js
const { running, summary } = backtestReport(out, { periodsPerYear: 24 * 365 });
running.drawdown;    // Float64Array(2000)
summary.sharpe;      // 8.045...
summary.num_trades;  // 304
```

The keys are those of Python's `backtest_report`.

## Portfolios

`BacktestPortfolioTarget` and `BacktestPortfolioOHLCTarget` trade K assets from one capital
account. They take one row of K values per input and step (an array of rows or a `[T, K]`
`NdArray`) and return a `[T, K, 4]` output. `PortfolioReport` reduces that output to
portfolio-level columns. Cross-sectional ops such as `CrossRank` and `CrossNormalize` turn
per-asset scores into weights; see [Calling conventions](./regimes.md) for how these row ops
are called.

```js
import { BacktestPortfolioTarget, PortfolioReport } from "@screamer-labs/screamer";

const weights = [[0.5, -0.5], [0.5, -0.5], [-0.5, 0.5]]; // two assets, three steps
const prices = [[100, 50], [101, 49], [103, 50]];
const book = BacktestPortfolioTarget({ fee: 1e-3, size: "weight", initialCapital: 1000 });
const nd = (rows) => ({ data: new Float64Array(rows.flat()), shape: [rows.length, 2] });
const result = book(nd(weights), nd(prices)); // NdArray, shape [3, 2, 4]
PortfolioReport()(result);                    // NdArray, shape [3, 6]
```

## Significance tests

A good backtest can be luck. A randomization test reruns the backtest many times with the link
between the strategy and the market broken, and reports the p-value: the share of runs that do
at least as well as the real one, counting the real run itself.

**`positionTest(engine, target, market, opts)`** keeps the market and the strategy's trades
and randomizes when the trades happen. Use it for a fixed strategy, one whose rules were not
tuned on this data. `method: "shift"` (the default) rotates the target series in time;
`"segments"` shuffles the order of its runs of constant position. `metric` names a summary
statistic (`"sharpe"` by default) or is a function of one run's `[T, 4]` engine output.

```js
import { positionTest } from "@screamer-labs/screamer";

const res = positionTest(engine, target, [open, high, low, close], { n: 500, seed: 1 });
res.observed;                   // 0.0860..., the per-step Sharpe ratio of the real run
res.pValue;                     // 0.001996..., 1 / 501
res.nullStats.max_drawdown;     // Float64Array(500), one value per randomized run
```

`observedStats` and `nullStats` hold every report statistic of the real run and of each
randomized run, for plotting their distributions. A `[T, K]` target tests K variants together
and adds `familyPValue`, which corrects for having picked the best of them. With a portfolio
engine, the target and the market inputs are `[T, K]` and the result is one test of the book.

**`strategyTest(strategy, inputs, opts)`** keeps the strategy and randomizes the market. It
calls `strategy(...inputs)` on the real data and on each randomized copy, so anything the
strategy fits or tunes is fitted again every time. Use it when the strategy was tuned on the
same data. `method: "bars"` (the default) permutes the OHLC bars, `"returns"` permutes a price
series, and `"shift"` rotates feature inputs to test whether they carry timing information.

```js
import { strategyTest } from "@screamer-labs/screamer";

function strategy(open, high, low, close) {
  const target = Sign()(FillNa(0)(EwMean({ span: 12 })(LogReturn()(close))));
  return BacktestOHLCTarget({ takerFee: 2e-4 })(target, open, high, low, close);
}

const st = strategyTest(strategy, [open, high, low, close], { n: 200, seed: 1 });
st.pValue; // 0.004975..., 1 / 201
```

`probabilisticSharpe` and `deflatedSharpe` give an analytic check with no resampling,
`walkForward` selects among candidate targets on rolling training windows, and
`timeSplits(n, nSplits, { labelLength })` gives the train and test rows of time-ordered
folds for a model, leaving out the training rows whose labels reach into the test block.

## The randomizers

The tests build their randomized data with functions you can call directly, for example to
plot a surrogate:

- `circularShift(x, n, opts)`: `n` copies of a series, each rotated in time by a random offset,
  as a `[T, n]` `NdArray`.
- `segmentShuffle(x, n, opts)`: `n` copies of a series with its runs in random order.
- `permuteBars(open, high, low, close, n, opts)`: `n` permuted copies of an OHLC series, as four
  `[T, n]` arrays, keeping the distribution of returns and intrabar shapes.
- `permuteReturns(x, n, opts)`: `n` copies of a price series with its returns permuted.

```js
import { permuteBars, circularShift } from "@screamer-labs/screamer";

const [o, h, l, c] = permuteBars(open, high, low, close, 100, { seed: 1 }); // each [2000, 100]
circularShift([0, 0, 1, 1, 1, -1, -1, 0], 3, { seed: 1 });                // NdArray, shape [8, 3]
```

The same `seed` gives the same randomized data as Python.
