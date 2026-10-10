# What you can build

This page groups screamer by task. Each tile names one area, links to a worked
example, and lists the functions to start with. The [gallery](gallery.rst) shows a plot for every
function.

```{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Indicators
:img-top: _static/gallery/recipes__pandas-ohlcv-indicators.webp
:img-alt: Bollinger Bands, RSI, and ATR on one-minute BTC bars

Rolling statistics and technical indicators on arrays or DataFrame columns,
from moving averages to RSI, MACD, Bollinger Bands, and ATR.
+++
Example: [Indicators on a pandas OHLCV DataFrame](recipes/pandas-ohlcv-indicators.md),
[Financial indicators notebook](notebooks/03-financial-indicators.ipynb).
Functions: [`RollingRSI`](functions_rolling/RollingRSI.md),
[`MACD`](functions_rolling/MACD.md),
[`BollingerBands`](functions_rolling/BollingerBands.md),
[all indicators](by_group/indicators.rst).
:::

:::{grid-item-card} Filtering and cycles
:img-top: _static/gallery/functions_signal__MAMA.webp
:img-alt: MAMA adaptive moving average on a cycling series

Low-lag smoothers, Butterworth filters, a Kalman filter, and Ehlers' cycle tools
that measure the dominant cycle and adapt to it.
+++
Example: [Signal processing notebook](notebooks/04-signal-processing.ipynb),
[Cycle and spectral analysis notebook](notebooks/19-cycle-and-spectral-analysis.ipynb).
Functions: [`SuperSmoother`](functions_signal/SuperSmoother.md),
[`KalmanFilter`](functions_signal/KalmanFilter.md),
[`DominantCycle`](functions_signal/DominantCycle.md),
[`MAMA`](functions_signal/MAMA.md).
:::

:::{grid-item-card} Bars from ticks
:img-top: _static/gallery/capability__bars.webp
:img-alt: BTC trades with ten-minute OHLC bars

Turn a trade tape into time, tick-count, volume, or dollar bars, with OHLCV and
buy/sell volume aggregations.
+++
Example: [Bars from ticks notebook](notebooks/09-bars-from-ticks.ipynb),
[Custom and multi-column bars](notebooks/10-custom-and-multi-column-bars.ipynb).
Functions: [`Resample`](functions_streams/Resample.md),
[`to_pandas`](functions_streams/to_pandas.md).
:::

:::{grid-item-card} Microstructure
:img-top: _static/gallery/capability__microstructure.webp
:img-alt: BTC price and order flow, true and inferred from the tick rule

Order flow and liquidity from trades and quotes: trade signing, order-flow
imbalance, price impact, VPIN, and spreads.
+++
Example: [Order flow and trade signs notebook](notebooks/11-microstructure-order-flow.ipynb),
[Market microstructure guide](microstructure.md).
Functions: [`TickRuleSign`](functions_micro/TickRuleSign.md),
[`OFI`](functions_micro/OFI.md), [`VPIN`](functions_micro/VPIN.md),
[`RollingKyleLambda`](functions_micro/RollingKyleLambda.md).
:::

:::{grid-item-card} Backtesting
:img-top: _static/gallery/recipes__ticks-to-backtest.webp
:img-alt: Equity curves with and without trading costs

Costed backtests on prices, bars, trades, or top-of-book quotes, with fees,
slippage, fill timing, and a randomization test of whether the result beats
chance.
+++
Example: [From ticks to a costed backtest](recipes/ticks-to-backtest.md),
[Is my backtest significant?](recipes/significance-test.md),
[Choosing a backtest engine](functions_fin/choosing_a_backtest_engine.md),
[Backtesting best practices](backtesting_best_practices.md).
Functions: [`BacktestOHLCTarget`](functions_fin/BacktestOHLCTarget.md),
[`backtest_report`](functions_fin/backtest_report.md),
[`position_test`](functions_fin/position_test.md).
:::

:::{grid-item-card} Forecasting and ML
:img-top: _static/gallery/recipes__ml-features.webp
:img-alt: Predicted against realized returns, and feature weights

Build training sets from causal features, fit online regressions, and filter a
latent signal with uncertainty bands.
+++
Example: [A feature matrix for machine learning](recipes/ml-features.md),
[Supervised forecasting notebook](notebooks/16-supervised-forecasting.ipynb),
[Filtering and forecasting with uncertainty](notebooks/17-filtering-and-forecasting-with-uncertainty.ipynb).
Functions: [`forecast_pairs`](functions_supervised/forecast_pairs.md),
[`BayesianRegression`](functions_fin/BayesianRegression.md),
[`KalmanFilter`](functions_signal/KalmanFilter.md).
:::

:::{grid-item-card} Live streaming and pipelines
:img-top: _static/gallery/recipes__live-loop.webp
:img-alt: Trades processed one at a time, with alerts and order flow

Call any function one value at a time on a live feed, align streams that tick
at different times, and wire functions into a reusable pipeline.
+++
Example: [A live event loop](recipes/live-loop.md),
[Streaming live events notebook](notebooks/06-streaming-live-events.ipynb),
[Pipelines notebook](notebooks/08-pipelines.ipynb).
Functions: [`Pipeline`](functions_dag/Pipeline.md),
[`CombineLatest`](functions_streams/CombineLatest.md),
[`Resample`](functions_streams/Resample.md).
:::

:::{grid-item-card} JavaScript in the browser
:img-top: _static/gallery/capability__javascript.webp
:img-alt: A screamer.js code snippet above a price trace

The same C++ operators compiled to WebAssembly, for Node.js and the browser,
with results that match the Python package.
+++
Example: [JavaScript and the browser](javascript.md),
[live trades demo](https://screamer-labs.github.io/screamer/live-trades.html).
Reference: [screamer.js documentation](https://screamer-labs.github.io/screamer/).
:::
```
