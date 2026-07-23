# Examples

Runnable notebooks, grouped by task. Each one executes on real or seeded data at
build time.

## Getting started

```{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Quickstart: the polymorphic API
:link: 01-quickstart-polymorphic-api
:link-type: doc
One operator on scalars, arrays, and live streams.
:::

:::{grid-item-card} NaN handling
:link: 05-nan-handling
:link-type: doc
How missing values flow through operators and warm up.
:::
```

## Statistics & indicators

```{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Window statistics
:link: 02-window-statistics
:link-type: doc
Rolling mean, dispersion, quantiles, and ranks.
:::

:::{grid-item-card} Financial indicators
:link: 03-financial-indicators
:link-type: doc
Moving averages, momentum, bands, and volume.
:::

:::{grid-item-card} Signal processing
:link: 04-signal-processing
:link-type: doc
Filters and smoothers on noisy series.
:::
```

## Streaming & pipelines

```{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Streaming live events
:link: 06-streaming-live-events
:link-type: doc
Feed events one at a time from an iterator or generator.
:::

:::{grid-item-card} Multi-stream operators
:link: 07-multi-stream-operators
:link-type: doc
Align and combine streams that do not tick together.
:::

:::{grid-item-card} Pipelines
:link: 08-pipelines
:link-type: doc
Wire operators into a reusable graph.
:::

:::{grid-item-card} Bars from ticks
:link: 09-bars-from-ticks
:link-type: doc
Resample a trade tape into OHLC bars.
:::

:::{grid-item-card} Custom and multi-column bars
:link: 10-custom-and-multi-column-bars
:link-type: doc
Build bespoke bar aggregations.
:::
```

## Market microstructure

```{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Order flow and trade signs
:link: 11-microstructure-order-flow
:link-type: doc
Recover trade direction and order-flow imbalance.
:::

:::{grid-item-card} Price impact and liquidity
:link: 12-microstructure-price-impact
:link-type: doc
Kyle's lambda, Amihud, Roll spread, and the propagator.
:::

:::{grid-item-card} Toxicity, book pressure, and spreads
:link: 13-microstructure-toxicity-and-book
:link-type: doc
VPIN, queue imbalance, micro-price, and spread decomposition.
:::
```

## Backtesting

```{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Backtesting a signal
:link: 14-backtesting-a-signal
:link-type: doc
From a position signal to a costed equity curve.
:::

:::{grid-item-card} Event-driven backtests
:link: 15-event-driven-backtests
:link-type: doc
Fills on bars, the trade tape, and top-of-book quotes.
:::
```

## Filtering & forecasting

```{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Filtering and forecasting with uncertainty
:link: 17-filtering-and-forecasting-with-uncertainty
:link-type: doc
Recover a latent signal with `KalmanFilter` and forecast online with a calibrated interval using `BayesianRegression`.
:::

:::{grid-item-card} Pairs trading and mean reversion
:link: 18-pairs-and-mean-reversion
:link-type: doc
Estimate a hedge ratio, extract a stationary spread, and fit an Ornstein-Uhlenbeck model using `RollingBeta`, `RollingSpread`, `RollingAlpha`, and `RollingOU`.
:::
```

## Machine learning

```{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Supervised forecasting
:link: 16-supervised-forecasting
:link-type: doc
Build a leak-safe training set with `forecast_pairs`, fit, and backtest.
:::
```

```{toctree}
:hidden:

01-quickstart-polymorphic-api
02-window-statistics
03-financial-indicators
04-signal-processing
05-nan-handling
06-streaming-live-events
07-multi-stream-operators
08-pipelines
09-bars-from-ticks
10-custom-and-multi-column-bars
11-microstructure-order-flow
12-microstructure-price-impact
13-microstructure-toxicity-and-book
14-backtesting-a-signal
15-event-driven-backtests
16-supervised-forecasting
17-filtering-and-forecasting-with-uncertainty
18-pairs-and-mean-reversion
```
