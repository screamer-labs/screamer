# Recipes

Short, complete solutions to common tasks. Each recipe runs on the committed
Deribit trade slice (six hours of BTC and ETH perpetual trades), executes when
the docs build, and ends with one plot.

```{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Indicators on a pandas OHLCV DataFrame
:link: pandas-ohlcv-indicators
:link-type: doc
:img-top: ../_static/gallery/recipes__pandas-ohlcv-indicators.webp
Add Bollinger Bands, RSI, and ATR as DataFrame columns.
:::

:::{grid-item-card} From ticks to a costed backtest
:link: ticks-to-backtest
:link-type: doc
:img-top: ../_static/gallery/recipes__ticks-to-backtest.webp
Trades to bars to a signal to a backtest with fees and slippage.
:::

:::{grid-item-card} Is my backtest significant?
:link: significance-test
:link-type: doc
:img-top: ../_static/gallery/recipes__significance-test.webp
Compare a strategy with randomly timed copies using `position_test`.
:::

:::{grid-item-card} A live event loop
:link: live-loop
:link-type: doc
:img-top: ../_static/gallery/recipes__live-loop.webp
Feed trades one at a time and act on each output.
:::

:::{grid-item-card} Many assets at once
:link: multi-asset
:link-type: doc
:img-top: ../_static/gallery/recipes__multi-asset.webp
Volatility and correlation of several assets with `(T, K)` arrays.
:::

:::{grid-item-card} A pairs trade
:link: pairs-trade
:link-type: doc
:img-top: ../_static/gallery/recipes__pairs-trade.webp
Kalman hedge ratio, cointegration gate, entry and exit bands, and a two-leg portfolio backtest with a significance test.
:::

:::{grid-item-card} A feature matrix for machine learning
:link: ml-features
:link-type: doc
:img-top: ../_static/gallery/recipes__ml-features.webp
Causal features paired with a future target by `forecast_pairs`.
:::

:::{grid-item-card} Migrating from pandas rolling and TA-Lib
:link: migrating
:link-type: doc
:img-top: ../_static/gallery/recipes__migrating.webp
The screamer equivalent of common pandas and TA-Lib calls.
:::
```

```{toctree}
:hidden:

pandas-ohlcv-indicators
ticks-to-backtest
significance-test
live-loop
multi-asset
pairs-trade
ml-features
migrating
```
