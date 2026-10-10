# Recipes

Short, complete solutions to one task each. Every recipe runs on real market
data committed with the docs, executes when the docs build, and ends with one
plot. The data are six hours of Deribit BTC and ETH perpetual trades, daily
Binance bars of 20 coins from 2022 to 2026, and the BTC perpetual's funding
rates. For a longer tour of one area, see the [example notebooks](../notebooks/index.md).

## Start here

```{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Your first strategy
:link: first-strategy
:link-type: doc
:img-top: ../_static/gallery/recipes__first-strategy.webp
From daily prices to a costed backtest, its statistics, and a p-value.
:::

:::{grid-item-card} Migrating from pandas rolling and TA-Lib
:link: migrating
:link-type: doc
:img-top: ../_static/gallery/recipes__migrating.webp
The screamer equivalent of common pandas and TA-Lib calls.
:::
```

## Data and indicators

```{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Indicators on a pandas OHLCV DataFrame
:link: pandas-ohlcv-indicators
:link-type: doc
:img-top: ../_static/gallery/recipes__pandas-ohlcv-indicators.webp
Add Bollinger Bands, RSI, and ATR as DataFrame columns.
:::

:::{grid-item-card} Many assets at once
:link: multi-asset
:link-type: doc
:img-top: ../_static/gallery/recipes__multi-asset.webp
Volatility and correlation of several assets with `(T, K)` arrays.
:::

:::{grid-item-card} A feature matrix for machine learning
:link: ml-features
:link-type: doc
:img-top: ../_static/gallery/recipes__ml-features.webp
Causal features paired with a future target by `forecast_pairs`.
:::
```

## Backtesting

```{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} From ticks to a costed backtest
:link: ticks-to-backtest
:link-type: doc
:img-top: ../_static/gallery/recipes__ticks-to-backtest.webp
Trades to bars to a signal to a backtest with fees and slippage.
:::

:::{grid-item-card} Stops, take-profit and trailing exits
:link: stops-and-exits
:link-type: doc
:img-top: ../_static/gallery/recipes__stops-and-exits.webp
The same trend rule with and without risk exits, and which exit fired when.
:::

:::{grid-item-card} Volatility-targeted position sizing
:link: volatility-sizing
:link-type: doc
:img-top: ../_static/gallery/recipes__volatility-sizing.webp
Size positions to a risk budget instead of a fixed ±1.
:::

:::{grid-item-card} Funding costs on a perpetual
:link: funding-carry
:link-type: doc
:img-top: ../_static/gallery/recipes__funding-carry.webp
Charge the BTC perpetual's funding rate on a position with `BacktestCarry`.
:::
```

## Is it more than luck?

```{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Is my backtest significant?
:link: significance-test
:link-type: doc
:img-top: ../_static/gallery/recipes__significance-test.webp
Compare a strategy with randomly timed copies using `position_test`.
:::

:::{grid-item-card} A parameter grid and its robustness
:link: parameter-grid
:link-type: doc
:img-top: ../_static/gallery/recipes__parameter-grid.webp
Backtest a grid of parameters at once, map its Sharpe ratios, and correct for the number tried.
:::

:::{grid-item-card} Walk-forward evaluation
:link: walk-forward
:link-type: doc
:img-top: ../_static/gallery/recipes__walk-forward.webp
Choose a parameter on each training window, trade it on the next, and measure the selection bias.
:::
```

## Portfolios and pairs

```{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} A cross-sectional momentum book
:link: cross-sectional-momentum
:link-type: doc
:img-top: ../_static/gallery/recipes__cross-sectional-momentum.webp
Rank 20 crypto assets by past return, hold a weekly dollar-neutral book with costs, and test it with `position_test`.
:::

:::{grid-item-card} A pairs trade
:link: pairs-trade
:link-type: doc
:img-top: ../_static/gallery/recipes__pairs-trade.webp
Kalman hedge ratio, cointegration gate, entry and exit bands, and a two-leg portfolio backtest with a significance test.
:::
```

## Live

```{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} A live event loop
:link: live-loop
:link-type: doc
:img-top: ../_static/gallery/recipes__live-loop.webp
Feed trades one at a time and act on each output.
:::

:::{grid-item-card} From research to a live loop
:link: strategy-to-live
:link-type: doc
:img-top: ../_static/gallery/recipes__strategy-to-live.webp
One pipeline from trades to a target, run on stored data and then event by event as a paper trader.
:::
```

```{toctree}
:hidden:

migrating
pandas-ohlcv-indicators
multi-asset
ml-features
ticks-to-backtest
stops-and-exits
volatility-sizing
funding-carry
significance-test
parameter-grid
walk-forward
cross-sectional-momentum
pairs-trade
live-loop
strategy-to-live
```
