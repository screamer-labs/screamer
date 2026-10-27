# Guide

Each chapter answers a few tasks with a short, runnable example. Read in order, or jump to a chapter.

## Part 0: Basics

0. [First steps](first-steps/index.md): Install screamer, apply a function, and handle missing values.
1. [Get your data in](get-data-in/index.md): Load OHLCV data, migrate from pandas and TA-Lib, and build bars from ticks.

## Part 1: Understand a series

2. [Smooth and denoise a series](denoise/index.md): Moving averages, low-lag smoothers, Kalman and Butterworth filters, despiking.
3. [Measure volatility](volatility/index.md): Close-to-close and OHLC estimators, and the average true range.
4. [Detect a trend](trend/index.md): Slope and regression, DMI and ADX, trend mode, and crossovers.
5. [Detect mean reversion](mean-reversion/index.md): Variance ratio, Hurst exponent, OU half-life, and z-score bands.
6. [Find cycles](cycles/index.md): Estimate the dominant cycle of a series.
7. [Detect regimes and change points](regimes/index.md): Hidden Markov models, Bayesian changepoints, and CUSUM events.
8. [Read order flow and liquidity](order-flow/index.md): Trade signing, imbalance, price impact, and toxicity.

## Part 2: Build a strategy

9. [Turn a signal into a position](signal-to-position/index.md): Sign, bands and hysteresis, holding, and volatility sizing.
10. [Backtest a signal on OHLCV bars](backtest/index.md): Costs, the report, the equity curve, and event-driven backtests.
11. [Strategy templates](strategies/index.md): Trend following, breakouts, reversion, pairs, momentum, and carry.
12. [Stops and exits](stops/index.md): Stop-loss, trailing stops, and take-profit exits.
13. [Portfolios of many assets](portfolios/index.md): Backtest and size positions across many assets at once.

## Part 3: Test a strategy

14. [Check that a backtest result is not luck](not-luck/index.md): Significance tests, walk-forward testing, and parameter grids.
15. [Prepare data for machine learning](machine-learning/index.md): Labels and features for supervised models.

## Part 4: Run a strategy live

16. [Run on live data](live/index.md): Streaming, pipelines, a live loop, and JavaScript in the browser.

```{toctree}
:hidden:

first-steps/index
get-data-in/index
denoise/index
volatility/index
trend/index
mean-reversion/index
cycles/index
regimes/index
order-flow/index
signal-to-position/index
backtest/index
strategies/index
stops/index
portfolios/index
not-luck/index
machine-learning/index
live/index
```
