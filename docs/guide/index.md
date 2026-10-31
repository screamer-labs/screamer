# Guide

Each chapter answers a few tasks with a short, runnable example. Read in order, or jump to a chapter.

## Part 0: Basics

0. [First steps](first-steps/index.md): Install screamer, apply a function to arrays and DataFrames, and handle NaN values and warmup.
1. [Get your data in](get-data-in/index.md): Load OHLCV data, build bars from ticks, and replace pandas, TA-Lib and mlfinlab code.

## Part 1: Understand a series

2. [Smooth and denoise a series](denoise/index.md): Moving averages, low-lag and adaptive smoothers (Hull, KAMA, MAMA), Kalman and Butterworth filters, Hampel despiking, and robust outlier detection (MAD).
3. [Measure volatility](volatility/index.md): Close-to-close and OHLC estimators (Yang-Zhang), average true range (ATR), jumps (bipower variation), and seasonal z-scores.
4. [Detect a trend](trend/index.md): Rolling slope (linear regression), moving-average crossover, ADX and directional movement (DMI), and trend mode.
5. [Compute classic technical indicators](indicators/index.md): Momentum oscillators (Stochastic RSI, Williams %R, CCI, TRIX), volume indicators (OBV, MFI, VWAP), and Keltner Channels.
6. [Detect mean reversion](mean-reversion/index.md): Variance ratio test, Hurst exponent, Ornstein-Uhlenbeck (OU) half-life, and z-score bands.
7. [Find cycles](cycles/index.md): Dominant cycle period (Ehlers, Hilbert transform, roofing filter) and cycle turning points (sinewave indicator).
8. [Detect regimes and change points](regimes/index.md): Hidden Markov models (HMM), Bayesian online changepoint detection, CUSUM filter events, and permutation entropy.
9. [Read order flow and liquidity](order-flow/index.md): Trade signing (tick rule), order-flow imbalance (OFI), price impact (Kyle's lambda), Amihud illiquidity, bid-ask spread from bars (Corwin-Schultz, Abdi-Ranaldo, Roll), effective and realized spread, VPIN toxicity, micro-price and queue imbalance, and Hawkes trade arrivals.

## Part 2: Build a strategy

10. [Turn a signal into a position](signal-to-position/index.md): Sign of a signal, entry and exit bands (hysteresis, Schmitt trigger), holding periods, and volatility position sizing.
11. [Backtest a strategy](backtest/index.md): Backtests on bars, tick trades and bid-ask quotes (L1), the backtest report, fees and slippage, limit and market orders, volume caps (participation), portfolio reports, drawdown, and liquidation distance.
12. [Measure performance and risk](performance/index.md): Rolling Sharpe, Sortino, Calmar and Omega ratios, the information ratio, CVaR (expected shortfall) and downside deviation, beta and alpha against a benchmark, and multi-factor regressions.
13. [Strategy templates](strategies/index.md): Trend following, breakouts, RSI and Bollinger Band reversion, pairs trading, cross-sectional momentum, funding carry, and volatility targeting.
14. [Stops and exits](stops/index.md): Stop-loss, take-profit and trailing-stop exits on a backtest.
15. [Portfolios of many assets](portfolios/index.md): Multi-asset backtests and sizing, cross-sectional z-scores, demeaning and winsorizing, and calculations that restart per day, contract or group (Segment).

## Part 3: Test a strategy

16. [Check that a backtest result is not luck](not-luck/index.md): Randomization and Monte Carlo permutation tests, block shuffles, time-shifted signals, synthetic price paths, walk-forward testing, parameter grids, and the deflated Sharpe ratio.
17. [Prepare data for machine learning](machine-learning/index.md): Features, labels (triple barrier), training sets, purged time series cross-validation, and fractional differencing.

## Part 4: Run a strategy live

18. [Run on live data](live/index.md): Streaming one value at a time, aligning streams, pipelines, a live trading loop, and JavaScript in the browser.

```{toctree}
:hidden:

first-steps/index
get-data-in/index
denoise/index
volatility/index
trend/index
indicators/index
mean-reversion/index
cycles/index
regimes/index
order-flow/index
signal-to-position/index
backtest/index
performance/index
strategies/index
stops/index
portfolios/index
not-luck/index
machine-learning/index
live/index
```
