# Strategy templates

A strategy template is a complete rule, from data to backtest, that you can
copy and change. Each page trades committed crypto data with fees, and ends
with the same three numbers: the Sharpe ratio, the maximum drawdown, and the
number of trades. The templates use standard settings, not tuned on the data,
and most of them lose money on this sample.

<!-- tasks:start -->

1. [Follow a trend with a moving-average crossover](trend-following.md)
2. [Trade a breakout of the recent high or low](breakout.md)
3. [Buy oversold and sell overbought with RSI](rsi-reversion.md)
4. [Fade moves outside the Bollinger Bands](bollinger-reversion.md)
5. [Trade a pair](pairs.md)
6. [Rank assets by momentum](cross-sectional-momentum.md)
7. [Earn the funding carry](carry.md)
8. [Target a constant volatility](volatility-targeting.md)

```{toctree}
:hidden:

/guide/strategies/trend-following
/guide/strategies/breakout
/guide/strategies/rsi-reversion
/guide/strategies/bollinger-reversion
/guide/strategies/pairs
/guide/strategies/cross-sectional-momentum
/guide/strategies/carry
/guide/strategies/volatility-targeting
```

<!-- tasks:end -->

## All functions for this chapter

| Intent | Function |
|:-------|:---------|
| Trend and breakout signals | `EwMean`, `Sign`, `DonchianChannels` |
| Reversion signals | `RollingRSI`, `BollingerBands`, `BandSignal` |
| Hedge ratio and spread test for a pair | `KalmanRegression`, `RollingZscore`, `RollingADF` |
| Rank and weight assets | `CrossRank`, `CrossNormalize` |
| Size by volatility | `EwStd` |
| Backtest one asset or a book | `BacktestOHLCTarget`, `BacktestPortfolioOHLCTarget` |
| Funding and other carry | `BacktestCarry` |
| Summary statistics | `backtest_report`, `PortfolioReport` |
