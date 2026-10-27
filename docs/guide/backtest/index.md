# Backtest a signal on OHLCV bars

A backtest replays a signal against past prices and records what it would have
earned. These pages run a signal through `BacktestOHLCTarget` on daily bars,
read the result with `backtest_report`, and compare one rule at three levels of
fees and slippage on bars built from trades.

<!-- tasks:start -->

1. [Backtest a signal on OHLCV bars](ohlcv.md)
2. [See how fees and slippage change a result](costs.md)

```{toctree}
:hidden:

/guide/backtest/ohlcv
/guide/backtest/costs
/notebooks/14-backtesting-a-signal
/notebooks/15-event-driven-backtests
```

<!-- tasks:end -->

## All functions for this chapter

| Intent | Function |
|:-------|:---------|
| One asset, target position, bar data | `BacktestOHLCTarget`, `BacktestPriceTarget` |
| Summary statistics and running statistics | `backtest_report` |
| A list of the trades | `backtest_trades` |
