# Backtest a strategy

A backtest replays a strategy against past prices and records what it would
have earned. These pages run a signal on daily bars and read the result with
`backtest_report`, add fees, slippage and a cap on each bar's volume, post
limit orders, trade on the raw trade tape and on bid and ask quotes, report a
portfolio as one book, and watch the drawdown and the distance to liquidation.

<!-- tasks:start -->

1. [Backtest a signal on OHLCV bars](ohlcv.md)
2. [See how fees and slippage change a result](costs.md)
3. [Backtest with limit and stop orders](limit-stop-orders.md)
4. [Cap trades at a share of each bar's volume](volume-cap.md)
5. [Backtest on tick trades instead of bars](tick-trades.md)
6. [Backtest against bid and ask quotes (L1)](quotes.md)
7. [Report a portfolio backtest as one book](portfolio-report.md)
8. [Track drawdown and distance to liquidation live](drawdown-liquidation.md)

```{toctree}
:hidden:

/guide/backtest/ohlcv
/guide/backtest/costs
/guide/backtest/limit-stop-orders
/guide/backtest/volume-cap
/guide/backtest/tick-trades
/guide/backtest/quotes
/guide/backtest/portfolio-report
/guide/backtest/drawdown-liquidation
/notebooks/14-backtesting-a-signal
/notebooks/15-event-driven-backtests
```

<!-- tasks:end -->

## All functions for this chapter

| Intent | Function |
|:-------|:---------|
| One asset, target position, bar data | `BacktestOHLCTarget` |
| Limit and market orders on bars | `BacktestOHLCOrders` |
| Cap each fill at a share of the bar's volume | `BacktestOHLCVTarget` |
| Target position or resting orders on the trade tape | `BacktestTradesTarget`, `BacktestTradesOrders` |
| Target position or resting orders against L1 quotes | `BacktestL1Target`, `BacktestL1Orders` |
| Many assets from one account | `BacktestPortfolioOHLCTarget` |
| Summary statistics and running statistics, for one asset or a book | `backtest_report` |
| Drawdown from the running peak | `Drawdown`, `MaxDrawdown` |
| Distance to the liquidation price of a leveraged position | `LiquidationDistance` |
