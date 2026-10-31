# Portfolios of many assets

A portfolio backtest trades many assets from one account, so the weights of
all assets share one equity. Inputs are `(T, K)` arrays with one asset per
column, and the output adds a last axis with the engine's four columns. The
cross-sectional functions compare the assets within each row, and `Segment`
restarts any calculation where a label such as the day changes.

<!-- tasks:start -->

1. [Backtest a portfolio of many assets](multi-asset.md)
2. [Z-score, demean and winsorize across assets each day (cross-sectional normalization)](cross-sectional-scores.md)
3. [Restart a calculation each day, contract or group (Segment, groupby)](per-group.md)

```{toctree}
:hidden:

/guide/portfolios/multi-asset
/guide/portfolios/cross-sectional-scores
/guide/portfolios/per-group
```

<!-- tasks:end -->

## All functions for this chapter

| Intent | Function |
|:-------|:---------|
| Many assets from one account | `BacktestPortfolioOHLCTarget` |
| Weights from scores | `CrossNormalize` |
| Statistics of the book | `backtest_report` |
| Compare assets within each day | `CrossDemean`, `CrossZscore`, `CrossWinsorize`, `CrossStd` |
| Restart per day, contract or group | `Segment` |
