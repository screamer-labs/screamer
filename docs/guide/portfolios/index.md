# Portfolios of many assets

A portfolio backtest trades many assets from one account, so the weights of
all assets share one equity. Inputs are `(T, K)` arrays with one asset per
column, and the output adds a last axis with the engine's four columns.

<!-- tasks:start -->

1. [Backtest a portfolio of many assets](multi-asset.md)

```{toctree}
:hidden:

/guide/portfolios/multi-asset
```

<!-- tasks:end -->

## All functions for this chapter

| Intent | Function |
|:-------|:---------|
| Many assets from one account | `BacktestPortfolioOHLCTarget`, `BacktestPortfolioTarget` |
| Weights from scores | `CrossRank`, `CrossNormalize`, `CrossTopBottom` |
| Statistics of the book | `PortfolioReport` |
