# Detect mean reversion

A series reverts when its moves away from a mean tend to be undone. The pages
follow one series, the one-minute log ratio of the ETH and BTC prices, through
three steps: test whether it reverts, measure how fast, and trade the
reversion.

<!-- tasks:start -->

1. [Test whether a series reverts with the variance ratio](variance-ratio.md)
2. [Estimate the Hurst exponent](hurst.md)
3. [Estimate how fast a series reverts](half-life.md)
4. [Trade the reversion with z-score bands](zscore-bands.md)

```{toctree}
:hidden:

/guide/mean-reversion/variance-ratio
/guide/mean-reversion/hurst
/guide/mean-reversion/half-life
/guide/mean-reversion/zscore-bands
/notebooks/18-pairs-and-mean-reversion
```

<!-- tasks:end -->

## All functions for this chapter

| Intent | Function |
|:-------|:---------|
| Test whether a series trends or reverts | `RollingVarianceRatio`, `RollingHurst` |
| Test the level for a unit root | `RollingADF` |
| Measure how fast it reverts | `RollingOU` |
| Standardize the series | `RollingZscore` |
| Enter and exit on bands | `BandSignal` |
| Backtest the position | `BacktestPriceTarget`, `backtest_report` |
