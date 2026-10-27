# Detect a trend

A trend is a sustained move in one direction. These pages measure its
direction from a fitted slope or from two moving averages, its strength with
ADX, and whether a price trends or moves in cycles with `TrendMode`.

<!-- tasks:start -->

1. [Measure a trend with a rolling slope](slope.md)
2. [Detect a trend with a moving-average crossover](crossover.md)
3. [Measure trend strength with ADX](adx.md)
4. [Tell trending periods from cycling periods](trend-mode.md)

```{toctree}
:hidden:

/guide/trend/slope
/guide/trend/crossover
/guide/trend/adx
/guide/trend/trend-mode
/notebooks/03-financial-indicators
```

<!-- tasks:end -->

## All functions for this chapter

| Intent | Function |
|:-------|:---------|
| Slope of a line fitted to a trailing window | `RollingPoly1`, `RollingPoly2`, `RollingLinearRegression` |
| Direction of a slope or a difference | `Sign` |
| Moving averages for a crossover | `EwMean` |
| Trend strength and direction from highs and lows | `ADX`, `PlusDI`, `MinusDI` |
| Trending or cycling | `TrendMode` |
