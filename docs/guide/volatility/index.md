# Measure volatility

Volatility is the standard deviation of returns, usually stated per year.
These pages estimate it from closing prices, from the open, high, low and close
of each bar, and as the average true range in price units.

<!-- tasks:start -->

1. [Measure volatility from closing prices](close-to-close.md)
2. [Measure volatility from OHLC bars](ohlc.md)
3. [Measure the average true range](atr.md)

```{toctree}
:hidden:

/guide/volatility/close-to-close
/guide/volatility/ohlc
/guide/volatility/atr
/notebooks/02-window-statistics
```

<!-- tasks:end -->

## All functions for this chapter

| Intent | Function |
|:-------|:---------|
| Returns of a price series | `LogReturn` |
| Close-to-close volatility | `RollingStd`, `EwStd` |
| Volatility from OHLC bars | `RollingYangZhangVol`, `RollingGarmanKlassVol`, `RollingParkinsonVol` |
| Average true range, in price units or as a percentage | `ATR`, `NATR` |
