# Measure volatility

Volatility is the standard deviation of returns, usually stated per year.
These pages estimate it from closing prices, from the open, high, low and close
of each bar, and as the average true range in price units, then separate jumps
from ordinary moves and compare each day with the same weekday in past weeks.

<!-- tasks:start -->

1. [Measure volatility from closing prices](close-to-close.md)
2. [Measure volatility from OHLC bars](ohlc.md)
3. [Measure the average true range](atr.md)
4. [Separate jumps from normal volatility (bipower variation)](jumps.md)
5. [Compare each day with the same weekday in past weeks (seasonality, seasonal z-score)](seasonality.md)

```{toctree}
:hidden:

/guide/volatility/close-to-close
/guide/volatility/ohlc
/guide/volatility/atr
/guide/volatility/jumps
/guide/volatility/seasonality
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
| Variance without jumps, and the share from jumps | `RollingSum`, `RollingBipowerVar`, `RollingJumpShare` |
| Compare a value with the same slot of past cycles | `RollingSeasonalZscore`, `RollingZscore` |
