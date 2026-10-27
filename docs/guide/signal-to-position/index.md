# Turn a signal into a position

A position is the amount of an asset to hold at each bar. These pages turn a
score into a long or short position with `Sign`, add a dead band so that a
score near a threshold does not switch the position back and forth, hold a
position for a fixed number of bars, and size it by volatility.

<!-- tasks:start -->

1. [Go long or short on the sign of a score](sign.md)
2. [Enter and exit at different levels](bands.md)
3. [Hold a position for a fixed number of bars](hold.md)
4. [Size a position by volatility](volatility-sizing.md)

```{toctree}
:hidden:

/guide/signal-to-position/sign
/guide/signal-to-position/bands
/guide/signal-to-position/hold
/guide/signal-to-position/volatility-sizing
```

<!-- tasks:end -->

## All functions for this chapter

| Intent | Function |
|:-------|:---------|
| Long or short on the sign of a score | `Sign` |
| Enter and exit at different levels | `SchmittTrigger`, `BandSignal` |
| Hold a signal for a fixed number of bars | `Hold` |
| Estimate the volatility to size by | `EwStd`, `RollingStd`, `ATR` |
