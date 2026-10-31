# Smooth and denoise a series

A smoother keeps the slow movement of a series and removes the fast, noisy
part. The smoother the output, the more it lags the series, and the pages
below trade smoothness against lag in different ways, some by adapting to the
measured cycle. A spike, a single bad value, pulls every average, so remove it
with a despiker before smoothing, and judge outliers on a robust scale.

<!-- tasks:start -->

1. [Smooth a series with a moving average](moving-averages.md)
2. [Smooth a series with less lag](low-lag.md)
3. [Track an adaptive trend (MAMA, instantaneous trendline)](adaptive-trend.md)
4. [Track a noisy level with a Kalman filter](kalman.md)
5. [Remove fast wiggles with a low-pass filter](lowpass.md)
6. [Remove spikes from a series](despike.md)
7. [Detect outliers robustly (median absolute deviation, sigma clipping)](robust-outliers.md)

```{toctree}
:hidden:

/guide/denoise/moving-averages
/guide/denoise/low-lag
/guide/denoise/adaptive-trend
/guide/denoise/kalman
/guide/denoise/lowpass
/guide/denoise/despike
/guide/denoise/robust-outliers
/notebooks/04-signal-processing
/notebooks/17-filtering-and-forecasting-with-uncertainty
```

<!-- tasks:end -->

## All functions for this chapter

| Intent | Function |
|:-------|:---------|
| Simple or exponentially weighted moving average | `RollingMean`, `EwMean` |
| Moving average with less lag | `HullMA`, `KAMA`, `SuperSmoother` |
| Adaptive trend set by the measured cycle | `MAMA`, `InstantaneousTrendline` |
| Track a level from noisy measurements | `KalmanFilter` |
| Low-pass filter with a cutoff frequency | `Butter` |
| Replace spikes with the window median | `Hampel` |
| Robust z-score and outlier flags | `RollingMedian`, `RollingMedianAD`, `RollingMad`, `RollingZscore`, `RollingSigmaClip` |
| Returns of a price series | `LogReturn` |
