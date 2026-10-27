# Smooth and denoise a series

A smoother keeps the slow movement of a series and removes the fast, noisy
part. The smoother its output, the more it lags the series, and the pages below
trade the two off in different ways. A spike, a single bad value, pulls every
average, so remove it with a despiker before smoothing.

<!-- tasks:start -->

1. [Smooth a series with a moving average](moving-averages.md)
2. [Smooth a series with less lag](low-lag.md)
3. [Track a noisy level with a Kalman filter](kalman.md)
4. [Remove fast wiggles with a low-pass filter](lowpass.md)
5. [Remove spikes from a series](despike.md)

```{toctree}
:hidden:

/guide/denoise/moving-averages
/guide/denoise/low-lag
/guide/denoise/kalman
/guide/denoise/lowpass
/guide/denoise/despike
/notebooks/04-signal-processing
/notebooks/17-filtering-and-forecasting-with-uncertainty
```

<!-- tasks:end -->

## All functions for this chapter

| Intent | Function |
|:-------|:---------|
| Simple or exponentially weighted moving average | `RollingMean`, `EwMean` |
| Moving average with less lag | `HullMA`, `KAMA`, `SuperSmoother` |
| Track a level from noisy measurements | `KalmanFilter` |
| Low-pass filter with a cutoff frequency | `Butter` |
| Replace spikes with the window median | `Hampel` |
