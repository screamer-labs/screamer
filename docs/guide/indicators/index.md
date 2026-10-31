# Compute classic technical indicators

Technical indicators are fixed formulas on price and volume, such as
oscillators that measure momentum, volume indicators that
weigh price moves by the volume traded, and channels around a moving average.
These pages compute the classic ones and show how to read them.

<!-- tasks:start -->

1. [Compute momentum oscillators (Stochastic RSI, Williams %R, CCI, TRIX, Ultimate Oscillator)](oscillators.md)
2. [Compute MACD and the stochastic oscillator (MACD, Stoch, ROC)](macd-stoch.md)
3. [Compute volume indicators (OBV, MFI, accumulation/distribution, VWAP)](volume.md)
4. [Compare Keltner Channels with Bollinger Bands (squeeze)](keltner.md)

```{toctree}
:hidden:

/guide/indicators/oscillators
/guide/indicators/macd-stoch
/guide/indicators/volume
/guide/indicators/keltner
/notebooks/03-financial-indicators
```

<!-- tasks:end -->

## All functions for this chapter

| Intent | Function |
|:-------|:---------|
| Momentum oscillators | `StochRSI`, `WilliamsR`, `CCI`, `TRIX`, `UltimateOscillator` |
| MACD, stochastic oscillator and rate of change | `MACD`, `Stoch`, `ROC`, `ROCP`, `ROCR` |
| Volume indicators | `OBV`, `AD`, `MFI`, `RollingVWAP` |
| Channels around an average | `KeltnerChannels`, `BollingerBands` |
