# Prepare data for machine learning

A supervised model learns to predict a label, the price move after a bar,
from features known at that bar. These pages build the features and the
labels with screamer functions, pair them into training rows, and split the
rows into cross-validation folds that do not leak the test period into training.

<!-- tasks:start -->

1. [Build a feature matrix for a model](features.md)
2. [Make a series stationary without losing its memory (fractional differencing)](fractional-differencing.md)
3. [Label each bar with its future move](labels.md)
4. [Pair features with labels without look-ahead](training-set.md)
5. [Cross-validate on time series without leakage (purged time series split)](time-series-cv.md)

```{toctree}
:hidden:

/guide/machine-learning/features
/guide/machine-learning/fractional-differencing
/guide/machine-learning/labels
/guide/machine-learning/training-set
/guide/machine-learning/time-series-cv
/notebooks/16-supervised-forecasting
/notebooks/23-classifying-direction
/notebooks/24-triple-barrier-labels
/notebooks/25-meta-labeling
```

<!-- tasks:end -->

## All functions for this chapter

| Intent | Function |
|:-------|:---------|
| Compute features, one column each | `Resample`, `LogReturn`, `RollingZscore`, `RollingStd`, `EwMean`, `OFI` |
| Make a series stationary and keep its memory | `FracDiff`, `RollingADF`, `Log` |
| Label each bar with its future move | `Sign`, `TripleBarrierOHLC` |
| Pair features with a label over a horizon | `forecast_pairs` |
| Split rows into purged cross-validation folds | `time_splits` |
