# Prepare data for machine learning

A supervised model learns to predict a label, the price move after a bar,
from features known at that bar. These pages build the features and the
labels with screamer functions and pair them into training rows.

<!-- tasks:start -->

1. [Build a feature matrix for a model](features.md)
2. [Label each bar with its future move](labels.md)
3. [Pair features with labels without look-ahead](training-set.md)

```{toctree}
:hidden:

/guide/machine-learning/features
/guide/machine-learning/labels
/guide/machine-learning/training-set
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
| Label each bar with its future move | `Sign`, `TripleBarrierOHLC` |
| Pair features with a label over a horizon | `forecast_pairs` |
