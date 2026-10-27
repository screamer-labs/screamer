# Prepare data for machine learning

A supervised model needs features known at each step and a target from the
steps after it. These pages build both with screamer functions and pair them
into a training set without look-ahead.

<!-- tasks:start -->

1. [A feature matrix for machine learning](../../recipes/ml-features.md)

```{toctree}
:hidden:

/recipes/ml-features
/notebooks/16-supervised-forecasting
/notebooks/23-classifying-direction
/notebooks/24-triple-barrier-labels
/notebooks/25-meta-labeling
```

<!-- tasks:end -->

## All functions for this chapter

| Intent | Function |
|:-------|:---------|
| Scale-free features | `RollingZscore`, `EwMean`, `LogReturn` |
| Pair features with a target over a horizon | `forecast_pairs` |
| Label which of two price levels is reached first | `TripleBarrier`, `TripleBarrierOHLC` |
