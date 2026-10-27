# First steps

Install screamer, then apply one function to a NumPy array, a pandas column,
and a stream of single values. The last page covers what a function returns for
a missing input and before its window is full.

<!-- tasks:start -->

1. [Install screamer](install.md)
2. [Apply a function to an array, a column and a live stream](first-function.md)
3. [Handle missing values and warmup](missing-values.md)

```{toctree}
:hidden:

/guide/first-steps/install
/guide/first-steps/first-function
/guide/first-steps/missing-values
/notebooks/01-quickstart-polymorphic-api
/notebooks/05-nan-handling
```

<!-- tasks:end -->

## All functions for this chapter

| Intent | Function |
|:-------|:---------|
| A trailing-window mean | `RollingMean` |
| Fill missing values | `Ffill`, `FillNa` |
| Mark missing or infinite values | `IsNan`, `IsFinite` |
