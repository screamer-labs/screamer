# Get your data in

screamer functions take NumPy arrays and pandas columns, so data you already
have in a DataFrame needs no conversion. These pages add indicators to an OHLCV
DataFrame, map pandas and TA-Lib calls to screamer, and build bars from a trade
tape.

<!-- tasks:start -->

1. [Compute indicators on an OHLCV DataFrame](ohlcv-dataframe.md)
2. [Replace pandas rolling and TA-Lib calls](from-pandas-talib.md)
3. [Build bars from trades](bars-from-ticks.md)

```{toctree}
:hidden:

/guide/get-data-in/ohlcv-dataframe
/guide/get-data-in/from-pandas-talib
/guide/get-data-in/bars-from-ticks
/notebooks/09-bars-from-ticks
/notebooks/10-custom-and-multi-column-bars
```

<!-- tasks:end -->

## All functions for this chapter

| Intent | Function |
|:-------|:---------|
| Build time bars or bars of a fixed number of events from ticks | `Resample` |
| Convert between pandas and a `(values, index)` pair | `from_pandas`, `to_pandas` |
| Combine streams, carrying the last value of a stream that has not ticked | `CombineLatest` |
| Interleave several index-sorted streams into one timeline | `Merge` |
| Drop rows that contain NaN from any input | `Dropna` |
| Keep only rows that satisfy a boolean condition | `Filter` |
| Pick a subset of columns from a multi-column stream | `Select` |
