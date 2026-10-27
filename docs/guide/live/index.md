# Run on live data

The functions that run on a whole array also run one event at a time on a
live feed and give the same numbers. These pages run a loop on single events
and turn a research pipeline into a live one.

<!-- tasks:start -->

1. [A live event loop](../../recipes/live-loop.md)
2. [From research to a live loop](../../recipes/strategy-to-live.md)

```{toctree}
:hidden:

/recipes/live-loop
/recipes/strategy-to-live
/notebooks/06-streaming-live-events
/notebooks/07-multi-stream-operators
/notebooks/08-pipelines
```

<!-- tasks:end -->

## All functions for this chapter

| Intent | Function |
|:-------|:---------|
| Define a computation once and bind data at call time | `Input`, `Pipeline` |
| Bars from a live trade feed | `Resample` |
| Combine feeds that tick at different times | `CombineLatest` |
| Pick columns from a multi-column stream | `Select` |
