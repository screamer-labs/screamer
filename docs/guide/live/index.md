# Run on live data

The functions that run on a whole array also run one event at a time on a
live feed and give the same numbers. These pages call a function value by
value, align feeds that tick at different times, wire functions into a
pipeline, and run a strategy in a live loop, in Python and in the browser.

<!-- tasks:start -->

1. [Process values one at a time](one-value-at-a-time.md)
2. [Combine streams that tick at different times](align-streams.md)
3. [Wire functions into a pipeline](pipeline.md)
4. [Run a strategy in a live loop](live-loop.md)
5. [Move a strategy from research to a live loop](research-to-live.md)
6. [Run screamer in the browser](browser.md)

```{toctree}
:hidden:

/guide/live/one-value-at-a-time
/guide/live/align-streams
/guide/live/pipeline
/guide/live/live-loop
/guide/live/research-to-live
/guide/live/browser
/notebooks/06-streaming-live-events
/notebooks/07-multi-stream-operators
/notebooks/08-pipelines
```

<!-- tasks:end -->

## All functions for this chapter

| Intent | Function |
|:-------|:---------|
| Process one value per event | every function, such as `RollingZscore` or `EwMean` |
| Combine feeds that tick at different times | `CombineLatest` |
| Define a computation once and bind data at call time | `Input`, `Pipeline` |
| Bars from a live trade feed | `Resample` |
| Pick columns from a multi-column stream | `Select` |
| Backtest bars as they complete | `BacktestOHLCTarget` |
