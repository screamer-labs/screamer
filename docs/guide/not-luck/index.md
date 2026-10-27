# Check that a backtest result is not luck

A good backtest can come from luck: from the timing of a few trades, or from
trying many settings and keeping the best. These pages compare a rule with
randomly timed copies of its own trades, check the best of a parameter grid
against the best that luck would give, and choose parameters out of sample.

<!-- tasks:start -->

1. [Test whether a strategy's timing beats chance](timing-test.md)
2. [Choose parameters with walk-forward testing](walk-forward.md)
3. [Check that a result holds across nearby parameters](parameter-grid.md)

```{toctree}
:hidden:

/guide/not-luck/timing-test
/guide/not-luck/walk-forward
/guide/not-luck/parameter-grid
/notebooks/20-is-the-timing-real
/notebooks/21-is-the-tuned-result-real
/notebooks/22-does-the-feature-carry-information
```

<!-- tasks:end -->

## All functions for this chapter

| Intent | Function |
|:-------|:---------|
| Compare a strategy with randomly timed copies of its trades | `position_test` |
| Correct for the number of variants tried | `deflated_sharpe`, `probabilistic_sharpe` |
| Choose parameters out of sample | `walk_forward` |
| Backtest each variant and summarize it | `BacktestOHLCTarget`, `backtest_report` |
