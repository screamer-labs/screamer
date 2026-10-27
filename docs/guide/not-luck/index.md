# Check that a backtest result is not luck

A good backtest can come from luck, from the timing of a few trades or from
trying many settings and keeping the best. These pages test a result against
randomly timed copies of its own trades, correct the Sharpe ratio for the
number of variants tried, and choose parameters out of sample.

<!-- tasks:start -->

1. [Is my backtest significant?](../../recipes/significance-test.md)
2. [Walk-forward parameter selection](../../recipes/walk-forward.md)
3. [A parameter grid and its robustness](../../recipes/parameter-grid.md)

```{toctree}
:hidden:

/recipes/significance-test
/recipes/walk-forward
/recipes/parameter-grid
/notebooks/20-is-the-timing-real
/notebooks/21-is-the-tuned-result-real
/notebooks/22-does-the-feature-carry-information
```

<!-- tasks:end -->

## All functions for this chapter

| Intent | Function |
|:-------|:---------|
| Compare a result with randomly timed copies of its trades | `position_test` |
| Correct the Sharpe ratio for the number of variants tried | `probabilistic_sharpe`, `deflated_sharpe` |
| Choose parameters out of sample | `walk_forward` |
| Backtest each variant | `BacktestOHLCTarget` |
