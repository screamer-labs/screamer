# Check that a backtest result is not luck

A good backtest can come from luck: from the timing of a few trades, or from
trying many settings and keeping the best. These pages compare a rule with
randomly timed or shuffled copies of its own trades, a signal with shifted
copies of itself, and a whole strategy with its results on permuted markets.
They also choose parameters out of sample and correct the best of many variants
for the search that found it.

<!-- tasks:start -->

1. [Test whether a strategy's timing beats chance](timing-test.md)
2. [Shuffle trades in blocks to test a strategy's timing (block randomization test)](block-shuffle.md)
3. [Test a signal against time-shifted copies of itself (circular shift test)](shifted-copies.md)
4. [Generate synthetic price paths with the same return distribution (permuted bars, surrogate data)](synthetic-paths.md)
5. [Run a Monte Carlo permutation test on a whole strategy](permutation-test.md)
6. [Choose parameters with walk-forward testing](walk-forward.md)
7. [Check that a result holds across nearby parameters](parameter-grid.md)
8. [Correct a Sharpe ratio for the number of strategies tried (deflated Sharpe ratio)](deflated-sharpe.md)

```{toctree}
:hidden:

/guide/not-luck/timing-test
/guide/not-luck/block-shuffle
/guide/not-luck/shifted-copies
/guide/not-luck/synthetic-paths
/guide/not-luck/permutation-test
/guide/not-luck/walk-forward
/guide/not-luck/parameter-grid
/guide/not-luck/deflated-sharpe
/notebooks/20-is-the-timing-real
/notebooks/21-is-the-tuned-result-real
/notebooks/22-does-the-feature-carry-information
```

<!-- tasks:end -->

## All functions for this chapter

| Intent | Function |
|:-------|:---------|
| Compare a strategy with randomly timed copies of its trades | `position_test` |
| Make shuffled or rotated copies of a series | `segment_shuffle`, `circular_shift` |
| Make price paths with the same returns and no serial structure | `permute_bars`, `permute_returns` |
| Test a whole strategy on permuted markets | `strategy_test` |
| Correct for the number of variants tried | `deflated_sharpe`, `probabilistic_sharpe` |
| Choose parameters out of sample | `walk_forward` |
| Backtest each variant and summarize it | `BacktestOHLCTarget`, `backtest_report` |
| Pair a signal with the return that follows it | `forecast_pairs` |
