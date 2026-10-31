# Detect regimes and change points

A regime is a stretch in which a series keeps the same behavior, such as
calm or volatile returns. These pages estimate the probability of a volatile
regime, detect when the mean or spread of a series changes, mark the
steps where the cumulative move passes a threshold, and measure how
predictable the order of returns is, on BTC daily returns.

<!-- tasks:start -->

1. [Estimate the probability of a volatile regime](hmm.md)
2. [Detect a change in mean or spread](changepoint.md)
3. [Sample events at large cumulative moves](cusum.md)
4. [Measure how predictable a series is (permutation entropy)](permutation-entropy.md)

```{toctree}
:hidden:

/guide/regimes/hmm
/guide/regimes/changepoint
/guide/regimes/cusum
/guide/regimes/permutation-entropy
```

<!-- tasks:end -->

## All functions for this chapter

| Intent | Function |
|:-------|:---------|
| Returns of a price series | `LogReturn` |
| Probability of a volatile regime | `RollingHMM` |
| Standardize a series | `RollingZscore` |
| Detect a change in mean or spread | `BayesianChangepoint` |
| Sample events at cumulative moves | `CusumFilter` |
| Predictability of the order of returns | `RollingPermutationEntropy`, `permute_returns` |
