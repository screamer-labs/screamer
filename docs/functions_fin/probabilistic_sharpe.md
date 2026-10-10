---
name: probabilistic_sharpe
title: probabilistic_sharpe
kind: function
short: Probability that a strategy's true Sharpe ratio is above a benchmark, given its sample length and fat tails.
topics:
- backtesting
---

# `probabilistic_sharpe`

Returns the probability that a strategy's true Sharpe ratio is above a
benchmark, given its observed per-step Sharpe ratio (Bailey and López de Prado
2012). The estimate accounts for the length of the record and for the skewness
and kurtosis of the per-step pnl: a short record, negative skew, or fat tails all
make a given Sharpe ratio less convincing.

With Sharpe ratio `SR`, `T` finite pnl values, skewness `g3`, and kurtosis `g4`
(3 for a normal distribution):

`PSR = Phi((SR - benchmark) * sqrt(T - 1) / sqrt(1 - g3 * SR + (g4 - 1) / 4 * SR^2))`

`pnl` is a per-step pnl series, or a backtest engine's `(T, 4)` output, whose pnl
column is used. `benchmark` is a per-step Sharpe ratio (default 0). NaN values
are skipped. The computation runs in C++ and needs no resampling. To account for
having tried several variants, use [`deflated_sharpe`](deflated_sharpe.md).

<!-- HELP_END -->

## Signature

`probabilistic_sharpe(pnl, benchmark=0.0)`

## Example

```{eval-rst}
.. exec_code::

    import numpy as np
    from screamer import probabilistic_sharpe

    rng = np.random.default_rng(0)
    pnl = rng.standard_t(4, size=500) * 0.01 + 0.001     # fat-tailed daily pnl
    print(f"P(true Sharpe > 0)    = {probabilistic_sharpe(pnl):.3f}")
    print(f"P(true Sharpe > 0.05) = {probabilistic_sharpe(pnl, 0.05):.3f}")
```

## References

- Bailey, D. H. and López de Prado, M. (2012). The Sharpe ratio efficient
  frontier. *Journal of Risk*, 15(2).
