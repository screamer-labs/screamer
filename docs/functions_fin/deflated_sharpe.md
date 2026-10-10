---
name: deflated_sharpe
title: deflated_sharpe
kind: function
short: Probabilistic Sharpe ratio against the Sharpe ratio the best of your tried variants
  would reach by luck.
tags:
- overfitting
- multiple testing
- selection bias
- sharpe ratio
- dsr
see_also:
- probabilistic_sharpe
- RollingSharpe
- strategy_test
references:
- 'Bailey, D. H. and López de Prado, M. (2014). The deflated Sharpe ratio: correcting for
  selection bias, backtest overfitting, and non-normality. *Journal of Portfolio Management*,
  40(5), 94-107.'
topics:
- significance
---

# `deflated_sharpe`

Corrects a Sharpe ratio for selection. If you try many strategy variants and
keep the best, its Sharpe ratio is inflated even when none of them has skill.
The deflated Sharpe ratio (Bailey and López de Prado 2014) estimates the Sharpe
ratio that the best of your `N` trials would reach by luck alone, from their
number and their spread, and uses it as the benchmark in
[`probabilistic_sharpe`](probabilistic_sharpe.md).

With `V` the variance of the trial Sharpe ratios and `gamma` Euler's constant,
the benchmark is

`SR0 = sqrt(V) * ((1 - gamma) * Phi^-1(1 - 1/N) + gamma * Phi^-1(1 - 1/(N e)))`

`trial_sharpes` holds the per-step Sharpe ratio of every variant you tried, this
one included. The function returns `(probability, benchmark)`: the probability
that this strategy's true Sharpe ratio is above `SR0`, and `SR0` itself.

This is the analytic counterpart of the family-wise p-value of
[`position_test`](position_test.md), which reaches the same correction by
resampling.

<!-- HELP_END -->

## Signature

`deflated_sharpe(pnl, trial_sharpes)`

## Examples

```{eval-rst}
.. exec_code::

    import numpy as np
    from screamer import BacktestPriceTarget, EwMean, LogReturn, deflated_sharpe

    rng = np.random.default_rng(1)
    price = 100 * np.exp(np.cumsum(rng.standard_normal(2000) * 0.01))
    returns = np.nan_to_num(LogReturn()(price))

    # Try 30 trend speeds on a random walk and keep the best.
    runs = [BacktestPriceTarget()(np.sign(np.nan_to_num(EwMean(span=k)(returns))), price)
            for k in range(2, 32)]
    sharpes = [r[:, 1].mean() / r[:, 1].std(ddof=1) for r in runs]
    best = runs[int(np.argmax(sharpes))]
    p, bench = deflated_sharpe(best, sharpes)
    print(f"best Sharpe {max(sharpes):.4f}, luck benchmark {bench:.4f}, probability {p:.3f}")
```
