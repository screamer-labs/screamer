# Measure performance and risk

A strategy's returns say more than its final equity. These pages compute
risk-adjusted ratios and tail risk in a moving window, estimate beta and alpha
against a benchmark, and regress returns on several factors at once.

<!-- tasks:start -->

1. [Compute a rolling Sharpe, Sortino and Calmar ratio](rolling-ratios.md)
2. [Measure tail risk with rolling CVaR and downside deviation (expected shortfall)](tail-risk.md)
3. [Estimate beta and alpha against a benchmark](beta-alpha.md)
4. [Regress returns on several factors in a moving window (rolling multi-factor regression)](factor-regression.md)

```{toctree}
:hidden:

/guide/performance/rolling-ratios
/guide/performance/tail-risk
/guide/performance/beta-alpha
/guide/performance/factor-regression
```

<!-- tasks:end -->

## All functions for this chapter

| Intent | Function |
|:-------|:---------|
| Risk-adjusted return over a window | `RollingSharpe`, `RollingSortino`, `RollingCalmar`, `RollingInfoRatio`, `RollingOmega` |
| Tail and downside risk | `RollingCVaR`, `RollingDownsideDeviation` |
| Worst drawdown inside a window | `RollingMaxDrawdown` |
| Beta and alpha against a benchmark | `RollingBeta`, `RollingAlpha`, `EwBeta` |
| Slope that treats both series alike | `RollingTLSBeta` |
| Regression on several factors | `RollingMultiRegression`, `EwMultiRegression`, `ExpandingMultiRegression` |
