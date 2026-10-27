# Stops and exits

A stop closes a position when the price has moved too far, whatever the signal
says. The backtest engine handles stop-loss, take-profit and trailing-stop
exits itself, so the signal stays as it is and the exit is one parameter.

<!-- tasks:start -->

1. [Add stops and exits](stops-and-exits.md)

```{toctree}
:hidden:

/guide/stops/stops-and-exits
```

<!-- tasks:end -->

## All functions for this chapter

| Intent | Function |
|:-------|:---------|
| Stop-loss, take-profit and trailing stop on bars | `BacktestOHLCTarget` |
| Compare runs with and without exits | `backtest_report` |
