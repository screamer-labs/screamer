# Using screamer: a guide for coding agents

screamer is a library of causal time-series functions for finance and signal
processing: rolling statistics, technical indicators, filters, market
microstructure, bars from ticks, backtests, and significance tests. The
functions are implemented in C++ and run the same way on stored arrays and on a
live feed, one value at a time. The same functions are available in JavaScript
as `@screamer-labs/screamer`.

## Find a function before writing code

screamer has close to 300 functions. Before writing a calculation by hand in
numpy or pandas, search for it:

```text
python -m screamer.docs search "order flow imbalance"
python -m screamer.docs show RollingStd
python -m screamer.docs topics
```

`show` prints the parameters with their types, defaults, and allowed values, the
number of inputs and outputs, how the function treats NaN, related functions,
and an example. Add `--json` for structured output. The same calls are
available in Python after `from screamer import docs`, as `docs.search()`,
`docs.show()`, and `docs.topics()`. They describe the installed version.

## Construct, then call

A function is an object: construct it with its parameters, then call it on
data. The object holds the state of the calculation, such as the samples in a
window.

```python
import numpy as np
from screamer import RollingMean

ma = RollingMean(window_size=20)
out = ma(prices)                 # prices: a 1-D array
```

- The output type follows the input: an array gives an array, a list gives a
  list, a float gives a float, and an iterator gives a lazy iterator.
- On a 2-D array, axis 0 is time and every column is an independent series. A
  `(T, K)` array is `K` series of length `T`. There is no `axis=` argument.
- A list is a series, never a scalar: `[4.0]` is a series of length 1.
- An array or list call starts from a fresh state. Calling with one float at a
  time continues the running series, which is the live path; `reset()` starts
  it over.
- Functions with several inputs take them as separate arguments or as the
  columns of one 2-D array: `ATR(14)(high, low, close)` or `ATR(14)(hlc)`.
- Functions with several outputs add a trailing axis: `BollingerBands(20)(x)`
  has shape `(T, 3)`.
- Functions compose by nesting: `RollingMax(3)(Diff(1)(x))`.

## Causality and warmup

Every function is causal: the output at step `t` uses only inputs up to `t`.
Do not shift the data to avoid look-ahead, and do not use backward fills or
centered windows. Until a window is full a function outputs NaN; the
`start_policy` parameter (`"strict"`, `"expanding"`, `"zero"`) chooses another
behavior where it exists. The IIR filters (`Butter` and its variants,
`SuperSmoother`, `Decycler`, `RoofingFilter`) have no NaN warmup; their
`start_policy` is `"steady"` (default, as if the input had always been at its
first value) or `"zero"`. Each function's NaN policy (`ignore`, `propagate`, or
`nan-aware`) is listed by `show`.

## Streams and pipelines

The same function object runs live: call it once per new value, or pass an
iterator. Streams that tick at different times are aligned with
`CombineLatest` or `Merge` and downsampled with `Resample` (time, tick, volume,
or dollar bars). `Pipeline` wires functions into a reusable unit:

```python
from screamer import Input, Pipeline, CombineLatest, Sub, RollingMean

a, b = Input("a"), Input("b")
signal = RollingMean(10)(Sub()(CombineLatest()(a, b)))
pipe = Pipeline(inputs=[a, b], outputs=[signal])
pipe(array_a, array_b)           # stored data; pass iterators to run live
```

## Backtests and significance

A target-position engine takes the desired position at each step and the
market data, and returns equity, pnl, position, and cost per step. The engine
fills at the next opportunity, so do not lag the target yourself.

```python
from screamer import BacktestOHLCTarget, backtest_report, position_test

engine = BacktestOHLCTarget(taker_fee=5e-4)
out = engine(target, open_, high, low, close)     # shape (T, 4)
series, summary = backtest_report(out, periods_per_year=365)
test = position_test(engine, target, open_, high, low, close, n=999)
print(summary["sharpe"], test.p_value)
```

`position_test` compares the strategy with randomly timed copies of its own
trades. If the strategy was tuned or selected on the same data, use
`strategy_test`, which reruns the whole strategy on permuted market data, or
correct for the number of settings tried with `deflated_sharpe`. Search the
`backtesting` and `significance` topics for the other engines (prices, trades,
quotes, portfolios) and tests.

## JavaScript

The JavaScript package has the same functions with the same names; plain
functions use camelCase (`positionTest`). Wait for the WebAssembly module once,
then call factories without `new`. Parameters are positional, in the Python
order, or an options object with camelCase names:

```javascript
import { ready, RollingRSI } from "@screamer-labs/screamer";
await ready();
const rsi = RollingRSI({ windowSize: 14 })(closes);   // closes: Float64Array
```

The JavaScript package has the same documentation in JavaScript names, with
each function's TypeScript signatures: run `npx screamer-docs search <keywords>`
and `npx screamer-docs show <Name>` in a project that has the package
installed, or import `search`, `show`, and `topics` from
`@screamer-labs/screamer/docs`.

## More

The documentation is at https://screamer.readthedocs.io. Every function page is
linked from the output of `show`, and https://screamer.readthedocs.io/en/latest/llms.txt
indexes the whole site for agents.
