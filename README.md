# Screamer

Screamingly fast rolling statistics, technical indicators, and signal filters for
time series. One C++ core, two languages: a simple Python API and a
JavaScript/WebAssembly build, producing identical results on batch arrays and
live streams.

[![License](https://img.shields.io/pypi/l/screamer?color=#28A745)](https://github.com/screamer-labs/screamer/blob/main/LICENSE)
![Python Versions](https://img.shields.io/pypi/pyversions/screamer)
[![tests](https://github.com/screamer-labs/screamer/actions/workflows/tests.yml/badge.svg)](https://github.com/screamer-labs/screamer/actions/workflows/tests.yml)
[![Docs](https://readthedocs.org/projects/screamer/badge/?version=latest)](https://screamer.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/screamer)](https://pypi.org/project/screamer/)

Screamer runs in Python and in JavaScript. Both bind the same C++ engine, so a
computation in one language matches the other to the last bit.

## Why screamer

- **Fast.** Every operator is implemented in C++ and routinely outruns equivalent
  NumPy and pandas code, often by a factor of two or more.
- **One API, batch or streaming.** The same operator runs on a stored array or a
  live, event-driven stream and produces identical results, so code tested on
  historical data deploys to production unchanged.
- **Causal by construction.** Output depends only on current and past inputs, never
  future ones, which eliminates look-ahead bias.
- **Batteries included.** 200+ rolling and exponentially-weighted statistics,
  technical indicators (MACD, RSI, Bollinger Bands, ATR, and more), OHLC volatility
  estimators, signal filters, plus stream operators and composable pipelines.
- **Python and JavaScript.** The same operators and the same pipeline model in both
  ecosystems, from one C++ source.

## Python

Python 3.12 or newer.

```bash
pip install screamer
```

The wheel is self-contained (a single abi3 wheel per platform, no build step and no
runtime dependencies). For development setup and the example notebooks see the
[Installation](https://screamer.readthedocs.io/en/latest/installation.html) page.

Fit a line to each sliding window of 50 values, take the slope, then its sign to get
the trend direction:

```python
import numpy as np
from screamer import RollingPoly2, Sign

data = np.cumsum(np.random.normal(size=300))

slope = RollingPoly2(window_size=50, derivative_order=1)
sign = Sign()

trend = sign(slope(data))   # the same calls work on a live stream, one value at a time
```

## JavaScript

Node 18 or newer, or any modern browser. Published to npm from v2.1.

```bash
npm install @screamer-labs/screamer
```

The package is self-contained: the WebAssembly module is embedded, so there is no
separate asset to configure. The WASM loads once via `await ready()`, then the
operators read exactly like the Python ones:

```javascript
import { ready, RollingPoly2, Sign } from "@screamer-labs/screamer";
await ready();

const data = Float64Array.from({ length: 300 }, () => Math.random());

const slope = RollingPoly2(50, 1);   // window_size, derivative_order
const sign = Sign();

const trend = sign(slope(data));     // a number, a Float64Array, an iterable, or an async stream
```

The JavaScript build covers every operator plus the full pipeline model:

```javascript
import { ready, Input, Pipeline, RollingMean, Diff } from "@screamer-labs/screamer";
await ready();

const x = Input("x");
const y = Diff(1)(RollingMean(3)(x));      // compose operators into a graph
const p = new Pipeline([x], [y]);
p(data);                                    // bind data at call time; p.live() streams
```

## Documentation

- **Python:** the full reference grouped by topic and runnable notebooks live at
  [screamer.readthedocs.io](https://screamer.readthedocs.io/en/latest/).
- **JavaScript:** see [`js/README.md`](js/README.md) for install and the API. JavaScript
  API reference and guide: [screamer-labs.github.io/screamer](https://screamer-labs.github.io/screamer/).

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to set up a
development environment, build the extension, run the tests, and open a pull request.
By participating you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## License

Screamer is released under the MIT License. See [LICENSE](LICENSE).
