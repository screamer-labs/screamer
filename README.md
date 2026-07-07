# Screamer

Screamingly fast rolling statistics, technical indicators, and signal filters for
time series. C++ performance with a simple Python API, and identical results on
batch NumPy arrays and live streams.

[![License](https://img.shields.io/pypi/l/screamer?color=#28A745)](https://github.com/screamer-labs/screamer/blob/main/LICENSE)
![Python Versions](https://img.shields.io/pypi/pyversions/screamer)
[![tests](https://github.com/screamer-labs/screamer/actions/workflows/tests.yml/badge.svg)](https://github.com/screamer-labs/screamer/actions/workflows/tests.yml)
[![Docs](https://readthedocs.org/projects/screamer/badge/?version=latest)](https://screamer.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/screamer)](https://pypi.org/project/screamer/)

```bash
pip install screamer
```

## Why screamer

- **Fast.** Every function is implemented in C++ and routinely outruns equivalent
  NumPy and pandas code, often by a factor of two or more.
- **One API, batch or streaming.** The same function runs on a stored NumPy array or
  a live, event-driven stream and produces identical results, so code tested on
  historical data deploys to production unchanged.
- **Causal by construction.** Output depends only on current and past inputs, never
  future ones, which eliminates look-ahead bias.
- **Batteries included.** 150+ rolling and exponentially-weighted statistics,
  technical indicators (MACD, RSI, Bollinger Bands, ATR, and more), OHLC volatility
  estimators, signal filters, plus stream operators and a computational DAG.

## Quick example

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

## Documentation

Full documentation, the function reference grouped by topic, and runnable example
notebooks live at [screamer.readthedocs.io](https://screamer.readthedocs.io/en/latest/).

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to set up a
development environment, build the extension, run the tests, and open a pull request.
By participating you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## License

Screamer is released under the MIT License. See [LICENSE](LICENSE).
