"""QuantLib-backed helper for the range-based volatility references.

Importing this module requires QuantLib. That is deliberate: a baseline whose
dependency is missing must fail at import so `devtools/baselines/__init__.py`
skips it and records why, rather than failing at call time and failing the
suite. References that do not need QuantLib must not import from here.

QuantLib implements two of the four estimators screamer ships, and it is the
reference used for those: it is an independent implementation of the formula
itself, which a transcription of the same formula cannot be. QuantLib returns
a *per-bar sigma*; screamer's Rolling* variants are the rolling mean of the
per-bar variance, so the references here square and then roll.

Rogers-Satchell and Yang-Zhang are not in QuantLib, so those are transcribed
from the documented formulas. `tests/test_ohlc_volatility.py` is what anchors
all four against ground truth: each must recover the sigma of a simulated GBM
path in the regime its page claims to handle.
"""
import numpy as np
import QuantLib as ql


def quantlib_per_bar_sigma(estimator, open_, high, low, close):
    """Run a QuantLib per-bar volatility estimator over OHLC arrays.

    QuantLib wants a TimeSeries<IntervalPrice> keyed by Date. The dates are
    arbitrary here: these estimators are per-bar and do not read the calendar.
    """
    n = len(close)
    start = ql.Date(1, 1, 2020)
    dates = ql.DateVector([start + i for i in range(n)])
    prices = ql.IntervalPriceVector([
        ql.IntervalPrice(float(open_[i]), float(close[i]), float(high[i]), float(low[i]))
        for i in range(n)
    ])
    series = estimator.calculate(ql.IntervalPriceTimeSeries(dates, prices))
    return np.array([series[d] for d in series.dates()], dtype=float)
