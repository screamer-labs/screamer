---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Replace mlfinlab and AFML functions

screamer has causal versions of the main tools of *Advances in Financial
Machine Learning* (López de Prado 2018, AFML) and mlfinlab, the library that
implements it: `CusumFilter` samples events, `TripleBarrier` labels them,
`FracDiff` makes a stationary feature, and `time_splits` and `deflated_sharpe`
test the result.

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
from screamer import (Return, LogReturn, Log, EwStd, CusumFilter, FracDiff,
                      TripleBarrier, forecast_pairs)

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
c = df[df["symbol"] == "BTC"].set_index("date")["close"]

X = pd.DataFrame({"close": c})
X["daily_vol"] = EwStd(span=100)(Return()(c))               # get_daily_vol
X["ffd"] = FracDiff(d=0.4, window_size=1000, threshold=1e-4)(Log()(c))
X["event"] = CusumFilter(threshold=0.1)(LogReturn()(c))     # cusum_filter
barriers = TripleBarrier(horizon=10, upper=1.0, lower=1.0,  # get_events, get_bins
                         barrier_unit="vol", vol_span=100)(c)
# Move each label back to the row of its entry
_, X["label"] = forecast_pairs(X.to_numpy(), barriers[:, 0], count=10)
events = X[X["event"] != 0]
events.groupby("event")["label"].value_counts().unstack()
```

```{code-cell} ipython3
part = events.loc["2024":"2025"]
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(X.loc["2024":"2025"].index, X.loc["2024":"2025", "close"],
        color="0.6", lw=0.8)
for value, color in [(1, "tab:green"), (-1, "tab:red"), (0, "0.3")]:
    at = part[part["label"] == value]
    ax.scatter(at.index, at["close"], color=color, s=18, zorder=3,
               label=f"{value:+d}" if value else "0")
ax.set_ylabel("BTC close (USD)")
ax.legend(title="label at a CUSUM event", fontsize=8)
plt.tight_layout()
```

The table counts the triple-barrier labels at the CUSUM events: rows are the
event direction, columns the label. With barriers one 10-day volatility away,
73 of the 116 labeled events reach neither barrier within 10 days. Each
screamer function returns one value per row, so a step that mlfinlab does by
selecting timestamps is a row filter here. `TripleBarrier` labels every row and
stamps each label `horizon` rows late, when its outcome is known;
`forecast_pairs` moves the labels back to the rows of their entries.

## Function map

| mlfinlab (AFML) | screamer | Guide page |
|---|---|---|
| `get_tick_bars` (ch. 2) | [`Resample`](../../functions_streams/Resample.md)`(count=N, agg="ohlc")(price, time)` | [Build bars from trades](bars-from-ticks.md) |
| `get_volume_bars`, `get_dollar_bars` (ch. 2) | [`Resample`](../../functions_streams/Resample.md)`((price, time), driver, threshold=T, agg="ohlc")`, with volume or price times volume as the driver | [Build bars from trades](bars-from-ticks.md) |
| `cusum_filter` (ch. 2) | [`CusumFilter`](../../functions_signal/CusumFilter.md)`(threshold)(returns)`: mlfinlab takes prices (log returns inside), screamer takes the returns; +1, -1 or 0 on every row instead of a list of timestamps | [Sample events at large cumulative moves](../regimes/cusum.md) |
| `get_daily_vol` (ch. 3) | [`EwStd`](../../functions_ew/EwStd.md)`(span=100)(Return()(close))` on daily bars | [Measure volatility from closing prices](../volatility/close-to-close.md) |
| `add_vertical_barrier`, `get_events`, `get_bins` (ch. 3) | [`TripleBarrier`](../../functions_fin/TripleBarrier.md)`(horizon, upper, lower)(close)`, or [`TripleBarrierOHLC`](../../functions_fin/TripleBarrierOHLC.md) on bars: label, return and holding period; label 0 at the vertical barrier; `barrier_unit="vol"` scales the barriers by the volatility over the horizon | [Label each bar with its future move](../machine-learning/labels.md) |
| meta-labeling, `get_events(side=...)` (ch. 3) | `TripleBarrier` with symmetric barriers: the trade worked when `label == side` | [Label each bar with its future move](../machine-learning/labels.md) |
| `frac_diff_ffd` (ch. 5) | [`FracDiff`](../../functions_signal/FracDiff.md)`(d, window_size, threshold)`, the fixed-width method; `window_size` caps the number of weights | [Make a series stationary without losing its memory (fractional differencing)](../machine-learning/fractional-differencing.md) |
| `PurgedKFold` (ch. 7) | [`time_splits`](../../functions_fin/time_splits.md)`(n, n_splits, label_length=h)`: purged folds that train only on rows before the test block, so no embargo is needed | [Cross-validate on time series without leakage (purged time series split)](../machine-learning/time-series-cv.md) |
| `probabilistic_sharpe_ratio` (ch. 14) | [`probabilistic_sharpe`](../../functions_fin/probabilistic_sharpe.md)`(pnl, benchmark)`: takes the pnl and computes its skewness and kurtosis | [Correct a Sharpe ratio for the number of strategies tried (deflated Sharpe ratio)](../not-luck/deflated-sharpe.md) |
| `deflated_sharpe_ratio` (ch. 14) | [`deflated_sharpe`](../../functions_fin/deflated_sharpe.md)`(pnl, trial_sharpes)` | [Correct a Sharpe ratio for the number of strategies tried (deflated Sharpe ratio)](../not-luck/deflated-sharpe.md) |
| tick rule (ch. 19) | [`TickRuleSign`](../../functions_micro/TickRuleSign.md)`()(price)` | [Sign trades as buys or sells](../order-flow/sign-trades.md) |
| `get_bvc_buy_volume` (ch. 19) | [`BulkVolumeClassifier`](../../functions_micro/BulkVolumeClassifier.md)`(window)(returns)` times volume | [Sign trades as buys or sells](../order-flow/sign-trades.md) |
| `get_roll_measure` (ch. 19) | [`RollSpread`](../../functions_micro/RollSpread.md)`(window)(price)`: `NaN` where the serial covariance is not negative | [Estimate the bid-ask spread from OHLC bars](../order-flow/spread-from-bars.md) |
| `get_corwin_schultz_estimator` (ch. 19) | [`CorwinSchultzSpread`](../../functions_micro/CorwinSchultzSpread.md)`(window)(high, low, close)`; the close is for the overnight adjustment, `adjust_overnight=False` turns it off | [Estimate the bid-ask spread from OHLC bars](../order-flow/spread-from-bars.md) |
| `get_bar_based_kyle_lambda` (ch. 19) | [`RollingKyleLambda`](../../functions_micro/RollingKyleLambda.md)`(window)(signed_flow, returns)` | [Measure price impact](../order-flow/price-impact.md) |
| `get_bar_based_amihud_lambda` (ch. 19) | [`AmihudIlliquidity`](../../functions_micro/AmihudIlliquidity.md)`(window)(returns, notional)` | [Measure illiquidity from daily bars](../order-flow/illiquidity.md) |
| `get_vpin` (ch. 19) | [`VPIN`](../../functions_micro/VPIN.md)`(bucket_volume, n_buckets)(buy, sell)`, on buckets of equal volume | [Measure order-flow toxicity (VPIN)](../order-flow/vpin.md) |

The chapters in brackets are those of AFML. screamer has no sample weights,
sequential bootstrap, bet sizing or feature importance, and no CUSUM or SADF
structural break tests (AFML ch. 17); for change points see [Detect a change in
mean or spread](../regimes/changepoint.md).

## Go deeper

- [Prepare data for machine learning](../machine-learning/index.md): features, labels and cross-validation step by step.
- [Triple-barrier labels notebook](../../notebooks/24-triple-barrier-labels.ipynb) and [meta-labeling notebook](../../notebooks/25-meta-labeling.ipynb): the AFML labeling methods in full.
- [Replace pandas rolling and TA-Lib calls](from-pandas-talib.md): the same kind of map for pandas and TA-Lib.
