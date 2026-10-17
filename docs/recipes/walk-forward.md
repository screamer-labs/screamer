---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Walk-forward parameter selection

A backtest of the best parameter, chosen after seeing the whole history, is
optimistic: the choice itself used the data it is scored on. `walk_forward`
removes that look-ahead. It splits the history into folds, picks the best
candidate on each training window, trades the pick on the test window that
follows, and reports the stitched result. This recipe runs a trend rule with
six lookback spans on daily BTC, with a rolling one-year training window and a
three-month test window, and compares the result with the span that looks best
in hindsight.

## Candidates

Each candidate is the same rule with a different span: long after recent gains
and short after recent losses, measured by an exponential mean of the daily log
return. The columns of `X` are the candidates.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import EwMean, LogReturn, BacktestOHLCTarget, walk_forward

df = pd.read_csv("../notebooks/data/binance_daily_20.csv", parse_dates=["date"])
btc = df[df.symbol == "BTC"].reset_index(drop=True)
o, h, l, c = (btc[k].to_numpy() for k in ["open", "high", "low", "close"])

spans = [5, 10, 20, 40, 80, 160]
ret = LogReturn()(c)
X = np.column_stack([np.sign(np.nan_to_num(EwMean(span=s)(ret))) for s in spans])
engine = BacktestOHLCTarget(taker_fee=5e-4)
```

## The walk-forward run

`train=365` and `test=91` give a rolling one-year training window and a test
window of about three months. The first test window starts after the first
year. `periods_per_year=365` annualizes the Sharpe ratio for daily bars that
trade every day.

```{code-cell} ipython3
wf = walk_forward(engine, X, o, h, l, c, train=365, test=91, metric="sharpe",
                  periods_per_year=365)
chosen = [spans[k] for k in wf.choice]
print(f"{len(wf.fold_start)} folds, first test day {btc.date[wf.fold_start[0]]:%Y-%m-%d}")
print("span chosen per fold:", chosen)
print(f"out-of-sample Sharpe {wf.stats['sharpe']:.2f}, pnl {wf.stats['total_pnl']:,.0f} USD")
```

The choice moves around: in this run the 16 folds pick every one of the six
spans, and the span changes between consecutive folds 8 times. The out-of-sample
Sharpe ratio is 0.42 per year, for one BTC of exposure.

## The hindsight pick

The same candidates, each backtested over the whole history, show which span
looks best with hindsight. The comparison uses the out-of-sample span, from the
first test day to the end, for both.

```{code-cell} ipython3
T, K = X.shape
engine.reset()  # the (T,) prices broadcast against the (T, K) candidates
full = np.asarray(engine(X, o, h, l, c))
t0 = wf.fold_start[0]

def sharpe(pnl):
    pnl = pnl[~np.isnan(pnl)]
    return pnl.mean() / pnl.std(ddof=1) * np.sqrt(365)

full_sharpe = [sharpe(full[:, k, 1]) for k in range(K)]
best = int(np.argmax(full_sharpe))
print("full-history Sharpe per span:", dict(zip(spans, np.round(full_sharpe, 2))))
print(f"hindsight best span {spans[best]}: Sharpe {sharpe(full[t0:, best, 1]):.2f} "
      f"over the out-of-sample span, walk-forward {wf.stats['sharpe']:.2f}")
```

Span 40 has the best Sharpe ratio over the full history (0.55), and over the
out-of-sample span alone it reaches 0.62. That number is not available to a
trader: picking span 40 required the whole history, including the years it is
scored on. The walk-forward result, 0.42, uses only data available at each
decision, and it is the better estimate of what the procedure "pick the best
span on the last year" delivers. The difference of 0.20 is the selection bias
of the hindsight pick.

The top panel shows both equity curves from the first test day. The bottom
panel shows the span chosen in each fold.

```{code-cell} ipython3
dates = btc.date[t0:]
eq_wf = wf.output[t0:, 0] - wf.output[t0 - 1, 0]
eq_best = full[t0:, best, 0] - full[t0 - 1, best, 0]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
ax1.plot(dates, eq_best, color="grey", lw=1.2, label=f"hindsight best (span {spans[best]})")
ax1.plot(dates, eq_wf, color="crimson", lw=1.5, label="walk-forward")
ax1.set_ylabel("pnl (USD)")
ax1.legend(fontsize=8, loc="upper left")
ax2.step(btc.date[wf.fold_start], chosen, where="post", color="crimson")
ax2.set_yscale("log")
ax2.set_yticks(spans, [str(s) for s in spans])
ax2.minorticks_off()
ax2.set_ylabel("chosen span")
plt.tight_layout()
```

The training window and the test window are choices too. Trying several of
them and reporting the best brings back the same selection bias, one level up.

## With funding

A perpetual pays funding on the position held. `carry=` takes the per-step
rate and scores every candidate after [`BacktestCarry`](../functions_fin/BacktestCarry.md),
marked with the close. A flat rate of 0.03% per day, about 11% a year, charges
the longs and pays the shorts.

```{code-cell} ipython3
funding = np.full(len(c), 3e-4)
wf_carry = walk_forward(engine, X, o, h, l, c, train=365, test=91, metric="sharpe",
                        periods_per_year=365, carry=funding)
print("span chosen per fold:", [spans[k] for k in wf_carry.choice])
print(f"out-of-sample Sharpe {wf_carry.stats['sharpe']:.2f}, "
      f"pnl {wf_carry.stats['total_pnl']:,.0f} USD after funding")
```

Related pages: [`walk_forward`](../functions_fin/walk_forward.md),
[`position_test`](../functions_fin/position_test.md),
[`deflated_sharpe`](../functions_fin/deflated_sharpe.md), and
[`BacktestOHLCTarget`](../functions_fin/BacktestOHLCTarget.md).
