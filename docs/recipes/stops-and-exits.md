---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Stops, take-profit and trailing exits

`BacktestOHLCTarget` can close a position on its own when the price moves too
far: `stop_loss` against the entry, `take_profit` in favour of it, and
`trailing_stop` back from the best price since entry. Each is a fraction of
price (0.10 = 10%), and 0 turns it off. This recipe runs one trend rule on daily
BTC bars without exits, with a 10% stop-loss, and with a 15% trailing stop, and
compares the results.

## The rule and three engines

The signal is the trend rule of [Backtest a signal on OHLCV bars](../guide/backtest/ohlcv.md): long
one BTC when the 20-day exponential average of the close is above the 100-day
one, short when it is below.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import EwMean, Sign, FillNa, BacktestOHLCTarget, backtest_report

bars = pd.read_csv("../notebooks/data/binance_daily_20.csv", parse_dates=["date"])
btc = bars[bars["symbol"] == "BTC"]
t = btc["date"].to_numpy()
o, h, l, c = (btc[k].to_numpy() for k in ["open", "high", "low", "close"])
signal = Sign()(FillNa(0.0)(EwMean(span=20)(c) - EwMean(span=100)(c)))

costs = dict(taker_fee=1e-3, slippage=5e-4)
runs = {
    "no exits": BacktestOHLCTarget(**costs),
    "10% stop-loss": BacktestOHLCTarget(stop_loss=0.10, **costs),
    "15% trailing stop": BacktestOHLCTarget(trailing_stop=0.15, **costs),
}
out = {name: engine(signal, o, h, l, c) for name, engine in runs.items()}

for name, res in out.items():
    s = backtest_report(res, periods_per_year=365)[1]
    print(f"{name:18s} pnl {s['total_pnl']:8.0f}  max drawdown {s['max_drawdown']:8.0f}  "
          f"Sharpe {s['sharpe']:5.2f}  exposure {s['exposure']:4.2f}")
```

The stop-loss compares the price with the average entry fill. The trailing stop
compares it with the highest high since entry for a long (the lowest low for a
short). When a bar opens beyond the exit level the position closes at that open;
otherwise it closes at the level, on the bar whose low or high reaches it. The
exit is a market order and pays the same fee and slippage as any other trade.

After an exit the engine ignores further targets on the same side, so the
position stays flat until the signal turns flat or reverses. That is why the
exposure, the share of days with a position, drops from 1.00 to 0.93 with the
stop-loss and to 0.47 with the trailing stop.

## Where the exits fired

An exit shows up in the output as a day on which the position returns to zero
while the target still asks for a position.

```{code-cell} ipython3
def exits(res):
    pos = res[:, 2]
    held = np.r_[0.0, pos[:-1]]
    wanted = np.r_[0.0, signal[:-1]]            # the target being executed on each day
    return np.flatnonzero((pos == 0) & (held != 0) & (wanted != 0))

for name in ["10% stop-loss", "15% trailing stop"]:
    print(f"{name}: {len(exits(out[name]))} exits")
```

With the default `fill_at="open"`, a position opened at a day's open can also
exit later in the same day.

```{code-cell} ipython3
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
ax0.plot(t, c, color="0.4", lw=0.8)
for name, marker, color in [("10% stop-loss", "v", "crimson"),
                            ("15% trailing stop", "o", "darkorange")]:
    k = exits(out[name])
    ax0.plot(t[k], c[k], marker, mfc="none", color=color, ms=6, label=name)
ax0.set_ylabel("BTC close (USD)")
ax0.legend(loc="upper left", fontsize=8)
for (name, res), color in zip(out.items(), ["0.6", "crimson", "darkorange"]):
    ax1.plot(t, res[:, 0], color=color, label=name)
ax1.axhline(0, color="k", lw=0.5)
ax1.set_ylabel("equity (USD, one BTC)")
ax1.legend(loc="upper left", fontsize=8)
plt.tight_layout()
```

On this sample both exits reduce the largest drawdown, from about $44,800 to
$33,800 with the stop-loss and $25,400 with the trailing stop. The stop-loss
also raises the total pnl, from $56,400 to $72,200. The trailing stop ends near
the no-exit pnl, at $54,100, while holding a position on fewer than half the
days. The markers sit at the close of each exit day. Picking an exit level by
comparing a few variants on the same data is a form of selection; the
[parameter grid recipe](parameter-grid.md) shows how to account for it.

Related pages: [`BacktestOHLCTarget`](../functions_fin/BacktestOHLCTarget.md),
[`backtest_report`](../functions_fin/backtest_report.md), and
[Choosing a backtest engine](../functions_fin/choosing_a_backtest_engine.md).
