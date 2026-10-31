---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Backtest with limit and market orders

`BacktestOHLCOrders` takes a bid and an ask for each bar, each a price and a
size, and fills them as resting limit orders when the bar's range reaches
them, or as market orders at the open when the price is NaN.

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
from screamer import (EwMean, Sign, Lag, FillNa, BacktestOHLCOrders,
                      backtest_report)

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
btc = df[df["symbol"] == "BTC"].set_index("date")
o, h, l, c = (btc[k].to_numpy() for k in ["open", "high", "low", "close"])
direction = Sign()(EwMean(span=20)(c) - EwMean(span=100)(c))
direction[:100] = 0.0
side = FillNa(0.0)(Lag(1)(direction))            # decided at yesterday's close
prev_close = FillNa(c[0])(Lag(1)(c))

engine = BacktestOHLCOrders(maker_fee=1e-3, taker_fee=1e-3, slippage=5e-4,
                            min_position=-1.0, max_position=1.0)
prices = {"market order": float("nan"), "limit 1% away": 0.01, "limit 3% away": 0.03}
out = {name: engine(prev_close * (1 - k), 2.0 * (side > 0),
                    prev_close * (1 + k), 2.0 * (side < 0), o, h, l, c)
       for name, k in prices.items()}
pd.DataFrame({n: backtest_report(r, periods_per_year=365)[1] for n, r in out.items()}
             ).loc[["total_pnl", "total_cost", "num_trades", "sharpe"]].round(2)
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(9, 3.5))
for (name, res), color in zip(out.items(), ["0.6", "steelblue", "darkorange"]):
    ax.plot(btc.index, res[:, 0], color=color, label=name)
ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel("profit (USD, 1 BTC)")
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()
```

The rule is the 20/100-day trend rule of [Backtest a signal on OHLCV
bars](ohlcv.md), long or short one BTC, on daily Binance bars. Each day it
posts one order, a bid to go long or an ask to go short: a market order, or a
limit order 1% or 3% from yesterday's close, below it for a bid and above it
for an ask. The size is 2 BTC, and the bounds `min_position=-1`
and `max_position=1` cut each fill to the room left, so a reversal trades 2
BTC, an order on the side the book already holds trades nothing, and the
position stays between -1 and 1.

The engine fills an order on the bar it is posted, not on the next one, so the
page lags the direction and the reference close by one day with `Lag`. With
the default `fill="touch"`, a resting bid fills when the bar's low reaches its
price (`low <= bid_price`) and a resting ask when the high reaches its price;
`fill="breach"` needs the bar to trade through it. A limit order fills at its
own price and pays `maker_fee`. A NaN price is a market order: it fills at the
open, moved against the trade by `slippage`, and pays `taker_fee`. Fees are
0.1% for both, and slippage is 0.05%. A limit order that the range does not
reach is posted again the next day, so the book keeps the old position until
the price comes to it.

Waiting for a better price saves the slippage and buys at a lower price when
it fills, but it misses the reversals that run away from the order. Here the
order 1% away made less than the market order, and the order 3% away made
more with two trades fewer; on one sample, neither difference shows that
waiting pays.

A stop order waits until the price moves past a level, up for a buy stop and
down for a sell stop, and then trades at the market. `BacktestOHLCOrders` does
not model it: a bid above the market is not a buy stop, because the bar's low
is below it, so it fills at once at its own price. For a stop-loss or a trailing stop on a position, use the
`stop_loss` and `trailing_stop` parameters of the target engines, as on [Add
stops and exits](../stops/stops-and-exits.md).

## Go deeper

- [`BacktestOHLCOrders`](../../functions_fin/BacktestOHLCOrders.md): fill rules, `participation_ratio`, and the order of two fills on one bar.
- [Add stops and exits](../stops/stops-and-exits.md): stop-loss, take-profit, and trailing-stop exits.
- [Backtest on tick trades instead of bars](tick-trades.md): resting orders filled by the trade tape.
- [Choosing a backtest engine](../../functions_fin/choosing_a_backtest_engine.md): the MARKET, NaN and infinity price encoding.
