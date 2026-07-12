"""Backtest the ETH/BTC ratio fade on Deribit perpetual trades, taker vs maker.

Two execution models, both grounded in the trade data (Deribit has no historical
best-bid/offer, so we reconstruct what we can from the prints):

  - taker: cross the book at the bar close. You pay the taker fee on both legs
    plus the crossed half-spread (estimated per leg with Roll's estimator). Fills
    are immediate.
  - maker: post a limit at the touch and fill only when a later print trades
    *through* your level. This is the honest fill model: you often miss (your
    sell sits above a market that is reverting down), and you fill at your limit
    price when the market comes to you. No `capture` fudge factor; the fills come
    from the intra-bar high and low of the spread.

Strategies: always-fade, regime-gated fade (only fade while `RollingOU` says the
ratio is reverting), and buy & hold. Reports gross Sharpe, taker and maker net
results, the maker fill rate, and a train/test split, and saves an equity plot.

The lesson on real data: the fade has a real but thin mean-reversion edge. As a
taker the two-leg fees erase it. As a maker you would earn the spread, but the
fill model shows you catch only about a third of the entries, and the edge that
survives is small and not stable out of sample. A liquid pair stays hard to beat.

Data: reads `*ETH-PERPETUAL*.csv` and `*BTC-PERPETUAL*.csv` from --data-dir
(default devtools/data/). Download more with deribit_download.py.

Run:
    python examples/deribit_pairs_backtest.py --data-dir examples --bar 30s
    python examples/deribit_pairs_backtest.py --bar 30s --entry 1.0 --exit 0.25
"""
import argparse
import glob
import math
import os

import numpy as np
import pandas as pd

from screamer import RollingZscore, RollingOU

REPO = os.path.dirname(os.path.dirname(__file__))


def roll_half_spread_bps(prices):
    """Roll's effective half-spread (bps) from the bid-ask bounce in trade prices."""
    dp = np.diff(prices)
    if len(dp) < 3:
        return 0.0
    cov = np.cov(dp[1:], dp[:-1])[0, 1]
    spread = 2 * math.sqrt(-cov) if cov < 0 else 0.0
    return (spread / 2) / np.median(prices) * 1e4


def load_ohlc_spread(data_dir, base, quote, bar):
    """Tick-align the two legs, then bar the log-ratio into OHLC.

    Returns (times, close, high, low, combined half-spread bps). The high and low
    are what the maker fill model needs: they say how far the spread printed
    inside each bar, hence whether a resting limit would have been hit.
    """
    half = 0.0
    legs = {}
    for key, symbol in (("e", base), ("q", quote)):
        matches = glob.glob(os.path.join(data_dir, f"*{symbol.lower()}*.csv"))
        if not matches:
            raise SystemExit(f"no CSV for {symbol} in {data_dir}")
        df = pd.read_csv(matches[0])
        half += roll_half_spread_bps(df["price"].values)
        s = pd.Series(df["price"].values, index=pd.to_datetime(df["timestamp"], unit="ms"))
        legs[key] = s.groupby(level=0).last()               # collapse same-ms prints
    idx = legs["e"].index.union(legs["q"].index)
    both = pd.DataFrame({k: v.reindex(idx).ffill() for k, v in legs.items()}).dropna()
    tick = np.log(both["e"]) - np.log(both["q"])
    o = tick.resample(bar).ohlc().dropna()
    return o.index.values, o["close"].values, o["high"].values, o["low"].values, half


def signals(close, z_window, regime_window):
    z = np.asarray(RollingZscore(z_window)(close))
    rate = np.asarray(RollingOU(regime_window, output=0)(close)) if regime_window else None
    return z, rate


def _target(pos, zt, rate_t, rate_thr, entry, exit_band):
    """Desired position: fade the z-score with hysteresis, gated on the regime."""
    reverting = True if rate_t is None else (rate_t == rate_t and rate_t > rate_thr)
    if pos == 0.0 and zt == zt and abs(zt) > entry and reverting:
        return -float(np.sign(zt))
    if pos != 0.0 and zt == zt and (abs(zt) < exit_band or not reverting):
        return 0.0
    return pos


def _stats(per_bar, bars_per_year, total=None):
    std = per_bar.std()
    sharpe = per_bar.mean() / std * math.sqrt(bars_per_year) if std > 0 else 0.0
    return dict(sharpe=sharpe, total=per_bar.sum() if total is None else total, per_bar=per_bar)


def taker_backtest(close, z, rate, entry, exit_band, cost_bps, bpy):
    """Cross at the close on every position change; pay cost_bps per unit turnover."""
    thr = np.nanmedian(rate) if rate is not None else None
    pos = 0.0
    path = np.zeros(len(close))
    for t in range(len(close)):
        pos = _target(pos, z[t], None if rate is None else rate[t], thr, entry, exit_band)
        path[t] = pos
    move = np.diff(close)
    turnover = np.abs(np.diff(np.concatenate([[0.0], path])))[:-1]
    net = path[:-1] * move - (cost_bps / 1e4) * turnover
    return _stats(net, bpy), turnover.sum() / 2


def maker_backtest(close, high, low, z, rate, entry, exit_band, half_bps, maker_fee_bps, bpy):
    """Post a limit at the touch; fill only when a print trades through it."""
    thr = np.nanmedian(rate) if rate is not None else None
    half = half_bps / 1e4
    fee = 2 * maker_fee_bps / 1e4
    pos = 0.0
    cash = 0.0
    equity = np.zeros(len(close))
    fills = attempts = 0
    for t in range(1, len(close)):
        ref = close[t - 1]
        target = _target(pos, z[t - 1], None if rate is None else rate[t - 1], thr, entry, exit_band)
        if target > pos:                                    # want to buy: post below the market
            attempts += 1
            limit = ref - half
            if low[t] < limit:                              # a print traded through the limit
                delta = target - pos
                cash -= limit * delta + fee * abs(delta)
                pos = target
                fills += 1
        elif target < pos:                                  # want to sell: post above the market
            attempts += 1
            limit = ref + half
            if high[t] > limit:
                delta = target - pos
                cash -= limit * delta + fee * abs(delta)
                pos = target
                fills += 1
        equity[t] = cash + pos * close[t]
    return _stats(np.diff(equity), bpy, total=equity[-1]), fills, fills / max(attempts, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=os.path.join(REPO, "devtools", "data"))
    p.add_argument("--base", default="ETH-PERPETUAL")
    p.add_argument("--quote", default="BTC-PERPETUAL")
    p.add_argument("--bar", default="30s")
    p.add_argument("--z-window", type=int, default=100)
    p.add_argument("--regime-window", type=int, default=200)
    p.add_argument("--entry", type=float, default=1.0)
    p.add_argument("--exit", type=float, default=0.25, dest="exit_band")
    p.add_argument("--taker-fee-bps", type=float, default=5.0)
    p.add_argument("--maker-fee-bps", type=float, default=0.0)
    p.add_argument("--half-spread-bps", type=float, default=None,
                   help="combined half-spread both legs; default is estimated from the data")
    p.add_argument("--plot", default=os.path.join(os.path.dirname(__file__), "pairs_equity.png"))
    args = p.parse_args()

    times, close, high, low, half_est = load_ohlc_spread(args.data_dir, args.base, args.quote, args.bar)
    half = args.half_spread_bps if args.half_spread_bps is not None else half_est
    bar_seconds = pd.Timedelta(args.bar).total_seconds()
    bpy = 365 * 24 * 3600 / bar_seconds
    minutes = len(close) * bar_seconds / 60
    taker_cost = 2 * args.taker_fee_bps + half
    ac1 = pd.Series(np.diff(close)).autocorr(1)
    print(f"{len(close)} bars of {args.bar} over {minutes/1440:.1f} days | ratio-return autocorr "
          f"{ac1:+.3f} | half-spread {half:.2f} bps | taker cost {taker_cost:.2f} bps/turn\n")

    z, rate = signals(close, args.z_window, args.regime_window)
    variants = {"always-fade": None, "regime-gated": rate}
    cut = int(len(close) * 2 / 3)

    print(f"{'strategy':<14} {'gross_S':>8} | {'taker%':>7} {'taker_S':>8} |"
          f" {'maker%':>7} {'maker_S':>8} {'fill':>5} {'test_mk_S':>9} | {'min/trd':>7}")
    curves = {}
    for name, gate in variants.items():
        gross, _ = taker_backtest(close, z, gate, args.entry, args.exit_band, 0.0, bpy)
        taker, n_taker = taker_backtest(close, z, gate, args.entry, args.exit_band, taker_cost, bpy)
        maker, fills, fillrate = maker_backtest(close, high, low, z, gate, args.entry,
                                                args.exit_band, half, args.maker_fee_bps, bpy)
        mk_test, _, _ = maker_backtest(close[cut:], high[cut:], low[cut:], z[cut:],
                                       None if gate is None else gate[cut:], args.entry,
                                       args.exit_band, half, args.maker_fee_bps, bpy)
        curves[name] = (taker, maker)
        print(f"{name:<14} {gross['sharpe']:>8.1f} | {taker['total']*100:>7.2f} {taker['sharpe']:>8.1f} |"
              f" {maker['total']*100:>7.2f} {maker['sharpe']:>8.1f} {fillrate:>5.0%} {mk_test['sharpe']:>9.1f} |"
              f" {minutes/max(fills,1):>7.1f}")

    _plot(times, close, curves, cut, args.plot)
    print(f"\nequity curves saved to {args.plot}")


def _plot(times, close, curves, cut, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = times[1:]
    plt.figure(figsize=(11, 5.5))
    plt.plot(t, np.cumsum(curves["regime-gated"][1]["per_bar"]) * 100, color="#28a745", lw=1.7,
             label="regime-gated fade (maker, fill model)")
    plt.plot(t, np.cumsum(curves["always-fade"][1]["per_bar"]) * 100, color="#0074a2", lw=1.3,
             label="always-fade (maker, fill model)")
    plt.plot(t, np.cumsum(curves["regime-gated"][0]["per_bar"]) * 100, color="#d62728", lw=1.3, ls="--",
             label="regime-gated fade (taker)")
    plt.plot(t, np.cumsum(np.diff(close)) * 100, color="#888", lw=1.0, label="buy & hold ratio")
    plt.axvline(t[cut], color="k", ls=":", lw=1)
    plt.text(t[cut], plt.ylim()[1] * 0.92, " test", fontsize=9)
    plt.axhline(0, color="k", lw=0.6)
    plt.title("ETH/BTC ratio fade: taker vs maker (realistic fill model)")
    plt.ylabel("cumulative P&L (%)")
    plt.xlabel("time")
    plt.legend(loc="best", fontsize=9)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
