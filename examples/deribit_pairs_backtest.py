"""Backtest the ETH/BTC ratio fade on Deribit perpetual trades, taker vs maker.

Resamples two instruments to a common bar clock, builds the log-ratio spread, and
runs positions through a P&L loop under two execution models:

  - taker: cross the book on every trade. You pay the taker fee on both legs plus
    half the bid-ask spread on each. Cost is positive and large.
  - maker: post a limit at the bid/ask. You pay the maker fee (often zero) and
    *earn* part of the spread instead of paying it, so the cost per trade can be
    negative. A `maker_capture` fraction accounts for the fills you miss and the
    adverse selection you cannot avoid without an order book in the data.

The spread each leg costs to cross is estimated from the trades themselves with
Roll's estimator, so the cost is grounded rather than guessed (override with
--half-spread-bps).

Strategies: always-fade, regime-gated fade (only fade while `RollingOU` says the
ratio is reverting), and buy & hold as a benchmark. It reports gross Sharpe and
both taker and maker net results, splits the sample into train and test, and
saves an equity-curve plot.

The lesson on real data: the fade has a real but thin mean-reversion edge. As a
taker the two-leg fees erase it. As a maker you are paid the spread, which can
turn the same signal net-positive; the regime gate then protects it when the
ratio switches from reverting to trending.

Data: reads `*ETH-PERPETUAL*.csv` and `*BTC-PERPETUAL*.csv` from --data-dir
(default devtools/data/, the committed sample). Download more with
deribit_download.py to make the train/test split meaningful.

Run:
    python examples/deribit_pairs_backtest.py --data-dir examples --bar 60s
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
        return float("nan")
    cov = np.cov(dp[1:], dp[:-1])[0, 1]
    spread = 2 * math.sqrt(-cov) if cov < 0 else 0.0
    return (spread / 2) / np.median(prices) * 1e4


def load_spread(data_dir, base, quote, bar):
    """Resample both legs to `bar`; return (times, log-ratio spread, combined half-spread bps)."""
    half = 0.0
    resampled = {}
    for key, symbol in (("b", base), ("q", quote)):
        matches = glob.glob(os.path.join(data_dir, f"*{symbol.lower()}*.csv"))
        if not matches:
            raise SystemExit(f"no CSV for {symbol} in {data_dir}")
        df = pd.read_csv(matches[0])
        half += roll_half_spread_bps(df["price"].values)
        df["t"] = pd.to_datetime(df["timestamp"], unit="ms")
        resampled[key] = df.set_index("t")["price"].resample(bar).last().ffill()
    both = pd.concat(resampled.values(), axis=1, keys=resampled.keys()).dropna()
    spread = (np.log(both["b"]) - np.log(both["q"])).values
    return both.index.values, spread, half


def fade_positions(spread, z_window, entry, exit_band, regime_window=None):
    """Fade the z-score with hysteresis. If regime_window is set, only fade while
    the OU mean-reversion rate is above its median (the reverting regime)."""
    z = np.asarray(RollingZscore(z_window)(spread))
    rate = np.asarray(RollingOU(regime_window, output=0)(spread)) if regime_window else None
    threshold = np.nanmedian(rate) if rate is not None else None

    pos = np.zeros(len(spread))
    held = 0.0
    for t in range(len(spread)):
        zt = z[t]
        reverting = True if rate is None else (rate[t] == rate[t] and rate[t] > threshold)
        if zt == zt:
            if held == 0.0 and abs(zt) > entry and reverting:
                held = -np.sign(zt)
            elif held != 0.0 and (abs(zt) < exit_band or not reverting):
                held = 0.0
        pos[t] = held
    return pos


def turnover_cost_bps(execution, taker_fee, maker_fee, half_spread, capture):
    """Cost in bps per unit of turnover (trading one unit in each of the two legs)."""
    if execution == "taker":
        return 2 * taker_fee + half_spread          # pay both fees, cross both spreads
    return 2 * maker_fee - capture * half_spread     # pay maker fees, earn captured spread


def evaluate(spread, pos, cost_bps, bars_per_year):
    move = np.diff(spread)                           # spread move, t -> t+1
    gross = pos[:-1] * move
    turnover = np.abs(np.diff(np.concatenate([[0.0], pos])))[:-1]
    net = gross - (cost_bps / 1e4) * turnover
    def sharpe(x):
        return x.mean() / x.std() * math.sqrt(bars_per_year) if x.std() > 0 else 0.0
    return dict(net=net, sharpe=sharpe(net), gross_sharpe=sharpe(gross),
                total=net.sum(), trades=turnover.sum() / 2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=os.path.join(REPO, "devtools", "data"))
    p.add_argument("--base", default="ETH-PERPETUAL")
    p.add_argument("--quote", default="BTC-PERPETUAL")
    p.add_argument("--bar", default="60s", help="bar size, e.g. 15s, 30s, 60s")
    p.add_argument("--z-window", type=int, default=100)
    p.add_argument("--regime-window", type=int, default=200)
    p.add_argument("--entry", type=float, default=1.5)
    p.add_argument("--exit", type=float, default=0.5, dest="exit_band")
    p.add_argument("--taker-fee-bps", type=float, default=5.0, help="taker fee per leg")
    p.add_argument("--maker-fee-bps", type=float, default=0.0, help="maker fee per leg")
    p.add_argument("--half-spread-bps", type=float, default=None,
                   help="combined half-spread both legs; default is estimated from the data")
    p.add_argument("--maker-capture", type=float, default=0.5,
                   help="fraction of the half-spread a resting order actually earns")
    p.add_argument("--plot", default=os.path.join(os.path.dirname(__file__), "pairs_equity.png"))
    args = p.parse_args()

    times, spread, half_est = load_spread(args.data_dir, args.base, args.quote, args.bar)
    half = args.half_spread_bps if args.half_spread_bps is not None else half_est
    bar_seconds = pd.Timedelta(args.bar).total_seconds()
    bpy = 365 * 24 * 3600 / bar_seconds
    minutes = len(spread) * bar_seconds / 60

    taker = turnover_cost_bps("taker", args.taker_fee_bps, args.maker_fee_bps, half, args.maker_capture)
    maker = turnover_cost_bps("maker", args.taker_fee_bps, args.maker_fee_bps, half, args.maker_capture)
    ac1 = pd.Series(np.diff(spread)).autocorr(1)
    print(f"{len(spread)} bars of {args.bar} over {minutes/1440:.1f} days | "
          f"ratio-return autocorr {ac1:+.3f} | est half-spread {half:.2f} bps")
    print(f"cost per turnover: taker {taker:+.2f} bps, maker {maker:+.2f} bps "
          f"(fee {args.taker_fee_bps}/{args.maker_fee_bps}, capture {args.maker_capture})\n")

    strategies = {
        "always-fade": fade_positions(spread, args.z_window, args.entry, args.exit_band),
        "regime-gated": fade_positions(spread, args.z_window, args.entry, args.exit_band, args.regime_window),
        "buy & hold": np.ones(len(spread)),
    }
    cut = int(len(spread) * 2 / 3)
    header = (f"{'strategy':<14} {'gross_S':>8} | {'taker%':>7} {'taker_S':>8} |"
              f" {'maker%':>7} {'maker_S':>8} | {'min/trd':>7} {'test_mk_S':>9}")
    print(header)
    for name, pos in strategies.items():
        g = evaluate(spread, pos, 0.0, bpy)
        tk = evaluate(spread, pos, taker, bpy)
        mk = evaluate(spread, pos, maker, bpy)
        mk_test = evaluate(spread[cut:], pos[cut:], maker, bpy)["sharpe"]
        per_trade = minutes / max(g["trades"], 1)
        print(f"{name:<14} {g['gross_sharpe']:>8.1f} | {tk['total']*100:>7.2f} {tk['sharpe']:>8.1f} |"
              f" {mk['total']*100:>7.2f} {mk['sharpe']:>8.1f} | {per_trade:>7.1f} {mk_test:>9.1f}")

    _plot(times, spread, strategies, taker, maker, cut, bpy, args.plot)
    print(f"\nequity curves saved to {args.plot}")


def _plot(times, spread, strategies, taker, maker, cut, bpy, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def equity(pos, cost):
        return np.cumsum(evaluate(spread, pos, cost, bpy)["net"]) * 100

    t = times[1:]
    plt.figure(figsize=(11, 5.5))
    plt.plot(t, equity(strategies["regime-gated"], maker), color="#28a745", lw=1.7,
             label="regime-gated fade (maker)")
    plt.plot(t, equity(strategies["always-fade"], maker), color="#0074a2", lw=1.3,
             label="always-fade (maker)")
    plt.plot(t, equity(strategies["regime-gated"], taker), color="#d62728", lw=1.3, ls="--",
             label="regime-gated fade (taker)")
    plt.plot(t, np.cumsum(np.diff(spread)) * 100, color="#888", lw=1.0, label="buy & hold ratio")
    plt.axvline(t[cut], color="k", ls=":", lw=1)
    plt.text(t[cut], plt.ylim()[1] * 0.92, " test", fontsize=9)
    plt.axhline(0, color="k", lw=0.6)
    plt.title("ETH/BTC ratio fade: taker vs maker execution")
    plt.ylabel("cumulative P&L (%)")
    plt.xlabel("time")
    plt.legend(loc="best", fontsize=9)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
