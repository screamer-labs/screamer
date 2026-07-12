"""Backtest the ETH/BTC ratio fade on Deribit perpetual trades.

Resamples two instruments to a common bar clock, builds the log-ratio spread, and
runs three positions through a P&L loop with transaction costs:

  - always-fade: short the ratio when its z-score is high, long when low, exit
    near the mean. This is the mean-reversion bet.
  - regime-gated fade: the same, but only while `RollingOU` says the ratio is
    actually mean-reverting. It stands aside when the ratio starts to trend.
  - buy & hold: long the ratio the whole time, as a benchmark.

It reports gross and net Sharpe and return, splits the sample into train and test
so you can see whether the edge holds out of sample, and saves an equity-curve
plot.

What you will see on real data: the ratio mean-reverts most of the time, so the
fade is strongly profitable gross. But the per-trade edge is a fraction of a basis
point, so realistic costs (two legs, a few bps a round-turn) erase it, and when
the ratio switches to trending the naive fade gives its gains back. The regime
gate cuts that give-back. In short, the signal is real but thin: it needs
maker-level costs to be tradeable, and it needs the regime check to survive.

Data: reads `*ETH-PERPETUAL*.csv` and `*BTC-PERPETUAL*.csv` from --data-dir
(default devtools/data/, the committed sample). Download more with
deribit_download.py, which is what makes the train/test split meaningful.

Run:
    python examples/deribit_pairs_backtest.py
    python examples/deribit_pairs_backtest.py --data-dir examples --bar 60s --cost-bps 2
"""
import argparse
import glob
import math
import os

import numpy as np
import pandas as pd

from screamer import RollingZscore, RollingOU

REPO = os.path.dirname(os.path.dirname(__file__))


def load_spread(data_dir, base, quote, bar):
    """Resample both legs to `bar` and return (times, log-ratio spread)."""
    def series(symbol):
        matches = glob.glob(os.path.join(data_dir, f"*{symbol.lower()}*.csv"))
        if not matches:
            raise SystemExit(f"no CSV for {symbol} in {data_dir}")
        df = pd.read_csv(matches[0])
        df["t"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df.set_index("t")["price"].resample(bar).last().ffill()

    both = pd.concat([series(base), series(quote)], axis=1, keys=["b", "q"]).dropna()
    spread = (np.log(both["b"]) - np.log(both["q"])).values
    return both.index.values, spread


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
                held = -np.sign(zt)                       # enter the fade
            elif held != 0.0 and (abs(zt) < exit_band or not reverting):
                held = 0.0                                # exit near the mean, or on regime change
        pos[t] = held
    return pos


def evaluate(spread, pos, cost_bps, bars_per_year, days):
    """P&L of holding `pos` through the next bar's spread move, net of costs."""
    move = np.diff(spread)                                # spread move, t -> t+1
    gross = pos[:-1] * move
    turnover = np.abs(np.diff(np.concatenate([[0.0], pos])))[:-1]
    net = gross - (cost_bps / 1e4) * turnover
    sharpe = net.mean() / net.std() * math.sqrt(bars_per_year) if net.std() > 0 else 0.0
    gross_sharpe = gross.mean() / gross.std() * math.sqrt(bars_per_year) if gross.std() > 0 else 0.0
    return dict(net=net, sharpe=sharpe, gross_sharpe=gross_sharpe,
                total=net.sum(), trades_per_day=turnover.sum() / 2 / max(days, 1e-9))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=os.path.join(REPO, "devtools", "data"))
    p.add_argument("--base", default="ETH-PERPETUAL")
    p.add_argument("--quote", default="BTC-PERPETUAL")
    p.add_argument("--bar", default="60s", help="bar size, e.g. 15s, 60s, 5min")
    p.add_argument("--z-window", type=int, default=100)
    p.add_argument("--regime-window", type=int, default=200)
    p.add_argument("--entry", type=float, default=2.5)
    p.add_argument("--exit", type=float, default=1.0, dest="exit_band")
    p.add_argument("--cost-bps", type=float, default=2.0, help="round-turn cost per unit turnover, both legs")
    p.add_argument("--plot", default=os.path.join(os.path.dirname(__file__), "pairs_equity.png"))
    args = p.parse_args()

    times, spread = load_spread(args.data_dir, args.base, args.quote, args.bar)
    bar_seconds = pd.Timedelta(args.bar).total_seconds()
    bpy = 365 * 24 * 3600 / bar_seconds
    days = len(spread) * bar_seconds / 86400
    ac1 = pd.Series(np.diff(spread)).autocorr(1)
    print(f"{len(spread)} bars of {args.bar} over {days:.1f} days, "
          f"ratio-return lag-1 autocorrelation {ac1:+.3f} "
          f"({'mean-reverting' if ac1 < -0.02 else 'random walk'})\n")

    strategies = {
        "always-fade": fade_positions(spread, args.z_window, args.entry, args.exit_band),
        "regime-gated": fade_positions(spread, args.z_window, args.entry, args.exit_band, args.regime_window),
        "buy & hold": np.ones(len(spread)),
    }

    cut = int(len(spread) * 2 / 3)
    print(f"{'strategy':<14} {'gross_S':>8} {'net_S':>7} {'net_ret%':>9} {'trd/day':>8}  |"
          f" {'train_S':>8} {'test_S':>7}")
    for name, pos in strategies.items():
        full = evaluate(spread, pos, args.cost_bps, bpy, days)
        train = evaluate(spread[:cut], pos[:cut], args.cost_bps, bpy, days * 2 / 3)
        test = evaluate(spread[cut:], pos[cut:], args.cost_bps, bpy, days / 3)
        print(f"{name:<14} {full['gross_sharpe']:>8.1f} {full['sharpe']:>7.1f} "
              f"{full['total'] * 100:>9.2f} {full['trades_per_day']:>8.1f}  |"
              f" {train['sharpe']:>8.1f} {test['sharpe']:>7.1f}")

    _plot(times, spread, strategies, args.cost_bps, cut, bpy, days, args.plot)
    print(f"\nequity curves saved to {args.plot}")


def _plot(times, spread, strategies, cost_bps, cut, bpy, days, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def equity(pos, cost):
        return np.cumsum(evaluate(spread, pos, cost, bpy, days)["net"]) * 100

    t = times[1:]
    plt.figure(figsize=(11, 5.5))
    plt.plot(t, equity(strategies["always-fade"], 0), color="#0074a2", lw=1.6, label="always-fade (gross)")
    plt.plot(t, equity(strategies["regime-gated"], 0), color="#28a745", lw=1.6, label="regime-gated fade (gross)")
    plt.plot(t, equity(strategies["always-fade"], cost_bps), color="#d62728", lw=1.3, ls="--",
             label=f"always-fade (net, {cost_bps:g}bp)")
    plt.plot(t, np.cumsum(np.diff(spread)) * 100, color="#888", lw=1.1, label="buy & hold ratio")
    plt.axvline(t[cut], color="k", ls=":", lw=1)
    plt.text(t[cut], plt.ylim()[1] * 0.92, " test", fontsize=9)
    plt.axhline(0, color="k", lw=0.6)
    plt.title("ETH/BTC ratio fade on Deribit perpetual trades")
    plt.ylabel("cumulative P&L (%)")
    plt.xlabel("time")
    plt.legend(loc="best", fontsize=9)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
