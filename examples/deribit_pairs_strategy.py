"""Regime-adaptive ETH/BTC pairs strategy on Deribit perpetual trades.

The idea: track the ETH/BTC ratio, measure whether it is mean-reverting or
wandering like a random walk, and trade it the way the regime warrants instead of
assuming one or the other.

The spread is the log-ratio, `log(ETH) - log(BTC)`. ETH and BTC move close to
one-for-one in log terms, so a fixed one-to-one hedge is the natural spread and
avoids estimating a ratio. (A rolling regression hedge like `RollingSpread` is
the right tool when the two legs move at genuinely different scales, but on
log-prices its large means amplify small beta wobble into the spread, so the
plain log-ratio is cleaner for ETH/BTC.)

The screamer functions that do the work:

  - regime detection: `RollingOU` gives the mean-reversion rate, the reliable
    measure here (near zero for a random walk, larger when the ratio pulls back
    to its mean). `RollingHurst` reports persistence as a second opinion, though
    it can be NaN on a very stable or repeated tick ratio, so the OU rate leads.
  - signal: the z-score of the spread, `RollingZscore`. The strategy fades the
    z-score when the ratio mean-reverts and follows it when it does not, acting
    only on a meaningful deviation. `SchmittTrigger` latches the regime so it
    does not flip on noise.

Two lessons that shaped this, both worth knowing:

  - Trade ticks are heavily autocorrelated (carry-forward alignment, bid-ask
    bounce), so slope significance / t-stat trend tests are spurious on them and
    are avoided here. A bounded z-score signal does not assume independent samples.
  - For ETH/BTC perpetuals the intraday ratio is close to a random walk (OU rate
    near zero, half-life in the thousands of ticks), so this strategy stays in
    the follow branch and never fades on the sample data. That is the detector
    working: it switches to fading only for a genuinely mean-reverting pair.

The position is a target in [-1, +1], where +1 means long the ratio (long ETH,
short BTC). The feeds come from deribit_streaming.py, so the same code runs on
replayed history and on a live Deribit feed.

Run:
    python examples/deribit_pairs_strategy.py
    python examples/deribit_pairs_strategy.py --limit 30000 --every 5000
    python examples/deribit_pairs_strategy.py --live
"""
import argparse
import math

from screamer import RollingOU, RollingHurst, RollingZscore, SchmittTrigger
from deribit_streaming import replay_feed, live_feed


def _finite(x, default=0.0):
    return x if x == x else default   # NaN check (NaN != NaN)


class PairStrategy:
    """A regime-adaptive ratio trader for one (base, quote) pair.

    Feed it trades one at a time with `on_trade`. It keeps the latest price of
    each leg, forms the log-ratio on every trade, and returns the current regime
    and target position.
    """

    def __init__(self, base, quote, regime=1000, z_window=500,
                 mr_low=0.02, mr_high=0.05, z_entry=1.5, z_scale=1.5):
        self.base, self.quote = base, quote
        self.rate = RollingOU(regime, output=0)         # OU mean-reversion rate
        self.hurst = RollingHurst(regime)
        self.mr_gate = SchmittTrigger(mr_low, mr_high)  # 1 mean-revert, held between
        self.z = RollingZscore(z_window)
        self.z_entry, self.z_scale = z_entry, z_scale

        self.last_base = self.last_quote = None
        self.state = None

    def on_trade(self, symbol, price):
        if symbol == self.base:
            self.last_base = price
        elif symbol == self.quote:
            self.last_quote = price
        else:
            return None
        if self.last_base is None or self.last_quote is None:
            return None

        spread = math.log(self.last_base) - math.log(self.last_quote)   # log-ratio
        rate = self.rate(spread)                 # mean-reversion rate
        hurst = self.hurst(spread)
        z = self.z(spread)

        mr_flag = self.mr_gate(_finite(rate))    # 1 mean-revert, 0 trend

        # One bounded signal, the z-score deviation, used both ways: fade it when
        # the ratio mean-reverts, follow it when it does not. Act only past the
        # entry band, so small wiggles leave the position flat.
        if z == z and abs(z) > self.z_entry:
            direction = math.tanh(z / self.z_scale)
            sign = 1.0 - 2.0 * mr_flag           # +1 follow (trend), -1 fade (mean-revert)
            position = sign * direction
        else:
            position = 0.0

        half_life = math.log(2) / rate if (rate == rate and rate > 1e-6) else float("inf")
        regime = "mean-revert" if mr_flag > 0.5 else "trend"

        self.state = dict(ratio=self.last_base / self.last_quote, spread=spread, rate=rate,
                          half_life=half_life, hurst=hurst, z=z, regime=regime, position=position)
        return self.state


def _fmt(x, width=8, prec=3):
    if x != x:
        return " " * (width - 3) + "nan"
    if x == float("inf"):
        return " " * (width - 3) + "inf"
    return f"{x:{width}.{prec}f}"


def print_state(n, s):
    print(
        f"[{n:>7} ticks]  {s['regime']:<12}"
        f"  ratio {_fmt(s['ratio'], 8, 5)}"
        f"  z {_fmt(s['z'], 6, 2)}"
        f"  rate {_fmt(s['rate'], 7, 4)}"
        f"  half_life {_fmt(s['half_life'], 8, 0)}"
        f"  position {_fmt(s['position'], 6, 2)}"
    )


def run(base, quote, live=False, limit=None, every=5000):
    strat = PairStrategy(base, quote)
    feed = live_feed([base, quote]) if live else replay_feed([base, quote])

    mode = "live Deribit feed" if live else "replaying sample trades"
    print(f"ETH/BTC regime-adaptive pairs strategy, {mode}")
    print(f"spread = log({base}) - log({quote}); position in [-1, +1]\n")

    n = 0
    for symbol, price in feed:
        state = strat.on_trade(symbol, price)
        if state is None:
            continue
        n += 1
        if (live and n % 50 == 0) or (not live and n % every == 0):
            print_state(n, state)
        if limit and n >= limit:
            break

    if strat.state is not None:
        print("\nfinal:")
        print_state(n, strat.state)
    print(f"\ndone, {n} paired ticks processed")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="ETH-PERPETUAL")
    p.add_argument("--quote", default="BTC-PERPETUAL")
    p.add_argument("--live", action="store_true", help="stream live from Deribit (needs websockets)")
    p.add_argument("--limit", type=int, default=None, help="stop after N paired ticks")
    p.add_argument("--every", type=int, default=5000, help="print a snapshot every N paired ticks")
    args = p.parse_args()
    run(args.base, args.quote, live=args.live, limit=args.limit, every=args.every)


if __name__ == "__main__":
    main()
