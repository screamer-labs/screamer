"""Stream Deribit perpetual trades into screamer indicators.

The same indicator code runs on two feeds, which is the whole point of screamer:
develop on stored history, then run unchanged on a live stream.

  - replay (default): the sample trade CSVs in devtools/data/, interleaved by
    timestamp so ETH and BTC arrive as one time-ordered feed.
  - live (--live): the Deribit public WebSocket, one trade at a time.

Run:
    python examples/deribit_streaming.py                # replay the sample data
    python examples/deribit_streaming.py --limit 5000   # replay the first 5000 trades
    python examples/deribit_streaming.py --live         # live feed (pip install websockets)
"""
import argparse
import csv
import glob
import heapq
import os

from screamer import (
    EwMean, RollingMean, RollingStd, RollingZscore, RollingMax, RollingMin, LogReturn,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "devtools", "data")
WS_URL = "wss://www.deribit.com/ws/api/v2"
DEFAULT_SYMBOLS = ["ETH-PERPETUAL", "BTC-PERPETUAL"]


class SymbolIndicators:
    """Streaming indicators for one instrument, fed one trade at a time.

    Each screamer object holds its own state, so calling it on successive
    prices continues the calculation. Feeding an array would give the same
    numbers; only the delivery differs.
    """

    def __init__(self, name, fast=50, slow=200, vol_window=200):
        self.name = name
        self.ewma = EwMean(span=fast)          # fast EWMA of price
        self.sma = RollingMean(slow)           # slower simple moving average
        self.logret = LogReturn(1)             # trade-to-trade log return
        self.vol = RollingStd(vol_window)      # realized vol from tick returns
        self.zscore = RollingZscore(slow)      # how far price sits from its mean
        self.hi = RollingMax(slow)             # recent high
        self.lo = RollingMin(slow)             # recent low
        self.count = 0

    def update(self, price):
        """Feed one trade price and return the current indicator values."""
        self.count += 1
        ret = self.logret(price)               # NaN on the first trade
        return {
            "price": price,
            "ewma": self.ewma(price),
            "sma": self.sma(price),
            "vol": self.vol(ret),              # std of recent tick returns
            "z": self.zscore(price),
            "hi": self.hi(price),
            "lo": self.lo(price),
        }


def _fmt(x, width=10, prec=2):
    """Format a float for the status table, showing blank for NaN."""
    return " " * (width - 3) + "nan" if x != x else f"{x:{width}.{prec}f}"


def print_row(name, v):
    print(
        f"  {name:<14}"
        f" px {_fmt(v['price'])}"
        f"  ewma {_fmt(v['ewma'])}"
        f"  sma {_fmt(v['sma'])}"
        f"  z {_fmt(v['z'], 7, 2)}"
        f"  ret_std {_fmt(v['vol'], 9, 6)}"
        f"  range [{_fmt(v['lo'], 9)},{_fmt(v['hi'], 9)}]"
    )


# --- source 1: replay the sample CSVs -------------------------------------

def _find_csv(symbol):
    """Locate the sample CSV for a symbol, e.g. ETH-PERPETUAL -> eth-perpetual."""
    tag = symbol.lower()
    matches = glob.glob(os.path.join(DATA_DIR, f"deribit.trades.{tag}.*.csv"))
    return matches[0] if matches else None


def replay_feed(symbols):
    """Yield (symbol, price) trades from the CSVs, time-ordered across symbols."""
    streams = []
    for symbol in symbols:
        path = _find_csv(symbol)
        if not path:
            print(f"  (no sample CSV for {symbol}, skipping)")
            continue

        def rows(path=path, symbol=symbol):
            with open(path) as fh:
                for r in csv.DictReader(fh):
                    yield int(r["timestamp"]), symbol, float(r["price"])

        streams.append(rows())
    # heapq.merge interleaves the per-symbol streams by timestamp
    for _ts, symbol, price in heapq.merge(*streams, key=lambda t: t[0]):
        yield symbol, price


# --- source 2: live Deribit WebSocket -------------------------------------

def live_feed(symbols):
    """Yield (symbol, price) trades from the Deribit public WebSocket."""
    try:
        import asyncio
        import json
        import websockets
    except ImportError:
        raise SystemExit("Live mode needs the 'websockets' package: pip install websockets")

    channels = [f"trades.{s}.raw" for s in symbols]
    subscribe = {"jsonrpc": "2.0", "id": 1, "method": "public/subscribe",
                 "params": {"channels": channels}}

    async def stream():
        async with websockets.connect(WS_URL) as ws:
            await ws.send(json.dumps(subscribe))
            print(f"  subscribed to {', '.join(channels)}")
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("method") != "subscription":
                    continue
                for trade in msg["params"]["data"]:
                    yield trade["instrument_name"], float(trade["price"])

    # bridge the async generator into a plain iterator
    loop = asyncio.new_event_loop()
    agen = stream().__aiter__()
    try:
        while True:
            yield loop.run_until_complete(agen.__anext__())
    except StopAsyncIteration:
        return
    finally:
        loop.close()


# --- driver ---------------------------------------------------------------

def run(symbols, live=False, limit=None, every=2000):
    books = {s: SymbolIndicators(s) for s in symbols}
    latest = {}
    feed = live_feed(symbols) if live else replay_feed(symbols)

    mode = "live Deribit feed" if live else "replaying sample trades"
    print(f"screamer streaming indicators, {mode}")
    print(f"symbols: {', '.join(symbols)}\n")

    processed = 0
    for symbol, price in feed:
        if symbol not in books:
            continue
        latest[symbol] = books[symbol].update(price)
        processed += 1

        # print a snapshot every `every` trades (or every trade when live)
        if (live and symbol == symbols[0]) or (not live and processed % every == 0):
            print(f"[{processed} trades]")
            for s in symbols:
                if s in latest:
                    print_row(s, latest[s])
            print()

        if limit and processed >= limit:
            break

    print(f"done, {processed} trades processed")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--live", action="store_true", help="stream live from Deribit (needs websockets)")
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS, help="instrument names")
    p.add_argument("--limit", type=int, default=None, help="stop after N trades (replay)")
    p.add_argument("--every", type=int, default=2000, help="print a snapshot every N trades (replay)")
    args = p.parse_args()
    run(args.symbols, live=args.live, limit=args.limit, every=args.every)


if __name__ == "__main__":
    main()
