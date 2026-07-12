"""Download recent Deribit perpetual trades to CSV, for backtesting.

Fetches the public trade history for one or more instruments over the last N
days and writes one CSV per instrument with columns `timestamp,volume,price`
(volume is signed: positive for buyer-initiated trades, negative for seller).
The format matches the sample files in devtools/data/, so the other examples
read either interchangeably.

Run:
    python examples/deribit_download.py --days 3
    python examples/deribit_download.py --days 7 --out-dir data --instruments ETH-PERPETUAL BTC-PERPETUAL
"""
import argparse
import csv
import os
import time

import requests

HISTORY = "https://history.deribit.com/api/v2/public/get_last_trades_by_instrument_and_time"
TIME_URL = "https://www.deribit.com/api/v2/public/get_time"


def _get_trades(params, tries=6):
    for i in range(tries):
        try:
            payload = requests.get(HISTORY, params=params, timeout=25).json()
            if "result" in payload:
                return payload["result"]["trades"]
            reason = str(payload.get("error", payload))[:100]
            print(f"   api error: {reason}; retry {i + 1}")
        except Exception as exc:
            print(f"   network error: {type(exc).__name__}; retry {i + 1}")
        time.sleep(1.5 * (i + 1))
    raise SystemExit("giving up after repeated errors")


def download(instrument, start_ms, end_ms, out_path):
    """Page through the trade history and write it to out_path."""
    total = 0
    start = start_ms
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp", "volume", "price"])
        while start < end_ms:
            trades = _get_trades(dict(instrument_name=instrument, start_timestamp=start,
                                      end_timestamp=end_ms, count=10000, sorting="asc"))
            if not trades:
                break
            for t in trades:
                volume = t["amount"] if t["direction"] == "buy" else -t["amount"]
                writer.writerow([t["timestamp"], volume, t["price"]])
            total += len(trades)
            last = trades[-1]["timestamp"]
            print(f"  {instrument}: {total} trades", end="\r")
            if last <= start or len(trades) < 10000:
                break
            start = last + 1
            time.sleep(0.15)
    print(f"  {instrument}: {total} trades -> {out_path}")
    return total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", type=int, default=3, help="how many days back to fetch")
    p.add_argument("--out-dir", default=os.path.dirname(__file__), help="where to write the CSVs")
    p.add_argument("--instruments", nargs="+", default=["ETH-PERPETUAL", "BTC-PERPETUAL"])
    args = p.parse_args()

    end = requests.get(TIME_URL, timeout=10).json()["result"]
    start = end - args.days * 24 * 3600 * 1000
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"downloading {args.days} days of trades")
    for instrument in args.instruments:
        out = os.path.join(args.out_dir, f"{instrument.lower()}.csv")
        download(instrument, start, end, out)


if __name__ == "__main__":
    main()
