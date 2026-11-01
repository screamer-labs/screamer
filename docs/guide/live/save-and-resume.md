---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Save a strategy and resume it later (save, load, state)

`save` writes a function's settings and state to a file, `load` reads it
back, and `resume` runs the next data from that state, so a job can stop at
the end of a day and continue the next day where it stopped.

```{code-cell} ipython3
import json, pathlib, tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import EwMean, Diff, Sign, BacktestPortfolioOHLCTarget, load, loads, from_settings

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv", parse_dates=["date"])
wide = df.pivot(index="date", columns="symbol")       # (T, K) per field
o, h, l, c = (wide[f].to_numpy() for f in ["open", "high", "low", "close"])

def run(book, o, h, l, c):                            # continues the book's state
    direction = Sign()(book["fast"].resume(c) - book["slow"].resume(c))
    weights = direction / c.shape[1]                  # 1/20 of equity each
    weights[book["turn"].resume(direction) == 0] = np.nan    # trade on a change only
    return book["engine"].resume(weights, o, h, l, c)

folder = pathlib.Path(tempfile.mkdtemp())
book = {"fast": EwMean(span=20), "slow": EwMean(span=100), "turn": Diff(),
        "engine": BacktestPortfolioOHLCTarget(fee=1e-3, slippage=5e-4)}
start, outs = len(c) - 30, []
for a, b in [(0, start)] + [(t, t + 1) for t in range(start, len(c))]:
    outs.append(run(book, o[a:b], h[a:b], l[a:b], c[a:b]))   # the first run, then a day
    for name, op in book.items():
        op.save(folder / f"{name}.json")
    book = {name: load(folder / f"{name}.json") for name in book}
```

```{code-cell} ipython3
profit = np.concatenate(outs)[:, :, 0].sum(axis=1)   # the book: equity summed over coins
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(wide.index, profit, color="steelblue")
ax.axvline(wide.index[start], color="0.5", lw=0.8, ls="--")
ax.set_ylabel("book profit (starting capital 1)")
plt.tight_layout()
```

The book is the 20/100-day trend rule of [Report a portfolio backtest as one
book](../backtest/portfolio-report.md) on 20 Binance coins: two `EwMean`s give
each coin's direction, `Diff` marks the days it changes, and
`BacktestPortfolioOHLCTarget` trades the changes from one account at the next
day's open, paying a 0.1% fee and 0.05% slippage. The first run trades the
history up to the dashed line, 30 days before the end of the data. Each daily
job after it loads the four files the previous job saved, trades one day, and
saves them again.

`resume` is the batch call that continues. A default call such as
`EwMean(span=20)(c)` starts from a fresh state and leaves the object fresh;
`resume(c)` starts from the object's state, which is the state of the file
after `load`, and keeps the end state in the object. On a fresh object it
warms the object up on history, ready to save or to feed one value at a time.
Called on a `(T, K)` array, `resume` keeps one state per column, so `fast`,
`slow` and `turn` hold 20 states each, and a call with a single event raises
until `reset()` drops them. The engine is one object for the `K` coins: its
`resume` continues the account, open positions included, and raises on rows
of another width.

A file holds the settings, the constructor arguments under their documented
names, and the state, what the object has learned from the data: for `slow`,
the two running sums of each coin. `save` writes JSON, or CBOR (a binary
encoding of the same content) for a path that ends in `.cbor`; `dumps` and
`loads` do the same with a string or bytes in memory. A `Pipeline` saves its
description, and a live session from `Pipeline.live()` saves its description
and its state, including the events that wait in a join, so `load` gives back
a session that continues.

## Load the file in JavaScript

The JavaScript package reads and writes the same files. In Node.js, the next
day's job can load the engine that Python saved:

```js
import { ready, load } from "@screamer-labs/screamer";
await ready();

const engine = await load("book/engine.json");
const out = engine.resume(weights, open, high, low, close);   // NdArray [T, K] each
await engine.save("book/engine.json");
```

In a browser, `loads(text)` and `engine.dumps()` take and return the file's
text, since a page has no file system. The settings of a file have the same
bytes in both languages.

## Edit a setting by hand

A file is meant to be read and edited, and an edited file goes through the
constructor's checks and the state's checks. An edit that the state does not
depend on, such as the engine's `fee`, loads, and the engine continues with
the new fee. An edit the state depends on raises `StateMismatch`, which names
the field: `Diff` holds the last value of each coin, and a window of 2 needs
two. With `on_mismatch="fresh"`, the load keeps the edited settings, starts
the object fresh and warns:

```{code-cell} ipython3
saved = json.loads((folder / "turn.json").read_text(encoding="utf-8"))
saved["object"]["settings"]["window_size"] = 2
turn = loads(json.dumps(saved), on_mismatch="fresh")
```

A fresh object needs a warm-up with `resume` on history before it trades
again.

## After an upgrade refuses a state

Settings load in every later release, and a renamed parameter is mapped to
its new name. A state loads when its state version, stored in the file, is the
one the library reads. A function's state version changes only when that
function's internals change, so after most upgrades the files load as they
are. A file with another state version raises `StateVersionError`, such as
`EwMean: state version 2 in the file, this library reads version 1`. To
recover, build the object from the file's settings with `from_settings`, which
ignores the state, and warm it up on the stored history:

```{code-cell} ipython3
fast = from_settings(json.loads((folder / "fast.json").read_text(encoding="utf-8"))["object"])
fast.resume(c)                                        # rebuilds the state from history
fast.save(folder / "fast.json")
```

`load(path, on_mismatch="fresh")` does the first step for every object in a
file, with a warning for each one that starts fresh. An engine is warmed up
the same way, by trading the history again.

## Go deeper

- [Save and restore](../../save_restore.md): the file layout, every call, the errors and the compatibility rule.
- [Process values one at a time](one-value-at-a-time.md): call a loaded function once per event.
- [Pipelines](../../pipelines.md): live sessions, which save their waiting events with their state.
- [Report a portfolio backtest as one book](../backtest/portfolio-report.md): the book's summary statistics.
