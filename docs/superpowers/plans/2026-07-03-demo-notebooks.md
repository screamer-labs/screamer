# Demo notebooks (Phase 1: local) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Author 10 small, self-contained `.ipynb` notebooks under `docs/notebooks/` that showcase screamer's capabilities, plus a one-command local execute-and-verify workflow (`make notebooks`).

**Architecture:** Each notebook is authored via `jupytext` from a percent-format cell layout, committed as `.ipynb` (the artifact the user opens in Jupyter and that phase-2 myst-nb will render). All notebooks are self-contained (seeded synthetic data, no network) and use only the public `screamer` API. `nbmake` executes every notebook as a pytest run; the load-bearing equalities (batch==stream, pandas cross-checks) are asserted in-cell so a regression fails the run.

**Tech Stack:** Python, Jupyter (`.ipynb`), `nbmake`, `jupytext`, numpy, pandas, matplotlib (all deps present except `nbmake`/`jupytext`).

## Global Constraints

- **`.ipynb` committed under `docs/notebooks/`**, named `NN-slug.ipynb` (numbered for order). Author via `jupytext --to ipynb` from a percent-`.py`; commit the `.ipynb`.
- **Self-contained & reproducible:** each notebook generates its own **seeded** synthetic data (`np.random.default_rng(SEED)`); no network, no external files; deterministic outputs.
- **Public API only:** import from `screamer` / `screamer.streams` (or top-level `from screamer import ...`); NEVER `screamer_bindings` or any `_underscore` name.
- **Acceptance gate:** `poetry run pytest --nbmake docs/notebooks/` executes every notebook top-to-bottom and passes (no cell errors). In-cell `assert`/`np.testing.assert_array_equal` guard the key equalities.
- **Structure per notebook:** a title + one-paragraph intro (markdown), 2–4 short sections (markdown + code, small captioned plots), a closing "Takeaways" markdown cell.
- Confirmed signatures: `RollingMean/RollingStd/RollingZscore(window, start_policy=...)`, `EwMean/EwStd(span=...)`, `ATR(period)` (3-in: high,low,close), `RollingRSI(period)`, `MACD()` (1→3), `BollingerBands(window)` (1→3: lower,mid,upper), `Stoch(period)` (3-in→2-out), `Butter(order, cutoff_freq)`, `HullMA/KAMA/Detrend(window)`, `FillNa(value)`, `Ffill()`, `RollingCorr(window)` (2-in), `Sub()` (2-in). Streams: `combine_latest(*series, emit=..., func=...)`, `merge(*series)->(keys,values,sources)`, `pace(*series, speed=..., sleep=...)` (async), `dropna(keys,values,how=...)`, `filter(keys,values,predicate)`, `split(keys,values,sources,n=...)`. DAG: `Input(name)`, `Dag(inputs=[...], outputs=[...], align_outputs=True)`, `dag(*feeds)`, `dag.stream(*feeds)`. A "stream" arg is `(keys_int64, values_float64)`.
- Never hand-edit `screamer/__init__.py` or version files.

---

## File Structure

- `pyproject.toml` (modify) — add `nbmake`, `jupytext` to the dev dependency group.
- `Makefile` (modify) — add a `notebooks` target (`install-dev` + nbmake run).
- `docs/notebooks/README.md` (create) — how to run/edit.
- `docs/notebooks/01-…​.ipynb` … `10-…​.ipynb` (create) — the notebooks.

Authoring method (all notebooks): write the cells as a percent-format `.py`
(`# %% [markdown]` for prose, `# %%` for code), then
`poetry run jupytext --to ipynb docs/notebooks/NN-slug.py` → commit the `.ipynb`
(delete or keep the `.py`; the `.ipynb` is the committed source of truth).

---

### Task 1: tooling + notebook 1 (quickstart) — establish the pattern

**Files:**
- Modify: `pyproject.toml`, `Makefile`
- Create: `docs/notebooks/README.md`, `docs/notebooks/01-quickstart-polymorphic-api.ipynb`

**Interfaces:**
- Produces: `make notebooks` (build + `pytest --nbmake docs/notebooks/`); the notebook conventions (seeded data, in-cell asserts, takeaways) that Tasks 2–3 follow.

- [ ] **Step 1: Add the dev deps**

In `pyproject.toml`, add to the dev/test dependency group (next to pytest) `nbmake` and `jupytext`. Then:

```bash
poetry lock && poetry install
```

- [ ] **Step 2: Add the `make notebooks` target**

In `Makefile`, add:

```makefile
notebooks: install-dev
	$(PYTEST) --nbmake docs/notebooks/
```

(uses the existing `$(PYTEST)` and `install-dev` targets, so `make notebooks` builds the lib+bindings, installs, then executes every notebook.)

- [ ] **Step 3: Write the quickstart notebook (percent-format source)**

Create `docs/notebooks/01-quickstart-polymorphic-api.py` with these cells, then convert to `.ipynb` (Step 4):

```python
# %% [markdown]
# # Quickstart: one callable, every input shape
#
# screamer's defining feature: **the same configured function works on a scalar,
# a NumPy array, a 2-D batch, a Python list, or a live iterator — and the numbers
# are identical across all of them.** That's what lets you backtest on arrays and
# run live on a stream with the same code and no train/serve skew.

# %%
import numpy as np
from screamer import RollingMean

rng = np.random.default_rng(0)
x = rng.standard_normal(20)

# %% [markdown]
# ## Scalars, arrays, and 2-D batches
# `axis=0` is always time; extra axes are independent parallel series.

# %%
ma = RollingMean(5)
print("scalar :", ma(1.5))                 # -> a float
print("array  :", RollingMean(5)(x)[:6])   # -> a (20,) array
batch = np.random.default_rng(1).standard_normal((20, 3))
print("2-D    :", RollingMean(5)(batch).shape)   # (20, 3): 3 independent columns

# %% [markdown]
# ## Lists and live iterators
# A list is processed eagerly; an iterator is processed lazily, one value at a time
# (this is the live-event path).

# %%
print("list   :", RollingMean(5)([1.0, 2.0, 3.0, 4.0, 5.0]))

def live_source(vals):
    for v in vals:            # imagine this yielding from a socket / clock
        yield v

streamed = list(RollingMean(5)(live_source(x)))

# %% [markdown]
# ## Batch == streaming (the key guarantee)

# %%
batch_result = RollingMean(5)(x)
np.testing.assert_array_equal(batch_result, np.array(streamed))
print("batch == streamed:", np.allclose(batch_result, streamed, equal_nan=True))

# %% [markdown]
# **Takeaways**
# - One configured functor handles scalar / array / 2-D / list / iterator.
# - `axis=0` is time; higher axes are independent series (no `axis=` argument).
# - The array (backtest) and iterator (live) paths give byte-identical results.
```

- [ ] **Step 4: Convert to `.ipynb` and add the README**

Run: `poetry run jupytext --to ipynb docs/notebooks/01-quickstart-polymorphic-api.py`
(then remove the `.py`: `rm docs/notebooks/01-quickstart-polymorphic-api.py`).

Create `docs/notebooks/README.md`:

```markdown
# Demo notebooks

Runnable notebooks showcasing screamer. Each is self-contained (seeded synthetic
data, no downloads) and uses only the public API.

## Run / verify

```bash
make notebooks     # builds the lib + bindings, then executes every notebook
# or, against an already-built install:
poetry run pytest --nbmake docs/notebooks/
```

Open any `.ipynb` in Jupyter to read/edit. A notebook that computes an equality
(e.g. batch == stream) asserts it in-cell, so a regression fails the run above.
```

- [ ] **Step 5: Run the acceptance gate**

Run: `make notebooks`
Expected: `01-quickstart-polymorphic-api.ipynb` executes clean (1 notebook passed).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml poetry.lock Makefile docs/notebooks/README.md docs/notebooks/01-quickstart-polymorphic-api.ipynb
git commit -m "docs(notebooks): tooling (nbmake) + quickstart notebook"
```

---

### Task 2: single-series notebooks (2–6)

**Files:**
- Create: `docs/notebooks/02-rolling-and-ew-statistics.ipynb`, `03-streaming-live-events.ipynb`, `04-financial-indicators.ipynb`, `05-nan-handling.ipynb`, `06-signal-processing.ipynb`

**Interfaces:**
- Consumes: the Task-1 conventions (seeded data, in-cell asserts, takeaways, jupytext→ipynb authoring).

Author each as a percent-`.py`, convert to `.ipynb`, and confirm each executes under nbmake before moving on. The **required code cells** (the load-bearing content — wrap with concise markdown intros/sections/takeaways per the conventions):

- [ ] **Step 1: `02-rolling-and-ew-statistics.ipynb`**

```python
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from screamer import RollingMean, RollingStd, RollingZscore, EwMean, EwStd
rng = np.random.default_rng(0)
price = 100 + np.cumsum(rng.standard_normal(300))

ma20 = RollingMean(20)(price)
z = RollingZscore(50)(price)

# pandas cross-check (equality proves screamer matches the reference)
pd_ma = pd.Series(price).rolling(20).mean().to_numpy()
np.testing.assert_allclose(ma20, pd_ma, equal_nan=True)

# start_policy variants: strict emits NaN during warmup; expanding fills early
strict = RollingMean(20, "strict")(price)
expanding = RollingMean(20, "expanding")(price)
assert np.isnan(strict[0]) and not np.isnan(expanding[0])

# EW family
ew = EwMean(span=20)(price)

fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
ax[0].plot(price, lw=.6, label="price"); ax[0].plot(ma20, label="RollingMean(20)")
ax[0].plot(ew, label="EwMean(20)"); ax[0].legend(loc="upper left")
ax[1].plot(z, label="RollingZscore(50)"); ax[1].axhline(0, color="k", lw=.5); ax[1].legend()
plt.tight_layout()
```
Sections: rolling vs EW smoothing; `start_policy` (strict/expanding/zero); pandas cross-check. Takeaways: the family is broad, windows/policies are constructor args, results match pandas.

- [ ] **Step 2: `03-streaming-live-events.ipynb`**

```python
import numpy as np
from screamer import RollingZscore
rng = np.random.default_rng(1)
x = rng.standard_normal(500)

# BATCH: whole array at once (the backtest path)
batch = RollingZscore(50)(x)

# LIVE: feed one event at a time (the production path) — same functor class
z = RollingZscore(50)
live = []
for v in x:                        # imagine v arriving from a socket / queue
    live.append(z(v))              # scalar in, scalar out; state persists

np.testing.assert_array_equal(batch, np.array(live))
print("backtest == live:", np.allclose(batch, live, equal_nan=True))
```
Sections: array path vs scalar-loop path; the "define once, run backtest and live" story; back-pressure (results come out as you pull). Takeaways: no train/serve skew; the scalar call is O(1) per event.

- [ ] **Step 3: `04-financial-indicators.ipynb`**

```python
import numpy as np, matplotlib.pyplot as plt
from screamer import ATR, RollingRSI, MACD, BollingerBands
rng = np.random.default_rng(2)
close = 100 + np.cumsum(rng.standard_normal(400))
high = close + np.abs(rng.standard_normal(400))
low  = close - np.abs(rng.standard_normal(400))

atr = ATR(14)(high, low, close)             # 3-input functor
rsi = RollingRSI(14)(close)
bb  = BollingerBands(20)(close)             # (400, 3): lower, mid, upper
macd = MACD()(close)                        # (400, 3): macd, signal, hist

# the (T, N) form: pass one (T, 3) array instead of three args
hlc = np.column_stack([high, low, close])
np.testing.assert_array_equal(atr, ATR(14)(hlc))

fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
ax[0].plot(close, lw=.7); ax[0].plot(bb[:, 0], "r--", lw=.5); ax[0].plot(bb[:, 2], "r--", lw=.5); ax[0].set_title("price + Bollinger")
ax[1].plot(rsi); ax[1].axhline(70, color="r", lw=.4); ax[1].axhline(30, color="g", lw=.4); ax[1].set_title("RSI(14)")
ax[2].plot(atr); ax[2].set_title("ATR(14)"); plt.tight_layout()
```
Sections: single-input indicators (RSI, MACD, Bollinger); multi-input OHLC (ATR, Stoch) + the `(T, N)` array form; a price panel with overlays. Takeaways: multi-output→trailing axis, multi-input→N args or one `(T,N)` array.

- [ ] **Step 4: `05-nan-handling.ipynb`**

```python
import numpy as np
from screamer import RollingMean, FillNa, Ffill
from screamer import dropna
x = np.arange(10.0); x[3] = np.nan; x[7] = np.nan     # gaps

ignore = RollingMean(3)(x)          # "ignore": NaN skipped from state, emitted at its index
filled = FillNa(0.0)(x)             # replace NaN with 0
ff     = Ffill()(x)                 # carry last valid value forward

# stream-level dropna removes the events entirely (cardinality changes)
keys = np.arange(x.size, dtype=np.int64)
dk, dv = dropna(keys, x)
assert dv.size == x.size - 2 and not np.isnan(dv).any()

print("ignore :", np.round(ignore, 2))
print("ffill  :", ff)
print("dropped:", dv)
```
Sections: the three policies (ignore/propagate/nan-aware) with a one-line description of each and a demo; `FillNa`/`Ffill` (shape-preserving) vs `dropna` (cardinality-changing); recovery after a gap. Takeaways: fill preserves the grid, drop compresses it; `bfill` doesn't exist (causal).

- [ ] **Step 5: `06-signal-processing.ipynb`**

```python
import numpy as np, matplotlib.pyplot as plt
from screamer import Butter, HullMA, Detrend
rng = np.random.default_rng(3)
t = np.linspace(0, 4 * np.pi, 500)
clean = np.sin(t)
noisy = clean + 0.4 * rng.standard_normal(t.size)

lp   = Butter(4, 0.05)(noisy)     # 4th-order low-pass, cutoff 0.05
hull = HullMA(20)(noisy)
detr = Detrend(50)(100 + t + noisy)   # remove a rolling trend

fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
ax[0].plot(noisy, lw=.4, alpha=.6, label="noisy"); ax[0].plot(lp, label="Butter(4, 0.05)")
ax[0].plot(hull, label="HullMA(20)"); ax[0].legend(loc="upper right")
ax[1].plot(detr, label="Detrend(50)"); ax[1].axhline(0, color="k", lw=.4); ax[1].legend()
plt.tight_layout()
```
Sections: low-pass filtering (`Butter`); smoothing (`HullMA`/`KAMA`); detrending. A note that all filters are **causal** (no lookahead — same result batch or live). Takeaways: filters are functors too; they compose with everything else.

- [ ] **Step 6: Convert all five, run, commit**

Convert each `.py` → `.ipynb` (`poetry run jupytext --to ipynb docs/notebooks/0*.py`), remove the `.py` sources, then:

Run: `make notebooks`
Expected: notebooks 1–6 pass.

```bash
git add docs/notebooks/02-*.ipynb docs/notebooks/03-*.ipynb docs/notebooks/04-*.ipynb docs/notebooks/05-*.ipynb docs/notebooks/06-*.ipynb
git commit -m "docs(notebooks): rolling/EW, streaming, financial, NaN, signal"
```

---

### Task 3: multi-stream + DAG notebooks (7–10)

**Files:**
- Create: `docs/notebooks/07-aligning-async-streams.ipynb`, `08-replay-backtest-live.ipynb`, `09-stream-shaping.ipynb`, `10-computational-dag.ipynb`

**Interfaces:**
- Consumes: the Task-1 conventions; the public `screamer.streams` + `Node`/`Input`/`Dag` API.

- [ ] **Step 1: `07-aligning-async-streams.ipynb`**

```python
import numpy as np, matplotlib.pyplot as plt
from screamer import combine_latest, merge
# two async feeds with DIFFERENT timestamps (keys)
a_k = np.array([1, 3, 5, 7, 9], dtype=np.int64);  a_v = np.array([10., 11., 12., 11., 13.])
b_k = np.array([2, 4, 6, 8], dtype=np.int64);      b_v = np.array([5., 6., 5.5, 6.5])

# combine_latest: emit whenever EITHER ticks, carrying each input's last value
keys, aligned = combine_latest((a_k, a_v), (b_k, b_v))     # emit="when_all" default
spread = aligned[:, 0] - aligned[:, 1]

# merge: interleave into one key-sorted, source-tagged stream
mk, mv, ms = merge((a_k, a_v), (b_k, b_v))
assert (np.diff(mk) >= 0).all()      # globally key-sorted

# on_any emits from the first event (NaN for the not-yet-seen input)
k2, a2 = combine_latest((a_k, a_v), (b_k, b_v), emit="on_any")

plt.figure(figsize=(8, 3))
plt.step(a_k, a_v, where="post", label="feed a"); plt.step(b_k, b_v, where="post", label="feed b")
plt.step(keys, spread, where="post", label="spread (a-b, aligned)"); plt.legend(); plt.tight_layout()
```
Sections: the order-key model (explicit timestamps vs row-number); `combine_latest` as-of join (spread-on-any-tick); `emit="when_all"` vs `"on_any"`; `merge` for one sorted tagged stream. Takeaways: alignment is a separate layer; feed the aligned columns into any functor (`RollingCorr(combine_latest(a,b))`).

- [ ] **Step 2: `08-replay-backtest-live.ipynb`**

```python
import asyncio, numpy as np
from screamer import pace, merge
a = (np.array([0, 10, 30], dtype=np.int64), np.array([1., 2., 3.]))
b = (np.array([5, 20], dtype=np.int64), np.array([.5, 2.5]))

async def drain(agen):
    out = []
    async for e in agen:
        out.append(e)
    return out

# backtest: max speed, order-preserving (speed=inf -> no sleeping)
backtest = asyncio.run(drain(pace(a, b, speed=float("inf"))))

# wall-clock replay: sleeps proportional to key-deltas / speed.
# inject a fake clock so the demo is deterministic and instant.
slept = []
async def fake_sleep(sec): slept.append(sec)
replay = asyncio.run(drain(pace(a, b, speed=2.0, sleep=fake_sleep)))

# identical events; only the TIMING differs
assert [e[:2] for e in backtest] == [e[:2] for e in replay]
print("events:", [(k, v) for k, v, _ in backtest])
print("sleeps (key-deltas / 2):", slept)
```
Sections: turning stored series into an event stream; backtest (`speed=inf`) vs wall-clock replay (`speed=`, injectable clock); values identical, only timing differs. Takeaways: `pace = merge + optional pacing`; the metric-key requirement for wall-clock; this is how "backtest then go live" stays honest.

- [ ] **Step 3: `09-stream-shaping.ipynb`**

```python
import numpy as np
from screamer import dropna, filter as keep, split, merge
keys = np.array([1, 2, 3, 4, 5], dtype=np.int64)
vals = np.array([1.0, np.nan, 3.0, -4.0, 5.0])

# dropna: remove bad readings (cardinality shrinks)
ck, cv = dropna(keys, vals)
assert not np.isnan(cv).any()

# filter: keep only events matching a predicate
pk, pv = keep(keys, vals, lambda v: (v > 0) if not np.isnan(v) else False)

# split: inverse of merge — route a merged tagged stream back per source
a = (np.array([1, 3], dtype=np.int64), np.array([10., 30.]))
b = (np.array([2, 4], dtype=np.int64), np.array([20., 40.]))
mk, mv, ms = merge(a, b)
parts = split(mk, mv, ms)
np.testing.assert_array_equal(parts[0][1], a[1])   # reconstructs feed a
print("clean:", cv, "| positive:", pv)
```
Sections: `dropna` (`how="any"/"all"`); `filter` with a predicate; `split` (inverse of `merge`). Takeaways: these change cardinality (clean before ingest); `dropna(combine_latest(...))` is a common idiom.

- [ ] **Step 4: `10-computational-dag.ipynb`**

```python
import numpy as np
from screamer import Input, Dag, RollingMean, Sub, combine_latest

# define the computation ONCE as a graph
a, b = Input("a"), Input("b")
spread = Sub()(combine_latest(a, b))     # align two async feeds, take the difference
signal = RollingMean(10)(spread)         # smooth it
dag = Dag(inputs=[a, b], outputs=[signal])

# feed two async series
fa = (np.array([1, 3, 5, 7, 9, 11, 13], dtype=np.int64), np.arange(7.0))
fb = (np.array([2, 4, 6, 8, 10, 12], dtype=np.int64), np.arange(6.0) * 2)

batch  = dag(fa, fb)          # run over arrays (backtest)
live   = dag.stream(fa, fb)   # run event-by-event (live)

bk, bv = batch; sk, sv = live
np.testing.assert_array_equal(bk, sk)
np.testing.assert_array_equal(bv, sv)
print("batch == live:", np.array_equal(bv, sv, equal_nan=True))
```
Sections: define a graph with `Input`/`combine_latest`/functors; a `Dag` is a plain N→M callable; run `dag(...)` (batch) and `dag.stream(...)` (live) and assert identical; note graph ops are C++ (use `Sub()` not a Python lambda). Takeaways: define once, run batch or live through one C++ engine, byte-identical — the whole library's thesis in one cell.

- [ ] **Step 5: Convert all four, run the full gate, commit**

Convert each `.py` → `.ipynb`, remove `.py`, then:

Run: `make notebooks`
Expected: all 10 notebooks pass.

Also confirm the public-API-only rule:
Run: `grep -rl "screamer_bindings\|import _\| _[a-z]" docs/notebooks/*.ipynb || echo "clean: no private imports"`
Expected: `clean: no private imports`.

```bash
git add docs/notebooks/07-*.ipynb docs/notebooks/08-*.ipynb docs/notebooks/09-*.ipynb docs/notebooks/10-*.ipynb
git commit -m "docs(notebooks): align, replay, shaping, DAG"
```

---

## Self-Review

**1. Spec coverage:**
- 10 notebooks, the exact topics/slugs from the spec → Tasks 1–3. ✓
- Self-contained + seeded + public-API-only → every notebook uses `default_rng(SEED)`, top-level imports; grep guard in Task 3. ✓
- `nbmake` acceptance gate + `make notebooks` target + README → Task 1. ✓
- In-cell asserts guard the key equalities (batch==stream, pandas cross-check, split inverts merge) → Tasks 1–3. ✓
- Plots via matplotlib; pandas cross-check → Tasks 2/4. ✓
- Phase-2 (RTD/myst-nb/release) correctly ABSENT. ✓

**2. Placeholder scan:** none — every notebook's load-bearing code cells are given in full; prose cells are directed per the fixed "intro → sections → takeaways" convention established concretely in Task 1's notebook. No "TBD".

**3. Type consistency:** the confirmed signatures in Global Constraints are used consistently in all code cells (`ATR(14)(high,low,close)` and the `(T,3)` form; `combine_latest((k,v),(k,v))`; `Dag(inputs=[...],outputs=[...])`; `dag(...)`/`dag.stream(...)`). Streams are `(keys_int64, values_float64)` throughout.

---

## Follow-on (Phase 2, separate effort)

- Add `myst-nb` to `docs/conf.py` extensions; add the notebooks to `docs/index.rst` toctree (a "Tutorials" section); add `.readthedocs.yaml`.
- Version release (`make minor`/`major`, user-approved), PyPI, RTD hosting, docs CI.
