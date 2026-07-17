# Backtest Fill Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the fill models of the committed `BacktestTrades` and `BacktestL1` engines and add a new `BacktestL1Trades` engine, per the settled fill-model spec.

**Architecture:** Each engine is a header-only `FunctorBase<Derived, N, 4>` subclass emitting `[equity, pnl, position, cost]` through the shared `detail::PnLAccount`. Fills follow one rule set: a trade/quote *through* the resting price fills the full remaining; a trade/quote *at* the price fills `min(remaining, participation_ratio * available_size)`; markets/marketable orders fill fully with `tick_size` slippage. Quotes-only `BacktestL1` uses a lock-episode state machine to avoid double-counting; `BacktestL1Trades` drives fills from trade events.

**Tech Stack:** C++17 header-only core (`include/screamer/`), pybind11 bindings (`bindings/bindings_fin.cpp`), Python packaging/regeneration (`make install-dev`, devtools regen scripts), pytest.

## Global Constraints

- All operator logic in C++; Python is thin bindings only. No numpy/pandas fill logic.
- Causal, no lookahead; batch and streaming give identical results; deterministic (no randomness).
- Orders are counterfactual: zero volume, no market impact; a fill may never exceed observed volume at the level.
- `participation_ratio ∈ (0, 1]`, default `1.0`; at-touch fill is `min(remaining, participation_ratio * available_size)` (scale available volume, cap by remaining).
- Through / swept fill = full remaining. Bars = full fill. Market/overflow = full fill + `tick_size` slippage.
- `fill = "breach"` (conservative) is the **default** for `BacktestL1`; `"touch"` (optimistic) is opt-in.
- Resting limit fills at its own limit price (never better); a quote submitted already crossing is a taker (`taker_fee`, market price + `tick_size`); a resting quote run through is a maker (`maker_fee`).
- Never edit version files. Commit as `simu.ai <claude@sitmo.com>` with the standard `Co-Authored-By` + `Claude-Session` footer. Do not push.
- No em-dashes in docs/prose (ASCII hyphens only).
- After any C++ change run `make install-dev` (not just `make build`) before importing from Python.

---

## File Structure

- `include/screamer/backtest_trades.h` (modify) - tape engine; through=full, at=participation.
- `include/screamer/backtest_ohlc.h` (audit) - confirm full-fill semantics; comment only.
- `include/screamer/backtest_l1.h` (rewrite) - quotes-only; lock-episode state machine, taker path, new params.
- `include/screamer/backtest_l1_trades.h` (create) - quotes+trades engine, 10 inputs.
- `bindings/bindings_fin.cpp` (modify) - new params on modified engines, register the new engine.
- `screamer/backtest.py` (unchanged) - `backtest_report` already generic over `[equity,pnl,position,cost]`.
- `tests/test_backtest.py` (modify) - corrected + new fill tests.
- `docs/functions_fin/BacktestTrades.md`, `BacktestL1.md` (modify), `BacktestL1Trades.md` (create) - reference pages + Limitations.
- `docs/notebooks/15-event-driven-backtests.ipynb` (modify) - reflect corrected fills; add L1Trades.
- `docs/index.rst` (modify) - no new notebook (15 already listed); no change unless a page is added to a toctree.

Regeneration after binding/doc changes: `make install-dev`, then
`poetry run python devtools/build_help_registry.py` (validates frontmatter),
then `poetry run python devtools/build_topic_pages.py`, then `make regen-init`.

---

## Task 1: Correct `BacktestTrades` fills + `participation_ratio`

**Files:**
- Modify: `include/screamer/backtest_trades.h`
- Modify: `bindings/bindings_fin.cpp:190-194`
- Test: `tests/test_backtest.py`

**Interfaces:**
- Produces: `BacktestTrades(maker_fee=0.0, fill="touch", participation_ratio=1.0)`, 4 inputs `(order_price, order_size, trade_price, trade_size)`, 4 outputs `[equity, pnl, position, cost]`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_backtest.py`)

```python
def test_trades_through_fills_full_order_not_print_size():
    from screamer import BacktestTrades
    # resting buy 10 @ 100; a print of size 2 at 99 trades THROUGH -> full 10 fills
    t = BacktestTrades()(np.array([100.]), np.array([10.]), np.array([99.]), np.array([2.]))
    assert t[0, 2] == 10.0                                  # swept: full order, not min(10,2)


def test_trades_at_fills_participation_of_trade_size():
    from screamer import BacktestTrades
    # resting buy 10 @ 100; a print of size 8 AT 100, participation 0.5 -> min(10, 0.5*8)=4
    t = BacktestTrades(participation_ratio=0.5)(
        np.array([100.]), np.array([10.]), np.array([100.]), np.array([8.]))
    assert t[0, 2] == 4.0


def test_trades_participation_capped_by_order_no_zeno():
    from screamer import BacktestTrades
    # participation*trade_size exceeds the order -> full fill, capped by remaining
    t = BacktestTrades(participation_ratio=0.5)(
        np.array([100.]), np.array([10.]), np.array([100.]), np.array([100.]))
    assert t[0, 2] == 10.0                                  # min(10, 0.5*100)=10, no dust


def test_trades_breach_ignores_at_price():
    from screamer import BacktestTrades
    # breach: a print exactly AT 100 does not fill; only strictly through does
    at = BacktestTrades(fill="breach")(np.array([100.]), np.array([10.]),
                                       np.array([100.]), np.array([8.]))
    assert at[0, 2] == 0.0
    through = BacktestTrades(fill="breach")(np.array([100.]), np.array([10.]),
                                            np.array([99.]), np.array([2.]))
    assert through[0, 2] == 10.0
```

- [ ] **Step 2: Run to verify failure**

Run: `poetry run python -m pytest tests/test_backtest.py -q -k "trades_through_fills_full or trades_at_fills_participation or participation_capped or breach_ignores"`
Expected: FAIL (`participation_ratio` is not a constructor arg yet; sizing differs).

- [ ] **Step 3: Rewrite the fill body** in `include/screamer/backtest_trades.h`

Add `#include <algorithm>` (already present). Change the constructor and `call`:

```cpp
BacktestTrades(double maker_fee = 0.0, const std::string& fill = "touch",
               double participation_ratio = 1.0)
    : maker_fee_(maker_fee), breach_(parse_fill(fill)),
      participation_ratio_(parse_participation(participation_ratio))
{}
```

Replace the fill computation inside `call` (the block that sets `fill_dpos`/`fill_price`):

```cpp
double fill_dpos = 0.0, fill_price = trade_price;
if (!isnan2(order_price) && !isnan2(order_size) && order_size != 0.0) {
    const bool buy = order_size > 0.0;
    const double remaining = std::abs(order_size);
    const bool through = buy ? (trade_price < order_price)
                             : (trade_price > order_price);
    const bool at = !breach_ && (trade_price == order_price);
    double filled = 0.0;
    if (through) {
        filled = remaining;                                     // swept: full remaining
    } else if (at) {
        filled = std::min(remaining, participation_ratio_ * trade_size);
    }
    if (filled > 0.0) {
        fill_dpos  = buy ? filled : -filled;
        fill_price = order_price;
    }
}
```

Add the private members and validator:

```cpp
static double parse_participation(double p) {
    if (!(p > 0.0) || p > 1.0)
        throw std::invalid_argument("participation_ratio must be in (0, 1].");
    return p;
}
double maker_fee_;
bool breach_;
double participation_ratio_;
detail::PnLAccount account_;
```

- [ ] **Step 4: Update the binding** in `bindings/bindings_fin.cpp`

```cpp
py::class_<screamer::BacktestTrades, screamer::EvalOp>(m, "BacktestTrades")
    .def(py::init<double, const std::string&, double>(),
         py::arg("maker_fee") = 0.0, py::arg("fill") = "touch",
         py::arg("participation_ratio") = 1.0)
    .def("__call__", &screamer::BacktestTrades::handle_input)
    .def("reset", &screamer::BacktestTrades::reset, "Reset.");
```

- [ ] **Step 5: Rebuild and run**

Run: `make install-dev && poetry run python -m pytest tests/test_backtest.py -q -k "trades"`
Expected: PASS (new tests + the existing `test_trades_*`). Update `test_trades_fill_and_adverse_selection` if its 1-lot example still holds (buy 1 @ 100, print 99 size 2: through -> full 1; unchanged) and `test_trades_partial_up_to_print_size` (buy 5 @ 100, print AT 100 size 2, default participation 1.0 -> min(5, 1.0*2)=2; unchanged). Verify both still pass; if `test_trades_partial_up_to_print_size` name no longer fits, keep it (behavior identical at participation 1.0).

- [ ] **Step 6: Commit**

```bash
git add include/screamer/backtest_trades.h bindings/bindings_fin.cpp tests/test_backtest.py
git commit -m "fix(backtest): BacktestTrades through=full, at=participation partial"
```

---

## Task 2: Audit `BacktestOHLC` full-fill semantics

**Files:**
- Modify: `include/screamer/backtest_ohlc.h` (comment only if code already correct)
- Test: `tests/test_backtest.py`

**Interfaces:**
- Consumes: nothing new. Confirms `BacktestOHLC` fills the full `dpos` on a triggered order (no volume, no participation).

- [ ] **Step 1: Write the confirming test**

```python
def test_ohlc_limit_fills_full_target_no_participation():
    from screamer import BacktestOHLC
    # a large target with a touched limit fills fully (bars carry no volume)
    out = BacktestOHLC()(np.array([1000.]), np.array([99.]), np.array([100.]),
                         np.array([101.]), np.array([98.]), np.array([100.]))
    assert out[0, 2] == 1000.0
```

- [ ] **Step 2: Run to verify it already passes**

Run: `poetry run python -m pytest tests/test_backtest.py -q -k "ohlc_limit_fills_full_target"`
Expected: PASS (the committed engine already fills full `dpos`). If it fails, the engine is capping fills; fix by ensuring `fill_dpos = dpos` on a hit (see `backtest_ohlc.h` limit branch).

- [ ] **Step 3: Add a clarifying comment** in `include/screamer/backtest_ohlc.h` above the limit branch:

```cpp
// Bars carry no per-level volume, so a triggered order fills its full target
// (no participation_ratio here). Intrabar path is unknown; see BacktestL1Trades
// for volume-aware fills.
```

- [ ] **Step 4: Commit**

```bash
git add include/screamer/backtest_ohlc.h tests/test_backtest.py
git commit -m "test(backtest): confirm BacktestOHLC full-fill semantics"
```

---

## Task 3: Rewrite `BacktestL1` (lock-episode, taker path, new params)

**Files:**
- Rewrite: `include/screamer/backtest_l1.h`
- Modify: `bindings/bindings_fin.cpp:196-202`
- Test: `tests/test_backtest.py`

**Interfaces:**
- Produces: `BacktestL1(maker_fee=0.0, taker_fee=0.0, fill="breach", participation_ratio=1.0, tick_size=0.0, max_position=inf, min_position=-inf)`, 8 inputs `(bid, ask, bid_size, ask_size, my_bid, my_bid_size, my_ask, my_ask_size)`, 4 outputs. Default `fill="breach"`.
- Produces (for Task 4 reuse): a private per-side fill helper `SideFill compute_side(...)` returning `{dpos, fill_price, fee, is_taker}`.

- [ ] **Step 1: Write the failing tests**

```python
def test_l1_default_is_breach_full_fill_on_cross():
    from screamer import BacktestL1
    # default breach: my_bid 100 rests passive (ask 101), then ask crosses to 99 -> full fill at 100 (maker)
    out = BacktestL1()(np.array([100., 99.]), np.array([101., 99.5]),
                       np.array([5., 5]), np.array([5., 5]),
                       np.array([100., 100.]), np.array([10., 10.]),
                       np.array([np.nan, np.nan]), np.array([np.nan, np.nan]))
    assert out[0, 2] == 0.0                                  # passive, no fill
    assert out[1, 2] == 10.0                                 # swept: full 10 at my_bid 100
    # bought 10 at 100, marks to mid (99+99.5)/2=99.25 -> pnl negative (adverse), maker fee 0
    assert out[1, 1] < 0.0


def test_l1_breach_no_fill_on_lock():
    from screamer import BacktestL1
    # breach: ask == my_bid (locked) must NOT fill
    out = BacktestL1(fill="breach")(np.array([100.]), np.array([100.]),
                                    np.array([5.]), np.array([5.]),
                                    np.array([100.]), np.array([10.]),
                                    np.array([np.nan]), np.array([np.nan]))
    assert out[0, 2] == 0.0


def test_l1_touch_lock_fills_participation_once():
    from screamer import BacktestL1
    # touch: my_bid 100 resting; then ask locks at 100 (size 8), participation 0.5 -> min(10, 0.5*8)=4;
    # a second identical locked snapshot must NOT add (edge-triggered)
    bid = np.array([100., 100., 100.]); ask = np.array([101., 100., 100.])
    out = BacktestL1(fill="touch", participation_ratio=0.5)(
        bid, ask, np.array([5., 5, 5]), np.array([8., 8, 8]),
        np.array([100., 100, 100]), np.array([10., 10, 10]),
        np.array([np.nan]*3), np.array([np.nan]*3))
    assert out[0, 2] == 0.0                                  # passive
    assert out[1, 2] == 4.0                                  # lock entry: min(10, 0.5*8)
    assert out[2, 2] == 4.0                                  # same lock: no further fill


def test_l1_submitted_crossing_is_taker_with_tick_slippage():
    from screamer import BacktestL1
    # quote appears already marketable (my_bid 100 > ask 99): taker, full fill, overflow at ask+tick.
    # ask_size 4, my_bid_size 10, tick 0.5 -> 4 @ 99, 6 @ 99.5; VWAP=(4*99+6*99.5)/10=99.3
    out = BacktestL1(taker_fee=0.0, tick_size=0.5)(
        np.array([98.]), np.array([99.]), np.array([5.]), np.array([4.]),
        np.array([100.]), np.array([10.]), np.array([np.nan]), np.array([np.nan]))
    assert out[0, 2] == 10.0                                 # full taker fill
    # marks to mid (98+99)/2=98.5; bought VWAP 99.3 -> immediate adverse cost = 10*(99.3-98.5)=8.0
    np.testing.assert_allclose(out[0, 3], 10 * (99.3 - 98.5), atol=1e-9)


def test_l1_inventory_cap_still_holds():
    from screamer import BacktestL1
    # ask 99 crosses through my_bid 100 (a taker sweep on first appearance);
    # room = max_position - 0 = 3 caps the fill at 3 of the 10 quoted
    out = BacktestL1(max_position=3.0)(
        np.array([98.]), np.array([99.]), np.array([5.]), np.array([5.]),
        np.array([100.]), np.array([10.]), np.array([np.nan]), np.array([np.nan]))
    assert out[0, 2] == 3.0                                  # capped even on a full sweep


def test_l1_stream_equals_batch_breach():
    from screamer import BacktestL1
    rng = np.random.default_rng(7); n = 200
    mid = 100 + np.cumsum(rng.standard_normal(n) * 0.1)
    bid, ask = mid - 0.05, mid + 0.05
    args = (bid, ask, np.full(n, 5.0), np.full(n, 5.0),
            bid - 0.01, np.full(n, 1.0), ask + 0.01, np.full(n, 1.0))
    op = BacktestL1()
    stream = np.array([op(*(float(a[i]) for a in args)) for i in range(n)])
    batch = BacktestL1()(*args)
    np.testing.assert_allclose(np.nan_to_num(stream), np.nan_to_num(batch))
```

- [ ] **Step 2: Run to verify failure**

Run: `poetry run python -m pytest tests/test_backtest.py -q -k "l1_default_is_breach or l1_breach_no_fill or l1_touch_lock or l1_submitted_crossing or l1_inventory_cap_still or stream_equals_batch_breach"`
Expected: FAIL (params and logic not present).

- [ ] **Step 3: Rewrite `include/screamer/backtest_l1.h`**

```cpp
#ifndef SCREAMER_BACKTEST_L1_H
#define SCREAMER_BACKTEST_L1_H

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include "screamer/common/functor_base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/pnl_account.h"

namespace screamer {

    // BacktestL1: 8 -> 4. Two-sided market maker against top-of-book quotes ONLY.
    // Fills are a documented heuristic (see the reference page Limitations box):
    // a resting quote fills full when the opposite side crosses through it, and in
    // "touch" mode also fills a participation partial once per lock episode. A quote
    // that appears already marketable is a taker (fills at market + tick_size).
    // Prefer BacktestL1Trades when a trade feed is available. Outputs
    // [equity, pnl, position, cost]. nan_policy: ignore on the market quote; a NaN
    // own-quote price means that side is not quoted.
    class BacktestL1 : public FunctorBase<BacktestL1, 8, 4> {
    public:
        BacktestL1(double maker_fee = 0.0, double taker_fee = 0.0,
                   const std::string& fill = "breach",
                   double participation_ratio = 1.0, double tick_size = 0.0,
                   double max_position = std::numeric_limits<double>::infinity(),
                   double min_position = -std::numeric_limits<double>::infinity())
            : maker_fee_(maker_fee), taker_fee_(taker_fee), breach_(parse_fill(fill)),
              participation_(parse_participation(participation_ratio)),
              tick_size_(tick_size), max_position_(max_position),
              min_position_(min_position)
        {
            if (min_position_ > max_position_)
                throw std::invalid_argument("min_position must not exceed max_position.");
            if (tick_size_ < 0.0)
                throw std::invalid_argument("tick_size must be non-negative.");
            reset();
        }

        void reset() override {
            account_.reset();
            bid_passive_ = false; ask_passive_ = false;
            bid_locked_ = false; ask_locked_ = false;
        }

        ResultTuple call(const InputArray& inputs) override {
            const double bid = inputs[0], ask = inputs[1];
            const double bid_size = inputs[2], ask_size = inputs[3];
            const double my_bid = inputs[4], my_bid_size = inputs[5];
            const double my_ask = inputs[6], my_ask_size = inputs[7];
            if (isnan2(bid) || isnan2(ask) || isnan2(bid_size) || isnan2(ask_size)) {
                const double nan = std::numeric_limits<double>::quiet_NaN();
                return std::make_tuple(nan, nan, nan, nan);   // ignore
            }
            const double mid = 0.5 * (bid + ask);

            // Buy side: resting buy at my_bid filled by the market ask.
            const double room_buy = max_position_ - account_.position();
            SideFill b = compute_side(/*buy=*/true, my_bid, my_bid_size, ask, ask_size,
                                      std::max(room_buy, 0.0), bid_passive_, bid_locked_);
            double eq = 0, pnl = 0, position = account_.position(), cost = 0; bool did = false;
            if (b.dpos != 0.0) {
                auto [e, p, pos, c] = account_.step(mid, b.dpos, b.fill_price,
                                                    b.is_taker ? taker_fee_ : maker_fee_);
                eq = e; pnl += p; position = pos; cost += c; did = true;
            }

            // Sell side: resting sell at my_ask filled by the market bid.
            const double room_sell = account_.position() - min_position_;
            SideFill s = compute_side(/*buy=*/false, my_ask, my_ask_size, bid, bid_size,
                                      std::max(room_sell, 0.0), ask_passive_, ask_locked_);
            if (s.dpos != 0.0) {
                auto [e, p, pos, c] = account_.step(mid, s.dpos, s.fill_price,
                                                    s.is_taker ? taker_fee_ : maker_fee_);
                eq = e; pnl += p; position = pos; cost += c; did = true;
            }

            if (!did) {
                auto [e, p, pos, c] = account_.step(mid, 0.0, mid, 0.0);  // mark only
                eq = e; pnl = p; position = pos; cost = c;
            }
            return std::make_tuple(eq, pnl, position, cost);
        }

    private:
        struct SideFill { double dpos; double fill_price; double fee; bool is_taker; };

        // Compute one side's fill and update that side's lock/passive state.
        // buy: resting buy at my_price hit by opp (=ask); sell: resting sell hit by opp (=bid).
        SideFill compute_side(bool buy, double my_price, double my_size,
                              double opp_price, double opp_size, double room,
                              bool& passive_prev, bool& locked) {
            SideFill r{0.0, my_price, buy ? maker_fee_ : taker_fee_, false};
            if (isnan2(my_price) || isnan2(my_size) || my_size <= 0.0 || room <= 0.0) {
                passive_prev = false; locked = false; return r;   // no quote this event
            }
            const bool through = buy ? (opp_price < my_price) : (opp_price > my_price);
            const bool lock    = (opp_price == my_price);
            const double remaining = std::min(my_size, room);

            if (through) {
                if (passive_prev) {                       // run over while resting: maker at my_price
                    r.dpos = buy ? remaining : -remaining;
                    r.fill_price = my_price; r.is_taker = false;
                } else {                                  // submitted already crossing: taker + slippage
                    const double disp = std::min(remaining, opp_size);
                    const double over = remaining - disp;
                    const double slip = buy ? tick_size_ : -tick_size_;
                    const double vwap = (disp * opp_price + over * (opp_price + slip))
                                        / remaining;
                    r.dpos = buy ? remaining : -remaining;
                    r.fill_price = vwap; r.is_taker = true;
                }
                passive_prev = false; locked = false;     // order consumed
            } else if (lock) {
                if (!breach_ && !locked) {                // touch mode, first event of this lock
                    const double filled = std::min(remaining, participation_ * opp_size);
                    r.dpos = buy ? filled : -filled;
                    r.fill_price = my_price; r.is_taker = false;
                }
                locked = true; passive_prev = true;       // still resting (a lock is passive)
            } else {                                       // opp on the far side: purely passive
                passive_prev = true; locked = false;
            }
            return r;
        }

        static bool parse_fill(const std::string& fill) {
            if (fill == "touch") return false;
            if (fill == "breach") return true;
            throw std::invalid_argument("fill must be \"touch\" or \"breach\".");
        }
        static double parse_participation(double p) {
            if (!(p > 0.0) || p > 1.0)
                throw std::invalid_argument("participation_ratio must be in (0, 1].");
            return p;
        }

        double maker_fee_, taker_fee_;
        bool breach_;
        double participation_, tick_size_, max_position_, min_position_;
        bool bid_passive_, ask_passive_, bid_locked_, ask_locked_;
        detail::PnLAccount account_;
    };

} // namespace screamer

#endif // SCREAMER_BACKTEST_L1_H
```

Note: the two `account_.step` calls in one event both mark to the same `mid`, so the second side's `mark_pnl` is zero (prev_close == mid); only its trade cost accrues. This matches the committed two-step aggregation.

- [ ] **Step 4: Update the binding** in `bindings/bindings_fin.cpp`

```cpp
py::class_<screamer::BacktestL1, screamer::EvalOp>(m, "BacktestL1")
    .def(py::init<double, double, const std::string&, double, double, double, double>(),
         py::arg("maker_fee") = 0.0, py::arg("taker_fee") = 0.0,
         py::arg("fill") = "breach", py::arg("participation_ratio") = 1.0,
         py::arg("tick_size") = 0.0,
         py::arg("max_position") = std::numeric_limits<double>::infinity(),
         py::arg("min_position") = -std::numeric_limits<double>::infinity())
    .def("__call__", &screamer::BacktestL1::handle_input)
    .def("reset", &screamer::BacktestL1::reset, "Reset.");
```

- [ ] **Step 5: Rebuild and run**

Run: `make install-dev && poetry run python -m pytest tests/test_backtest.py -q -k "l1"`
Expected: PASS for the new tests. The committed `test_l1_two_sided_fills_and_spread_capture` and `test_l1_stream_equals_batch` will likely need updating for the new default (`breach`) and taker/lock semantics; rewrite them to assert the corrected behavior (e.g. pass `fill="touch"` where a lock fill is intended, and recompute expected positions). Keep `test_l1_inventory_cap` (still valid).

- [ ] **Step 6: Commit**

```bash
git add include/screamer/backtest_l1.h bindings/bindings_fin.cpp tests/test_backtest.py
git commit -m "fix(backtest): BacktestL1 lock-episode fills, taker path, breach default"
```

---

## Task 4: Create `BacktestL1Trades` (quotes + trades, 10 inputs)

**Files:**
- Create: `include/screamer/backtest_l1_trades.h`
- Modify: `bindings/bindings_fin.cpp` (add include + registration)
- Test: `tests/test_backtest.py`

**Interfaces:**
- Consumes: the fill rules from Task 3 (through=full, at=participation, run-over vs taker) applied to trade events.
- Produces: `BacktestL1Trades(maker_fee=0.0, taker_fee=0.0, fill="touch", participation_ratio=1.0, tick_size=0.0, max_position=inf, min_position=-inf)`, 10 inputs `(bid, ask, bid_size, ask_size, my_bid, my_bid_size, my_ask, my_ask_size, trade_price, trade_size)`, 4 outputs. Default `fill="touch"` (trades make touch fills honest here).

- [ ] **Step 1: Write the failing tests**

```python
def test_l1trades_passive_fill_from_trade():
    from screamer import BacktestL1Trades
    # my_bid 100 resting inside (market 99.5/100.5); a sell-print AT 100 size 8, participation 0.5
    # -> min(10, 0.5*8)=4 at 100 (maker). NaN trade rows do not fill.
    q = dict(bid=99.5, ask=100.5, bs=5., asz=5., mb=100., mbs=10., ma=np.nan, mas=np.nan)
    out = BacktestL1Trades(participation_ratio=0.5)(
        np.array([q['bid'], q['bid']]), np.array([q['ask'], q['ask']]),
        np.array([q['bs'], q['bs']]), np.array([q['asz'], q['asz']]),
        np.array([q['mb'], q['mb']]), np.array([q['mbs'], q['mbs']]),
        np.array([q['ma'], q['ma']]), np.array([q['mas'], q['mas']]),
        np.array([np.nan, 100.]), np.array([np.nan, 8.]))     # quote row, then a trade row
    assert out[0, 2] == 0.0                                    # NaN trade -> no fill, mark only
    assert out[1, 2] == 4.0                                    # trade at 100 -> participation fill


def test_l1trades_through_trade_fills_full():
    from screamer import BacktestL1Trades
    # a sell-print THROUGH my_bid (99 < 100) fills the full resting size
    out = BacktestL1Trades()(
        np.array([99.5]), np.array([100.5]), np.array([5.]), np.array([5.]),
        np.array([100.]), np.array([10.]), np.array([np.nan]), np.array([np.nan]),
        np.array([99.]), np.array([2.]))
    assert out[0, 2] == 10.0


def test_l1trades_run_over_fills_without_a_trade():
    from screamer import BacktestL1Trades
    # no trade (NaN), but the quote ask crosses my_bid -> run-over full fill at my_bid
    out = BacktestL1Trades()(
        np.array([100., 99.]), np.array([101., 99.5]),
        np.array([5., 5]), np.array([5., 5]),
        np.array([100., 100.]), np.array([10., 10.]),
        np.array([np.nan, np.nan]), np.array([np.nan, np.nan]),
        np.array([np.nan, np.nan]), np.array([np.nan, np.nan]))
    assert out[0, 2] == 0.0                                    # passive
    assert out[1, 2] == 10.0                                   # run-over on the cross


def test_l1trades_stream_equals_batch():
    from screamer import BacktestL1Trades
    rng = np.random.default_rng(9); n = 200
    mid = 100 + np.cumsum(rng.standard_normal(n) * 0.05)
    bid, ask = mid - 0.05, mid + 0.05
    tp = np.where(rng.standard_normal(n) > 0.5, mid, np.nan)   # sparse trades
    ts = np.where(np.isnan(tp), np.nan, 1.0)
    args = (bid, ask, np.full(n, 5.0), np.full(n, 5.0),
            bid, np.full(n, 1.0), ask, np.full(n, 1.0), tp, ts)
    op = BacktestL1Trades()
    stream = np.array([op(*(float(a[i]) for a in args)) for i in range(n)])
    batch = BacktestL1Trades()(*args)
    np.testing.assert_allclose(np.nan_to_num(stream), np.nan_to_num(batch))
```

- [ ] **Step 2: Run to verify failure**

Run: `poetry run python -m pytest tests/test_backtest.py -q -k "l1trades"`
Expected: FAIL (`BacktestL1Trades` does not exist).

- [ ] **Step 3: Create `include/screamer/backtest_l1_trades.h`**

```cpp
#ifndef SCREAMER_BACKTEST_L1_TRADES_H
#define SCREAMER_BACKTEST_L1_TRADES_H

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include "screamer/common/functor_base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/pnl_account.h"

namespace screamer {

    // BacktestL1Trades: 10 -> 4. The preferred market-making engine: quotes mark the
    // position and seed context, TRADES drive passive fills (unambiguous execution
    // events), and a quote cross with no explaining trade is the run-over fallback.
    // Inputs (bid, ask, bid_size, ask_size, my_bid, my_bid_size, my_ask, my_ask_size,
    // trade_price, trade_size). trades are NOT forward-filled: a NaN trade is a
    // quote-only update (mark, no fill), which is nan_policy: ignore, so each real
    // trade fills at most once. Outputs [equity, pnl, position, cost].
    class BacktestL1Trades : public FunctorBase<BacktestL1Trades, 10, 4> {
    public:
        BacktestL1Trades(double maker_fee = 0.0, double taker_fee = 0.0,
                         const std::string& fill = "touch",
                         double participation_ratio = 1.0, double tick_size = 0.0,
                         double max_position = std::numeric_limits<double>::infinity(),
                         double min_position = -std::numeric_limits<double>::infinity())
            : maker_fee_(maker_fee), taker_fee_(taker_fee), breach_(parse_fill(fill)),
              participation_(parse_participation(participation_ratio)),
              tick_size_(tick_size), max_position_(max_position),
              min_position_(min_position)
        {
            if (min_position_ > max_position_)
                throw std::invalid_argument("min_position must not exceed max_position.");
            if (tick_size_ < 0.0)
                throw std::invalid_argument("tick_size must be non-negative.");
            reset();
        }

        void reset() override { account_.reset(); bid_passive_ = false; ask_passive_ = false; }

        ResultTuple call(const InputArray& inputs) override {
            const double bid = inputs[0], ask = inputs[1];
            const double bid_size = inputs[2], ask_size = inputs[3];
            const double my_bid = inputs[4], my_bid_size = inputs[5];
            const double my_ask = inputs[6], my_ask_size = inputs[7];
            const double trade_price = inputs[8], trade_size = inputs[9];
            if (isnan2(bid) || isnan2(ask) || isnan2(bid_size) || isnan2(ask_size)) {
                const double nan = std::numeric_limits<double>::quiet_NaN();
                return std::make_tuple(nan, nan, nan, nan);   // ignore
            }
            const double mid = 0.5 * (bid + ask);
            const bool has_trade = !isnan2(trade_price) && !isnan2(trade_size);

            double eq = 0, pnl = 0, position = account_.position(), cost = 0; bool did = false;

            // Buy side.
            const double room_buy = std::max(max_position_ - account_.position(), 0.0);
            double b_dpos = 0.0, b_price = my_bid; bool b_taker = false;
            resolve_side(true, my_bid, my_bid_size, ask, ask_size, room_buy,
                         has_trade, trade_price, trade_size, bid_passive_,
                         b_dpos, b_price, b_taker);
            if (b_dpos != 0.0) {
                auto [e, p, pos, c] = account_.step(mid, b_dpos, b_price,
                                                    b_taker ? taker_fee_ : maker_fee_);
                eq = e; pnl += p; position = pos; cost += c; did = true;
            }

            // Sell side.
            const double room_sell = std::max(account_.position() - min_position_, 0.0);
            double s_dpos = 0.0, s_price = my_ask; bool s_taker = false;
            resolve_side(false, my_ask, my_ask_size, bid, bid_size, room_sell,
                         has_trade, trade_price, trade_size, ask_passive_,
                         s_dpos, s_price, s_taker);
            if (s_dpos != 0.0) {
                auto [e, p, pos, c] = account_.step(mid, s_dpos, s_price,
                                                    s_taker ? taker_fee_ : maker_fee_);
                eq = e; pnl += p; position = pos; cost += c; did = true;
            }

            if (!did) {
                auto [e, p, pos, c] = account_.step(mid, 0.0, mid, 0.0);
                eq = e; pnl = p; position = pos; cost = c;
            }
            return std::make_tuple(eq, pnl, position, cost);
        }

    private:
        // Passive fills come from the trade tape; a quote cross with no explaining
        // trade is the run-over fallback (maker at my_price).
        void resolve_side(bool buy, double my_price, double my_size,
                          double opp_price, double opp_size, double room,
                          bool has_trade, double trade_price, double trade_size,
                          bool& passive_prev,
                          double& dpos, double& fill_price, bool& is_taker) {
            if (isnan2(my_price) || isnan2(my_size) || my_size <= 0.0 || room <= 0.0) {
                passive_prev = false; return;
            }
            const double remaining = std::min(my_size, room);
            const bool quote_through = buy ? (opp_price < my_price) : (opp_price > my_price);

            if (has_trade) {                                   // trade drives the fill
                const bool t_through = buy ? (trade_price < my_price) : (trade_price > my_price);
                const bool t_at = !breach_ && (trade_price == my_price);
                double filled = 0.0;
                if (t_through) filled = remaining;
                else if (t_at)  filled = std::min(remaining, participation_ * trade_size);
                if (filled > 0.0) { dpos = buy ? filled : -filled; fill_price = my_price; is_taker = false; }
                passive_prev = !quote_through;                 // still resting unless the quote also crossed
                return;
            }

            // No trade this event: only a quote cross fills (run-over), else stay passive.
            if (quote_through && passive_prev) {
                dpos = buy ? remaining : -remaining; fill_price = my_price; is_taker = false;
                passive_prev = false;
            } else {
                passive_prev = !quote_through;                 // passive if not crossed
            }
        }

        static bool parse_fill(const std::string& fill) {
            if (fill == "touch") return false;
            if (fill == "breach") return true;
            throw std::invalid_argument("fill must be \"touch\" or \"breach\".");
        }
        static double parse_participation(double p) {
            if (!(p > 0.0) || p > 1.0)
                throw std::invalid_argument("participation_ratio must be in (0, 1].");
            return p;
        }

        double maker_fee_, taker_fee_;
        bool breach_;
        double participation_, tick_size_, max_position_, min_position_;
        bool bid_passive_, ask_passive_;
        detail::PnLAccount account_;
    };

} // namespace screamer

#endif // SCREAMER_BACKTEST_L1_TRADES_H
```

- [ ] **Step 4: Register the binding** in `bindings/bindings_fin.cpp` (add include near the other backtest includes, and the class after `BacktestL1`)

```cpp
#include "screamer/backtest_l1_trades.h"
```
```cpp
py::class_<screamer::BacktestL1Trades, screamer::EvalOp>(m, "BacktestL1Trades")
    .def(py::init<double, double, const std::string&, double, double, double, double>(),
         py::arg("maker_fee") = 0.0, py::arg("taker_fee") = 0.0,
         py::arg("fill") = "touch", py::arg("participation_ratio") = 1.0,
         py::arg("tick_size") = 0.0,
         py::arg("max_position") = std::numeric_limits<double>::infinity(),
         py::arg("min_position") = -std::numeric_limits<double>::infinity())
    .def("__call__", &screamer::BacktestL1Trades::handle_input)
    .def("reset", &screamer::BacktestL1Trades::reset, "Reset.");
```

- [ ] **Step 5: Rebuild and run**

Run: `make install-dev && poetry run python -m pytest tests/test_backtest.py -q -k "l1trades"`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add include/screamer/backtest_l1_trades.h bindings/bindings_fin.cpp tests/test_backtest.py
git commit -m "feat(backtest): BacktestL1Trades engine (quotes + trades)"
```

---

## Task 5: Docs pages (Limitations) + notebook 15 correction

**Files:**
- Modify: `docs/functions_fin/BacktestTrades.md`, `docs/functions_fin/BacktestL1.md`
- Create: `docs/functions_fin/BacktestL1Trades.md`
- Modify: `docs/notebooks/15-event-driven-backtests.ipynb`

**Interfaces:**
- Consumes: the final constructor signatures from Tasks 1, 3, 4 (params + defaults).

- [ ] **Step 1: Update `BacktestTrades.md`** frontmatter to add the `participation_ratio` parameter and a Description note; add a Limitations subsection:

```markdown
- name: participation_ratio
  type: float
  default: 1.0
  min: 0.0
  max: 1.0
  description: Fraction of the at-price trade volume captured (front-of-queue at 1.0). A trade through your price fills the full order.
```

Add under Description:

```markdown
## Limitations

Fills are front-of-queue and optimistic: a trade at your price fills
`min(remaining, participation_ratio * trade_size)`, a trade through it fills the
full order. There is no queue-position modelling; orders are counterfactual (zero
volume, no market impact).
```

- [ ] **Step 2: Update `BacktestL1.md`** frontmatter (params: `maker_fee, taker_fee, fill` default `breach`, `participation_ratio, tick_size, max_position, min_position`) and add the Limitations box:

```markdown
## Limitations

Quotes-only fills are a heuristic and cannot tell a trade from a cancel. The
default `breach` fills only when the market trades through your quote (it can
under-fill). `touch` additionally fills a participation partial once per lock
episode and can over-fill when a lock's size came from cancels. Prefer
`BacktestL1Trades` when a trade feed is available. Orders are counterfactual
(zero volume, no market impact); a market/marketable order fills beyond the
displayed size only under a `tick_size` slippage assumption.
```

- [ ] **Step 3: Create `docs/functions_fin/BacktestL1Trades.md`** mirroring the `BacktestL1.md` structure (frontmatter `name/title/implementation_family: fin/topics: [risk]/inputs: 10/outputs: 4`, the seven parameters, `nan_policy: ignore`, `see_also: [BacktestL1, BacktestTrades, backtest_report]`), a Description explaining trades-drive-fills + quotes-mark, a `.. plotly::` example (adapt notebook 15's L1Trades cell), and a Limitations box noting the `tick_size` assumption and counterfactual orders. Use the plotted-example pattern from `BacktestL1.md`.

- [ ] **Step 4: Correct notebook 15** (`docs/notebooks/15-event-driven-backtests.ipynb`): the L1 cell currently relies on the old fill semantics. Update the Trades and L1 cells to the corrected API (pass `fill="touch"` where a touch fill is intended; the default L1 is now `breach`), and add a short `BacktestL1Trades` section building a trades-driven maker from the tape (trades on their own rows, quotes forward-filled). Re-execute to confirm non-degenerate, bounded results:

```bash
cd docs/notebooks && poetry run jupyter nbconvert --to notebook --execute --stdout \
  15-event-driven-backtests.ipynb > /tmp/nb15.ipynb 2>/tmp/nb15.err; echo "exit: $?"
```
Expected: exit 0. Strip outputs before committing (the notebook is committed without stored outputs).

- [ ] **Step 5: Commit**

```bash
git add docs/functions_fin/BacktestTrades.md docs/functions_fin/BacktestL1.md \
        docs/functions_fin/BacktestL1Trades.md docs/notebooks/15-event-driven-backtests.ipynb
git commit -m "docs(backtest): corrected fill semantics + BacktestL1Trades page and Limitations"
```

---

## Task 6: Regenerate, full suite, docs build, final commit

**Files:**
- Modify: `screamer/__init__.py`, `screamer/data/help.json`, `docs/by_topic/risk.rst`, `docs/by_topic_index.rst` (all regenerated), `CHANGELOG.md`.

- [ ] **Step 1: Regenerate registries** (help_registry BEFORE topic pages)

```bash
poetry run python devtools/build_help_registry.py    # validates every frontmatter vs binding
poetry run python devtools/build_topic_pages.py
make regen-init
```
Expected: `BacktestL1Trades` listed with 1 example; no validation errors (param names/defaults match the bindings, including `.inf` for the infinities and `participation_ratio` in `(0,1]`).

- [ ] **Step 2: Run the full test suite**

Run: `poetry run python -m pytest -q`
Expected: all pass (previously 4461 + the new tests). Investigate any compliance-harness pickups of `BacktestL1Trades` (it is auto-discovered by `test_nan_input_compliance.py`; the NaN-at-index and no-sticky-nan tests should pass as they do for `BacktestL1`).

- [ ] **Step 3: Build the docs** (executes notebooks + plotly examples)

Run: `make docs`
Expected: exit 0; `docs/_build/html/functions_fin/BacktestL1Trades.html` exists and the changed pages render (2 plotly iframes each, matching `BacktestSignal`).

- [ ] **Step 4: Update `CHANGELOG.md`** `[Unreleased]` to note the fill-model correction and the new engine:

```markdown
* Backtest fill models corrected and extended:
  * `BacktestTrades`/`BacktestL1`: a trade or quote through your price fills the
    full order; a fill at your price is `min(remaining, participation_ratio *
    available_size)` (new `participation_ratio`, default 1.0). `BacktestL1` now
    defaults to the conservative `breach` fill, adds a `touch` lock-episode mode,
    a taker path with `tick_size` slippage, and `taker_fee`.
  * `BacktestL1Trades` (new, 10 inputs): the preferred market-making engine, with
    quotes marking the position and the trade tape driving fills. Reference page,
    plotted example, tests, and a demo section in notebook 15.
```

- [ ] **Step 5: Final commit**

```bash
git add screamer/__init__.py screamer/data/help.json docs/by_topic/risk.rst \
        docs/by_topic_index.rst CHANGELOG.md
git commit -m "chore(backtest): regen registries + changelog for fill-model correction"
```

---

## Self-Review

**Spec coverage:**
- Principles (counterfactual, deterministic, event-driven fills): enforced across Tasks 1, 3, 4 and stated in headers/docs. Covered.
- Parameters (`participation_ratio`, `tick_size`, `fill`): Tasks 1 (ratio), 3 (all), 4 (all). Covered.
- Per-engine rules: OHLC (Task 2), Trades (Task 1), L1 (Task 3), L1Trades (Task 4), market/marketable (Task 3 taker path, reused in 4). Covered.
- Fill price / fee side (limit price; run-over maker vs submitted-crossing taker): Task 3 `compute_side`, Task 4 `resolve_side`. Covered.
- Transparency (Limitations sections): Task 5. Covered.
- Change scope list: matches Tasks 1-6. Covered.

**Placeholder scan:** No TBD/TODO; every code step shows complete code. The only prose-directed step is Task 5 Step 3 (new doc page), which specifies exact frontmatter fields and the pattern file to mirror.

**Type consistency:** `participation_ratio` (Python arg) maps to `participation_` (C++ member) consistently; `SideFill{dpos, fill_price, fee, is_taker}` used only within Task 3; Task 4 uses out-params of the same meaning. Constructor signatures in each task's binding step match the `py::init<...>` arity. `fill` default is `"touch"` for Trades/OHLC/L1Trades and `"breach"` for L1, as the spec requires.

---

## Notes for the implementer

- `isnan2`, `quiet_NaN`, `parse_start_policy` live in `screamer/common/float_info.h`; `FunctorBase`/`EvalOp`/`handle_input` in `screamer/common/functor_base.h` and `eval_op.h`. Follow the committed `backtest_l1.h` for the binding/registration idiom.
- The two-`account_.step`-per-event pattern (buy then sell) marks both to the same `mid`, so the second step contributes only trade cost. Preserve it.
- When rewriting the committed `test_l1_*` expectations, recompute by hand from the rules; do not weaken assertions to pass. Where a test asserted the old (wrong) sizing, its expected value changes.
