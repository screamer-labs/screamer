# Backtest Engine Suite (data model x order def) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reshape the backtest engines into a clean `Backtest<DataModel><OrderDef>` grid: eight engines over shared per-data-model fill cores and one reusable target front, with a static fill cap and the `MARKET` price encoding on every engine.

**Architecture:** Three data models that carry both order definitions (`OHLC`, `Trades`, `L1`) get a `detail/*_fill.h` core struct (config + `PnLAccount` + a `quote()` method taking a two-sided quote) reused by their `Orders` and `Target` engines. The two single-engine rows (`Price`, `L1Trades`) stay direct engines. A `Target` engine is a thin wrapper that converts a target position into a marketable one-sided quote via `detail/target_front.h` and feeds the core. Each engine is a thin `FunctorBase<Derived, N, 4>` with one fixed input schema.

**Tech Stack:** C++17 header-only core (`include/screamer/`), pybind11 bindings (`bindings/bindings_fin.cpp`), Python packaging/regeneration, pytest.

## Global Constraints

- Base branch: create `feat/backtest-engine-grid` from `feat/unified-backtest-api` (NOT from `main`). That branch already holds the reviewed fill logic (the two makers, the static cap on every engine, and `market_limit` + `MARKET`); this redesign renames and refactors it into the grid. The renames erase the old names, so the end state matches the spec's "fresh from main" intent while reusing reviewed code.
- Naming grid: `Backtest<DataModel><OrderDef>`, `DataModel` in {`Price`, `OHLC`, `Trades`, `L1`, `L1Trades`}, `OrderDef` in {`Target`, `Orders`}. Build exactly eight: `BacktestPriceTarget`, `BacktestOHLCTarget`, `BacktestOHLCOrders`, `BacktestTradesTarget`, `BacktestTradesOrders`, `BacktestL1Target`, `BacktestL1Orders`, `BacktestL1TradesOrders`. Do NOT build `BacktestPriceOrders` or `BacktestL1TradesTarget`.
- `Target` engines are market only: input is `(target_position, <market columns>)`, the engine sizes `target - position` and submits it as a marketable order (via the target front). Clamp `target` to `[min_position, max_position]` before sizing, so position stays in bounds the same way on every Target engine.
- `Orders` engines take `(bid_price, bid_size, ask_price, ask_size, <market columns>)`. A `size <= 0` or `NaN` size means no order that side; a one-sided order zeroes the other side.
- Market-order encoding via `screamer::market_limit(price, buy)` (already in `include/screamer/common/market_price.h`): `NaN` -> `+inf` on a buy / `-inf` on a sell; `+inf` buy = market, `-inf` buy = never fill; finite = limit. `screamer.MARKET` (Python, = `inf`) is the convenience spelling.
- Every engine returns `[equity, pnl, position, cost]` via the shared `detail::PnLAccount`. Do NOT change `PnLAccount`, `backtest_report`, or `BacktestReport`.
- Static fill cap `min_position` (default `-inf`) / `max_position` (default `+inf`) on every engine; a fill is `min(order size, counterparty volume, room to cap)`. Constructor throws if `min_position > max_position`.
- Cost/fidelity config carried by the cores: `maker_fee`, `taker_fee`, `fill` (`"touch"`/`"breach"`), `participation_ratio` in `(0,1]`, `tick_size >= 0`. `Price` keeps its `spread`/`fee` cost model (a value series has no book).
- After any C++ change: `make install-dev`, then `poetry run python devtools/build_help_registry.py`, `poetry run python devtools/build_topic_pages.py`, `make regen-init`. YAML infinity in docs frontmatter is `-.inf` / `.inf`.
- No em-dashes in prose/comments/docstrings (ASCII hyphens). Commit as `simu.ai <claude@sitmo.com>` with the footer:

      Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
      Claude-Session: https://claude.ai/code/session_018q4wFbrQaLrzUFc1H5NpJx

  Do not edit version files. Do not push.
- Every populated cell needs a `docs/functions_fin/<Name>.md` page (frontmatter with `nan_policy`, `topics: [backtesting]`) or `build_help_registry.py` fails.

## Naming map (source of each engine)

| new engine | built from (on `feat/unified-backtest-api`) | new logic |
|---|---|---|
| `BacktestPriceTarget` | `BacktestSignal` (rename) | none |
| `BacktestOHLCOrders` | `BacktestOHLCMaker` (-> `OHLCFill` core) | none |
| `BacktestOHLCTarget` | `BacktestOHLC` directional, market path only | drop the limit branch; defer to next open via the front |
| `BacktestTradesOrders` | `BacktestTradesMaker` (-> `TradesFill` core) | subsumes the old one-sided `BacktestTrades` (zero a side) |
| `BacktestTradesTarget` | new (old `BacktestTrades` was orders-style, now subsumed above) | target -> marketable print-taking via the front + `TradesFill` |
| `BacktestL1Orders` | `BacktestL1` (-> `L1Fill` core) | none |
| `BacktestL1Target` | new | target -> marketable book-taking via the front + `L1Fill` |
| `BacktestL1TradesOrders` | `BacktestL1Trades` (rename) | none |

Old headers removed as their replacements land: `backtest_signal.h`, `backtest_ohlc.h`, `backtest_ohlc_maker.h`, `backtest_trades.h`, `backtest_trades_maker.h`, `backtest_l1.h`, `backtest_l1_trades.h`.

---

## File Structure

- `include/screamer/detail/target_front.h` (create) - `target_to_quote(target, position)` helper.
- `include/screamer/detail/ohlc_fill.h`, `trades_fill.h`, `l1_fill.h` (create) - the three shared fill cores.
- `include/screamer/backtest_price_target.h` (create; replaces `backtest_signal.h`).
- `include/screamer/backtest_ohlc_orders.h`, `backtest_ohlc_target.h` (create; replace `backtest_ohlc.h`, `backtest_ohlc_maker.h`).
- `include/screamer/backtest_trades_orders.h`, `backtest_trades_target.h` (create; replace `backtest_trades.h`, `backtest_trades_maker.h`).
- `include/screamer/backtest_l1_orders.h`, `backtest_l1_target.h` (create; replace `backtest_l1.h`).
- `include/screamer/backtest_l1trades_orders.h` (create; replaces `backtest_l1_trades.h`).
- `bindings/bindings_fin.cpp` (rewrite the backtest block, lines ~181-245).
- `docs/functions_fin/` (rename the eight engine pages; rewrite `choosing_a_backtest_engine.md`).
- `tests/test_backtest.py`, `tests/param_cases.py`, `CHANGELOG.md`.

---

## Task 1: the target front + branch setup

**Files:**
- Create branch, create: `include/screamer/detail/target_front.h`

**Interfaces:**
- Consumes: `screamer::market_limit` (exists).
- Produces: `screamer::detail::target_to_quote(double target, double position, double min_pos, double max_pos) -> std::tuple<double,double,double,double>` returning `(bid_price, bid_size, ask_price, ask_size)`: the marketable one-sided order that moves `position` toward `clamp(target, min_pos, max_pos)`. Buy side uses `bid_price = +inf`; sell side uses `ask_price = -inf`; the idle side has size `0`.

- [ ] **Step 1: Create the branch**

Run: `git checkout feat/unified-backtest-api && git checkout -b feat/backtest-engine-grid`
Expected: on a new branch with the reviewed fill logic present.

- [ ] **Step 2: Create `include/screamer/detail/target_front.h`**

```cpp
#ifndef SCREAMER_DETAIL_TARGET_FRONT_H
#define SCREAMER_DETAIL_TARGET_FRONT_H

#include <algorithm>
#include <limits>
#include <tuple>

namespace screamer { namespace detail {

    // Convert a desired target position into a marketable one-sided order that
    // moves the live position toward clamp(target, [min_pos, max_pos]). Returns a
    // canonical two-sided quote (bid_price, bid_size, ask_price, ask_size) with a
    // market price (+inf buy / -inf sell) on the trading side and zero size on the
    // idle side, so any fill core's marketable path executes it. The core still caps
    // the fill to the room, so a target beyond the cap lands exactly on the cap.
    inline std::tuple<double, double, double, double>
    target_to_quote(double target, double position, double min_pos, double max_pos) {
        const double clamped = std::clamp(target, min_pos, max_pos);
        const double delta = clamped - position;
        const double inf = std::numeric_limits<double>::infinity();
        if (delta > 0.0) return std::make_tuple(inf, delta, -inf, 0.0);   // market buy
        if (delta < 0.0) return std::make_tuple(inf, 0.0, -inf, -delta);  // market sell
        return std::make_tuple(inf, 0.0, -inf, 0.0);                      // no trade
    }

}} // namespace screamer::detail

#endif // SCREAMER_DETAIL_TARGET_FRONT_H
```

`target_to_quote` is a pure header-only helper with no Python binding; its behavior is verified through the Target engines' parity/cap tests in Tasks 3-5. This task's deliverable is that the header compiles into the extension.

- [ ] **Step 3: Verify it compiles**

Run: `make install-dev`
Expected: builds clean (the header is included by later engines; confirm no syntax error by adding a temporary `#include "screamer/detail/target_front.h"` to `bindings/bindings_fin.cpp` if you want an immediate compile check, then remove it).

- [ ] **Step 4: Commit**

```bash
git add include/screamer/detail/target_front.h
git commit -m "feat(backtest): target_to_quote front helper for Target engines"
```

---

## Task 2: `BacktestPriceTarget` (rename of `BacktestSignal`)

**Files:**
- Create: `include/screamer/backtest_price_target.h`; Delete: `include/screamer/backtest_signal.h`
- Modify: `bindings/bindings_fin.cpp` (the `BacktestSignal` block), `screamer/data/help.json` (regen)
- Rename doc: `docs/functions_fin/BacktestSignal.md` -> `BacktestPriceTarget.md`
- Test: `tests/test_backtest.py`

**Interfaces:**
- Produces: `BacktestPriceTarget(spread=0.0, fee=0.0, min_position=-inf, max_position=+inf)`, inputs `(target_position, price)`, outputs `[equity, pnl, position, cost]`. Identical behavior to the old `BacktestSignal`.

- [ ] **Step 1: Write the failing parity test**

```python
def test_price_target_reaches_target_and_costs():
    import numpy as np
    from screamer import BacktestPriceTarget
    out = BacktestPriceTarget(spread=0.0, fee=0.001)(
        np.array([1., 1., 0.]), np.array([100., 110., 121.]))
    np.testing.assert_allclose(out[:, 2], [1., 1., 0.])          # position track
    assert out[0, 3] > 0.0                                        # taker fee charged on the buy
```

- [ ] **Step 2: Run to verify failure**

Run: `poetry run python -m pytest tests/test_backtest.py -q -k price_target_reaches`
Expected: FAIL (no `BacktestPriceTarget`).

- [ ] **Step 3: Create `include/screamer/backtest_price_target.h`** by copying `include/screamer/backtest_signal.h` verbatim, then: rename the class `BacktestSignal` -> `BacktestPriceTarget`, rename the include guard `SCREAMER_BACKTEST_SIGNAL_H` -> `SCREAMER_BACKTEST_PRICE_TARGET_H`, update the leading docstring's name and to say "value series (a price/mark); reaches a target position by taking liquidity." Keep the logic identical (it already clamps the target to the cap and fills market at `price * (1 +/- spread/2)`). Delete `include/screamer/backtest_signal.h`.

- [ ] **Step 4: Update the binding** in `bindings/bindings_fin.cpp` (replace the `BacktestSignal` block):

```cpp
py::class_<screamer::BacktestPriceTarget, screamer::EvalOp>(m, "BacktestPriceTarget")
    .def(py::init<double, double, double, double>(),
         py::arg("spread") = 0.0, py::arg("fee") = 0.0,
         py::arg("min_position") = -std::numeric_limits<double>::infinity(),
         py::arg("max_position") = std::numeric_limits<double>::infinity())
    .def("__call__", &screamer::BacktestPriceTarget::handle_input)
    .def("reset", &screamer::BacktestPriceTarget::reset, "Reset.");
```

Update the `#include "screamer/backtest_signal.h"` near the top of the bindings file to `#include "screamer/backtest_price_target.h"`.

- [ ] **Step 5: Rename the docs page** `git mv docs/functions_fin/BacktestSignal.md docs/functions_fin/BacktestPriceTarget.md`, then edit its frontmatter `name`/`title` to `BacktestPriceTarget` and update the prose to the "reach a target on a value series" framing. Keep `nan_policy` and `topics: [backtesting]`.

- [ ] **Step 6: Rebuild, regen, run**

Run: `make install-dev && poetry run python devtools/build_help_registry.py && make regen-init && poetry run python -m pytest tests/test_backtest.py -q -k "price_target or target_front_via_price"`
Expected: PASS (both the parity test and the Task 1 file-level test go green).

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat(backtest): BacktestPriceTarget (renames BacktestSignal into the grid)"
```

---

## Task 3: OHLC core + `BacktestOHLCOrders` + `BacktestOHLCTarget`

**Files:**
- Create: `include/screamer/detail/ohlc_fill.h`, `include/screamer/backtest_ohlc_orders.h`, `include/screamer/backtest_ohlc_target.h`
- Delete: `include/screamer/backtest_ohlc.h`, `include/screamer/backtest_ohlc_maker.h`
- Modify: `bindings/bindings_fin.cpp`, `tests/param_cases.py`
- Docs: `git mv docs/functions_fin/BacktestOHLCMaker.md -> BacktestOHLCOrders.md`; `git mv docs/functions_fin/BacktestOHLC.md -> BacktestOHLCTarget.md`
- Test: `tests/test_backtest.py`

**Interfaces:**
- Consumes: `detail::target_to_quote`, `market_limit`, `detail::PnLAccount`.
- Produces:
  - `detail::OHLCFill(maker_fee, taker_fee, breach, participation, tick_size, min_pos, max_pos)` with `void reset()`, `double position() const`, and `std::tuple<double,double,double,double> quote(double bid_price, double bid_size, double ask_price, double ask_size, double open, double high, double low, double close)`.
  - `BacktestOHLCOrders(maker_fee=0.0, taker_fee=0.0, fill="touch", participation_ratio=1.0, tick_size=0.0, min_position=-inf, max_position=+inf)`, 8 inputs `(bid_price, bid_size, ask_price, ask_size, open, high, low, close)`.
  - `BacktestOHLCTarget(taker_fee=0.0, tick_size=0.0, min_position=-inf, max_position=+inf)`, 5 inputs `(target_position, open, high, low, close)`; decides on bar t, executes at bar t+1's open (deferred, causal).

- [ ] **Step 1: Write the failing tests**

```python
def test_ohlc_orders_two_sided_fill():
    import numpy as np
    from screamer import BacktestOHLCOrders
    out = BacktestOHLCOrders()(
        np.array([99.]), np.array([1.]), np.array([np.nan]), np.array([0.]),
        np.array([100.]), np.array([101.]), np.array([98.]), np.array([100.]))
    assert out[0, 2] == 1.0                                # resting bid hit on the low

def test_ohlc_target_defers_to_next_open():
    import numpy as np
    from screamer import BacktestOHLCTarget
    # target +1 decided on bar 0 executes at bar 1's open (deferred); bar 0 stays flat
    out = BacktestOHLCTarget()(
        np.array([1., 1.]), np.array([100., 105.]),
        np.array([100., 105.]), np.array([100., 105.]), np.array([100., 105.]))
    assert out[0, 2] == 0.0                                # nothing executes on bar 0
    assert out[1, 2] == 1.0                                # filled at bar 1 open

def test_ohlc_target_market_capped():
    import numpy as np
    from screamer import BacktestOHLCTarget
    out = BacktestOHLCTarget(max_position=1.0)(
        np.array([5., 5.]), np.array([100., 100.]),
        np.array([100., 100.]), np.array([100., 100.]), np.array([100., 100.]))
    assert out[1, 2] == 1.0                                # target 5 clamped to the cap
```

- [ ] **Step 2: Run to verify failure**

Run: `poetry run python -m pytest tests/test_backtest.py -q -k "ohlc_orders or ohlc_target"`
Expected: FAIL (engines do not exist).

- [ ] **Step 3: Create `include/screamer/detail/ohlc_fill.h`** by refactoring the reviewed `include/screamer/backtest_ohlc_maker.h` (commit `79dffb1`): move its class body into a `struct OHLCFill` in namespace `screamer::detail`. The constructor takes `(double maker_fee, double taker_fee, bool breach, double participation, double tick_size, double min_pos, double max_pos)` (the parsed values; parsing of `fill`/`participation` moves to the engine wrappers). Rename its `call(const InputArray&)` to `quote(double bid_price, double bid_size, double ask_price, double ask_size, double open, double high, double low, double close)` reading the eight named arguments instead of `inputs[i]`. Keep the two-sided fill logic, the room cap, and the mark-to-close fallback byte-for-byte. Expose `void reset() { account_.reset(); }`, `double position() const { return account_.position(); }`, and a public `detail::PnLAccount account_`.

- [ ] **Step 4: Create `include/screamer/backtest_ohlc_orders.h`** (thin wrapper over the core):

```cpp
#ifndef SCREAMER_BACKTEST_OHLC_ORDERS_H
#define SCREAMER_BACKTEST_OHLC_ORDERS_H

#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include "screamer/common/functor_base.h"
#include "screamer/detail/ohlc_fill.h"

namespace screamer {

    // BacktestOHLCOrders: 8 -> 4. Post a two-sided quote against OHLC bars; each side
    // fills when the bar's low/high reaches it (touch/breach), a marketable order
    // fills at the open. Inventory is capped to [min_position, max_position].
    // Outputs [equity, pnl, position, cost].
    class BacktestOHLCOrders : public FunctorBase<BacktestOHLCOrders, 8, 4> {
    public:
        BacktestOHLCOrders(double maker_fee = 0.0, double taker_fee = 0.0,
                           const std::string& fill = "touch",
                           double participation_ratio = 1.0, double tick_size = 0.0,
                           double min_position = -std::numeric_limits<double>::infinity(),
                           double max_position = std::numeric_limits<double>::infinity())
            : core_(maker_fee, taker_fee, parse_fill(fill),
                    parse_participation(participation_ratio), tick_size,
                    min_position, max_position) {}
        void reset() override { core_.reset(); }
        ResultTuple call(const InputArray& in) override {
            return core_.quote(in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7]);
        }
    private:
        static bool parse_fill(const std::string& f) {
            if (f == "touch") return false;
            if (f == "breach") return true;
            throw std::invalid_argument("fill must be \"touch\" or \"breach\".");
        }
        static double parse_participation(double p) {
            if (!(p > 0.0) || p > 1.0)
                throw std::invalid_argument("participation_ratio must be in (0, 1].");
            return p;
        }
        detail::OHLCFill core_;
    };

} // namespace screamer

#endif // SCREAMER_BACKTEST_OHLC_ORDERS_H
```

- [ ] **Step 5: Create `include/screamer/backtest_ohlc_target.h`** (deferred market-to-target over the same core):

```cpp
#ifndef SCREAMER_BACKTEST_OHLC_TARGET_H
#define SCREAMER_BACKTEST_OHLC_TARGET_H

#include <limits>
#include <tuple>
#include "screamer/common/functor_base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/ohlc_fill.h"
#include "screamer/detail/target_front.h"

namespace screamer {

    // BacktestOHLCTarget: 5 -> 4. Reach a target position on OHLC bars by taking
    // liquidity. Causal: the target decided on bar t (from its close) executes at bar
    // t+1's open as a market order, sized to clamp(target, cap) - position. Inventory
    // is capped to [min_position, max_position]. Outputs [equity, pnl, position, cost].
    class BacktestOHLCTarget : public FunctorBase<BacktestOHLCTarget, 5, 4> {
    public:
        BacktestOHLCTarget(double taker_fee = 0.0, double tick_size = 0.0,
                           double min_position = -std::numeric_limits<double>::infinity(),
                           double max_position = std::numeric_limits<double>::infinity())
            : core_(0.0, taker_fee, /*breach=*/false, /*participation=*/1.0, tick_size,
                    min_position, max_position),
              min_(min_position), max_(max_position) { reset(); }
        void reset() override { core_.reset(); has_pending_ = false; pending_ = 0.0; }
        ResultTuple call(const InputArray& in) override {
            const double target = in[0];
            const double open = in[1], high = in[2], low = in[3], close = in[4];
            double bp = std::numeric_limits<double>::infinity(), bs = 0.0;
            double ap = -std::numeric_limits<double>::infinity(), as = 0.0;
            if (has_pending_) {
                auto [b_p, b_s, a_p, a_s] =
                    detail::target_to_quote(pending_, core_.position(), min_, max_);
                bp = b_p; bs = b_s; ap = a_p; as = a_s;
            }
            auto out = core_.quote(bp, bs, ap, as, open, high, low, close);
            if (isnan2(target)) has_pending_ = false;       // hold if no target
            else { has_pending_ = true; pending_ = target; }
            return out;
        }
    private:
        detail::OHLCFill core_;
        double min_, max_;
        bool has_pending_ = false;
        double pending_ = 0.0;
    };

} // namespace screamer

#endif // SCREAMER_BACKTEST_OHLC_TARGET_H
```

Delete `include/screamer/backtest_ohlc.h` and `include/screamer/backtest_ohlc_maker.h`.

- [ ] **Step 6: Update the bindings** in `bindings/bindings_fin.cpp`: replace the `#include`s for `backtest_ohlc.h` / `backtest_ohlc_maker.h` with `backtest_ohlc_orders.h` / `backtest_ohlc_target.h`, and replace the `BacktestOHLC` + `BacktestOHLCMaker` class blocks with:

```cpp
py::class_<screamer::BacktestOHLCOrders, screamer::EvalOp>(m, "BacktestOHLCOrders")
    .def(py::init<double, double, const std::string&, double, double, double, double>(),
         py::arg("maker_fee") = 0.0, py::arg("taker_fee") = 0.0,
         py::arg("fill") = "touch", py::arg("participation_ratio") = 1.0,
         py::arg("tick_size") = 0.0,
         py::arg("min_position") = -std::numeric_limits<double>::infinity(),
         py::arg("max_position") = std::numeric_limits<double>::infinity())
    .def("__call__", &screamer::BacktestOHLCOrders::handle_input)
    .def("reset", &screamer::BacktestOHLCOrders::reset, "Reset.");

py::class_<screamer::BacktestOHLCTarget, screamer::EvalOp>(m, "BacktestOHLCTarget")
    .def(py::init<double, double, double, double>(),
         py::arg("taker_fee") = 0.0, py::arg("tick_size") = 0.0,
         py::arg("min_position") = -std::numeric_limits<double>::infinity(),
         py::arg("max_position") = std::numeric_limits<double>::infinity())
    .def("__call__", &screamer::BacktestOHLCTarget::handle_input)
    .def("reset", &screamer::BacktestOHLCTarget::reset, "Reset.");
```

- [ ] **Step 7: Update `tests/param_cases.py`**: replace any `BacktestOHLCMaker` / `BacktestOHLC` entries in `_NO_ARG_AUTO_EXCLUDE` with `BacktestOHLCOrders` and `BacktestOHLCTarget` (both take args / multi-output).

- [ ] **Step 8: Rename the docs pages** `git mv docs/functions_fin/BacktestOHLCMaker.md docs/functions_fin/BacktestOHLCOrders.md` and `git mv docs/functions_fin/BacktestOHLC.md docs/functions_fin/BacktestOHLCTarget.md`; update each frontmatter `name`/`title`, the input list (Orders 8 inputs; Target 5 inputs `(target_position, open, high, low, close)`), the parameter list (Target: `taker_fee, tick_size, min_position, max_position`), and the prose. Keep `nan_policy: ignore`, `topics: [backtesting]`.

- [ ] **Step 9: Rebuild, regen, run**

Run: `make install-dev && poetry run python devtools/build_help_registry.py && poetry run python devtools/build_topic_pages.py && make regen-init && poetry run python -m pytest tests/test_backtest.py -q -k "ohlc"`
Expected: PASS (the three new tests; no stale `BacktestOHLC*`/`Maker` references remain).

- [ ] **Step 10: Commit**

```bash
git add -A && git commit -m "feat(backtest): OHLC core + BacktestOHLCOrders/BacktestOHLCTarget"
```

---

## Task 4: Trades core + `BacktestTradesOrders` + `BacktestTradesTarget`

**Files:**
- Create: `include/screamer/detail/trades_fill.h`, `include/screamer/backtest_trades_orders.h`, `include/screamer/backtest_trades_target.h`
- Delete: `include/screamer/backtest_trades.h`, `include/screamer/backtest_trades_maker.h`
- Modify: `bindings/bindings_fin.cpp`, `tests/param_cases.py`
- Docs: `git mv docs/functions_fin/BacktestTradesMaker.md -> BacktestTradesOrders.md`; delete `docs/functions_fin/BacktestTrades.md` and create `BacktestTradesTarget.md`
- Test: `tests/test_backtest.py`

**Interfaces:**
- Produces:
  - `detail::TradesFill(maker_fee, taker_fee, breach, participation, tick_size, min_pos, max_pos)` with `reset()`, `position()`, `std::tuple<...> quote(double bid_price, double bid_size, double ask_price, double ask_size, double trade_price, double trade_size)`.
  - `BacktestTradesOrders(maker_fee=0.0, taker_fee=0.0, fill="touch", participation_ratio=1.0, tick_size=0.0, min_position=-inf, max_position=+inf)`, 6 inputs `(bid_price, bid_size, ask_price, ask_size, trade_price, trade_size)`.
  - `BacktestTradesTarget(taker_fee=0.0, tick_size=0.0, min_position=-inf, max_position=+inf)`, 3 inputs `(target_position, trade_price, trade_size)`; each print, size `clamp(target,cap)-position` and take it at the print.

- [ ] **Step 1: Write the failing tests**

```python
def test_trades_orders_one_sided_equals_resting_limit():
    import numpy as np
    from screamer import BacktestTradesOrders
    # a resting bid at 100 size 5; a sell-print at 99 sweeps it -> buy 5
    out = BacktestTradesOrders()(
        np.array([100.]), np.array([5.]), np.array([np.nan]), np.array([0.]),
        np.array([99.]), np.array([8.]))
    assert out[0, 2] == 5.0

def test_trades_target_takes_prints_to_reach_target():
    import numpy as np
    from screamer import BacktestTradesTarget
    # target +2 taken against the print; participation caps to the print size where needed
    out = BacktestTradesTarget()(
        np.array([2., 2.]), np.array([100., 100.]), np.array([10., 10.]))
    assert out[-1, 2] == 2.0

def test_trades_target_capped():
    import numpy as np
    from screamer import BacktestTradesTarget
    out = BacktestTradesTarget(max_position=1.0)(
        np.array([9.]), np.array([100.]), np.array([100.]))
    assert out[0, 2] == 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `poetry run python -m pytest tests/test_backtest.py -q -k "trades_orders or trades_target"`
Expected: FAIL.

- [ ] **Step 3: Create `include/screamer/detail/trades_fill.h`** by refactoring the reviewed `include/screamer/backtest_trades_maker.h` (commit `f7a8da1`) into `struct TradesFill` in `screamer::detail`, exactly as Task 3 refactored the OHLC maker: constructor takes the parsed values, `call` becomes `quote(double bid_price, double bid_size, double ask_price, double ask_size, double trade_price, double trade_size)` reading named args, the two-sided crossing-print fill + room cap + mark-to-trade fallback kept byte-for-byte; expose `reset()`, `position()`, public `account_`.

- [ ] **Step 4: Create `include/screamer/backtest_trades_orders.h`** (thin wrapper), mirroring Task 3's `backtest_ohlc_orders.h` but `FunctorBase<BacktestTradesOrders, 6, 4>`, forwarding `in[0..5]` to `core_.quote`, with the same `parse_fill`/`parse_participation` and constructor arg list. Include `screamer/detail/trades_fill.h`.

- [ ] **Step 5: Create `include/screamer/backtest_trades_target.h`** (immediate market-to-target, no deferral):

```cpp
#ifndef SCREAMER_BACKTEST_TRADES_TARGET_H
#define SCREAMER_BACKTEST_TRADES_TARGET_H

#include <limits>
#include <tuple>
#include "screamer/common/functor_base.h"
#include "screamer/detail/trades_fill.h"
#include "screamer/detail/target_front.h"

namespace screamer {

    // BacktestTradesTarget: 3 -> 4. Reach a target position on the trade tape by
    // taking prints. Each event, size clamp(target, cap) - position and submit it as
    // a marketable order that the current print fills (taker). Inventory capped to
    // [min_position, max_position]. Outputs [equity, pnl, position, cost].
    class BacktestTradesTarget : public FunctorBase<BacktestTradesTarget, 3, 4> {
    public:
        BacktestTradesTarget(double taker_fee = 0.0, double tick_size = 0.0,
                             double min_position = -std::numeric_limits<double>::infinity(),
                             double max_position = std::numeric_limits<double>::infinity())
            : core_(0.0, taker_fee, /*breach=*/false, /*participation=*/1.0, tick_size,
                    min_position, max_position),
              min_(min_position), max_(max_position) {}
        void reset() override { core_.reset(); }
        ResultTuple call(const InputArray& in) override {
            const double target = in[0], trade_price = in[1], trade_size = in[2];
            auto [bp, bs, ap, as] =
                detail::target_to_quote(target, core_.position(), min_, max_);
            return core_.quote(bp, bs, ap, as, trade_price, trade_size);
        }
    private:
        detail::TradesFill core_;
        double min_, max_;
    };

} // namespace screamer

#endif // SCREAMER_BACKTEST_TRADES_TARGET_H
```

Delete `include/screamer/backtest_trades.h` and `include/screamer/backtest_trades_maker.h`.

- [ ] **Step 6: Update the bindings**: swap the includes; replace the `BacktestTrades` + `BacktestTradesMaker` blocks with `BacktestTradesOrders` (7 args, like OHLCOrders) and `BacktestTradesTarget` (4 args: `taker_fee, tick_size, min_position, max_position`), following Task 3's binding shape.

- [ ] **Step 7: Update `tests/param_cases.py`**: replace `BacktestTradesMaker`/`BacktestTrades` in `_NO_ARG_AUTO_EXCLUDE` with `BacktestTradesOrders` and `BacktestTradesTarget`.

- [ ] **Step 8: Docs**: `git mv docs/functions_fin/BacktestTradesMaker.md docs/functions_fin/BacktestTradesOrders.md` (update name/title/prose, 6 inputs); `git rm docs/functions_fin/BacktestTrades.md`; create `docs/functions_fin/BacktestTradesTarget.md` (frontmatter: `name`, `title`, 3 inputs `(target_position, trade_price, trade_size)`, params `taker_fee, tick_size, min_position, max_position`, `nan_policy: ignore`, `topics: [backtesting]`, a plotly example modeled on `BacktestTradesOrders.md`).

- [ ] **Step 9: Rebuild, regen, run**

Run: `make install-dev && poetry run python devtools/build_help_registry.py && poetry run python devtools/build_topic_pages.py && make regen-init && poetry run python -m pytest tests/test_backtest.py -q -k "trades"`
Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add -A && git commit -m "feat(backtest): Trades core + BacktestTradesOrders/BacktestTradesTarget"
```

---

## Task 5: L1 core + `BacktestL1Orders` + `BacktestL1Target`

**Files:**
- Create: `include/screamer/detail/l1_fill.h`, `include/screamer/backtest_l1_orders.h`, `include/screamer/backtest_l1_target.h`
- Delete: `include/screamer/backtest_l1.h`
- Modify: `bindings/bindings_fin.cpp`, `tests/param_cases.py`
- Docs: `git mv docs/functions_fin/BacktestL1.md -> BacktestL1Orders.md`; create `BacktestL1Target.md`
- Test: `tests/test_backtest.py`

**Interfaces:**
- Produces:
  - `detail::L1Fill(maker_fee, taker_fee, breach, participation, tick_size, min_pos, max_pos)` with `reset()`, `position()`, `std::tuple<...> quote(double bid_price, double bid_size, double ask_price, double ask_size, double market_bid, double market_ask, double market_bid_size, double market_ask_size)`.
  - `BacktestL1Orders(maker_fee=0.0, taker_fee=0.0, fill="breach", participation_ratio=1.0, tick_size=0.0, min_position=-inf, max_position=+inf)`, 8 inputs `(bid_price, bid_size, ask_price, ask_size, market_bid, market_ask, market_bid_size, market_ask_size)`. (Preserve `BacktestL1`'s current default `fill="breach"`.)
  - `BacktestL1Target(taker_fee=0.0, tick_size=0.0, min_position=-inf, max_position=+inf)`, 5 inputs `(target_position, market_bid, market_ask, market_bid_size, market_ask_size)`; each event, take toward `clamp(target,cap)` against the book.

- [ ] **Step 1: Write the failing tests**

```python
def test_l1_orders_parity_resting_quote():
    import numpy as np
    from screamer import BacktestL1Orders
    # resting bid at 100 fills when the market ask drops to it (breach default)
    out = BacktestL1Orders()(
        np.array([100.]), np.array([1.]), np.array([np.nan]), np.array([0.]),
        np.array([100.5]), np.array([99.9]), np.array([5.]), np.array([5.]))
    assert out[0, 2] == 1.0

def test_l1_target_takes_book_to_reach_target():
    import numpy as np
    from screamer import BacktestL1Target
    # target +1 taken against the displayed ask
    out = BacktestL1Target()(
        np.array([1.]), np.array([100.]), np.array([100.1]),
        np.array([5.]), np.array([5.]))
    assert out[0, 2] == 1.0

def test_l1_target_capped():
    import numpy as np
    from screamer import BacktestL1Target
    out = BacktestL1Target(max_position=1.0)(
        np.array([9.]), np.array([100.]), np.array([100.1]),
        np.array([50.]), np.array([50.]))
    assert out[0, 2] == 1.0
```

(Adjust the expected values in Step 6 if the parity port marks/fees differently; the invariant asserted here is fill direction and the cap.)

- [ ] **Step 2: Run to verify failure**

Run: `poetry run python -m pytest tests/test_backtest.py -q -k "l1_orders or l1_target"`
Expected: FAIL.

- [ ] **Step 3: Create `include/screamer/detail/l1_fill.h`** by refactoring `include/screamer/backtest_l1.h` into `struct L1Fill` in `screamer::detail`: constructor takes the parsed values, `call` becomes `quote(double bid_price, double bid_size, double ask_price, double ask_size, double market_bid, double market_ask, double market_bid_size, double market_ask_size)` reading named args; keep the resting-vs-marketable fill, `tick_size` overflow, room cap, and mark logic byte-for-byte; expose `reset()`, `position()`, public `account_`.

- [ ] **Step 4: Create `include/screamer/backtest_l1_orders.h`** (thin wrapper, `FunctorBase<BacktestL1Orders, 8, 4>`, default `fill="breach"`), mirroring Task 3's Orders wrapper but forwarding `in[0..7]` to `core_.quote` and including `screamer/detail/l1_fill.h`.

- [ ] **Step 5: Create `include/screamer/backtest_l1_target.h`** (immediate market-to-target over the L1 core), mirroring `backtest_trades_target.h` but `FunctorBase<BacktestL1Target, 5, 4>`, reading `(target, market_bid, market_ask, market_bid_size, market_ask_size)` and calling `core_.quote(bp, bs, ap, as, market_bid, market_ask, market_bid_size, market_ask_size)`. Delete `include/screamer/backtest_l1.h`.

- [ ] **Step 6: Update the bindings**: swap the include; replace the `BacktestL1` block with `BacktestL1Orders` (7 args, `fill="breach"` default) and add `BacktestL1Target` (4 args). Run the tests from Step 1 and, if a parity value differs, correct the test's expected number to the hand-verified fill (keep the direction + cap assertions).

- [ ] **Step 7: Update `tests/param_cases.py`**: replace `BacktestL1` with `BacktestL1Orders` and add `BacktestL1Target` to `_NO_ARG_AUTO_EXCLUDE`.

- [ ] **Step 8: Docs**: `git mv docs/functions_fin/BacktestL1.md docs/functions_fin/BacktestL1Orders.md` (update name/title/prose); create `docs/functions_fin/BacktestL1Target.md` (frontmatter: 5 inputs, params `taker_fee, tick_size, min_position, max_position`, `nan_policy: ignore`, `topics: [backtesting]`, a plotly example).

- [ ] **Step 9: Rebuild, regen, run**

Run: `make install-dev && poetry run python devtools/build_help_registry.py && poetry run python devtools/build_topic_pages.py && make regen-init && poetry run python -m pytest tests/test_backtest.py -q -k "l1_"`
Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add -A && git commit -m "feat(backtest): L1 core + BacktestL1Orders/BacktestL1Target"
```

---

## Task 6: `BacktestL1TradesOrders` (rename of `BacktestL1Trades`)

**Files:**
- Create: `include/screamer/backtest_l1trades_orders.h`; Delete: `include/screamer/backtest_l1_trades.h`
- Modify: `bindings/bindings_fin.cpp`, `tests/param_cases.py`
- Docs: `git mv docs/functions_fin/BacktestL1Trades.md -> BacktestL1TradesOrders.md`
- Test: `tests/test_backtest.py`

**Interfaces:**
- Produces: `BacktestL1TradesOrders(maker_fee=0.0, taker_fee=0.0, fill="touch", participation_ratio=1.0, tick_size=0.0, min_position=-inf, max_position=+inf)`, 10 inputs `(bid_price, bid_size, ask_price, ask_size, market_bid, market_ask, market_bid_size, market_ask_size, trade_price, trade_size)`. Identical behavior to `BacktestL1Trades`.

- [ ] **Step 1: Write the failing test**

```python
def test_l1trades_orders_exists_and_fills():
    import numpy as np
    from screamer import BacktestL1TradesOrders
    # resting bid filled when a trade explains the cross
    out = BacktestL1TradesOrders()(
        np.array([100.]), np.array([1.]), np.array([np.nan]), np.array([0.]),
        np.array([100.5]), np.array([99.9]), np.array([5.]), np.array([5.]),
        np.array([99.9]), np.array([3.]))
    assert out[0, 2] == 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `poetry run python -m pytest tests/test_backtest.py -q -k l1trades_orders_exists`
Expected: FAIL.

- [ ] **Step 3: Create `include/screamer/backtest_l1trades_orders.h`** by copying `include/screamer/backtest_l1_trades.h` verbatim, renaming the class `BacktestL1Trades` -> `BacktestL1TradesOrders`, the include guard, and the docstring name. Keep the logic identical. Delete `include/screamer/backtest_l1_trades.h`. (This row has one engine, so no core split.)

- [ ] **Step 4: Update the bindings**: swap the include; rename the `BacktestL1Trades` class block to `BacktestL1TradesOrders` (same 7 `py::arg`s).

- [ ] **Step 5: Update `tests/param_cases.py`**: replace `BacktestL1Trades` with `BacktestL1TradesOrders`.

- [ ] **Step 6: Docs**: `git mv docs/functions_fin/BacktestL1Trades.md docs/functions_fin/BacktestL1TradesOrders.md`; update frontmatter `name`/`title` and prose.

- [ ] **Step 7: Rebuild, regen, run**

Run: `make install-dev && poetry run python devtools/build_help_registry.py && poetry run python devtools/build_topic_pages.py && make regen-init && poetry run python -m pytest tests/test_backtest.py -q -k l1trades`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add -A && git commit -m "feat(backtest): BacktestL1TradesOrders (renames BacktestL1Trades)"
```

---

## Task 7: grid docs page, changelog, and full verification

**Files:**
- Rewrite: `docs/functions_fin/choosing_a_backtest_engine.md`, `CHANGELOG.md`
- Sweep: any remaining references to old engine names in `docs/`, notebooks under `docs/notebooks/`, and `tests/`.

- [ ] **Step 1: Grep for stale names**

Run: `grep -rn "BacktestSignal\|BacktestOHLCMaker\|BacktestTradesMaker\|BacktestL1Trades\b\|BacktestOHLC\b\|BacktestTrades\b\|BacktestL1\b" docs/ tests/ screamer/ | grep -v "Orders\|Target"`
Expected: only intended hits. Update every stale reference to the new grid name (e.g. `BacktestSignal` -> `BacktestPriceTarget`, `BacktestL1` -> `BacktestL1Orders`).

- [ ] **Step 2: Rewrite `docs/functions_fin/choosing_a_backtest_engine.md`** with the grid: a 5x2 table (rows = data model, cols = `Target`/`Orders`) naming each engine, the two empty cells marked "not provided" with the redirect (`PriceOrders` -> use OHLC/L1 Orders; `L1TradesTarget` -> use `BacktestL1Target`), the two order-definition interfaces (Target = `(target_position, ...)` market to a position; Orders = `(bid_price, bid_size, ask_price, ask_size, ...)`), the `MARKET`/`NaN`/`inf` encoding table, and the static fill-cap three-way-minimum rule. Plain language, no em-dashes, `topics: [backtesting]` frontmatter.

- [ ] **Step 3: Rewrite the `[Unreleased]` backtest changelog** in `CHANGELOG.md` to describe the grid: the engines are renamed to `Backtest<DataModel><OrderDef>`; every engine takes the static `[min_position, max_position]` cap and the `MARKET` encoding; `Target` engines reach a position by taking liquidity, `Orders` engines post two-sided quotes; new `BacktestL1Target` and `BacktestTradesTarget` complete the useful cells; `BacktestPriceOrders` and `BacktestL1TradesTarget` are intentionally not provided.

- [ ] **Step 4: Full regen + suite**

Run:
```bash
poetry run python devtools/build_help_registry.py
poetry run python devtools/build_topic_pages.py
make regen-init
poetry run python -m pytest -q
```
Expected: help registry validates all eight engine pages + the grid page; the whole suite passes (the eight engines' tests, the NaN-policy/param sweeps picking up the new names, no import errors from removed engines).

- [ ] **Step 5: Docs build**

Run: `make docs`
Expected: exit 0; the eight engine pages and the grid page render and are homed; no new orphan warnings.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "docs(backtest): engine grid page + changelog; sweep old names"
```

---

## Self-Review

**Spec coverage:**
- Naming grid (`Backtest<DataModel><OrderDef>`, eight engines, two empty cells): Tasks 2-6 build the eight and remove the old headers; Task 1 the front; Task 7 the grid docs + empty-cell redirects. Covered.
- `Target` market-only, clamps target to cap: `target_to_quote` (Task 1) clamps and emits a marketable one-sided order; each Target engine (Tasks 3-5) uses it. Covered.
- `Orders` two-sided, one-sided by zeroing a side: Orders wrappers forward the four quote columns to the core (Tasks 3-5); the core is the reviewed maker logic that already handles a zeroed side. Covered.
- `MARKET` encoding: `market_limit` reused inside the cores (unchanged from the reviewed makers); Target submits `+inf`/`-inf` via the front. Covered.
- Static fill cap everywhere: carried by the cores and the Price engine (all from the reviewed cap work); Target clamps in the front. Covered.
- Output schema + `PnLAccount`/`backtest_report`/`BacktestReport` unchanged: cores wrap `PnLAccount`; no task touches those files. Covered.
- Fill cores one-per-data-model reused by both order defs: Tasks 3-5 create `OHLCFill`/`TradesFill`/`L1Fill`, each used by its Orders and Target engine. Price and L1Trades stay direct (single-engine rows), per YAGNI. Covered.
- Docs grid + per-cell pages + changelog + notebook/name sweep: Tasks 2-7. Covered.
- SchmittTrigger follow-up: explicitly out of scope in the spec; no task. Correct.

**Placeholder scan:** No TBD/TODO. Core-creation steps name the exact reviewed source header + commit to refactor and show the struct's public interface, the full thin-wrapper code, the full bindings, and full tests. The one deliberately-red test (Task 1 Step 2) is documented as turning green in Task 2.

**Type consistency:** Every core exposes the same trio (`reset()`, `position()`, `quote(...)`) and a public `account_`. Constructors across the Orders wrappers share the arg list `(maker_fee, taker_fee, fill, participation_ratio, tick_size, min_position, max_position)`; Target wrappers share `(taker_fee, tick_size, min_position, max_position)`. `target_to_quote(target, position, min_pos, max_pos)` has one signature, used by all three Target engines. Input arities: PriceTarget 2, OHLCTarget 5, OHLCOrders 8, TradesTarget 3, TradesOrders 6, L1Target 5, L1Orders 8, L1TradesOrders 10 - each matches its binding `py::init` and its tests.

---

## Notes for the implementer

- The three cores are mechanical refactors of already-reviewed headers (the two makers at `79dffb1`/`f7a8da1`, and `backtest_l1.h`), so parity is preserved by construction; the per-engine tests plus the full suite are the guardrail.
- The only genuinely new logic is `target_to_quote` (Task 1) and the three Target wrappers that drive a core with a marketable one-sided order. `BacktestOHLCTarget` is the one with deferral (decide on close, execute next open); `BacktestTradesTarget` and `BacktestL1Target` are immediate.
- Keep default behavior of the renamed engines identical: `BacktestL1Orders` keeps `fill="breach"`; the Orders defaults match their maker origins; `BacktestPriceTarget` keeps `BacktestSignal`'s `spread`/`fee` model.
- `make docs` executes notebooks; it is the final gate in Task 7.
