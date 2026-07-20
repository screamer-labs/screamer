# Unified Backtest Engine API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the backtest engines uniform: a static `[min, max]` fill cap and the `NaN`/`inf` market-order encoding on every engine, and add the missing market-making engines on OHLC bars and the trade tape.

**Architecture:** Each engine stays a header-only `FunctorBase<Derived, N, 4>` emitting `[equity, pnl, position, cost]` through `detail::PnLAccount`. Directional engines (`BacktestSignal`, `BacktestOHLC`, `BacktestTrades`) gain a static fill cap. Two new engines (`BacktestOHLCMaker`, `BacktestTradesMaker`) add two-sided market-making on bars and the tape, mirroring `BacktestL1`'s per-side fill + cap structure. A shared `market_price` helper and a Python `MARKET` constant standardize the market-order encoding.

**Tech Stack:** C++17 header-only core (`include/screamer/`), pybind11 bindings (`bindings/bindings_fin.cpp`), Python packaging/regeneration, pytest.

## Global Constraints

- Every engine returns `[equity, pnl, position, cost]` and shares `detail::PnLAccount`. Do not change `PnLAccount`, `backtest_report`, or `BacktestReport`.
- **Market order = a limit at the maximally aggressive price.** On a buy, `+inf` fills always, `-inf` never; on a sell, the mirror. `NaN` is the side-agnostic market shorthand: substitute `+inf` on a buy, `-inf` on a sell. Wrong-direction infinities are harmless never-fill limits. This encoding is a shared helper, not per-engine code.
- **Static fill cap:** `min_position` (default `-inf`), `max_position` (default `+inf`). A fill is the minimum of the order size, the counterparty volume, and the room to the cap (`max_position - position` buying, `position - min_position` selling). Constructor throws if `min_position > max_position`.
- **Keep the current engine names.** New engines are `BacktestOHLCMaker` and `BacktestTradesMaker`.
- `screamer.MARKET` is a Python constant backed by `float("inf")`; `NaN`, `+inf`, `-inf` are all accepted market spellings.
- `participation_ratio` is in `(0, 1]`; `fill` is `"touch"` or `"breach"`; `tick_size >= 0`; `spread >= 0`.
- After any C++ change: `make install-dev`, then `poetry run python devtools/build_help_registry.py`, `poetry run python devtools/build_topic_pages.py`, `make regen-init`. Build docs with `make docs` (executes notebooks + examples).
- No em-dashes in docs (ASCII hyphens). Commit as `simu.ai <claude@sitmo.com>` with the standard `Co-Authored-By` + `Claude-Session` footer. Do not push.
- Every new documented functor needs a `docs/functions_fin/<Name>.md` page with frontmatter (`nan_policy`, a `topics:` slug), or `test_doc_coverage.py` fails.

---

## File Structure

- `include/screamer/common/market_price.h` (create) - the `market_limit(price, buy)` encoding helper.
- `include/screamer/backtest_signal.h` (modify) - add the static cap (clamp target).
- `include/screamer/backtest_ohlc.h` (modify) - add the static cap (clamp target).
- `include/screamer/backtest_trades.h` (modify) - add the static cap (truncate fill to room).
- `include/screamer/backtest_ohlc_maker.h` (create) - two-sided market-making on bars.
- `include/screamer/backtest_trades_maker.h` (create) - two-sided market-making on the tape.
- `screamer/backtest.py` (modify) - add the `MARKET` constant to the package.
- `bindings/bindings_fin.cpp` (modify) - new params on the modified engines, register the two new engines.
- `docs/functions_fin/BacktestOHLCMaker.md`, `BacktestTradesMaker.md` (create).
- `docs/functions_fin/choosing_a_backtest_engine.md` (create) - the coverage matrix.
- `tests/test_backtest.py` (modify), `tests/param_cases.py` (modify).
- `CHANGELOG.md` (modify).

---

## Task 1: the market-order encoding helper + `MARKET` constant

**Files:**
- Create: `include/screamer/common/market_price.h`
- Modify: `screamer/backtest.py`
- Test: `tests/test_backtest.py`

**Interfaces:**
- Produces: `screamer::market_limit(double price, bool buy) -> double` returning the effective limit (NaN maps to `+inf` for a buy, `-inf` for a sell; finite and inf pass through). Python `screamer.MARKET == float("inf")`.

- [ ] **Step 1: Write the failing test**

```python
def test_market_constant_and_encoding():
    import math
    from screamer import MARKET, BacktestSignal
    assert MARKET == math.inf
    # a directional market buy to +1 via MARKET fills fully at the price (frictionless)
    import numpy as np
    out = BacktestSignal()(np.array([1., 1]), np.array([100., 101.]))
    assert out[1, 2] == 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `poetry run python -m pytest tests/test_backtest.py -q -k market_constant`
Expected: FAIL (`cannot import name 'MARKET'`).

- [ ] **Step 3: Create `include/screamer/common/market_price.h`**

```cpp
#ifndef SCREAMER_COMMON_MARKET_PRICE_H
#define SCREAMER_COMMON_MARKET_PRICE_H

#include <limits>
#include "screamer/common/float_info.h"

namespace screamer {

    // A market order is a limit at the maximally aggressive price. A NaN price is
    // the side-agnostic market shorthand: it becomes +inf on a buy (clears any
    // offer) and -inf on a sell (hits any bid). Finite and +/-inf prices pass
    // through unchanged, so the normal fill comparison turns a +inf buy into a
    // market fill and a -inf buy into a never-fill limit.
    inline double market_limit(double price, bool buy) {
        if (isnan2(price)) {
            return buy ? std::numeric_limits<double>::infinity()
                       : -std::numeric_limits<double>::infinity();
        }
        return price;
    }

} // namespace screamer

#endif // SCREAMER_COMMON_MARKET_PRICE_H
```

- [ ] **Step 4: Add the `MARKET` constant** in `screamer/backtest.py` (after the imports, before `backtest_report`)

```python
import math

#: Market-order price sentinel for the backtest engines. A quote or limit price of
#: ``MARKET`` (or any non-finite price) is a market order in the aggressive
#: direction; ``NaN`` works as a side-agnostic shorthand.
MARKET = math.inf
```

Add `"MARKET"` to `__all__`: change `__all__ = ["backtest_report"]` to `__all__ = ["backtest_report", "MARKET"]`.

- [ ] **Step 5: Rebuild, regen, and run**

Run: `make install-dev && make regen-init && poetry run python -m pytest tests/test_backtest.py -q -k market_constant`
Expected: PASS. `make regen-init` picks up `MARKET` into `screamer/__init__.py`.

- [ ] **Step 6: Commit**

```bash
git add include/screamer/common/market_price.h screamer/backtest.py screamer/__init__.py tests/test_backtest.py
git commit -m "feat(backtest): market_limit encoding helper + MARKET constant"
```

---

## Task 2: static fill cap on `BacktestSignal`

**Files:**
- Modify: `include/screamer/backtest_signal.h`
- Modify: `bindings/bindings_fin.cpp:180-181`
- Test: `tests/test_backtest.py`

**Interfaces:**
- Produces: `BacktestSignal(spread=0.0, fee=0.0, min_position=-inf, max_position=+inf)`. A target beyond the cap is clamped.

- [ ] **Step 1: Write the failing test**

```python
def test_signal_position_cap_clamps_target():
    from screamer import BacktestSignal
    import numpy as np
    out = BacktestSignal(max_position=1.0, min_position=-1.0)(
        np.array([5., -5., 0.]), np.array([100., 100., 100.]))
    np.testing.assert_allclose(out[:, 2], [1, -1, 0])   # target 5 clamped to 1, -5 to -1
```

- [ ] **Step 2: Run to verify failure**

Run: `poetry run python -m pytest tests/test_backtest.py -q -k signal_position_cap`
Expected: FAIL (`__init__` has no `max_position`).

- [ ] **Step 3: Add the cap** in `include/screamer/backtest_signal.h`

Add `#include <algorithm>` and `#include <limits>` (limits already present). Change the constructor:

```cpp
BacktestSignal(double spread = 0.0, double fee = 0.0,
               double min_position = -std::numeric_limits<double>::infinity(),
               double max_position = std::numeric_limits<double>::infinity())
    : spread_(spread), fee_(fee),
      min_position_(min_position), max_position_(max_position)
{
    if (spread_ < 0.0) throw std::invalid_argument("spread must be non-negative.");
    if (min_position_ > max_position_)
        throw std::invalid_argument("min_position must not exceed max_position.");
}
```

In `call`, clamp the target before computing `dpos`:

```cpp
const double target = std::clamp(signal, min_position_, max_position_);
const double dpos = target - account_.position();
```

Add members: `double min_position_; double max_position_;`.

- [ ] **Step 4: Update the binding** in `bindings/bindings_fin.cpp`

```cpp
py::class_<screamer::BacktestSignal, screamer::EvalOp>(m, "BacktestSignal")
    .def(py::init<double, double, double, double>(),
         py::arg("spread") = 0.0, py::arg("fee") = 0.0,
         py::arg("min_position") = -std::numeric_limits<double>::infinity(),
         py::arg("max_position") = std::numeric_limits<double>::infinity())
    .def("__call__", &screamer::BacktestSignal::handle_input)
    .def("reset", &screamer::BacktestSignal::reset, "Reset.");
```

- [ ] **Step 5: Rebuild and run**

Run: `make install-dev && poetry run python -m pytest tests/test_backtest.py -q -k "signal"`
Expected: PASS (new cap test + the existing `test_frictionless*`, `test_taker_cost*`, etc. still pass; the default cap is unbounded so old behavior is unchanged).

- [ ] **Step 6: Update the docs frontmatter** in `docs/functions_fin/BacktestSignal.md`: add `min_position` and `max_position` to the `parameters:` list (defaults `-.inf` / `.inf`, a one-line description each), and a sentence in the Description that a target beyond `[min_position, max_position]` is clamped. Run `poetry run python devtools/build_help_registry.py` to confirm the frontmatter validates.

- [ ] **Step 7: Commit**

```bash
git add include/screamer/backtest_signal.h bindings/bindings_fin.cpp docs/functions_fin/BacktestSignal.md screamer/data/help.json tests/test_backtest.py
git commit -m "feat(backtest): static position cap on BacktestSignal"
```

---

## Task 3: static fill cap on `BacktestOHLC`

**Files:**
- Modify: `include/screamer/backtest_ohlc.h`
- Modify: `bindings/bindings_fin.cpp` (the `BacktestOHLC` binding)
- Test: `tests/test_backtest.py`

**Interfaces:**
- Produces: `BacktestOHLC(spread=0.0, taker_fee=0.0, maker_fee=0.0, fill="touch", min_position=-inf, max_position=+inf)`. The deferred target is clamped to the cap before the fill.

- [ ] **Step 1: Write the failing test**

```python
def test_ohlc_position_cap_clamps_target():
    from screamer import BacktestOHLC
    import numpy as np
    # target 5 decided on bar 0, executed on bar 1 (deferred), clamped to max 1
    out = BacktestOHLC(max_position=1.0)(
        np.array([5., 5.]), np.array([np.nan, np.nan]),
        np.array([100., 100.]), np.array([101., 101.]),
        np.array([100., 100.]), np.array([100., 100.]))
    assert out[1, 2] == 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `poetry run python -m pytest tests/test_backtest.py -q -k ohlc_position_cap`
Expected: FAIL (no `max_position`).

- [ ] **Step 3: Add the cap** in `include/screamer/backtest_ohlc.h`

Add `#include <algorithm>`. Add `min_position`/`max_position` to the constructor (same pattern as Task 2, with the `min_position_ > max_position_` guard), add the two members. In `call`, where the deferred target is applied, clamp it:

```cpp
const double target = std::clamp(pending_target_, min_position_, max_position_);
const double dpos = target - account_.position();
```

(Replace the existing `const double dpos = pending_target_ - account_.position();`.)

- [ ] **Step 4: Update the binding** to add `py::arg("min_position")` / `py::arg("max_position")` after `fill`, with the infinity defaults, and widen the `py::init<...>` to `<double, double, double, const std::string&, double, double>`.

- [ ] **Step 5: Rebuild and run**

Run: `make install-dev && poetry run python -m pytest tests/test_backtest.py -q -k "ohlc"`
Expected: PASS (new cap test + all existing OHLC tests unchanged under the default unbounded cap).

- [ ] **Step 6: Docs frontmatter** in `docs/functions_fin/BacktestOHLC.md`: add `min_position`/`max_position` params + a Description sentence. Run `build_help_registry.py` to validate.

- [ ] **Step 7: Commit**

```bash
git add include/screamer/backtest_ohlc.h bindings/bindings_fin.cpp docs/functions_fin/BacktestOHLC.md screamer/data/help.json tests/test_backtest.py
git commit -m "feat(backtest): static position cap on BacktestOHLC"
```

---

## Task 4: static fill cap on `BacktestTrades`

**Files:**
- Modify: `include/screamer/backtest_trades.h`
- Modify: `bindings/bindings_fin.cpp` (the `BacktestTrades` binding)
- Test: `tests/test_backtest.py`

**Interfaces:**
- Produces: `BacktestTrades(maker_fee=0.0, fill="touch", participation_ratio=1.0, min_position=-inf, max_position=+inf)`. A fill is truncated so the position stays in the cap.

- [ ] **Step 1: Write the failing test**

```python
def test_trades_fill_truncated_by_cap():
    from screamer import BacktestTrades
    import numpy as np
    # resting buy 10 @ 100, a through-print would fill 10, but max_position 3 caps it
    out = BacktestTrades(max_position=3.0)(
        np.array([100.]), np.array([10.]), np.array([99.]), np.array([2.]))
    assert out[0, 2] == 3.0
```

- [ ] **Step 2: Run to verify failure**

Run: `poetry run python -m pytest tests/test_backtest.py -q -k trades_fill_truncated`
Expected: FAIL (no `max_position`; the through-fill returns 10).

- [ ] **Step 3: Add the cap** in `include/screamer/backtest_trades.h`

Add `min_position`/`max_position` to the constructor (with the guard) and members. In `call`, cap the `filled` amount by the room:

```cpp
const double room = buy ? std::max(max_position_ - account_.position(), 0.0)
                        : std::max(account_.position() - min_position_, 0.0);
double filled = 0.0;
if (through)  filled = remaining;
else if (at)  filled = std::min(remaining, participation_ratio_ * trade_size);
filled = std::min(filled, room);                       // truncate to the position cap
```

(Insert the `room` computation and the final `std::min(filled, room)` into the existing fill block, before `fill_dpos` is set.)

- [ ] **Step 4: Update the binding** to add `min_position`/`max_position` args (infinity defaults) and widen `py::init` to `<double, const std::string&, double, double, double>`.

- [ ] **Step 5: Rebuild and run**

Run: `make install-dev && poetry run python -m pytest tests/test_backtest.py -q -k "trades"`
Expected: PASS (cap test + all existing `test_trades_*` unchanged under the unbounded default).

- [ ] **Step 6: Docs frontmatter** in `docs/functions_fin/BacktestTrades.md`: add the two params + a Description note. Run `build_help_registry.py`.

- [ ] **Step 7: Commit**

```bash
git add include/screamer/backtest_trades.h bindings/bindings_fin.cpp docs/functions_fin/BacktestTrades.md screamer/data/help.json tests/test_backtest.py
git commit -m "feat(backtest): static position cap on BacktestTrades"
```

---

## Task 5: `BacktestOHLCMaker` (market-making on bars)

**Files:**
- Create: `include/screamer/backtest_ohlc_maker.h`
- Modify: `bindings/bindings_fin.cpp` (include + register), `tests/param_cases.py`
- Create: `docs/functions_fin/BacktestOHLCMaker.md`
- Test: `tests/test_backtest.py`

**Interfaces:**
- Consumes: `market_limit` (Task 1); `detail::PnLAccount`.
- Produces: `BacktestOHLCMaker(maker_fee=0.0, taker_fee=0.0, fill="touch", participation_ratio=1.0, tick_size=0.0, min_position=-inf, max_position=+inf)`, 8 inputs `(bid_price, bid_size, ask_price, ask_size, open, high, low, close)`, 4 outputs. Two-sided passive quotes fill on the bar's low/high reaching them; marketable quotes fill at the open. Marks to the close.

- [ ] **Step 1: Write the failing tests**

```python
def test_ohlc_maker_two_sided_fills_on_range():
    from screamer import BacktestOHLCMaker
    import numpy as np
    # bid 99 rests; the bar low 98 reaches it -> buy 1 at 99; no ask this bar
    out = BacktestOHLCMaker()(
        np.array([99.]), np.array([1.]), np.array([np.nan]), np.array([0.]),
        np.array([100.]), np.array([101.]), np.array([98.]), np.array([100.]))
    assert out[0, 2] == 1.0                       # bought 1 at the bid
    # marks to close 100 vs fill 99 -> +1 mark, cost 0 (maker), equity +1
    np.testing.assert_allclose(out[0, 0], 1.0, atol=1e-9)

def test_ohlc_maker_inventory_cap():
    from screamer import BacktestOHLCMaker
    import numpy as np
    # bid 99 size 10, low reaches it, but max_position 2 caps the buy
    out = BacktestOHLCMaker(max_position=2.0)(
        np.array([99.]), np.array([10.]), np.array([np.nan]), np.array([0.]),
        np.array([100.]), np.array([101.]), np.array([98.]), np.array([100.]))
    assert out[0, 2] == 2.0

def test_ohlc_maker_stream_equals_batch():
    from screamer import BacktestOHLCMaker
    import numpy as np
    rng = np.random.default_rng(0); n = 200
    close = 100 + np.cumsum(rng.standard_normal(n) * 0.2)
    o, h, l = close - 0.1, close + 0.3, close - 0.3
    bid, ask = close - 0.2, close + 0.2
    one = np.ones(n)
    args = (bid, one, ask, one, o, h, l, close)
    op = BacktestOHLCMaker(max_position=5.0, min_position=-5.0)
    stream = np.array([op(*(float(a[i]) for a in args)) for i in range(n)])
    op.reset()
    batch = BacktestOHLCMaker(max_position=5.0, min_position=-5.0)(*args)
    np.testing.assert_allclose(np.nan_to_num(stream), np.nan_to_num(batch))
```

- [ ] **Step 2: Run to verify failure**

Run: `poetry run python -m pytest tests/test_backtest.py -q -k ohlc_maker`
Expected: FAIL (`BacktestOHLCMaker` does not exist).

- [ ] **Step 3: Create `include/screamer/backtest_ohlc_maker.h`**

```cpp
#ifndef SCREAMER_BACKTEST_OHLC_MAKER_H
#define SCREAMER_BACKTEST_OHLC_MAKER_H

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include "screamer/common/functor_base.h"
#include "screamer/common/float_info.h"
#include "screamer/common/market_price.h"
#include "screamer/detail/pnl_account.h"

namespace screamer {

    // BacktestOHLCMaker: 8 -> 4. Two-sided market-making on OHLC bars. Each bar the
    // strategy posts a resting bid (bid_price, bid_size) and ask (ask_price,
    // ask_size); a NaN/inf price is a market order (see market_limit). A resting buy
    // fills when the bar's low reaches the bid (touch: low <= bid, breach: low <
    // bid) for min(bid_size, participation * bid_size, room) at the bid, paying
    // maker_fee; a marketable buy fills at the open plus tick_size overflow, paying
    // taker_fee. The sell side is symmetric on the high. Fills are capped so the
    // position stays in [min_position, max_position]. Positions mark to the close.
    // Outputs [equity, pnl, position, cost]. nan_policy: ignore on the bar fields.
    class BacktestOHLCMaker : public FunctorBase<BacktestOHLCMaker, 8, 4> {
    public:
        BacktestOHLCMaker(double maker_fee = 0.0, double taker_fee = 0.0,
                          const std::string& fill = "touch",
                          double participation_ratio = 1.0, double tick_size = 0.0,
                          double min_position = -std::numeric_limits<double>::infinity(),
                          double max_position = std::numeric_limits<double>::infinity())
            : maker_fee_(maker_fee), taker_fee_(taker_fee), breach_(parse_fill(fill)),
              participation_(parse_participation(participation_ratio)),
              tick_size_(tick_size), min_position_(min_position),
              max_position_(max_position)
        {
            if (min_position_ > max_position_)
                throw std::invalid_argument("min_position must not exceed max_position.");
            if (tick_size_ < 0.0)
                throw std::invalid_argument("tick_size must be non-negative.");
            reset();
        }

        void reset() override { account_.reset(); }

        ResultTuple call(const InputArray& inputs) override {
            const double bid_price = inputs[0], bid_size = inputs[1];
            const double ask_price = inputs[2], ask_size = inputs[3];
            const double open = inputs[4], high = inputs[5];
            const double low = inputs[6], close = inputs[7];
            if (isnan2(open) || isnan2(high) || isnan2(low) || isnan2(close)) {
                const double nan = std::numeric_limits<double>::quiet_NaN();
                return std::make_tuple(nan, nan, nan, nan);   // ignore the bad bar
            }

            double eq = 0, pnl = 0, position = account_.position(), cost = 0; bool did = false;

            // Buy side: resting bid at bid_price, hit when the bar trades down to it.
            if (!isnan2(bid_size) && bid_size > 0.0) {
                const double room = std::max(max_position_ - account_.position(), 0.0);
                const double limit = market_limit(bid_price, /*buy=*/true);
                double dpos = 0.0, fill_price = close, fee = maker_fee_;
                if (std::isinf(limit) && limit > 0.0) {       // market buy at the open
                    dpos = std::min(std::min(bid_size, participation_ * bid_size), room);
                    fill_price = open + tick_size_; fee = taker_fee_;
                } else {                                       // resting limit buy
                    const bool hit = breach_ ? (low < limit) : (low <= limit);
                    if (hit) {
                        dpos = std::min(std::min(bid_size, participation_ * bid_size), room);
                        fill_price = limit;
                    }
                }
                if (dpos > 0.0) {
                    auto [e, p, pos, c] = account_.step(close, dpos, fill_price, fee);
                    eq = e; pnl += p; position = pos; cost += c; did = true;
                }
            }

            // Sell side: resting ask at ask_price, hit when the bar trades up to it.
            if (!isnan2(ask_size) && ask_size > 0.0) {
                const double room = std::max(account_.position() - min_position_, 0.0);
                const double limit = market_limit(ask_price, /*buy=*/false);
                double dpos = 0.0, fill_price = close, fee = maker_fee_;
                if (std::isinf(limit) && limit < 0.0) {       // market sell at the open
                    dpos = -std::min(std::min(ask_size, participation_ * ask_size), room);
                    fill_price = open - tick_size_; fee = taker_fee_;
                } else {                                       // resting limit sell
                    const bool hit = breach_ ? (high > limit) : (high >= limit);
                    if (hit) {
                        dpos = -std::min(std::min(ask_size, participation_ * ask_size), room);
                        fill_price = limit;
                    }
                }
                if (dpos != 0.0) {
                    auto [e, p, pos, c] = account_.step(close, dpos, fill_price, fee);
                    eq = e; pnl += p; position = pos; cost += c; did = true;
                }
            }

            if (!did) {
                auto [e, p, pos, c] = account_.step(close, 0.0, close, 0.0);  // mark only
                eq = e; pnl = p; position = pos; cost = c;
            }
            return std::make_tuple(eq, pnl, position, cost);
        }

    private:
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
        double participation_, tick_size_, min_position_, max_position_;
        detail::PnLAccount account_;
    };

} // namespace screamer

#endif // SCREAMER_BACKTEST_OHLC_MAKER_H
```

- [ ] **Step 4: Register the binding** in `bindings/bindings_fin.cpp` (add the include near the other backtest includes, and the class near `BacktestOHLC`)

```cpp
#include "screamer/backtest_ohlc_maker.h"
```
```cpp
py::class_<screamer::BacktestOHLCMaker, screamer::EvalOp>(m, "BacktestOHLCMaker")
    .def(py::init<double, double, const std::string&, double, double, double, double>(),
         py::arg("maker_fee") = 0.0, py::arg("taker_fee") = 0.0,
         py::arg("fill") = "touch", py::arg("participation_ratio") = 1.0,
         py::arg("tick_size") = 0.0,
         py::arg("min_position") = -std::numeric_limits<double>::infinity(),
         py::arg("max_position") = std::numeric_limits<double>::infinity())
    .def("__call__", &screamer::BacktestOHLCMaker::handle_input)
    .def("reset", &screamer::BacktestOHLCMaker::reset, "Reset.");
```

- [ ] **Step 5: Exclude from the no-arg auto-sweep** in `tests/param_cases.py`: add `'BacktestOHLCMaker'` to the `_NO_ARG_AUTO_EXCLUDE` set (multi-input functor with args), with a one-line comment.

- [ ] **Step 6: Rebuild and run**

Run: `make install-dev && poetry run python -m pytest tests/test_backtest.py -q -k ohlc_maker`
Expected: PASS (all three).

- [ ] **Step 7: Docs page** `docs/functions_fin/BacktestOHLCMaker.md`: frontmatter (`name`, `title`, `implementation_family: fin`, `topics: [backtesting]`, `inputs: 8`, `outputs: 4`, the seven parameters, `nan_policy: ignore`, `see_also`), a Description of the two-sided bar maker, a Limitations box (bars have no intra-bar path, so fill timing is coarse), and a `.. plotly::` example (a two-sided maker on a synthetic mean-reverting bar series, mirroring `BacktestL1.md`). Run `build_help_registry.py` then `build_topic_pages.py`.

- [ ] **Step 8: Commit**

```bash
git add include/screamer/backtest_ohlc_maker.h bindings/bindings_fin.cpp tests/param_cases.py tests/test_backtest.py docs/functions_fin/BacktestOHLCMaker.md screamer/data/help.json screamer/__init__.py docs/by_group screamer/data
git commit -m "feat(backtest): BacktestOHLCMaker (two-sided market-making on bars)"
```

---

## Task 6: `BacktestTradesMaker` (market-making on the tape)

**Files:**
- Create: `include/screamer/backtest_trades_maker.h`
- Modify: `bindings/bindings_fin.cpp`, `tests/param_cases.py`
- Create: `docs/functions_fin/BacktestTradesMaker.md`
- Test: `tests/test_backtest.py`

**Interfaces:**
- Consumes: `market_limit`, `detail::PnLAccount`.
- Produces: `BacktestTradesMaker(maker_fee=0.0, taker_fee=0.0, fill="touch", participation_ratio=1.0, tick_size=0.0, min_position=-inf, max_position=+inf)`, 6 inputs `(bid_price, bid_size, ask_price, ask_size, trade_price, trade_size)`, 4 outputs. Two-sided resting quotes fill when a print crosses them; marks to the last trade.

- [ ] **Step 1: Write the failing tests**

```python
def test_trades_maker_two_sided_fills_on_prints():
    from screamer import BacktestTradesMaker
    import numpy as np
    # resting bid 100 size 5; a sell-print at 99 (<=100) size 8, participation 1.0 -> buy 5 at 100
    out = BacktestTradesMaker()(
        np.array([100.]), np.array([5.]), np.array([np.nan]), np.array([0.]),
        np.array([99.]), np.array([8.]))
    assert out[0, 2] == 5.0

def test_trades_maker_cap_and_participation():
    from screamer import BacktestTradesMaker
    import numpy as np
    # at-price print size 8, participation 0.5 -> min(remaining, 0.5*8)=4, capped by max 3
    out = BacktestTradesMaker(participation_ratio=0.5, max_position=3.0)(
        np.array([100.]), np.array([10.]), np.array([np.nan]), np.array([0.]),
        np.array([100.]), np.array([8.]))
    assert out[0, 2] == 3.0

def test_trades_maker_stream_equals_batch():
    from screamer import BacktestTradesMaker
    import numpy as np
    rng = np.random.default_rng(1); n = 200
    price = 100 + np.cumsum(rng.standard_normal(n) * 0.1)
    size = np.abs(rng.standard_normal(n)) + 0.5
    bid, ask = price - 0.05, price + 0.05
    one = np.ones(n)
    args = (bid, one, ask, one, price, size)
    op = BacktestTradesMaker(max_position=8.0, min_position=-8.0)
    stream = np.array([op(*(float(a[i]) for a in args)) for i in range(n)])
    op.reset()
    batch = BacktestTradesMaker(max_position=8.0, min_position=-8.0)(*args)
    np.testing.assert_allclose(np.nan_to_num(stream), np.nan_to_num(batch))
```

- [ ] **Step 2: Run to verify failure**

Run: `poetry run python -m pytest tests/test_backtest.py -q -k trades_maker`
Expected: FAIL (`BacktestTradesMaker` does not exist).

- [ ] **Step 3: Create `include/screamer/backtest_trades_maker.h`**

```cpp
#ifndef SCREAMER_BACKTEST_TRADES_MAKER_H
#define SCREAMER_BACKTEST_TRADES_MAKER_H

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include "screamer/common/functor_base.h"
#include "screamer/common/float_info.h"
#include "screamer/common/market_price.h"
#include "screamer/detail/pnl_account.h"

namespace screamer {

    // BacktestTradesMaker: 6 -> 4. Two-sided market-making against the trade tape.
    // Each event is a print (trade_price, trade_size) and the strategy's resting bid
    // (bid_price, bid_size) and ask (ask_price, ask_size); a NaN/inf price is a
    // market order. A resting buy fills when a sell-print crosses it (touch:
    // trade_price <= bid, breach: <) for min(bid_size, participation * trade_size,
    // room) at the bid, paying maker_fee; a through-print sweeps the order. The sell
    // side is symmetric. Fills are capped to [min_position, max_position]. Marks to
    // the last trade. Outputs [equity, pnl, position, cost]. nan_policy: ignore on
    // the trade fields (a NaN trade emits an all-NaN row).
    class BacktestTradesMaker : public FunctorBase<BacktestTradesMaker, 6, 4> {
    public:
        BacktestTradesMaker(double maker_fee = 0.0, double taker_fee = 0.0,
                            const std::string& fill = "touch",
                            double participation_ratio = 1.0, double tick_size = 0.0,
                            double min_position = -std::numeric_limits<double>::infinity(),
                            double max_position = std::numeric_limits<double>::infinity())
            : maker_fee_(maker_fee), taker_fee_(taker_fee), breach_(parse_fill(fill)),
              participation_(parse_participation(participation_ratio)),
              tick_size_(tick_size), min_position_(min_position),
              max_position_(max_position)
        {
            if (min_position_ > max_position_)
                throw std::invalid_argument("min_position must not exceed max_position.");
            if (tick_size_ < 0.0)
                throw std::invalid_argument("tick_size must be non-negative.");
            reset();
        }

        void reset() override { account_.reset(); }

        ResultTuple call(const InputArray& inputs) override {
            const double bid_price = inputs[0], bid_size = inputs[1];
            const double ask_price = inputs[2], ask_size = inputs[3];
            const double trade_price = inputs[4], trade_size = inputs[5];
            if (isnan2(trade_price) || isnan2(trade_size)) {
                const double nan = std::numeric_limits<double>::quiet_NaN();
                return std::make_tuple(nan, nan, nan, nan);   // ignore: need a print
            }

            double eq = 0, pnl = 0, position = account_.position(), cost = 0; bool did = false;

            // Buy side: resting bid filled by a sell-print crossing it.
            if (!isnan2(bid_size) && bid_size > 0.0) {
                const double room = std::max(max_position_ - account_.position(), 0.0);
                const double limit = market_limit(bid_price, /*buy=*/true);
                const bool through = trade_price < limit;
                const bool at = !breach_ && (trade_price == limit);
                double dpos = 0.0;
                if (through) dpos = std::min(bid_size, room);
                else if (at) dpos = std::min(std::min(bid_size, participation_ * trade_size), room);
                if (dpos > 0.0) {
                    const double fp = std::isinf(limit) ? trade_price : limit;
                    const double fee = std::isinf(limit) ? taker_fee_ : maker_fee_;
                    auto [e, p, pos, c] = account_.step(trade_price, dpos, fp, fee);
                    eq = e; pnl += p; position = pos; cost += c; did = true;
                }
            }

            // Sell side: resting ask filled by a buy-print crossing it.
            if (!isnan2(ask_size) && ask_size > 0.0) {
                const double room = std::max(account_.position() - min_position_, 0.0);
                const double limit = market_limit(ask_price, /*buy=*/false);
                const bool through = trade_price > limit;
                const bool at = !breach_ && (trade_price == limit);
                double dpos = 0.0;
                if (through) dpos = -std::min(ask_size, room);
                else if (at) dpos = -std::min(std::min(ask_size, participation_ * trade_size), room);
                if (dpos != 0.0) {
                    const double fp = std::isinf(limit) ? trade_price : limit;
                    const double fee = std::isinf(limit) ? taker_fee_ : maker_fee_;
                    auto [e, p, pos, c] = account_.step(trade_price, dpos, fp, fee);
                    eq = e; pnl += p; position = pos; cost += c; did = true;
                }
            }

            if (!did) {
                auto [e, p, pos, c] = account_.step(trade_price, 0.0, trade_price, 0.0);
                eq = e; pnl = p; position = pos; cost = c;
            }
            return std::make_tuple(eq, pnl, position, cost);
        }

    private:
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
        double participation_, tick_size_, min_position_, max_position_;
        detail::PnLAccount account_;
    };

} // namespace screamer

#endif // SCREAMER_BACKTEST_TRADES_MAKER_H
```

Note: `tick_size_` is accepted for interface uniformity but the tape maker has no displayed-size overflow to walk (a marketable order fills at the print), so it is currently unused; keep the parameter for the uniform contract and note it in the docs.

- [ ] **Step 4: Register the binding** (include + class), mirroring Task 5's binding with the name `BacktestTradesMaker` and the same seven `py::arg`s.

- [ ] **Step 5: Exclude from the no-arg sweep**: add `'BacktestTradesMaker'` to `_NO_ARG_AUTO_EXCLUDE` in `tests/param_cases.py`.

- [ ] **Step 6: Rebuild and run**

Run: `make install-dev && poetry run python -m pytest tests/test_backtest.py -q -k trades_maker`
Expected: PASS.

- [ ] **Step 7: Docs page** `docs/functions_fin/BacktestTradesMaker.md`: frontmatter (6 inputs, 4 outputs, seven params, `nan_policy: ignore`, `topics: [backtesting]`), Description, a Limitations note (the `tick_size` overflow is inert on the tape), and a `.. plotly::` example. Run `build_help_registry.py` then `build_topic_pages.py`.

- [ ] **Step 8: Commit**

```bash
git add include/screamer/backtest_trades_maker.h bindings/bindings_fin.cpp tests/param_cases.py tests/test_backtest.py docs/functions_fin/BacktestTradesMaker.md screamer/data/help.json screamer/__init__.py docs/by_group
git commit -m "feat(backtest): BacktestTradesMaker (two-sided market-making on the tape)"
```

---

## Task 7: coverage-matrix docs page + changelog + full verification

**Files:**
- Create: `docs/functions_fin/choosing_a_backtest_engine.md`
- Modify: `docs/index.rst` (link the page), `CHANGELOG.md`
- Modify: `docs/topics.yml` only if a new topic is needed (it is not; use `backtesting`).

- [ ] **Step 1: Create the overview page** `docs/functions_fin/choosing_a_backtest_engine.md` with `kind: function` frontmatter style (like `backtest_report.md`: `name`, `title`, `topics: [backtesting]`), a short intro, and the coverage matrix from the spec (data model x {market, limit, market-making}), each cell naming the engine. Include the market-order encoding table and the fill-cap rule. No em-dashes.

- [ ] **Step 2: Link it** from the backtest overview: add `choosing_a_backtest_engine` to the References or a suitable toctree in `docs/index.rst`, or reference it from the group landing. Confirm it is homed (no orphan warning) by building docs in Step 5.

- [ ] **Step 3: Rewrite the `[Unreleased]` backtest changelog entry** in `CHANGELOG.md` to describe the unified contract: every engine takes a static `[min_position, max_position]` fill cap and the `MARKET`/`NaN`/`inf` market-order encoding; new `BacktestOHLCMaker` and `BacktestTradesMaker` add two-sided market-making on bars and the tape, closing the coverage matrix.

- [ ] **Step 4: Full regen + suite**

Run:
```bash
poetry run python devtools/build_help_registry.py
poetry run python devtools/build_topic_pages.py
make regen-init
poetry run python -m pytest -q
```
Expected: help registry validates all pages (including the two new engines and the overview); full suite passes (the new engine tests, the cap tests, and the NaN-compliance harness auto-picking up `BacktestOHLCMaker`/`BacktestTradesMaker`, which should pass as they do for `BacktestL1`).

- [ ] **Step 5: Docs build**

Run: `make docs`
Expected: exit 0; the two new engine pages render (plotly iframes), the coverage matrix page renders and is homed, no orphan warnings.

- [ ] **Step 6: Commit**

```bash
git add docs/functions_fin/choosing_a_backtest_engine.md docs/index.rst CHANGELOG.md screamer/data/help.json docs/by_group docs/by_group_index.rst
git commit -m "docs(backtest): coverage matrix overview + unified-API changelog"
```

---

## Self-Review

**Spec coverage:**
- Axis 1 (data model as the only difference): preserved; engines keep their data columns. Covered.
- Axis 2 (order intent: directional target + market-making quote, with the market encoding): directional engines unchanged (Tasks 2-4 add only the cap); market-making added on bars/tape (Tasks 5-6); the `market_limit` encoding + `MARKET` (Task 1). Covered.
- Axis 3 (static fill cap, three-way minimum): Tasks 2-4 (clamp target) and 5-6 (truncate fill to room). Covered.
- Axis 4 (cost + fill fidelity uniform): the new engines carry maker/taker fee, fill, participation, tick_size; the directional engines already had their cost params. Covered.
- Coverage matrix + overview page: Task 7. Covered.
- Keep current names, no renames: honored (new engines are additive `*Maker`). Covered.
- Changelog rewrite: Task 7. Covered.
- `SchmittTrigger` initial-state: explicitly out of scope in the spec (separate follow-up); no task, by design.

**Placeholder scan:** No TBD/TODO. New-engine tasks carry full header code; cap tasks show the exact constructor, `call` edits, and binding. Doc-page steps name the exact frontmatter fields and the pattern page to mirror.

**Type consistency:** All seven engines use the same param names (`maker_fee`, `taker_fee`, `fill`, `participation_ratio`, `tick_size`, `min_position`, `max_position`) and the same output schema. `market_limit(price, buy)` has one signature, used by both new engines. The `BacktestOHLCMaker` (8 inputs) and `BacktestTradesMaker` (6 inputs) input orders match their tests and bindings. `min_position` precedes `max_position` in every constructor and binding.

---

## Notes for the implementer

- Model the two new engines on `include/screamer/backtest_l1.h` (per-side fill, `room` cap, `account_.step` per side, mark-only fallback). The differences are the fill trigger (bar low/high vs quote cross vs print cross) and the market-data columns.
- `std::isinf(limit)` distinguishes a marketable order (taker) from a resting limit (maker) after `market_limit` has mapped `NaN` to the aggressive infinity.
- Keep the existing engines' default behavior identical: the cap defaults to `+/-inf`, so all prior tests must still pass unchanged.
- `make docs` executes every notebook; keep it as the final gate in Task 7.
