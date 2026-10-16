#ifndef SCREAMER_TRIPLE_BARRIER_H
#define SCREAMER_TRIPLE_BARRIER_H

// TripleBarrier and TripleBarrierOHLC: a three-state label over a window
// (Lopez de Prado, 2018). Every step e is an entry at its price (the close for
// bars). Over the next `horizon` steps the label is +1 when the price reaches
// the upper barrier first, -1 when it reaches the lower barrier first, and 0
// when it reaches neither. No position or side is involved.
//
//     1 -> 3  TripleBarrier      (price)
//     4 -> 3  TripleBarrierOHLC  (open, high, low, close)
//     outputs (label, return, holding)
//
// The output at step t is the entry e = t - horizon, the first step at which
// that entry is certain to be resolved; the first `horizon` outputs are NaN.
// `return` is exit / entry - 1 and `holding` the steps from entry to exit.
//
// Barriers, for an entry price p (0 turns a barrier off):
//     "fraction"  U = p * (1 + upper)        L = p * (1 - lower)
//     "price"     U = p + upper              L = p - lower
//     "vol"       U = p * (1 + upper * s)    L = p * (1 - lower * s)
// with s = EwStd(span=vol_span) of the one-step log returns up to and
// including e, times sqrt(horizon).
//
// A bar is tested with the rules of detail::RiskExits::bar_exit: an open
// through a barrier exits at the open (a gap); otherwise a high at or above U
// exits at U and a low at or below L at L, each clamped into the bar. When
// both barriers lie inside one bar the order is unknown and the entry's
// outputs are NaN. A price is a bar with open = high = low = close, so one
// rule set serves both operators. A NaN or non-positive price at entry or
// inside an entry's window makes that entry's outputs NaN.
//
// State: a ring of `horizon` entries (the open entries of the last `horizon`
// steps), so each step costs O(horizon) and allocates nothing.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include "screamer/common/functor_base.h"
#include "screamer/ew_std.h"

namespace screamer {
namespace detail {

class TripleBarrierCore {
public:
    using Result = std::tuple<double, double, double>;

    TripleBarrierCore(int horizon, double upper, double lower,
                      const std::string& barrier_unit, double vol_span)
        : horizon_(check_horizon(horizon)),
          upper_(check_distance(upper, "upper")),
          lower_(check_distance(lower, "lower")),
          unit_(parse_unit(barrier_unit)),
          sqrt_h_(std::sqrt(static_cast<double>(horizon))),
          vol_(std::nullopt, check_span(vol_span)),
          ring_(static_cast<std::size_t>(horizon))
    {
        if (unit_ == Unit::fraction && lower_ >= 1.0)
            throw std::invalid_argument(
                "lower must be below 1 when barrier_unit is \"fraction\".");
        reset();
    }

    void reset() {
        for (Entry& en : ring_) en = Entry{};
        cursor_ = 0;
        prev_close_ = nan();
        sigma_ = nan();
        vol_.reset();
    }

    // One step: test the bar against every open entry, emit the entry from
    // `horizon` steps ago, and open an entry at this step's close.
    template <bool kBar>
    Result step(double open, double high, double low, double close) {
        const bool valid = kBar
            ? (positive(open) && positive(high) && positive(low) && positive(close))
            : positive(close);
        for (Entry& en : ring_) {
            if (!en.open) continue;
            ++en.age;
            if (!valid) { en.fail(); continue; }
            test_bar<kBar>(en, open, high, low, close);
        }

        Entry& slot = ring_[cursor_];
        const Result out{slot.label, slot.ret, slot.hold};
        open_entry(slot, close);
        if (++cursor_ == ring_.size()) cursor_ = 0;
        return out;
    }

private:
    enum class Unit { fraction, price, vol };

    struct Entry {
        double price = std::numeric_limits<double>::quiet_NaN();
        double upper = std::numeric_limits<double>::infinity();
        double lower = -std::numeric_limits<double>::infinity();
        double label = std::numeric_limits<double>::quiet_NaN();
        double ret = std::numeric_limits<double>::quiet_NaN();
        double hold = std::numeric_limits<double>::quiet_NaN();
        int age = 0;
        bool open = false;

        void fail() { open = false; }       // outputs stay NaN
        void resolve(double lab, double exit) {
            label = lab;
            ret = exit / price - 1.0;
            hold = static_cast<double>(age);
            open = false;
        }
    };

    static double nan() { return std::numeric_limits<double>::quiet_NaN(); }
    static bool positive(double x) { return std::isfinite(x) && x > 0.0; }

    template <bool kBar>
    void test_bar(Entry& en, double open, double high, double low, double close) const {
        if (open >= en.upper) { en.resolve(1.0, open); return; }
        if (open <= en.lower) { en.resolve(-1.0, open); return; }
        if constexpr (kBar) {
            const bool up = high >= en.upper;
            const bool down = low <= en.lower;
            if (up && down) { en.fail(); return; }     // order inside the bar unknown
            if (up) { en.resolve(1.0, std::clamp(en.upper, low, high)); return; }
            if (down) { en.resolve(-1.0, std::clamp(en.lower, low, high)); return; }
        }
        if (en.age == horizon_) en.resolve(0.0, close);
    }

    void open_entry(Entry& en, double close) {
        en = Entry{};
        double s = 1.0;
        if (unit_ == Unit::vol) {
            if (positive(close) && positive(prev_close_)) {
                const double r = std::log(close / prev_close_);
                if (std::isfinite(r)) sigma_ = vol_.process_scalar(r) * sqrt_h_;
            }
            s = sigma_;
        }
        prev_close_ = close;
        if (!positive(close) || !(s > 0.0)) return;   // NaN outputs
        en.price = close;
        en.open = true;
        if (unit_ == Unit::price) {
            if (upper_ > 0.0) en.upper = close + upper_;
            if (lower_ > 0.0) en.lower = close - lower_;
        } else {
            if (upper_ > 0.0) en.upper = close * (1.0 + upper_ * s);
            if (lower_ > 0.0) en.lower = close * (1.0 - lower_ * s);
        }
    }

    static int check_horizon(int horizon) {
        if (horizon < 1) throw std::invalid_argument("horizon must be at least 1.");
        return horizon;
    }

    static double check_distance(double v, const char* name) {
        if (!std::isfinite(v) || v < 0.0)
            throw std::invalid_argument(std::string(name) + " must be finite and non-negative.");
        return v;
    }

    static double check_span(double span) {
        if (!std::isfinite(span) || span <= 1.0)
            throw std::invalid_argument("vol_span must be finite and greater than 1.");
        return span;
    }

    static Unit parse_unit(const std::string& unit) {
        if (unit == "fraction") return Unit::fraction;
        if (unit == "price") return Unit::price;
        if (unit == "vol") return Unit::vol;
        throw std::invalid_argument(
            "barrier_unit must be \"fraction\", \"price\", or \"vol\", got \"" + unit + "\".");
    }

    int horizon_;
    double upper_;
    double lower_;
    Unit unit_;
    double sqrt_h_;
    EwStd vol_;
    std::vector<Entry> ring_;
    std::size_t cursor_ = 0;
    double prev_close_ = std::numeric_limits<double>::quiet_NaN();
    double sigma_ = std::numeric_limits<double>::quiet_NaN();
};

}  // namespace detail

class TripleBarrier : public FunctorBase<TripleBarrier, 1, 3> {
public:
    explicit TripleBarrier(int horizon = 20, double upper = 0.0, double lower = 0.0,
                           const std::string& barrier_unit = "fraction",
                           double vol_span = 100.0)
        : core_(horizon, upper, lower, barrier_unit, vol_span) {}

    void reset() override { core_.reset(); }

    ResultTuple call(const InputArray& in) override {
        const double p = in[0];
        return core_.step<false>(p, p, p, p);
    }

private:
    detail::TripleBarrierCore core_;
};

class TripleBarrierOHLC : public FunctorBase<TripleBarrierOHLC, 4, 3> {
public:
    explicit TripleBarrierOHLC(int horizon = 20, double upper = 0.0, double lower = 0.0,
                               const std::string& barrier_unit = "fraction",
                               double vol_span = 100.0)
        : core_(horizon, upper, lower, barrier_unit, vol_span) {}

    void reset() override { core_.reset(); }

    ResultTuple call(const InputArray& in) override {
        return core_.step<true>(in[0], in[1], in[2], in[3]);
    }

private:
    detail::TripleBarrierCore core_;
};

}  // namespace screamer

#endif  // SCREAMER_TRIPLE_BARRIER_H
