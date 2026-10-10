#ifndef SCREAMER_DETAIL_RISK_EXITS_H
#define SCREAMER_DETAIL_RISK_EXITS_H

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include "screamer/common/float_info.h"

namespace screamer { namespace detail {

    // RiskExits: stop-loss, take-profit, and trailing-stop exits for the target
    // engines. All three are fractions of a price (0 turns an exit off):
    //   stop_loss      exit when the price moves this fraction against the
    //                  average entry price
    //   take_profit    exit when the price moves this fraction in favour of it
    //   trailing_stop  exit when the price falls this fraction from the best
    //                  price since entry (the highest for a long, the lowest
    //                  for a short)
    // The entry price is the average fill price of the open position; adding
    // to the position updates it, reducing it does not.
    //
    // After an exit the engine is latched: a target on the same side as the
    // exited position is treated as flat, until a target that is flat or on the
    // other side arrives. Without the latch the engine would re-enter on the
    // next step because the strategy's target has not changed.
    class RiskExits {
    public:
        RiskExits(double stop_loss = 0.0, double take_profit = 0.0,
                  double trailing_stop = 0.0)
            : stop_(stop_loss), take_(take_profit), trail_(trailing_stop)
        {
            check(stop_, "stop_loss", true);
            check(take_, "take_profit", false);
            check(trail_, "trailing_stop", true);
            reset();
        }

        bool active() const { return stop_ > 0.0 || take_ > 0.0 || trail_ > 0.0; }

        void reset() {
            entry_ = std::numeric_limits<double>::quiet_NaN();
            best_ = entry_;
            blocked_ = 0.0;
        }

        // Record a fill that moved the position from `before` to `after`.
        void on_fill(double before, double after, double fill_price) {
            if (after == 0.0) { entry_ = best_ = std::numeric_limits<double>::quiet_NaN(); return; }
            const bool opened = before == 0.0 || (before > 0.0) != (after > 0.0);
            if (opened) { entry_ = best_ = fill_price; return; }
            if (std::abs(after) > std::abs(before)) {          // added on the same side
                const double added = std::abs(after) - std::abs(before);
                entry_ = (std::abs(before) * entry_ + added * fill_price) / std::abs(after);
            }
        }

        // Track the best price since entry, for the trailing stop.
        void update_best(double position, double high, double low) {
            if (position == 0.0 || isnan2(best_)) return;
            if (position > 0.0) { if (high > best_) best_ = high; }
            else if (low < best_) best_ = low;
        }

        // Exit price for an event at `price` (price, trades, and L1 engines):
        // the event price if it crosses a level, else NaN.
        double event_exit(double position, double price) const {
            return bar_exit(position, price, price, price, true);
        }

        // Exit price within an OHLC bar for the position held through it. With
        // `include_open`, a bar that opens past a level exits at the open (a
        // gap). Otherwise the exit is at the level the low or high reaches. When
        // both the stop and the take-profit are inside the bar, the stop wins,
        // because the order of the high and the low is unknown. The exit price is
        // kept inside the bar's range: a level that the slipped entry fill left
        // outside it exits at the nearest price the bar traded. NaN: no exit.
        double bar_exit(double position, double open, double high, double low,
                        bool include_open) const {
            const double nan = std::numeric_limits<double>::quiet_NaN();
            if (position == 0.0 || isnan2(entry_) || !active()) return nan;
            const double inf = std::numeric_limits<double>::infinity();
            if (position > 0.0) {
                double stop = -inf;
                if (stop_ > 0.0) stop = entry_ * (1.0 - stop_);
                if (trail_ > 0.0) stop = std::max(stop, best_ * (1.0 - trail_));
                const double take = take_ > 0.0 ? entry_ * (1.0 + take_) : inf;
                if (include_open && open <= stop) return open;
                if (include_open && open >= take) return open;
                if (low <= stop) return std::clamp(stop, low, high);
                if (high >= take) return std::clamp(take, low, high);
            } else {
                double stop = inf;
                if (stop_ > 0.0) stop = entry_ * (1.0 + stop_);
                if (trail_ > 0.0) stop = std::min(stop, best_ * (1.0 + trail_));
                const double take = take_ > 0.0 ? entry_ * (1.0 - take_) : -inf;
                if (include_open && open >= stop) return open;
                if (include_open && open <= take) return open;
                if (high >= stop) return std::clamp(stop, low, high);
                if (low <= take) return std::clamp(take, low, high);
            }
            return nan;
        }

        // Latch after exiting a position with this sign.
        void exited(double position) { blocked_ = position > 0.0 ? 1.0 : -1.0; }

        // Apply the latch to a target. NaN (hold) passes through unchanged.
        double filter(double target) {
            if (isnan2(target) || blocked_ == 0.0) return target;
            if (target * blocked_ > 0.0) return 0.0;           // same side: stay flat
            blocked_ = 0.0;                                    // flat or reversed: release
            return target;
        }

    private:
        static void check(double v, const char* name, bool below_one) {
            if (!std::isfinite(v) || v < 0.0 || (below_one && v >= 1.0))
                throw std::invalid_argument(std::string(name) + (below_one
                    ? " must be in [0, 1)." : " must be finite and non-negative."));
        }

        double stop_, take_, trail_;
        double entry_ = std::numeric_limits<double>::quiet_NaN();
        double best_ = std::numeric_limits<double>::quiet_NaN();
        double blocked_ = 0.0;
    };

}} // namespace screamer::detail

#endif // SCREAMER_DETAIL_RISK_EXITS_H
