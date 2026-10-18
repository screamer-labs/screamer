#ifndef SCREAMER_LIQUIDATION_DISTANCE_H
#define SCREAMER_LIQUIDATION_DISTANCE_H

// LiquidationDistance: 4 -> 3. How close an isolated-margin position comes to
// liquidation, bar by bar. Inputs are the signed `position` and the `open`,
// `high`, `low` of the trigger price (ideally mark-price bars). The position
// is the one a backtest engine reports for the bar, already filled at that
// bar's open.
//
// Position changes follow exchange accounting, with every fill at the bar's
// open:
//     from flat, or a sign flip:  a fresh entry at the open.
//     same-sign increase:         the entry blends,
//                                 entry = (|p0| * e0 + (|p1| - |p0|) * open) / |p1|,
//                                 and margin is posted on the added size.
//     same-sign reduce:           the entry, and so the liquidation price,
//                                 are unchanged.
// The position holds |size| * entry / leverage of margin, with no top-up, so
// its liquidation price is
//     long:  entry * (1 - 1 / leverage + maintenance)
//     short: entry * (1 + 1 / leverage - maintenance)
//
// Outputs, per bar:
//     distance  the adverse move still needed from the bar's worst price to
//               reach liquidation, as a fraction of that price:
//               long (low - liq) / low, short (liq - high) / high.
//               0 or less means the bar reached the liquidation price.
//     liq_price the liquidation price of the current position.
//     breached  1 from the first bar that reached it until the position goes
//               flat or flips, 0 before. A resize does not clear it: the
//               position was liquidated in reality.
// All three are NaN while flat.
//
// NaN handling: a NaN position holds the previous position. A NaN open, high,
// or low gives a NaN row and leaves the state as it is, except that a
// position change on that bar is still taken; a fill that needs the open (a
// fresh entry or an increase) then happens at the next finite open. The
// operator only measures: it does not close positions or change any pnl.

#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include "screamer/common/functor_base.h"
#include "screamer/common/finite_settings.h"
#include "screamer/common/float_info.h"

namespace screamer {

class LiquidationDistance : public FunctorBase<LiquidationDistance, 4, 3> {
public:
    // State is a few plain values: the column loop runs on a register copy.
    static constexpr bool kColumnCopy = true;

    explicit LiquidationDistance(double leverage, double maintenance = 0.005)
    {
        require_finite("leverage", leverage);
        require_finite("maintenance", maintenance);
        if (!(leverage > 1.0)) throw std::invalid_argument("leverage must be more than 1.");
        if (!(maintenance >= 0.0 && maintenance < 1.0 / leverage))
            throw std::invalid_argument("maintenance must be 0 or more and less than 1 / leverage.");
        long_factor_ = 1.0 - 1.0 / leverage + maintenance;
        short_factor_ = 1.0 + 1.0 / leverage - maintenance;
        reset();
    }

    void reset() override {
        prev_position_ = 0.0;
        target_ = 0.0;
        filled_ = 0.0;
        entry_ = 0.0;
        liq_ = 0.0;
        side_ = 0;
        breached_ = false;
    }

    ResultTuple call(const InputArray& in) override {
        const double position = in[0], open = in[1], high = in[2], low = in[3];
        constexpr double nan = std::numeric_limits<double>::quiet_NaN();
        if (!isnan2(position) && position != prev_position_) {
            prev_position_ = position;
            const int side = (position > 0.0) - (position < 0.0);
            target_ = std::fabs(position);
            if (side != side_) {                 // flat, fresh entry, or flip
                side_ = side;
                filled_ = 0.0;
                breached_ = false;
            } else if (target_ < filled_) {      // same-sign reduce: entry kept
                filled_ = target_;
            }
        }
        if (side_ == 0) return std::make_tuple(nan, nan, nan);
        if (target_ > filled_ && !isnan2(open)) {   // fill the added size at the open
            entry_ = (filled_ * entry_ + (target_ - filled_) * open) / target_;
            filled_ = target_;
            liq_ = entry_ * (side_ > 0 ? long_factor_ : short_factor_);
        }
        if (target_ > filled_ || isnan2(open) || isnan2(high) || isnan2(low))
            return std::make_tuple(nan, nan, nan);
        const double distance = side_ > 0 ? (low - liq_) / low : (liq_ - high) / high;
        if (distance <= 0.0) breached_ = true;
        return std::make_tuple(distance, liq_, breached_ ? 1.0 : 0.0);
    }

private:
    double long_factor_ = 0.0;
    double short_factor_ = 0.0;
    double prev_position_ = 0.0;
    double target_ = 0.0;     // |position| to hold
    double filled_ = 0.0;     // |position| filled at entry_
    double entry_ = 0.0;
    double liq_ = 0.0;
    int side_ = 0;
    bool breached_ = false;
};

} // namespace screamer

#endif // SCREAMER_LIQUIDATION_DISTANCE_H
