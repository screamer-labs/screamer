#ifndef SCREAMER_LIQUIDATION_DISTANCE_H
#define SCREAMER_LIQUIDATION_DISTANCE_H

// LiquidationDistance: 4 -> 3. How close an isolated-margin position comes to
// liquidation, bar by bar. Inputs are the signed `position` and the `open`,
// `high`, `low` of the trigger price (ideally mark-price bars).
//
// A segment starts on a bar where the position changes to a nonzero value (a
// new position, a sign flip, or a resize) and fills at that bar's open. The
// segment posts |size| * entry / leverage of margin, with no top-up, so its
// liquidation price is
//     long:  entry * (1 - 1 / leverage + maintenance)
//     short: entry * (1 + 1 / leverage - maintenance)
// A resize starts a new segment at the new fill: the margin is re-posted at
// the current price, with no credit for earlier gains.
//
// Outputs, per bar:
//     distance  the adverse move still needed from the bar's worst price to
//               reach liquidation, as a fraction of that price:
//               long (low - liq) / low, short (liq - high) / high.
//               0 or less means the bar reached the liquidation price.
//     liq_price the segment's liquidation price.
//     breached  1 from the first bar that reached it to the end of the
//               segment, 0 before.
// All three are NaN while flat.
//
// NaN handling: a NaN position holds the previous position. A NaN open, high,
// or low gives a NaN row and leaves the segment as it is; a position change
// and a fill at a finite open on that bar are still taken. When the open of a
// segment's first bar is NaN, the segment fills at the next finite open. The operator only
// measures: it does not close positions or change any pnl.

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
        liq_ = 0.0;
        side_ = 0;
        pending_ = false;
        breached_ = false;
    }

    ResultTuple call(const InputArray& in) override {
        const double position = in[0], open = in[1], high = in[2], low = in[3];
        constexpr double nan = std::numeric_limits<double>::quiet_NaN();
        if (!isnan2(position) && position != prev_position_) {
            prev_position_ = position;
            side_ = (position > 0.0) - (position < 0.0);
            pending_ = side_ != 0;
            breached_ = false;
        }
        if (side_ == 0) return std::make_tuple(nan, nan, nan);
        if (pending_ && !isnan2(open)) {
            liq_ = open * (side_ > 0 ? long_factor_ : short_factor_);
            pending_ = false;
        }
        if (pending_ || isnan2(open) || isnan2(high) || isnan2(low))
            return std::make_tuple(nan, nan, nan);
        const double distance = side_ > 0 ? (low - liq_) / low : (liq_ - high) / high;
        if (distance <= 0.0) breached_ = true;
        return std::make_tuple(distance, liq_, breached_ ? 1.0 : 0.0);
    }

private:
    double long_factor_ = 0.0;
    double short_factor_ = 0.0;
    double prev_position_ = 0.0;
    double liq_ = 0.0;
    int side_ = 0;
    bool pending_ = false;
    bool breached_ = false;
};

} // namespace screamer

#endif // SCREAMER_LIQUIDATION_DISTANCE_H
