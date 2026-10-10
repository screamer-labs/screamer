#ifndef SCREAMER_BACKTEST_CARRY_H
#define SCREAMER_BACKTEST_CARRY_H

#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include "screamer/common/functor_base.h"
#include "screamer/common/float_info.h"

namespace screamer {

    // BacktestCarry: 6 -> 4. Adds a carry charge (perpetual funding, borrow fee,
    // financing) to any backtest engine's output. Inputs are the engine's
    // [equity, pnl, position, cost] plus the mark `price` and the per-step carry
    // `rate` of the same step. The position held into step t pays
    //     carry_t = position_{t-1} * price_t * rate_t * multiplier
    // so a positive rate charges longs and pays shorts, as perpetual funding
    // does. A negative rate reverses that. The output is the engine's columns
    // with the carry moved from pnl into cost:
    //     equity - cumulative carry, pnl - carry_t, position, cost + carry_t.
    // The rate is per step: divide an annual rate by the steps per year, or pass
    // the funding rate on funding steps and 0 in between.
    //
    // nan_policy: ignore. A NaN in any input skips the row (all-NaN output,
    // state held). Pass a rate of 0 on steps without carry.
    class BacktestCarry : public FunctorBase<BacktestCarry, 6, 4> {
    public:
        explicit BacktestCarry(double multiplier = 1.0) : multiplier_(multiplier) {
            if (!(multiplier_ > 0.0) || !std::isfinite(multiplier_))
                throw std::invalid_argument("multiplier must be finite and positive.");
            reset();
        }

        void reset() override { cum_carry_ = 0.0; prev_position_ = 0.0; }

        ResultTuple call(const InputArray& in) override {
            const double equity = in[0], pnl = in[1], position = in[2], cost = in[3];
            const double price = in[4], rate = in[5];
            if (isnan2(equity) || isnan2(pnl) || isnan2(position) || isnan2(cost) ||
                isnan2(price) || isnan2(rate)) {
                const double nan = std::numeric_limits<double>::quiet_NaN();
                return std::make_tuple(nan, nan, nan, nan);
            }
            const double carry = prev_position_ * price * rate * multiplier_;
            cum_carry_ += carry;
            prev_position_ = position;
            return std::make_tuple(equity - cum_carry_, pnl - carry, position, cost + carry);
        }

    private:
        double multiplier_;
        double cum_carry_ = 0.0;
        double prev_position_ = 0.0;
    };

} // namespace screamer

#endif // SCREAMER_BACKTEST_CARRY_H
