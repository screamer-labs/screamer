#ifndef SCREAMER_BACKTEST_L1_TARGET_H
#define SCREAMER_BACKTEST_L1_TARGET_H

#include <limits>
#include "screamer/common/functor_base.h"
#include "screamer/detail/l1_fill.h"
#include "screamer/detail/target_front.h"

namespace screamer {

    // BacktestL1Target: 5 -> 4. Reach a target position each event by taking
    // against the displayed L1 book immediately (taker). Inputs
    // (target_position, market_bid, market_ask, market_bid_size,
    // market_ask_size). The order size is clamp(target, cap) - position,
    // executed as a marketable order that sweeps the opposite side.
    // Inventory capped to [min_position, max_position].
    // Outputs [equity, pnl, position, cost]. nan_policy: ignore.
    class BacktestL1Target : public FunctorBase<BacktestL1Target, 5, 4> {
    public:
        BacktestL1Target(double taker_fee = 0.0, double tick_size = 0.0,
                         double min_position = -std::numeric_limits<double>::infinity(),
                         double max_position = std::numeric_limits<double>::infinity())
            : core_(0.0, taker_fee, /*breach=*/false, /*participation=*/1.0,
                    tick_size, min_position, max_position),
              min_(min_position), max_(max_position) {}

        void reset() override { core_.reset(); }

        ResultTuple call(const InputArray& in) override {
            const double target        = in[0];
            const double market_bid    = in[1];
            const double market_ask    = in[2];
            const double market_bid_sz = in[3];
            const double market_ask_sz = in[4];
            auto [bp, bs, ap, as] =
                detail::target_to_quote(target, core_.position(), min_, max_);
            return core_.quote(bp, bs, ap, as,
                               market_bid, market_ask, market_bid_sz, market_ask_sz);
        }

    private:
        detail::L1Fill core_;
        double min_, max_;
    };

} // namespace screamer

#endif // SCREAMER_BACKTEST_L1_TARGET_H
