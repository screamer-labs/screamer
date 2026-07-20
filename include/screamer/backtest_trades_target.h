#ifndef SCREAMER_BACKTEST_TRADES_TARGET_H
#define SCREAMER_BACKTEST_TRADES_TARGET_H

#include <limits>
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
