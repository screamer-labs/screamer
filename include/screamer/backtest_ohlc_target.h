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
