#ifndef SCREAMER_BACKTEST_TRADES_TARGET_H
#define SCREAMER_BACKTEST_TRADES_TARGET_H

#include <limits>
#include <string>
#include "screamer/common/functor_base.h"
#include "screamer/detail/trades_fill.h"
#include "screamer/detail/risk_exits.h"
#include "screamer/detail/target_front.h"

namespace screamer {

    // BacktestTradesTarget: 3 -> 4. Reach a target position on the trade tape by
    // taking prints. Each event, size clamp(target, cap) - position and submit it as
    // a marketable order that the current print fills (taker). Inventory capped to
    // [min_position, max_position]. Outputs [equity, pnl, position, cost].
    // Optional risk exits (fractions; 0 = off): stop_loss, take_profit, and
    // trailing_stop, measured from the average entry fill (see RiskExits). On an
    // event whose trade price crosses a level, the target is forced flat, so the
    // position closes as a market order on that event; same-side targets are
    // then ignored until the target turns flat or reverses.
    class BacktestTradesTarget : public FunctorBase<BacktestTradesTarget, 3, 4> {
    public:
        BacktestTradesTarget(double taker_fee = 0.0, double tick_size = 0.0,
                             double min_position = -std::numeric_limits<double>::infinity(),
                             double max_position = std::numeric_limits<double>::infinity(),
                           double multiplier = 1.0,
                           double fee_per_contract = 0.0,
                           double slippage = 0.0,
                           const std::string& slippage_unit = "fraction",
                           double stop_loss = 0.0, double take_profit = 0.0,
                           double trailing_stop = 0.0)
            : core_(0.0, taker_fee, /*breach=*/false, /*participation=*/1.0,
                    detail::Slippage(slippage, slippage_unit, tick_size),
                    min_position, max_position,
                    multiplier, 0.0, fee_per_contract),
              min_(min_position), max_(max_position),
              exits_(stop_loss, take_profit, trailing_stop) {}
        void reset() override { core_.reset(); exits_.reset(); }
        ResultTuple call(const InputArray& in) override {
            double target = in[0];
            const double trade_price = in[1], trade_size = in[2];
            const double before = core_.position();
            if (exits_.active() && !isnan2(trade_price) && !isnan2(trade_size)) {
                const bool hit = !isnan2(exits_.event_exit(before, trade_price));
                if (hit) exits_.exited(before);
                target = exits_.filter(target);
                if (hit && isnan2(target)) target = 0.0;   // a hold still exits
            }
            auto [bp, bs, ap, as] =
                detail::target_to_quote(target, before, min_, max_);
            auto r = core_.quote(bp, bs, ap, as, trade_price, trade_size);
            if (exits_.active() && !isnan2(trade_price) && !isnan2(trade_size)) {
                if (core_.position() != before)
                    exits_.on_fill(before, core_.position(), core_.last_fill_price());
                exits_.update_best(core_.position(), trade_price, trade_price);
            }
            return r;
        }
    private:
        detail::TradesFill core_;
        double min_, max_;
        detail::RiskExits exits_;
    };

} // namespace screamer

#endif // SCREAMER_BACKTEST_TRADES_TARGET_H
