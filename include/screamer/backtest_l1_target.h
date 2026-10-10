#ifndef SCREAMER_BACKTEST_L1_TARGET_H
#define SCREAMER_BACKTEST_L1_TARGET_H

#include <limits>
#include <string>
#include "screamer/common/functor_base.h"
#include "screamer/detail/l1_fill.h"
#include "screamer/detail/risk_exits.h"
#include "screamer/detail/sizer.h"
#include "screamer/detail/target_front.h"

namespace screamer {

    // BacktestL1Target: 5 -> 4. Reach a target position each event by taking
    // against the displayed L1 book immediately (taker). Inputs
    // (target_position, market_bid, market_ask, market_bid_size,
    // market_ask_size). The order size is clamp(target, cap) - position,
    // executed as a marketable order that sweeps the opposite side.
    // Inventory capped to [min_position, max_position].
    // Outputs [equity, pnl, position, cost]. nan_policy: ignore.
    // Optional risk exits (fractions; 0 = off): stop_loss, take_profit, and
    // trailing_stop, measured from the average entry fill (see RiskExits). On an
    // event whose mid crosses a level, the target is forced flat, so the
    // position closes as a market order on that event; same-side targets are
    // then ignored until the target turns flat or reverses.
    // `size` reads the target as units (default), a notional amount, or a
    // weight of initial_capital + equity, converted at the event's mid; the
    // units are kept until the target value changes or, with rebalance_band,
    // until the holding drifts (see detail::Sizer). A NaN target keeps working
    // toward the last finite target.
    class BacktestL1Target : public FunctorBase<BacktestL1Target, 5, 4> {
    public:
        BacktestL1Target(double taker_fee = 0.0, double tick_size = 0.0,
                         double min_position = -std::numeric_limits<double>::infinity(),
                         double max_position = std::numeric_limits<double>::infinity(),
                           double multiplier = 1.0,
                           double fee_per_contract = 0.0,
                           double slippage = 0.0,
                           const std::string& slippage_unit = "fraction",
                           double stop_loss = 0.0, double take_profit = 0.0,
                           double trailing_stop = 0.0,
                           const std::string& size = "units",
                           double initial_capital = 0.0,
                           double rebalance_band = 0.0)
            : core_(0.0, taker_fee, /*breach=*/false, /*participation=*/1.0,
                    detail::Slippage(slippage, slippage_unit, tick_size),
                    min_position, max_position,
                    multiplier, 0.0, fee_per_contract),
              min_(min_position), max_(max_position),
              exits_(stop_loss, take_profit, trailing_stop),
              sizer_(size, initial_capital, rebalance_band, multiplier) {}

        void reset() override {
            core_.reset(); exits_.reset(); sizer_.reset();
            working_ = std::numeric_limits<double>::quiet_NaN();
        }

        ResultTuple call(const InputArray& in) override {
            const double market_bid    = in[1];
            const double market_ask    = in[2];
            const double market_bid_sz = in[3];
            const double market_ask_sz = in[4];
            const double before = core_.position();
            const bool valid = !isnan2(market_bid) && !isnan2(market_ask) &&
                               !isnan2(market_bid_sz) && !isnan2(market_ask_sz);
            const double mid = 0.5 * (market_bid + market_ask);
            if (valid && !isnan2(in[0])) working_ = in[0];
            double target = working_;
            if (exits_.active() && valid) {
                const bool hit = !isnan2(exits_.event_exit(before, mid));
                if (hit) exits_.exited(before);
                target = exits_.filter(target);
                if (hit && isnan2(target)) target = 0.0;   // a hold still exits
            }
            if (valid) target = sizer_.units(target, mid, before, core_.equity_at(mid));
            auto [bp, bs, ap, as] =
                detail::target_to_quote(target, before, min_, max_);
            auto r = core_.quote(bp, bs, ap, as,
                                 market_bid, market_ask, market_bid_sz, market_ask_sz);
            if (exits_.active() && valid) {
                if (core_.position() != before)
                    exits_.on_fill(before, core_.position(), core_.last_fill_price());
                exits_.update_best(core_.position(), mid, mid);
            }
            return r;
        }

    private:
        detail::L1Fill core_;
        double min_, max_;
        detail::RiskExits exits_;
        detail::Sizer sizer_;
        double working_ = std::numeric_limits<double>::quiet_NaN();
    };

} // namespace screamer

#endif // SCREAMER_BACKTEST_L1_TARGET_H
