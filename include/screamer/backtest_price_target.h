#ifndef SCREAMER_BACKTEST_PRICE_TARGET_H
#define SCREAMER_BACKTEST_PRICE_TARGET_H

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include "screamer/common/functor_base.h"
#include "screamer/common/finite_settings.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/pnl_account.h"
#include "screamer/detail/slippage.h"
#include "screamer/detail/risk_exits.h"
#include "screamer/detail/sizer.h"

namespace screamer {

    // BacktestPriceTarget: 2 -> 4. Backtest a target position against a value
    // series (a price/mark); reaches a target position by taking liquidity.
    // `signal` is the target position in units (sign is long / short / flat,
    // any magnitude); `price` is the mark (mid). Each bar the position moves
    // to the signal via a market order that crosses half of the fractional
    // `spread` (a buy fills at price*(1 + spread/2), a sell at
    // price*(1 - spread/2)) and pays the fractional `fee` on the traded
    // notional. The slippage model then moves that fill further against the
    // trader and rounds it to the `tick_size` grid. It emits four positional columns:
    //     0 = equity (cumulative dollar PnL), 1 = pnl (per-step),
    //     2 = position, 3 = cost (per-step).
    // Causal: `signal_t` enters PnL only at t+1 (through the held position),
    // so a future signal never changes a past row. Default spread = fee = 0 is
    // frictionless. nan_policy: ignore - a NaN signal or price emits an
    // all-NaN row and leaves the account untouched.
    // The optional [min_position, max_position] cap clamps the target before
    // any order is computed; signals outside the range are treated as if they
    // were at the nearest boundary.
    // Optional risk exits (fractions; 0 = off): stop_loss, take_profit, and
    // trailing_stop, measured from the average entry fill (see RiskExits). On an
    // event whose price crosses a level, the target is forced flat, so the
    // position closes as a market order on that event; same-side targets are
    // then ignored until the target turns flat or reverses.
    // `size` reads the target as units (default), a notional amount, or a
    // weight of initial_capital + equity, converted at the event's price; the
    // units are kept until the target value changes or, with rebalance_band,
    // until the holding drifts (see detail::Sizer). A NaN signal skips the
    // step (nan_policy: ignore).
    class BacktestPriceTarget : public FunctorBase<BacktestPriceTarget, 2, 4> {
    public:
        // The market input (after the target) that marks a position: the price.
        // walk_forward's carry marks with it.
        static constexpr std::size_t mark_input = 0;

        BacktestPriceTarget(double spread = 0.0, double fee = 0.0,
                            double min_position = -std::numeric_limits<double>::infinity(),
                            double max_position = std::numeric_limits<double>::infinity(),
                            double multiplier = 1.0,
                            double fee_per_contract = 0.0,
                            double slippage = 0.0,
                            const std::string& slippage_unit = "fraction",
                            double tick_size = 0.0,
                            double stop_loss = 0.0, double take_profit = 0.0,
                            double trailing_stop = 0.0,
                            const std::string& size = "units",
                            double initial_capital = 0.0,
                            double rebalance_band = 0.0)
            : spread_(spread), fee_(fee),
              min_position_(min_position), max_position_(max_position),
              slippage_(slippage, slippage_unit, tick_size),
              exits_(stop_loss, take_profit, trailing_stop),
              sizer_(size, initial_capital, rebalance_band, multiplier),
              account_(multiplier, fee_per_contract)
        {
            require_finite("spread", spread_);
            require_finite("fee", fee_);
            require_finite("min_position", min_position_, kTakesNegInf);
            require_finite("max_position", max_position_, kTakesPosInf);
            if (spread_ < 0.0) throw std::invalid_argument("spread must be non-negative.");
            if (min_position_ > max_position_)
                throw std::invalid_argument("min_position must not exceed max_position.");
        }

        void reset() override { account_.reset(); exits_.reset(); sizer_.reset(); }

        ResultTuple call(const InputArray& inputs) override {
            const double signal = inputs[0];
            const double price = inputs[1];
            if (isnan2(signal) || isnan2(price)) {
                const double nan = std::numeric_limits<double>::quiet_NaN();
                return std::make_tuple(nan, nan, nan, nan);   // ignore
            }
            double sig = signal;
            const double before = account_.position();
            if (exits_.active()) {
                if (!isnan2(exits_.event_exit(before, price))) exits_.exited(before);
                sig = exits_.filter(sig);
            }
            sig = sizer_.units(sig, price, before, account_.equity_at(price));
            if (isnan2(sig)) sig = before;                  // cannot size: hold
            const double target = std::clamp(sig, min_position_, max_position_);
            const double dpos = target - account_.position();
            const double side = (dpos > 0.0) ? 1.0 : (dpos < 0.0 ? -1.0 : 0.0);
            const double half_spread = price * (1.0 + side * spread_ / 2.0);
            const double fill_price = side == 0.0 ? price
                                    : slippage_.fill(half_spread, side > 0.0);
            auto [equity, pnl, position, cost] =
                account_.step(price, dpos, fill_price, fee_);
            if (exits_.active()) {
                if (dpos != 0.0) exits_.on_fill(before, position, fill_price);
                exits_.update_best(position, price, price);
            }
            return std::make_tuple(equity, pnl, position, cost);
        }

    private:
        double spread_;
        double fee_;
        double min_position_;
        double max_position_;
        detail::Slippage slippage_;
        detail::RiskExits exits_;
        detail::Sizer sizer_;
        detail::PnLAccount account_;
    };

} // namespace screamer

#endif // SCREAMER_BACKTEST_PRICE_TARGET_H
