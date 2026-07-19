#ifndef SCREAMER_BACKTEST_SIGNAL_H
#define SCREAMER_BACKTEST_SIGNAL_H

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <tuple>
#include "screamer/common/functor_base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/pnl_account.h"

namespace screamer {

    // BacktestSignal: 2 -> 4. Backtest a position signal against a price series,
    // marking to market and charging a taker cost. `signal` is the target
    // position in units (sign is long / short / flat, any magnitude); `price` is
    // the mark (mid). Each bar the position moves to the signal via a market
    // order that crosses half of the fractional `spread` (a buy fills at
    // price*(1 + spread/2), a sell at price*(1 - spread/2)) and pays the
    // fractional `fee` on the traded notional. It emits four positional columns:
    //     0 = equity (cumulative dollar PnL), 1 = pnl (per-step),
    //     2 = position, 3 = cost (per-step).
    // Causal: `signal_t` enters PnL only at t+1 (through the held position), so a
    // future signal never changes a past row. Default spread = fee = 0 is
    // frictionless. nan_policy: ignore - a NaN signal or price emits an all-NaN
    // row and leaves the account untouched.
    // The optional [min_position, max_position] cap clamps the target before any
    // order is computed; signals outside the range are treated as if they were at
    // the nearest boundary.
    class BacktestSignal : public FunctorBase<BacktestSignal, 2, 4> {
    public:
        BacktestSignal(double spread = 0.0, double fee = 0.0,
                       double min_position = -std::numeric_limits<double>::infinity(),
                       double max_position = std::numeric_limits<double>::infinity())
            : spread_(spread), fee_(fee),
              min_position_(min_position), max_position_(max_position)
        {
            if (spread_ < 0.0) throw std::invalid_argument("spread must be non-negative.");
            if (min_position_ > max_position_)
                throw std::invalid_argument("min_position must not exceed max_position.");
        }

        void reset() override { account_.reset(); }

        ResultTuple call(const InputArray& inputs) override {
            const double signal = inputs[0];
            const double price = inputs[1];
            if (isnan2(signal) || isnan2(price)) {
                const double nan = std::numeric_limits<double>::quiet_NaN();
                return std::make_tuple(nan, nan, nan, nan);   // ignore
            }
            const double target = std::clamp(signal, min_position_, max_position_);
            const double dpos = target - account_.position();
            const double side = (dpos > 0.0) ? 1.0 : (dpos < 0.0 ? -1.0 : 0.0);
            const double fill_price = price * (1.0 + side * spread_ / 2.0);
            auto [equity, pnl, position, cost] =
                account_.step(price, dpos, fill_price, fee_);
            return std::make_tuple(equity, pnl, position, cost);
        }

    private:
        double spread_;
        double fee_;
        double min_position_;
        double max_position_;
        detail::PnLAccount account_;
    };

} // namespace screamer

#endif // SCREAMER_BACKTEST_SIGNAL_H
