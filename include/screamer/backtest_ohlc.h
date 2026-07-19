#ifndef SCREAMER_BACKTEST_OHLC_H
#define SCREAMER_BACKTEST_OHLC_H

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include "screamer/common/functor_base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/pnl_account.h"

namespace screamer {

    // BacktestOHLC: 6 -> 4. A lean, directional backtest on OHLC bars. Causal by
    // design: the `target_position` and `limit_price` you pass on bar t are decided
    // from data through bar t's close, and the engine executes them on bar t+1, so
    // no manual lag is needed (feed the raw signal). The deferred order is live over
    // the next bar:
    //   - a market order (limit_price NaN) fills at that bar's open, crossing half
    //     the fractional `spread`, and pays `taker_fee`;
    //   - a limit order fills only if the bar's range reaches its price -
    //     `fill = "touch"` when the range touches the level, `"breach"` when it
    //     trades through - at the limit price, paying `maker_fee` (negative for a
    //     rebate). An unreached limit holds the position.
    // Positions are marked to the close. Inputs are
    // (target_position, limit_price, open, high, low, close); the four outputs are
    // [equity, pnl, position, cost]. A bar has no intra-bar path, so two-sided
    // market-making is out of scope here (use BacktestL1). nan_policy: ignore for
    // the bar fields; a NaN limit_price is a market order (not a skip); a NaN target
    // places no order for the next bar (the position holds).
    class BacktestOHLC : public FunctorBase<BacktestOHLC, 6, 4> {
    public:
        BacktestOHLC(double spread = 0.0, double taker_fee = 0.0,
                     double maker_fee = 0.0, const std::string& fill = "touch",
                     double min_position = -std::numeric_limits<double>::infinity(),
                     double max_position = std::numeric_limits<double>::infinity())
            : spread_(spread), taker_fee_(taker_fee), maker_fee_(maker_fee),
              breach_(parse_fill(fill)),
              min_position_(min_position), max_position_(max_position)
        {
            if (spread_ < 0.0) {
                throw std::invalid_argument("spread must be non-negative.");
            }
            if (min_position_ > max_position_) {
                throw std::invalid_argument("min_position must be <= max_position.");
            }
            reset();
        }

        void reset() override {
            account_.reset();
            has_pending_ = false;
            pending_target_ = 0.0;
            pending_limit_ = 0.0;
        }

        ResultTuple call(const InputArray& inputs) override {
            const double target = inputs[0];
            const double limit  = inputs[1];   // NaN = market order
            const double open   = inputs[2];
            const double high   = inputs[3];
            const double low    = inputs[4];
            const double close  = inputs[5];
            if (isnan2(open) || isnan2(high) || isnan2(low) || isnan2(close)) {
                const double nan = std::numeric_limits<double>::quiet_NaN();
                return std::make_tuple(nan, nan, nan, nan);   // ignore the bad bar; hold pending
            }

            // Execute the order decided on the PREVIOUS bar against THIS bar (causal:
            // decide on the close, trade the next bar). Bars carry no per-level
            // volume, so a triggered order fills its full target (no participation);
            // see BacktestL1Trades for volume-aware fills.
            double fill_dpos = 0.0, fill_price = close, fee = 0.0;
            if (has_pending_) {
                const double target = std::clamp(pending_target_, min_position_, max_position_);
                const double dpos = target - account_.position();
                if (dpos != 0.0) {
                    const double side = (dpos > 0.0) ? 1.0 : -1.0;
                    if (isnan2(pending_limit_)) {             // market at open (taker)
                        fill_dpos  = dpos;
                        fill_price = open * (1.0 + side * spread_ / 2.0);
                        fee        = taker_fee_;
                    } else {                                  // resting limit (maker)
                        const bool buy = dpos > 0.0;
                        const bool hit = buy
                            ? (breach_ ? (low  < pending_limit_) : (low  <= pending_limit_))
                            : (breach_ ? (high > pending_limit_) : (high >= pending_limit_));
                        if (hit) {
                            fill_dpos  = dpos;
                            fill_price = pending_limit_;
                            fee        = maker_fee_;
                        }
                    }
                }
            }
            auto [equity, pnl, position, cost] =
                account_.step(close, fill_dpos, fill_price, fee);

            // Store this bar's decision to execute on the next bar.
            if (isnan2(target)) {
                has_pending_ = false;                         // no order next bar; hold
            } else {
                has_pending_ = true;
                pending_target_ = target;
                pending_limit_ = limit;
            }
            return std::make_tuple(equity, pnl, position, cost);
        }

    private:
        static bool parse_fill(const std::string& fill) {
            if (fill == "touch") return false;
            if (fill == "breach") return true;
            throw std::invalid_argument("fill must be \"touch\" or \"breach\".");
        }

        double spread_;
        double taker_fee_;
        double maker_fee_;
        bool breach_;
        double min_position_;
        double max_position_;
        bool has_pending_;
        double pending_target_;
        double pending_limit_;
        detail::PnLAccount account_;
    };

} // namespace screamer

#endif // SCREAMER_BACKTEST_OHLC_H
