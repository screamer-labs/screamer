#ifndef SCREAMER_BACKTEST_OHLCV_TARGET_H
#define SCREAMER_BACKTEST_OHLCV_TARGET_H

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include "screamer/common/functor_base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/ohlc_target_engine.h"

namespace screamer {

    // BacktestOHLCVTarget: 6 -> 4. BacktestOHLCTarget with the bar's volume as a
    // sixth input and a participation limit: an order trades at most
    // max_participation * volume units on a bar, and the rest carries to the
    // next bars (the engine keeps working toward the last target). Inputs
    // (target, open, high, low, close, volume); outputs
    // [equity, pnl, position, cost]. Risk exits are not limited by volume.
    // A NaN or negative volume skips the bar (nan_policy: ignore).
    class BacktestOHLCVTarget : public FunctorBase<BacktestOHLCVTarget, 6, 4> {
    public:
        BacktestOHLCVTarget(double max_participation = 1.0,
                            double taker_fee = 0.0, double tick_size = 0.0,
                            double min_position = -std::numeric_limits<double>::infinity(),
                            double max_position = std::numeric_limits<double>::infinity(),
                            double multiplier = 1.0,
                            double fee_per_contract = 0.0,
                            double slippage = 0.0,
                            const std::string& slippage_unit = "fraction",
                            const std::string& fill_at = "open",
                            int delay = 1,
                            double stop_loss = 0.0, double take_profit = 0.0,
                            double trailing_stop = 0.0,
                            const std::string& size = "units",
                            double initial_capital = 0.0,
                            double rebalance_band = 0.0)
            : participation_(max_participation),
              engine_(taker_fee, tick_size, min_position, max_position, multiplier,
                      fee_per_contract, slippage, slippage_unit, fill_at, delay,
                      stop_loss, take_profit, trailing_stop,
                      size, initial_capital, rebalance_band)
        {
            if (!(participation_ > 0.0) || !std::isfinite(participation_))
                throw std::invalid_argument("max_participation must be finite and positive.");
        }

        void reset() override { engine_.reset(); }

        ResultTuple call(const InputArray& in) override {
            const double volume = in[5];
            const double cap = (volume >= 0.0) ? participation_ * volume
                                               : std::numeric_limits<double>::quiet_NaN();
            return engine_.step(in[0], in[1], in[2], in[3], in[4], cap);
        }

    private:
        double participation_;
        detail::OHLCTargetEngine engine_;
    };

} // namespace screamer

#endif // SCREAMER_BACKTEST_OHLCV_TARGET_H
