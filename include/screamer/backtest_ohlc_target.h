#ifndef SCREAMER_BACKTEST_OHLC_TARGET_H
#define SCREAMER_BACKTEST_OHLC_TARGET_H

#include <limits>
#include <string>
#include "screamer/common/functor_base.h"
#include "screamer/detail/ohlc_target_engine.h"

namespace screamer {

    // BacktestOHLCTarget: 5 -> 4. Reach a target position on OHLC bars by taking
    // liquidity. Inputs (target, open, high, low, close); outputs
    // [equity, pnl, position, cost]. The target decided on bar t executes
    // `delay` bars later at the `fill_at` price, moved by the slippage model;
    // `size` reads the target as units, a notional amount, or a weight of
    // equity; stop_loss / take_profit / trailing_stop add risk exits. See
    // detail::OHLCTargetEngine for the timing and event order. Orders are not
    // limited by volume; BacktestOHLCVTarget adds that.
    class BacktestOHLCTarget : public FunctorBase<BacktestOHLCTarget, 5, 4> {
    public:
        BacktestOHLCTarget(double taker_fee = 0.0, double tick_size = 0.0,
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
            : engine_(taker_fee, tick_size, min_position, max_position, multiplier,
                      fee_per_contract, slippage, slippage_unit, fill_at, delay,
                      stop_loss, take_profit, trailing_stop,
                      size, initial_capital, rebalance_band) {}

        void reset() override { engine_.reset(); }

        ResultTuple call(const InputArray& in) override {
            return engine_.step(in[0], in[1], in[2], in[3], in[4],
                                std::numeric_limits<double>::infinity());
        }

    private:
        detail::OHLCTargetEngine engine_;
    };

} // namespace screamer

#endif // SCREAMER_BACKTEST_OHLC_TARGET_H
