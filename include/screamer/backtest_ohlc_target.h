#ifndef SCREAMER_BACKTEST_OHLC_TARGET_H
#define SCREAMER_BACKTEST_OHLC_TARGET_H

#include <cstddef>
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
        static constexpr const char* kTypeName = "BacktestOHLCTarget";
        static constexpr int kStateVersion = 1;

        template <class Ar>
        void settings(Ar& ar) {
            const auto& core = engine_.core();
            ar("taker_fee", core.taker_fee());
            ar("tick_size", core.slippage().tick_size());
            ar("min_position", core.min_position());
            ar("max_position", core.max_position());
            ar("multiplier", core.multiplier());
            ar("fee_per_contract", core.taker_fee_per_contract());
            ar("slippage", core.slippage().amount());
            ar("slippage_unit", std::string(core.slippage().unit_name()));
            ar("fill_at", std::string(engine_.fill_at_name()));
            ar("delay", engine_.delay());
            ar("stop_loss", engine_.exits().stop_loss());
            ar("take_profit", engine_.exits().take_profit());
            ar("trailing_stop", engine_.exits().trailing_stop());
            ar("size", std::string(engine_.sizer().size_name()));
            ar("initial_capital", engine_.sizer().initial_capital());
            ar("rebalance_band", engine_.sizer().rebalance_band());
        }

        template <class Ar>
        void state(Ar& ar) { engine_.state(ar); }

        // The market input (after the target) that marks a position: the close.
        // walk_forward's carry marks with it.
        static constexpr std::size_t mark_input = 3;

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
