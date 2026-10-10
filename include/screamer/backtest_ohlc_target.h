#ifndef SCREAMER_BACKTEST_OHLC_TARGET_H
#define SCREAMER_BACKTEST_OHLC_TARGET_H

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "screamer/common/functor_base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/ohlc_fill.h"
#include "screamer/detail/slippage.h"
#include "screamer/detail/target_front.h"

namespace screamer {

    // BacktestOHLCTarget: 5 -> 4. Reach a target position on OHLC bars by taking
    // liquidity. Causal: the target decided on bar t (from its close) executes
    // `delay` bars later as a market order, sized to clamp(target, cap) - position.
    // `fill_at` picks the execution bar's reference price: "open" (default),
    // "close", "typical" = (high + low + close) / 3, or "ohlc4" =
    // (open + high + low + close) / 4. The slippage model moves the fill against
    // the trader and rounds it to the tick grid. delay = 0 is allowed only with
    // fill_at = "close" (market-on-close of the signal bar); any other price of
    // the signal bar was already in the past when the signal was known.
    // Inventory is capped to [min_position, max_position].
    // Outputs [equity, pnl, position, cost].
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
                           int delay = 1)
            : core_(0.0, taker_fee, /*breach=*/false, /*participation=*/1.0,
                    detail::Slippage(slippage, slippage_unit, tick_size),
                    min_position, max_position,
                    multiplier, 0.0, fee_per_contract),
              min_(min_position), max_(max_position),
              fill_at_(parse_fill_at(fill_at)), delay_(delay)
        {
            if (delay_ < 0)
                throw std::invalid_argument("delay must be non-negative.");
            if (delay_ == 0 && fill_at_ != FillAt::Close)
                throw std::invalid_argument(
                    "delay=0 requires fill_at=\"close\": a target decided at a bar's "
                    "close cannot trade at an earlier price of that bar.");
            queue_.assign(static_cast<std::size_t>(delay_), 0.0);
            reset();
        }

        void reset() override {
            core_.reset();
            // A NaN slot is "no order": the first `delay` bars only mark.
            std::fill(queue_.begin(), queue_.end(),
                      std::numeric_limits<double>::quiet_NaN());
            head_ = 0;
        }

        ResultTuple call(const InputArray& in) override {
            const double target = in[0];
            const double open = in[1], high = in[2], low = in[3], close = in[4];
            double due = target;                     // delay 0: trade this bar
            if (delay_ > 0) {                        // FIFO of the last `delay` targets
                due = queue_[head_];
                queue_[head_] = target;
                head_ = (head_ + 1) % queue_.size();
            }
            const double inf = std::numeric_limits<double>::infinity();
            double bp = inf, bs = 0.0, ap = -inf, as = 0.0;
            if (!isnan2(due)) {                      // NaN target: hold, no order
                auto [b_p, b_s, a_p, a_s] =
                    detail::target_to_quote(due, core_.position(), min_, max_);
                bp = b_p; bs = b_s; ap = a_p; as = a_s;
            }
            return core_.quote(bp, bs, ap, as, open, high, low, close,
                               reference(open, high, low, close));
        }

    private:
        enum class FillAt { Open, Close, Typical, OHLC4 };

        static FillAt parse_fill_at(const std::string& f) {
            if (f == "open") return FillAt::Open;
            if (f == "close") return FillAt::Close;
            if (f == "typical") return FillAt::Typical;
            if (f == "ohlc4") return FillAt::OHLC4;
            throw std::invalid_argument(
                "fill_at must be \"open\", \"close\", \"typical\", or \"ohlc4\".");
        }

        double reference(double open, double high, double low, double close) const {
            switch (fill_at_) {
                case FillAt::Open:    return open;
                case FillAt::Close:   return close;
                case FillAt::Typical: return (high + low + close) / 3.0;
                case FillAt::OHLC4:   return (open + high + low + close) / 4.0;
            }
            return open;
        }

        detail::OHLCFill core_;
        double min_, max_;
        FillAt fill_at_;
        int delay_;
        std::vector<double> queue_;
        std::size_t head_ = 0;
    };

} // namespace screamer

#endif // SCREAMER_BACKTEST_OHLC_TARGET_H
