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
#include "screamer/detail/risk_exits.h"
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
    //
    // Optional risk exits (fractions; 0 = off): stop_loss, take_profit, and
    // trailing_stop, measured from the average entry fill (see RiskExits). In a
    // bar, the events run in this order:
    //   fill_at = "open": a gap exit at the open for the position carried into
    //     the bar, then the due order at the open, then an exit at a level the
    //     bar's low or high reaches for the position now held.
    //   other fill_at: an exit for the carried position (at the open on a gap,
    //     else at the level), then the due order at its reference price.
    // An exit is a market order at that price, moved by the slippage model and
    // paying taker_fee. A bar's own high (low) moves the trailing level only
    // after the bar, because the bar does not say whether the extreme came
    // before or after its low (high). After an exit, same-side targets are ignored until the
    // target turns flat or reverses.
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
                           int delay = 1,
                           double stop_loss = 0.0, double take_profit = 0.0,
                           double trailing_stop = 0.0)
            : core_(0.0, taker_fee, /*breach=*/false, /*participation=*/1.0,
                    detail::Slippage(slippage, slippage_unit, tick_size),
                    min_position, max_position,
                    multiplier, 0.0, fee_per_contract),
              min_(min_position), max_(max_position),
              fill_at_(parse_fill_at(fill_at)), delay_(delay),
              exits_(stop_loss, take_profit, trailing_stop)
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
            exits_.reset();
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
            const double ref = reference(open, high, low, close);
            if (!exits_.active())
                return order(due, open, high, low, close, ref);
            if (isnan2(open) || isnan2(high) || isnan2(low) || isnan2(close))
                return core_.quote(0, 0, 0, 0, open, high, low, close);   // all-NaN row

            // Several fills can happen in one bar. Each core call marks to the
            // close, so the pnl and cost of the calls add up; equity and
            // position are those of the last call.
            double pnl = 0.0, cost = 0.0, eq = 0.0, pos = 0.0;
            auto add = [&](const ResultTuple& r) {
                eq = std::get<0>(r); pnl += std::get<1>(r);
                pos = std::get<2>(r); cost += std::get<3>(r);
            };
            const bool at_open = fill_at_ == FillAt::Open;
            const double carried = core_.position();
            // 1. Exit for the position carried into the bar.
            double px = exits_.bar_exit(carried, open, high, low, /*include_open=*/true);
            if (at_open && !isnan2(px) && px != open) px = std::numeric_limits<double>::quiet_NaN();
            if (!isnan2(px)) add(exit_at(px, open, high, low, close));
            // 2. The due order, after the latch.
            add(order(exits_.filter(due), open, high, low, close, ref));
            // 3. With fill_at = "open", the position held after the open can still
            //    reach a level during the bar.
            if (at_open) {
                exits_.update_best(core_.position(), open, open);
                const double held = core_.position();
                const double lv = exits_.bar_exit(held, open, high, low, /*include_open=*/false);
                if (!isnan2(lv)) add(exit_at(lv, open, high, low, close));
                exits_.update_best(core_.position(), high, low);
            } else {
                // A position carried through the bar on the same side (resized or
                // not) saw the whole bar; one opened at the reference price did not.
                const double now = core_.position();
                if (carried != 0.0 && now != 0.0 && (carried > 0.0) == (now > 0.0))
                    exits_.update_best(now, high, low);
            }
            return std::make_tuple(eq, pnl, pos, cost);
        }

    private:
        enum class FillAt { Open, Close, Typical, OHLC4 };

        // Execute a target as a market order at `ref`, and record the fill.
        ResultTuple order(double target, double open, double high, double low,
                          double close, double ref) {
            const double inf = std::numeric_limits<double>::infinity();
            double bp = inf, bs = 0.0, ap = -inf, as = 0.0;
            if (!isnan2(target)) {                   // NaN target: hold, no order
                auto [b_p, b_s, a_p, a_s] =
                    detail::target_to_quote(target, core_.position(), min_, max_);
                bp = b_p; bs = b_s; ap = a_p; as = a_s;
            }
            const double before = core_.position();
            auto r = core_.quote(bp, bs, ap, as, open, high, low, close, ref);
            if (core_.position() != before)
                exits_.on_fill(before, core_.position(), core_.last_fill_price());
            return r;
        }

        // Close the whole position at `price` and latch.
        ResultTuple exit_at(double price, double open, double high, double low,
                            double close) {
            const double held = core_.position();
            exits_.exited(held);
            auto r = order(0.0, open, high, low, close, price);
            return r;
        }

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
        detail::RiskExits exits_;
        std::vector<double> queue_;
        std::size_t head_ = 0;
    };

} // namespace screamer

#endif // SCREAMER_BACKTEST_OHLC_TARGET_H
