#ifndef SCREAMER_DETAIL_OHLC_TARGET_ENGINE_H
#define SCREAMER_DETAIL_OHLC_TARGET_ENGINE_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include "screamer/common/float_info.h"
#include "screamer/detail/ohlc_fill.h"
#include "screamer/detail/risk_exits.h"
#include "screamer/detail/sizer.h"
#include "screamer/detail/slippage.h"
#include "screamer/detail/target_front.h"

namespace screamer { namespace detail {

    // OHLCTargetEngine: the shared logic of BacktestOHLCTarget and
    // BacktestOHLCVTarget. Reach a target on OHLC bars by taking liquidity.
    //
    // Timing: the target decided on bar t (from its close) executes `delay` bars
    // later at the execution bar's `fill_at` price ("open", "close", "typical",
    // "ohlc4"), moved by the slippage model. delay = 0 needs fill_at = "close".
    //
    // Working target: the engine works toward the last finite target. A NaN
    // target keeps working toward it, so the unfilled part of a capped order
    // carries to the next bars. With complete fills this is the same as holding.
    //
    // Size: the target is in units, a notional amount, or a weight of equity
    // (see Sizer); it is converted to units at the execution reference price.
    // Inventory is capped to [min_position, max_position].
    //
    // Fill cap: step() takes `max_fill`, the most units this bar's order may
    // trade (infinity for no cap); the rest carries to later bars through the
    // working target. Risk exits are not capped.
    //
    // Risk exits (stop_loss, take_profit, trailing_stop; see RiskExits) run in
    // this order within a bar:
    //   fill_at = "open": a gap exit at the open for the position carried into
    //     the bar, then the due order at the open, then an exit at a level the
    //     bar's low or high reaches for the position now held.
    //   other fill_at: an exit for the carried position (at the open on a gap,
    //     else at the level), then the due order at its reference price.
    // A bar's own high (low) moves the trailing level only after the bar.
    class OHLCTargetEngine {
    public:
        using Row = std::tuple<double, double, double, double>;

        OHLCTargetEngine(double taker_fee, double tick_size,
                         double min_position, double max_position,
                         double multiplier, double fee_per_contract,
                         double slippage, const std::string& slippage_unit,
                         const std::string& fill_at, int delay,
                         double stop_loss, double take_profit, double trailing_stop,
                         const std::string& size, double initial_capital,
                         double rebalance_band)
            : core_(0.0, taker_fee, /*breach=*/false, /*participation=*/1.0,
                    Slippage(slippage, slippage_unit, tick_size),
                    min_position, max_position,
                    multiplier, 0.0, fee_per_contract),
              min_(min_position), max_(max_position),
              fill_at_(parse_fill_at(fill_at)), delay_(delay),
              exits_(stop_loss, take_profit, trailing_stop),
              sizer_(size, initial_capital, rebalance_band, multiplier)
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

        void reset() {
            core_.reset();
            exits_.reset();
            sizer_.reset();
            working_ = std::numeric_limits<double>::quiet_NaN();
            // A NaN slot is "no new target": the first `delay` bars only mark.
            std::fill(queue_.begin(), queue_.end(),
                      std::numeric_limits<double>::quiet_NaN());
            head_ = 0;
        }

        Row step(double target, double open, double high, double low, double close,
                 double max_fill) {
            double due = target;                     // delay 0: trade this bar
            if (delay_ > 0) {                        // FIFO of the last `delay` targets
                due = queue_[head_];
                queue_[head_] = target;
                head_ = (head_ + 1) % queue_.size();
            }
            if (isnan2(open) || isnan2(high) || isnan2(low) || isnan2(close) ||
                isnan2(max_fill)) {
                const double nan = std::numeric_limits<double>::quiet_NaN();
                return std::make_tuple(nan, nan, nan, nan);   // skipped bar
            }
            if (!isnan2(due)) working_ = due;
            const double ref = reference(open, high, low, close);
            if (!exits_.active())
                return order(working_, ref, max_fill, open, high, low, close);

            // Several fills can happen in one bar. Each core call marks to the
            // close, so the pnl and cost of the calls add up; equity and
            // position are those of the last call.
            double pnl = 0.0, cost = 0.0, eq = 0.0, pos = 0.0;
            auto add = [&](const Row& r) {
                eq = std::get<0>(r); pnl += std::get<1>(r);
                pos = std::get<2>(r); cost += std::get<3>(r);
            };
            const bool at_open = fill_at_ == FillAt::Open;
            const double carried = core_.position();
            // 1. Exit for the position carried into the bar.
            double px = exits_.bar_exit(carried, open, high, low, /*include_open=*/true);
            if (at_open && !isnan2(px) && px != open)
                px = std::numeric_limits<double>::quiet_NaN();
            if (!isnan2(px)) add(exit_at(px, open, high, low, close));
            // 2. The due order, after the latch.
            add(order(exits_.filter(working_), ref, max_fill, open, high, low, close));
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

        // Execute a target value as a market order at `ref`: convert it to units,
        // cap the trade at `max_fill`, and record the fill.
        Row order(double value, double ref, double max_fill,
                  double open, double high, double low, double close) {
            const double inf = std::numeric_limits<double>::infinity();
            double bp = inf, bs = 0.0, ap = -inf, as = 0.0;
            const double before = core_.position();
            const double units = sizer_.units(value, ref, before, core_.equity_at(ref));
            if (!isnan2(units)) {                    // NaN: hold, no order
                double goal = std::clamp(units, min_, max_);
                const double delta = goal - before;
                if (std::abs(delta) > max_fill) goal = before + (delta > 0.0 ? max_fill : -max_fill);
                auto [b_p, b_s, a_p, a_s] = target_to_quote(goal, before, min_, max_);
                bp = b_p; bs = b_s; ap = a_p; as = a_s;
            }
            auto r = core_.quote(bp, bs, ap, as, open, high, low, close, ref);
            if (core_.position() != before)
                exits_.on_fill(before, core_.position(), core_.last_fill_price());
            return r;
        }

        // Close the whole position at `price` and latch. Exits are not capped.
        Row exit_at(double price, double open, double high, double low, double close) {
            const double held = core_.position();
            exits_.exited(held);
            auto [bp, bs, ap, as] = target_to_quote(0.0, held, min_, max_);
            auto r = core_.quote(bp, bs, ap, as, open, high, low, close, price);
            if (core_.position() != held)
                exits_.on_fill(held, core_.position(), core_.last_fill_price());
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

        OHLCFill core_;
        double min_, max_;
        FillAt fill_at_;
        int delay_;
        RiskExits exits_;
        Sizer sizer_;
        double working_ = std::numeric_limits<double>::quiet_NaN();
        std::vector<double> queue_;
        std::size_t head_ = 0;
    };

}} // namespace screamer::detail

#endif // SCREAMER_DETAIL_OHLC_TARGET_ENGINE_H
