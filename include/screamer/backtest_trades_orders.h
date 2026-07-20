#ifndef SCREAMER_BACKTEST_TRADES_ORDERS_H
#define SCREAMER_BACKTEST_TRADES_ORDERS_H

#include <limits>
#include <stdexcept>
#include <string>
#include "screamer/common/functor_base.h"
#include "screamer/detail/trades_fill.h"

namespace screamer {

    // BacktestTradesOrders: 6 -> 4. Post a two-sided quote on the trade tape;
    // a resting bid fills when a sell-print crosses it (touch/breach), a
    // marketable bid (market_limit returns +inf) fills at the print price as a
    // taker. The sell side is symmetric. Inventory is capped to
    // [min_position, max_position]. Outputs [equity, pnl, position, cost].
    class BacktestTradesOrders : public FunctorBase<BacktestTradesOrders, 6, 4> {
    public:
        BacktestTradesOrders(double maker_fee = 0.0, double taker_fee = 0.0,
                             const std::string& fill = "touch",
                             double participation_ratio = 1.0, double tick_size = 0.0,
                             double min_position = -std::numeric_limits<double>::infinity(),
                             double max_position = std::numeric_limits<double>::infinity())
            : core_(maker_fee, taker_fee, parse_fill(fill),
                    parse_participation(participation_ratio), tick_size,
                    min_position, max_position) {}
        void reset() override { core_.reset(); }
        ResultTuple call(const InputArray& in) override {
            return core_.quote(in[0], in[1], in[2], in[3], in[4], in[5]);
        }
    private:
        static bool parse_fill(const std::string& f) {
            if (f == "touch") return false;
            if (f == "breach") return true;
            throw std::invalid_argument("fill must be \"touch\" or \"breach\".");
        }
        static double parse_participation(double p) {
            if (!(p > 0.0) || p > 1.0)
                throw std::invalid_argument("participation_ratio must be in (0, 1].");
            return p;
        }
        detail::TradesFill core_;
    };

} // namespace screamer

#endif // SCREAMER_BACKTEST_TRADES_ORDERS_H
