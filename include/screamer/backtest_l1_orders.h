#ifndef SCREAMER_BACKTEST_L1_ORDERS_H
#define SCREAMER_BACKTEST_L1_ORDERS_H

#include <limits>
#include <stdexcept>
#include <string>
#include "screamer/common/functor_base.h"
#include "screamer/detail/l1_fill.h"

namespace screamer {

    // BacktestL1Orders: 8 -> 4. Post a two-sided quote against an L1 book.
    // Inputs (bid_price, bid_size, ask_price, ask_size,
    //         market_bid, market_ask, market_bid_size, market_ask_size).
    // A resting bid fills when the market ask trades through it (breach) or
    // locks it in touch mode; a quote submitted already crossing is a taker.
    // The sell side is symmetric. Inventory capped to [min_position,
    // max_position]. Outputs [equity, pnl, position, cost].
    // nan_policy: ignore on the market quote; a NaN own-quote price means
    // that side is not quoted.
    class BacktestL1Orders : public FunctorBase<BacktestL1Orders, 8, 4> {
    public:
        BacktestL1Orders(double maker_fee = 0.0, double taker_fee = 0.0,
                         const std::string& fill = "breach",
                         double participation_ratio = 1.0, double tick_size = 0.0,
                         double min_position = -std::numeric_limits<double>::infinity(),
                         double max_position = std::numeric_limits<double>::infinity())
            : core_(maker_fee, taker_fee, parse_fill(fill),
                    parse_participation(participation_ratio), tick_size,
                    min_position, max_position) {}

        void reset() override { core_.reset(); }

        ResultTuple call(const InputArray& in) override {
            return core_.quote(in[0], in[1], in[2], in[3],
                               in[4], in[5], in[6], in[7]);
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

        detail::L1Fill core_;
    };

} // namespace screamer

#endif // SCREAMER_BACKTEST_L1_ORDERS_H
