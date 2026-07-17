#ifndef SCREAMER_BACKTEST_L1_H
#define SCREAMER_BACKTEST_L1_H

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include "screamer/common/functor_base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/pnl_account.h"

namespace screamer {

    // BacktestL1: 8 -> 4. A market-making backtest against a top-of-book (L1)
    // stream. Each event carries the market quote (bid, ask, bid_size, ask_size)
    // and the strategy's two-sided quote (my_bid, my_bid_size, my_ask, my_ask_size)
    // (align your quote stream to the market with combine_latest upstream). Both
    // sides rest; either or both can be lifted on one event, so the position is
    // whatever the market fills.
    //
    // The resting buy at my_bid fills when the market can sell into it - the market
    // ask reaches my_bid (`ask <= my_bid` for `"touch"`, strict for `"breach"`) -
    // for min(my_bid_size, ask_size) at my_bid; the resting sell fills symmetrically
    // when `bid >= my_ask`. Both are maker fills at the quoted price paying
    // `maker_fee` (negative for a rebate); the favorable fill versus the mid is the
    // captured spread. Fills are capped so the position stays within
    // [min_position, max_position] (defaults unbounded), full up to the available
    // size (no queue-position modelling). Positions mark to the mid. Outputs are
    // [equity, pnl, position, cost]. nan_policy: ignore for the market quote; a NaN
    // own-quote price means that side is not quoted.
    class BacktestL1 : public FunctorBase<BacktestL1, 8, 4> {
    public:
        BacktestL1(double maker_fee = 0.0, const std::string& fill = "touch",
                   double max_position = std::numeric_limits<double>::infinity(),
                   double min_position = -std::numeric_limits<double>::infinity())
            : maker_fee_(maker_fee), breach_(parse_fill(fill)),
              max_position_(max_position), min_position_(min_position)
        {
            if (min_position_ > max_position_) {
                throw std::invalid_argument("min_position must not exceed max_position.");
            }
        }

        void reset() override { account_.reset(); }

        ResultTuple call(const InputArray& inputs) override {
            const double bid = inputs[0], ask = inputs[1];
            const double bid_size = inputs[2], ask_size = inputs[3];
            const double my_bid = inputs[4], my_bid_size = inputs[5];
            const double my_ask = inputs[6], my_ask_size = inputs[7];
            if (isnan2(bid) || isnan2(ask) || isnan2(bid_size) || isnan2(ask_size)) {
                const double nan = std::numeric_limits<double>::quiet_NaN();
                return std::make_tuple(nan, nan, nan, nan);   // ignore
            }
            const double mid = 0.5 * (bid + ask);
            const double pos = account_.position();

            // Resting buy: filled when the market ask reaches our bid.
            double buy_dpos = 0.0;
            if (!isnan2(my_bid) && !isnan2(my_bid_size) && my_bid_size > 0.0) {
                const bool hit = breach_ ? (ask < my_bid) : (ask <= my_bid);
                if (hit) {
                    const double room = max_position_ - pos;   // inventory cap
                    buy_dpos = std::min(std::min(my_bid_size, ask_size),
                                        std::max(room, 0.0));
                }
            }
            // Resting sell: filled when the market bid reaches our ask.
            double sell_dpos = 0.0;
            if (!isnan2(my_ask) && !isnan2(my_ask_size) && my_ask_size > 0.0) {
                const bool hit = breach_ ? (bid > my_ask) : (bid >= my_ask);
                if (hit) {
                    const double room = pos - min_position_;
                    sell_dpos = -std::min(std::min(my_ask_size, bid_size),
                                          std::max(room, 0.0));
                }
            }

            // First step always marks (against the mid); a second step for the
            // other side adds only its trade cost (its mark move is zero).
            auto [eq, pnl, position, cost] = account_.step(
                mid, buy_dpos, (buy_dpos != 0.0) ? my_bid : mid,
                (buy_dpos != 0.0) ? maker_fee_ : 0.0);
            if (sell_dpos != 0.0) {
                auto [eq2, pnl2, pos2, cost2] =
                    account_.step(mid, sell_dpos, my_ask, maker_fee_);
                eq = eq2; pnl += pnl2; position = pos2; cost += cost2;
            }
            return std::make_tuple(eq, pnl, position, cost);
        }

    private:
        static bool parse_fill(const std::string& fill) {
            if (fill == "touch") return false;
            if (fill == "breach") return true;
            throw std::invalid_argument("fill must be \"touch\" or \"breach\".");
        }

        double maker_fee_;
        bool breach_;
        double max_position_;
        double min_position_;
        detail::PnLAccount account_;
    };

} // namespace screamer

#endif // SCREAMER_BACKTEST_L1_H
