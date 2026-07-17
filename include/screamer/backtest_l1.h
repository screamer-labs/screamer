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

    // BacktestL1: 8 -> 4. Two-sided market maker against top-of-book quotes ONLY.
    // Fills are a documented heuristic (see the reference page Limitations box):
    // a resting quote fills full when the opposite side crosses through it, and in
    // "touch" mode also fills a participation partial once per lock episode. A quote
    // that appears already marketable is a taker (fills at market + tick_size). A
    // resting quote the market runs through is a maker fill at the quoted price.
    // Inputs (bid, ask, bid_size, ask_size, my_bid, my_bid_size, my_ask,
    // my_ask_size); positions mark to the mid. Prefer BacktestL1Trades when a trade
    // feed is available. Outputs [equity, pnl, position, cost]. nan_policy: ignore on
    // the market quote; a NaN own-quote price means that side is not quoted.
    class BacktestL1 : public FunctorBase<BacktestL1, 8, 4> {
    public:
        BacktestL1(double maker_fee = 0.0, double taker_fee = 0.0,
                   const std::string& fill = "breach",
                   double participation_ratio = 1.0, double tick_size = 0.0,
                   double max_position = std::numeric_limits<double>::infinity(),
                   double min_position = -std::numeric_limits<double>::infinity())
            : maker_fee_(maker_fee), taker_fee_(taker_fee), breach_(parse_fill(fill)),
              participation_(parse_participation(participation_ratio)),
              tick_size_(tick_size), max_position_(max_position),
              min_position_(min_position)
        {
            if (min_position_ > max_position_)
                throw std::invalid_argument("min_position must not exceed max_position.");
            if (tick_size_ < 0.0)
                throw std::invalid_argument("tick_size must be non-negative.");
            reset();
        }

        void reset() override {
            account_.reset();
            bid_passive_ = false; ask_passive_ = false;
            bid_locked_ = false; ask_locked_ = false;
        }

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
            double eq = 0, pnl = 0, position = account_.position(), cost = 0; bool did = false;

            // Buy side: resting buy at my_bid filled by the market ask.
            const double room_buy = std::max(max_position_ - account_.position(), 0.0);
            SideFill b = compute_side(/*buy=*/true, my_bid, my_bid_size, ask, ask_size,
                                      room_buy, bid_passive_, bid_locked_);
            if (b.dpos != 0.0) {
                auto [e, p, pos, c] = account_.step(mid, b.dpos, b.fill_price,
                                                    b.is_taker ? taker_fee_ : maker_fee_);
                eq = e; pnl += p; position = pos; cost += c; did = true;
            }

            // Sell side: resting sell at my_ask filled by the market bid.
            const double room_sell = std::max(account_.position() - min_position_, 0.0);
            SideFill s = compute_side(/*buy=*/false, my_ask, my_ask_size, bid, bid_size,
                                      room_sell, ask_passive_, ask_locked_);
            if (s.dpos != 0.0) {
                auto [e, p, pos, c] = account_.step(mid, s.dpos, s.fill_price,
                                                    s.is_taker ? taker_fee_ : maker_fee_);
                eq = e; pnl += p; position = pos; cost += c; did = true;
            }

            if (!did) {
                auto [e, p, pos, c] = account_.step(mid, 0.0, mid, 0.0);  // mark only
                eq = e; pnl = p; position = pos; cost = c;
            }
            return std::make_tuple(eq, pnl, position, cost);
        }

    private:
        struct SideFill { double dpos; double fill_price; bool is_taker; };

        // Compute one side's fill and update that side's lock/passive state.
        // buy: resting buy at my_price hit by opp (=ask); sell: resting sell at
        // my_price hit by opp (=bid).
        SideFill compute_side(bool buy, double my_price, double my_size,
                              double opp_price, double opp_size, double room,
                              bool& passive_prev, bool& locked) {
            SideFill r{0.0, my_price, false};
            if (isnan2(my_price) || isnan2(my_size) || my_size <= 0.0 || room <= 0.0) {
                passive_prev = false; locked = false; return r;   // no quote this event
            }
            const bool through = buy ? (opp_price < my_price) : (opp_price > my_price);
            const bool lock    = (opp_price == my_price);
            const double remaining = std::min(my_size, room);

            if (through) {
                if (passive_prev) {                       // run over while resting: maker at my_price
                    r.dpos = buy ? remaining : -remaining;
                    r.fill_price = my_price; r.is_taker = false;
                } else {                                  // submitted already crossing: taker + slippage
                    const double disp = std::min(remaining, opp_size);
                    const double over = remaining - disp;
                    const double slip = buy ? tick_size_ : -tick_size_;
                    const double vwap = (disp * opp_price + over * (opp_price + slip))
                                        / remaining;
                    r.dpos = buy ? remaining : -remaining;
                    r.fill_price = vwap; r.is_taker = true;
                }
                passive_prev = false; locked = false;     // order consumed
            } else if (lock) {
                if (!breach_ && !locked) {                // touch mode, first event of this lock
                    const double filled = std::min(remaining, participation_ * opp_size);
                    if (filled > 0.0) {
                        r.dpos = buy ? filled : -filled;
                        r.fill_price = my_price; r.is_taker = false;
                    }
                }
                locked = true; passive_prev = true;       // a lock is still resting
            } else {                                       // opp on the far side: purely passive
                passive_prev = true; locked = false;
            }
            return r;
        }

        static bool parse_fill(const std::string& fill) {
            if (fill == "touch") return false;
            if (fill == "breach") return true;
            throw std::invalid_argument("fill must be \"touch\" or \"breach\".");
        }
        static double parse_participation(double p) {
            if (!(p > 0.0) || p > 1.0)
                throw std::invalid_argument("participation_ratio must be in (0, 1].");
            return p;
        }

        double maker_fee_, taker_fee_;
        bool breach_;
        double participation_, tick_size_, max_position_, min_position_;
        bool bid_passive_, ask_passive_, bid_locked_, ask_locked_;
        detail::PnLAccount account_;
    };

} // namespace screamer

#endif // SCREAMER_BACKTEST_L1_H
