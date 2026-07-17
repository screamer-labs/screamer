#ifndef SCREAMER_BACKTEST_L1_TRADES_H
#define SCREAMER_BACKTEST_L1_TRADES_H

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include "screamer/common/functor_base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/pnl_account.h"

namespace screamer {

    // BacktestL1Trades: 10 -> 4. The preferred market-making engine: quotes mark the
    // position and seed context, TRADES drive passive fills (unambiguous execution
    // events), and a quote cross with no explaining trade is the run-over fallback.
    // Inputs (bid, ask, bid_size, ask_size, my_bid, my_bid_size, my_ask, my_ask_size,
    // trade_price, trade_size). Trades are NOT forward-filled: a NaN trade is a
    // quote-only update (mark, no fill), which is nan_policy: ignore, so each real
    // trade fills at most once. Positions mark to the mid. Outputs
    // [equity, pnl, position, cost].
    class BacktestL1Trades : public FunctorBase<BacktestL1Trades, 10, 4> {
    public:
        BacktestL1Trades(double maker_fee = 0.0, double taker_fee = 0.0,
                         const std::string& fill = "touch",
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

        void reset() override { account_.reset(); bid_passive_ = false; ask_passive_ = false; }

        ResultTuple call(const InputArray& inputs) override {
            const double bid = inputs[0], ask = inputs[1];
            const double bid_size = inputs[2], ask_size = inputs[3];
            const double my_bid = inputs[4], my_bid_size = inputs[5];
            const double my_ask = inputs[6], my_ask_size = inputs[7];
            const double trade_price = inputs[8], trade_size = inputs[9];
            if (isnan2(bid) || isnan2(ask) || isnan2(bid_size) || isnan2(ask_size)) {
                const double nan = std::numeric_limits<double>::quiet_NaN();
                return std::make_tuple(nan, nan, nan, nan);   // ignore
            }
            const double mid = 0.5 * (bid + ask);
            const bool has_trade = !isnan2(trade_price) && !isnan2(trade_size);

            double eq = 0, pnl = 0, position = account_.position(), cost = 0; bool did = false;

            // Buy side.
            const double room_buy = std::max(max_position_ - account_.position(), 0.0);
            double b_dpos = 0.0, b_price = my_bid; bool b_taker = false;
            resolve_side(true, my_bid, my_bid_size, ask, ask_size, room_buy,
                         has_trade, trade_price, trade_size, bid_passive_,
                         b_dpos, b_price, b_taker);
            if (b_dpos != 0.0) {
                auto [e, p, pos, c] = account_.step(mid, b_dpos, b_price,
                                                    b_taker ? taker_fee_ : maker_fee_);
                eq = e; pnl += p; position = pos; cost += c; did = true;
            }

            // Sell side.
            const double room_sell = std::max(account_.position() - min_position_, 0.0);
            double s_dpos = 0.0, s_price = my_ask; bool s_taker = false;
            resolve_side(false, my_ask, my_ask_size, bid, bid_size, room_sell,
                         has_trade, trade_price, trade_size, ask_passive_,
                         s_dpos, s_price, s_taker);
            if (s_dpos != 0.0) {
                auto [e, p, pos, c] = account_.step(mid, s_dpos, s_price,
                                                    s_taker ? taker_fee_ : maker_fee_);
                eq = e; pnl += p; position = pos; cost += c; did = true;
            }

            if (!did) {
                auto [e, p, pos, c] = account_.step(mid, 0.0, mid, 0.0);
                eq = e; pnl = p; position = pos; cost = c;
            }
            return std::make_tuple(eq, pnl, position, cost);
        }

    private:
        // Passive fills come from the trade tape; a quote cross with no explaining
        // trade is the run-over fallback (maker at my_price).
        void resolve_side(bool buy, double my_price, double my_size,
                          double opp_price, double opp_size, double room,
                          bool has_trade, double trade_price, double trade_size,
                          bool& passive_prev,
                          double& dpos, double& fill_price, bool& is_taker) {
            if (isnan2(my_price) || isnan2(my_size) || my_size <= 0.0 || room <= 0.0) {
                passive_prev = false; return;
            }
            const double remaining = std::min(my_size, room);
            const bool quote_through = buy ? (opp_price < my_price) : (opp_price > my_price);

            // Submitted already crossing (marketable on first appearance): taker,
            // fills at market + tick_size slippage, regardless of any print.
            if (quote_through && !passive_prev) {
                const double disp = std::min(remaining, opp_size);
                const double over = remaining - disp;
                const double slip = buy ? tick_size_ : -tick_size_;
                const double vwap = (disp * opp_price + over * (opp_price + slip)) / remaining;
                dpos = buy ? remaining : -remaining; fill_price = vwap; is_taker = true;
                passive_prev = false;
                return;
            }

            if (has_trade) {                                   // trade drives the passive fill
                const bool t_through = buy ? (trade_price < my_price) : (trade_price > my_price);
                const bool t_at = !breach_ && (trade_price == my_price);
                double filled = 0.0;
                if (t_through) filled = remaining;
                else if (t_at)  filled = std::min(remaining, participation_ * trade_size);
                if (filled > 0.0) {
                    dpos = buy ? filled : -filled; fill_price = my_price; is_taker = false;
                }
                passive_prev = !quote_through;                 // still resting unless the quote also crossed
                return;
            }

            // No trade this event: a quote cross of a resting order is the run-over
            // fallback (maker at my_price); otherwise stay passive.
            if (quote_through) {                               // passive_prev is true here
                dpos = buy ? remaining : -remaining; fill_price = my_price; is_taker = false;
                passive_prev = false;
            } else {
                passive_prev = true;                           // not crossed: resting
            }
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
        bool bid_passive_, ask_passive_;
        detail::PnLAccount account_;
    };

} // namespace screamer

#endif // SCREAMER_BACKTEST_L1_TRADES_H
