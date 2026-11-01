#ifndef SCREAMER_BACKTEST_L1TRADES_ORDERS_H
#define SCREAMER_BACKTEST_L1TRADES_ORDERS_H

#include "screamer/common/finite_settings.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include "screamer/common/functor_base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/pnl_account.h"
#include "screamer/detail/slippage.h"

namespace screamer {

    // BacktestL1TradesOrders: 10 -> 4. The preferred market-making engine: quotes mark the
    // position and seed context, TRADES drive passive fills (unambiguous execution
    // events), and a quote cross with no explaining trade is the run-over fallback.
    // A side arrives when it was not quoted on the previous event, its price
    // changed, or its previous order was fully filled; an arriving bid at or above the
    // market ask is a taker from the touch, before any print is considered.
    // When the quote crosses a resting order but the event's print does not
    // reach it, the order gets no fill and is treated as gone, so the same quote
    // on the next event is a new order.
    // Inputs (bid_price, bid_size, ask_price, ask_size, market_bid, market_ask,
    // market_bid_size, market_ask_size, trade_price, trade_size). Trades are NOT
    // forward-filled: a NaN trade is a quote-only update (mark, no fill), which is
    // nan_policy: ignore, so each real trade fills at most once. Positions mark to
    // the mid. Outputs [equity, pnl, position, cost]; a resting fill's cost is its
    // fee (reference = its own price), a taker fill's is measured against the mid.
    class BacktestL1TradesOrders : public FunctorBase<BacktestL1TradesOrders, 10, 4> {
    public:
        static constexpr const char* kTypeName = "BacktestL1TradesOrders";
        static constexpr int kStateVersion = 1;

        template <class Ar>
        void settings(Ar& ar) {
            ar("maker_fee", maker_fee_);
            ar("taker_fee", taker_fee_);
            ar("fill", std::string(breach_ ? "breach" : "touch"));
            ar("participation_ratio", participation_);
            ar("tick_size", slippage_.tick_size());
            ar("min_position", min_position_);
            ar("max_position", max_position_);
            ar("multiplier", account_.multiplier());
            ar("maker_fee_per_contract", maker_fee_per_contract_);
            ar("taker_fee_per_contract", taker_fee_per_contract_);
            ar("slippage", slippage_.amount());
            ar("slippage_unit", std::string(slippage_.unit_name()));
        }

        template <class Ar>
        void state(Ar& ar) {
            ar.object("account", account_);
            ar("bid_prev", bid_prev_);
            ar("ask_prev", ask_prev_);
        }

        BacktestL1TradesOrders(double maker_fee = 0.0, double taker_fee = 0.0,
                         const std::string& fill = "touch",
                         double participation_ratio = 1.0, double tick_size = 0.0,
                         double min_position = -std::numeric_limits<double>::infinity(),
                         double max_position = std::numeric_limits<double>::infinity(),
                         double multiplier = 1.0,
                         double maker_fee_per_contract = 0.0,
                         double taker_fee_per_contract = 0.0,
                         double slippage = 0.0,
                         const std::string& slippage_unit = "fraction")
            : maker_fee_(maker_fee), taker_fee_(taker_fee), breach_(parse_fill(fill)),
              participation_(parse_participation(participation_ratio)),
              slippage_(slippage, slippage_unit, tick_size), max_position_(max_position),
              min_position_(min_position),
              maker_fee_per_contract_(maker_fee_per_contract),
              taker_fee_per_contract_(taker_fee_per_contract),
              account_(multiplier)
        {
            require_finite("maker_fee", maker_fee);
            require_finite("taker_fee", taker_fee);
            require_finite("maker_fee_per_contract", maker_fee_per_contract);
            require_finite("taker_fee_per_contract", taker_fee_per_contract);
            require_finite("min_position", min_position, kTakesNegInf);
            require_finite("max_position", max_position, kTakesPosInf);
            if (min_position_ > max_position_)
                throw std::invalid_argument("min_position must not exceed max_position.");
            reset();
        }

        void reset() override {
            account_.reset();
            bid_prev_ = std::numeric_limits<double>::quiet_NaN();
            ask_prev_ = std::numeric_limits<double>::quiet_NaN();
        }

        ResultTuple call(const InputArray& inputs) override {
            // Own quote (strategy's resting orders).
            const double my_bid = inputs[0], my_bid_size = inputs[1];
            const double my_ask = inputs[2], my_ask_size = inputs[3];
            // Market L1 (top of book).
            const double market_bid = inputs[4], market_ask = inputs[5];
            const double market_bid_size = inputs[6], market_ask_size = inputs[7];
            // Trade tape.
            const double trade_price = inputs[8], trade_size = inputs[9];

            if (isnan2(market_bid) || isnan2(market_ask) ||
                isnan2(market_bid_size) || isnan2(market_ask_size)) {
                const double nan = std::numeric_limits<double>::quiet_NaN();
                return std::make_tuple(nan, nan, nan, nan);   // ignore
            }
            const double mid = 0.5 * (market_bid + market_ask);
            const bool has_trade = !isnan2(trade_price) && !isnan2(trade_size);

            double eq = 0, pnl = 0, position = account_.position(), cost = 0; bool did = false;

            // Buy side: my resting bid vs the market ask.
            const double room_buy = std::max(max_position_ - account_.position(), 0.0);
            double b_dpos = 0.0, b_price = my_bid; bool b_taker = false;
            resolve_side(true, my_bid, my_bid_size, market_ask, market_ask_size, room_buy,
                         has_trade, trade_price, trade_size, bid_prev_,
                         b_dpos, b_price, b_taker);
            if (b_dpos != 0.0) {
                auto [e, p, pos, c] = account_.step(mid, b_dpos, b_price,
                                                    b_taker ? taker_fee_ : maker_fee_,
                                                    b_taker ? taker_fee_per_contract_
                                                            : maker_fee_per_contract_,
                                                    b_taker ? mid : b_price);
                eq = e; pnl += p; position = pos; cost += c; did = true;
            }

            // Sell side: my resting ask vs the market bid.
            const double room_sell = std::max(account_.position() - min_position_, 0.0);
            double s_dpos = 0.0, s_price = my_ask; bool s_taker = false;
            resolve_side(false, my_ask, my_ask_size, market_bid, market_bid_size, room_sell,
                         has_trade, trade_price, trade_size, ask_prev_,
                         s_dpos, s_price, s_taker);
            if (s_dpos != 0.0) {
                auto [e, p, pos, c] = account_.step(mid, s_dpos, s_price,
                                                    s_taker ? taker_fee_ : maker_fee_,
                                                    s_taker ? taker_fee_per_contract_
                                                            : maker_fee_per_contract_,
                                                    s_taker ? mid : s_price);
                eq = e; pnl += p; position = pos; cost += c; did = true;
            }

            if (!did) {
                auto [e, p, pos, c] = account_.step(mid, 0.0, mid, 0.0, 0.0, mid);   // mark only
                eq = e; pnl = p; position = pos; cost = c;
            }
            return std::make_tuple(eq, pnl, position, cost);
        }

    private:
        // An arriving marketable order is a taker from the touch. Otherwise passive
        // fills come from the trade tape, and a quote cross with no explaining trade
        // is the run-over fallback (maker at my_price). prev_price is the price of
        // the order still resting from the previous event, NaN if there is none.
        void resolve_side(bool buy, double my_price, double my_size,
                          double opp_price, double opp_size, double room,
                          bool has_trade, double trade_price, double trade_size,
                          double& prev_price,
                          double& dpos, double& fill_price, bool& is_taker) {
            const double nan = std::numeric_limits<double>::quiet_NaN();
            if (isnan2(my_price) || isnan2(my_size) || my_size <= 0.0 || room <= 0.0) {
                prev_price = nan; return;
            }
            const double remaining = std::min(my_size, room);
            const bool arrival = !(my_price == prev_price);   // new, re-priced or re-entered
            const bool quote_through = buy ? (opp_price < my_price) : (opp_price > my_price);

            // Marketable on arrival (crossing or locking the opposite touch): taker,
            // takes the displayed size at the touch, walks one tick_size for the
            // rest, then the slippage model moves the average, never beyond the
            // limit, regardless of any print.
            if (arrival && (quote_through || opp_price == my_price)) {
                dpos = buy ? remaining : -remaining;
                fill_price = detail::marketable_fill(buy, my_price, remaining,
                                                     opp_price, opp_size, slippage_);
                is_taker = true;
                prev_price = nan;                              // order consumed
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
                // still resting unless the quote also crossed or the print filled it fully
                prev_price = (quote_through || filled >= remaining) ? nan : my_price;
                return;
            }

            // No trade this event: a quote cross of a resting order is the run-over
            // fallback (maker at my_price); otherwise stay resting.
            if (quote_through) {
                dpos = buy ? remaining : -remaining; fill_price = my_price; is_taker = false;
                prev_price = nan;
            } else {
                prev_price = my_price;
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
        double maker_fee_per_contract_, taker_fee_per_contract_;
        bool breach_;
        double participation_;
        detail::Slippage slippage_;
        double max_position_, min_position_;
        double bid_prev_ = std::numeric_limits<double>::quiet_NaN();   // resting bid price, NaN if none
        double ask_prev_ = std::numeric_limits<double>::quiet_NaN();
        detail::PnLAccount account_;
    };

} // namespace screamer

#endif // SCREAMER_BACKTEST_L1TRADES_ORDERS_H
