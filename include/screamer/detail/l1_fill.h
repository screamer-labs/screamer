#ifndef SCREAMER_DETAIL_L1_FILL_H
#define SCREAMER_DETAIL_L1_FILL_H

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include "screamer/common/finite_settings.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/pnl_account.h"
#include "screamer/detail/slippage.h"

namespace screamer { namespace detail {

    // L1Fill: shared fill core for L1 quote-based backtest engines. Posts a
    // two-sided quote each event. A side arrives when it was not quoted on the
    // previous event, its price changed, or its previous order was fully
    // filled; a size change alone is not an arrival. An arriving bid at or above the
    // market ask is a marketable taker: it takes the displayed size at the
    // touch and walks one tick_size for the rest, and the slippage model then
    // moves that average price against the trader, never beyond the limit. A
    // resting bid fills as a maker at its price when the market ask trades
    // through it, or for a participation share when the ask locks it in touch
    // mode. The sell side is symmetric. Fills are capped so the position stays in
    // [min_pos, max_pos]. Positions mark to the (market_bid + market_ask) / 2
    // mid. A resting fill's cost reference is its own price (cost = fee); a
    // taker fill's is the mid. A NaN in any market field emits all-NaN (ignore policy).
    struct L1Fill {
        L1Fill(double maker_fee, double taker_fee, bool breach,
               double participation, const Slippage& slippage,
               double min_pos, double max_pos,
                 double multiplier = 1.0,
                 double maker_fee_per_contract = 0.0,
                 double taker_fee_per_contract = 0.0)
            : maker_fee_(maker_fee), taker_fee_(taker_fee), breach_(breach),
              participation_(participation), slippage_(slippage),
              min_pos_(min_pos), max_pos_(max_pos),
              maker_fee_per_contract_(maker_fee_per_contract),
              taker_fee_per_contract_(taker_fee_per_contract),
              account_(multiplier)
        {
            // An infinite position bound is no bound; fees are finite.
            require_finite("maker_fee", maker_fee);
            require_finite("taker_fee", taker_fee);
            require_finite("min_position", min_pos, kTakesNegInf);
            require_finite("max_position", max_pos, kTakesPosInf);
            if (min_pos_ > max_pos_)
                throw std::invalid_argument("min_position must not exceed max_position.");
            if (!std::isfinite(maker_fee_per_contract_) ||
                !std::isfinite(taker_fee_per_contract_))
                throw std::invalid_argument("fee_per_contract values must be finite.");
            reset();
        }

        void reset() {
            account_.reset();
            bid_prev_ = std::numeric_limits<double>::quiet_NaN();
            ask_prev_ = std::numeric_limits<double>::quiet_NaN();
            bid_locked_ = false; ask_locked_ = false;
        }

        double position() const { return account_.position(); }
        double equity() const { return account_.equity(); }
        double equity_at(double price) const { return account_.equity_at(price); }
        // Fill price of the last fill in the most recent quote(), NaN if none.
        double last_fill_price() const { return last_fill_; }

        std::tuple<double, double, double, double>
        quote(double bid_price, double bid_size,
              double ask_price, double ask_size,
              double market_bid, double market_ask,
              double market_bid_size, double market_ask_size)
        {
            if (isnan2(market_bid) || isnan2(market_ask) ||
                isnan2(market_bid_size) || isnan2(market_ask_size)) {
                const double nan = std::numeric_limits<double>::quiet_NaN();
                return std::make_tuple(nan, nan, nan, nan);   // ignore
            }
            const double mid = 0.5 * (market_bid + market_ask);
            double eq = 0, pnl = 0, position = account_.position(), cost = 0;
            bool did = false;
            last_fill_ = std::numeric_limits<double>::quiet_NaN();

            // Buy side: resting bid at bid_price filled by the market ask.
            const double room_buy = std::max(max_pos_ - account_.position(), 0.0);
            SideFill b = compute_side(/*buy=*/true, bid_price, bid_size,
                                      market_ask, market_ask_size,
                                      room_buy, bid_prev_, bid_locked_);
            if (b.dpos != 0.0) {
                last_fill_ = b.fill_price;
                auto [e, p, pos, c] = account_.step(mid, b.dpos, b.fill_price,
                                                    b.is_taker ? taker_fee_ : maker_fee_,
                                                    b.is_taker ? taker_fee_per_contract_
                                                               : maker_fee_per_contract_,
                                                    b.is_taker ? mid : b.fill_price);
                eq = e; pnl += p; position = pos; cost += c; did = true;
            }

            // Sell side: resting ask at ask_price filled by the market bid.
            const double room_sell = std::max(account_.position() - min_pos_, 0.0);
            SideFill s = compute_side(/*buy=*/false, ask_price, ask_size,
                                      market_bid, market_bid_size,
                                      room_sell, ask_prev_, ask_locked_);
            if (s.dpos != 0.0) {
                last_fill_ = s.fill_price;
                auto [e, p, pos, c] = account_.step(mid, s.dpos, s.fill_price,
                                                    s.is_taker ? taker_fee_ : maker_fee_,
                                                    s.is_taker ? taker_fee_per_contract_
                                                               : maker_fee_per_contract_,
                                                    s.is_taker ? mid : s.fill_price);
                eq = e; pnl += p; position = pos; cost += c; did = true;
            }

            if (!did) {
                auto [e, p, pos, c] = account_.step(mid, 0.0, mid, 0.0, 0.0, mid);  // mark only
                eq = e; pnl = p; position = pos; cost = c;
            }
            return std::make_tuple(eq, pnl, position, cost);
        }

    private:
        struct SideFill { double dpos; double fill_price; bool is_taker; };

        // Compute one side's fill and update that side's resting state.
        // buy: bid at my_price against opp (= market_ask); sell: ask at my_price
        // against opp (= market_bid). prev_price is the price of the order still
        // resting from the previous event, NaN if there is none.
        SideFill compute_side(bool buy, double my_price, double my_size,
                              double opp_price, double opp_size, double room,
                              double& prev_price, bool& locked) {
            SideFill r{0.0, my_price, false};
            if (isnan2(my_price) || isnan2(my_size) || my_size <= 0.0 || room <= 0.0) {
                prev_price = std::numeric_limits<double>::quiet_NaN();   // no quote this event
                locked = false; return r;
            }
            const bool arrival = !(my_price == prev_price);   // new, re-priced or re-entered
            const bool through = buy ? (opp_price < my_price) : (opp_price > my_price);
            const bool lock    = (opp_price == my_price);
            const double remaining = std::min(my_size, room);
            const bool was_locked = locked;

            if (arrival && (through || lock)) {          // marketable: taker from the touch
                r.dpos = buy ? remaining : -remaining;
                r.fill_price = marketable_fill(buy, my_price, remaining,
                                               opp_price, opp_size, slippage_);
                r.is_taker = true;
                prev_price = std::numeric_limits<double>::quiet_NaN();   // order consumed
                locked = false;
            } else if (through) {                         // run over while resting: maker at my_price
                r.dpos = buy ? remaining : -remaining;
                r.fill_price = my_price; r.is_taker = false;
                prev_price = std::numeric_limits<double>::quiet_NaN();   // order consumed
                locked = false;
            } else if (lock) {
                locked = true; prev_price = my_price;     // a lock is still resting
                if (!breach_ && !was_locked) {            // touch mode, first event of this lock
                    const double filled = std::min(remaining, participation_ * opp_size);
                    if (filled > 0.0) {
                        r.dpos = buy ? filled : -filled;
                        r.fill_price = my_price; r.is_taker = false;
                    }
                    if (filled >= remaining) {            // fully filled: the order is gone
                        prev_price = std::numeric_limits<double>::quiet_NaN();
                        locked = false;
                    }
                }
            } else {                                       // opp on the far side: resting
                prev_price = my_price; locked = false;
            }
            return r;
        }

        double maker_fee_, taker_fee_;
        double maker_fee_per_contract_, taker_fee_per_contract_;
        bool breach_;
        double participation_;
        Slippage slippage_;
        double min_pos_, max_pos_;
        double bid_prev_ = std::numeric_limits<double>::quiet_NaN();   // resting bid price, NaN if none
        double ask_prev_ = std::numeric_limits<double>::quiet_NaN();
        bool bid_locked_ = false, ask_locked_ = false;
        detail::PnLAccount account_;
        double last_fill_ = std::numeric_limits<double>::quiet_NaN();
    };

}} // namespace screamer::detail

#endif // SCREAMER_DETAIL_L1_FILL_H
