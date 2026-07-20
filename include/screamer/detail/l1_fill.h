#ifndef SCREAMER_DETAIL_L1_FILL_H
#define SCREAMER_DETAIL_L1_FILL_H

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <tuple>
#include "screamer/common/float_info.h"
#include "screamer/detail/pnl_account.h"

namespace screamer { namespace detail {

    // L1Fill: shared fill core for L1 quote-based backtest engines. Posts a
    // two-sided resting quote each event: the strategy's bid fills when the
    // market ask trades through it (breach) or locks it in touch mode; a
    // quote that appears already crossing the spread is a marketable taker.
    // The sell side is symmetric. Fills are capped so the position stays in
    // [min_pos, max_pos]. Positions mark to the (market_bid + market_ask) / 2
    // mid. A NaN in any market field emits all-NaN (ignore policy).
    struct L1Fill {
        L1Fill(double maker_fee, double taker_fee, bool breach,
               double participation, double tick_size,
               double min_pos, double max_pos)
            : maker_fee_(maker_fee), taker_fee_(taker_fee), breach_(breach),
              participation_(participation), tick_size_(tick_size),
              min_pos_(min_pos), max_pos_(max_pos)
        {
            if (min_pos_ > max_pos_)
                throw std::invalid_argument("min_position must not exceed max_position.");
            if (tick_size_ < 0.0)
                throw std::invalid_argument("tick_size must be non-negative.");
            reset();
        }

        void reset() {
            account_.reset();
            bid_passive_ = false; ask_passive_ = false;
            bid_locked_ = false; ask_locked_ = false;
        }

        double position() const { return account_.position(); }

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

            // Buy side: resting bid at bid_price filled by the market ask.
            const double room_buy = std::max(max_pos_ - account_.position(), 0.0);
            SideFill b = compute_side(/*buy=*/true, bid_price, bid_size,
                                      market_ask, market_ask_size,
                                      room_buy, bid_passive_, bid_locked_);
            if (b.dpos != 0.0) {
                auto [e, p, pos, c] = account_.step(mid, b.dpos, b.fill_price,
                                                    b.is_taker ? taker_fee_ : maker_fee_);
                eq = e; pnl += p; position = pos; cost += c; did = true;
            }

            // Sell side: resting ask at ask_price filled by the market bid.
            const double room_sell = std::max(account_.position() - min_pos_, 0.0);
            SideFill s = compute_side(/*buy=*/false, ask_price, ask_size,
                                      market_bid, market_bid_size,
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

        detail::PnLAccount account_;

    private:
        struct SideFill { double dpos; double fill_price; bool is_taker; };

        // Compute one side's fill and update that side's lock/passive state.
        // buy: resting buy at my_price hit by opp (=market_ask);
        // sell: resting sell at my_price hit by opp (=market_bid).
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

        double maker_fee_, taker_fee_;
        bool breach_;
        double participation_, tick_size_, min_pos_, max_pos_;
        bool bid_passive_, ask_passive_, bid_locked_, ask_locked_;
    };

}} // namespace screamer::detail

#endif // SCREAMER_DETAIL_L1_FILL_H
