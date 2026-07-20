#ifndef SCREAMER_DETAIL_OHLC_FILL_H
#define SCREAMER_DETAIL_OHLC_FILL_H

#include <algorithm>
#include <limits>
#include <tuple>
#include "screamer/common/float_info.h"
#include "screamer/common/market_price.h"
#include "screamer/detail/pnl_account.h"

namespace screamer { namespace detail {

    // OHLCFill: shared fill core for OHLC-bar backtest engines. Posts a two-sided
    // quote each bar: a resting bid fills when the bar's low reaches it (touch:
    // low <= bid, breach: low < bid) for min(bid_size, participation * bid_size,
    // room) at the bid, paying maker_fee; a marketable bid (market_limit returns
    // +inf) fills at open + tick_size, paying taker_fee. The sell side is symmetric
    // on the high. Fills are capped so the position stays in [min_pos, max_pos].
    // The caller builds the quote; this struct executes and accounts for it.
    struct OHLCFill {
        OHLCFill(double maker_fee, double taker_fee, bool breach,
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

        void reset() { account_.reset(); }
        double position() const { return account_.position(); }

        std::tuple<double, double, double, double>
        quote(double bid_price, double bid_size,
              double ask_price, double ask_size,
              double open, double high, double low, double close)
        {
            if (isnan2(open) || isnan2(high) || isnan2(low) || isnan2(close)) {
                const double nan = std::numeric_limits<double>::quiet_NaN();
                return std::make_tuple(nan, nan, nan, nan);   // ignore the bad bar
            }

            double eq = 0, pnl = 0, position = account_.position(), cost = 0; bool did = false;

            // Buy side: resting bid at bid_price, hit when the bar trades down to it.
            if (!isnan2(bid_size) && bid_size > 0.0) {
                const double room = std::max(max_pos_ - account_.position(), 0.0);
                const double limit = market_limit(bid_price, /*buy=*/true);
                double dpos = 0.0, fill_price = close, fee = maker_fee_;
                if (std::isinf(limit) && limit > 0.0) {       // market buy at the open
                    dpos = std::min(std::min(bid_size, participation_ * bid_size), room);
                    fill_price = open + tick_size_; fee = taker_fee_;
                } else {                                       // resting limit buy
                    const bool hit = breach_ ? (low < limit) : (low <= limit);
                    if (hit) {
                        dpos = std::min(std::min(bid_size, participation_ * bid_size), room);
                        fill_price = limit;
                    }
                }
                if (dpos > 0.0) {
                    auto [e, p, pos, c] = account_.step(close, dpos, fill_price, fee);
                    eq = e; pnl += p; position = pos; cost += c; did = true;
                }
            }

            // Sell side: resting ask at ask_price, hit when the bar trades up to it.
            if (!isnan2(ask_size) && ask_size > 0.0) {
                const double room = std::max(account_.position() - min_pos_, 0.0);
                const double limit = market_limit(ask_price, /*buy=*/false);
                double dpos = 0.0, fill_price = close, fee = maker_fee_;
                if (std::isinf(limit) && limit < 0.0) {       // market sell at the open
                    dpos = -std::min(std::min(ask_size, participation_ * ask_size), room);
                    fill_price = open - tick_size_; fee = taker_fee_;
                } else {                                       // resting limit sell
                    const bool hit = breach_ ? (high > limit) : (high >= limit);
                    if (hit) {
                        dpos = -std::min(std::min(ask_size, participation_ * ask_size), room);
                        fill_price = limit;
                    }
                }
                if (dpos != 0.0) {
                    auto [e, p, pos, c] = account_.step(close, dpos, fill_price, fee);
                    eq = e; pnl += p; position = pos; cost += c; did = true;
                }
            }

            if (!did) {
                auto [e, p, pos, c] = account_.step(close, 0.0, close, 0.0);  // mark only
                eq = e; pnl = p; position = pos; cost = c;
            }
            return std::make_tuple(eq, pnl, position, cost);
        }

        detail::PnLAccount account_;

    private:
        double maker_fee_, taker_fee_;
        bool breach_;
        double participation_, tick_size_, min_pos_, max_pos_;
    };

}} // namespace screamer::detail

#endif // SCREAMER_DETAIL_OHLC_FILL_H
