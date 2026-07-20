#ifndef SCREAMER_DETAIL_TRADES_FILL_H
#define SCREAMER_DETAIL_TRADES_FILL_H

#include <algorithm>
#include <limits>
#include <tuple>
#include "screamer/common/float_info.h"
#include "screamer/common/market_price.h"
#include "screamer/detail/pnl_account.h"

namespace screamer { namespace detail {

    // TradesFill: shared fill core for trade-tape backtest engines. Posts a
    // two-sided quote each event: a resting bid fills when a sell-print crosses
    // it (touch: trade_price <= bid, breach: <) for min(bid_size,
    // participation * trade_size, room) at the bid, paying maker_fee; a
    // through-print sweeps the full resting size. A NaN/inf bid price is a
    // market order (+inf limit) that sweeps on any print, paying taker_fee. The
    // sell side is symmetric. Fills are capped so the position stays in
    // [min_pos, max_pos]. Marks to the last trade. A NaN trade_price or
    // trade_size emits all-NaN (ignore policy). The caller builds the quote;
    // this struct executes and accounts for it.
    struct TradesFill {
        TradesFill(double maker_fee, double taker_fee, bool breach,
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
              double trade_price, double trade_size)
        {
            if (isnan2(trade_price) || isnan2(trade_size)) {
                const double nan = std::numeric_limits<double>::quiet_NaN();
                return std::make_tuple(nan, nan, nan, nan);   // ignore: need a print
            }

            double eq = 0, pnl = 0, position = account_.position(), cost = 0; bool did = false;

            // Buy side: resting bid filled by a sell-print crossing it.
            if (!isnan2(bid_size) && bid_size > 0.0) {
                const double room = std::max(max_pos_ - account_.position(), 0.0);
                const double limit = market_limit(bid_price, /*buy=*/true);
                const bool through = trade_price < limit;
                const bool at = !breach_ && (trade_price == limit);
                double dpos = 0.0;
                if (through) dpos = std::min(bid_size, room);
                else if (at) dpos = std::min(std::min(bid_size, participation_ * trade_size), room);
                if (dpos > 0.0) {
                    const double fp = std::isinf(limit) ? trade_price : limit;
                    const double fee = std::isinf(limit) ? taker_fee_ : maker_fee_;
                    auto [e, p, pos, c] = account_.step(trade_price, dpos, fp, fee);
                    eq = e; pnl += p; position = pos; cost += c; did = true;
                }
            }

            // Sell side: resting ask filled by a buy-print crossing it.
            if (!isnan2(ask_size) && ask_size > 0.0) {
                const double room = std::max(account_.position() - min_pos_, 0.0);
                const double limit = market_limit(ask_price, /*buy=*/false);
                const bool through = trade_price > limit;
                const bool at = !breach_ && (trade_price == limit);
                double dpos = 0.0;
                if (through) dpos = -std::min(ask_size, room);
                else if (at) dpos = -std::min(std::min(ask_size, participation_ * trade_size), room);
                if (dpos != 0.0) {
                    const double fp = std::isinf(limit) ? trade_price : limit;
                    const double fee = std::isinf(limit) ? taker_fee_ : maker_fee_;
                    auto [e, p, pos, c] = account_.step(trade_price, dpos, fp, fee);
                    eq = e; pnl += p; position = pos; cost += c; did = true;
                }
            }

            if (!did) {
                auto [e, p, pos, c] = account_.step(trade_price, 0.0, trade_price, 0.0);
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

#endif // SCREAMER_DETAIL_TRADES_FILL_H
