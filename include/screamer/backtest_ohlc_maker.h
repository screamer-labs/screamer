#ifndef SCREAMER_BACKTEST_OHLC_MAKER_H
#define SCREAMER_BACKTEST_OHLC_MAKER_H

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include "screamer/common/functor_base.h"
#include "screamer/common/float_info.h"
#include "screamer/common/market_price.h"
#include "screamer/detail/pnl_account.h"

namespace screamer {

    // BacktestOHLCMaker: 8 -> 4. Two-sided market-making on OHLC bars. Each bar the
    // strategy posts a resting bid (bid_price, bid_size) and ask (ask_price,
    // ask_size); a NaN/inf price is a market order (see market_limit). A resting buy
    // fills when the bar's low reaches the bid (touch: low <= bid, breach: low <
    // bid) for min(bid_size, participation * bid_size, room) at the bid, paying
    // maker_fee; a marketable buy fills at the open plus tick_size overflow, paying
    // taker_fee. The sell side is symmetric on the high. Fills are capped so the
    // position stays in [min_position, max_position]. Positions mark to the close.
    // Outputs [equity, pnl, position, cost]. nan_policy: ignore on the bar fields.
    class BacktestOHLCMaker : public FunctorBase<BacktestOHLCMaker, 8, 4> {
    public:
        BacktestOHLCMaker(double maker_fee = 0.0, double taker_fee = 0.0,
                          const std::string& fill = "touch",
                          double participation_ratio = 1.0, double tick_size = 0.0,
                          double min_position = -std::numeric_limits<double>::infinity(),
                          double max_position = std::numeric_limits<double>::infinity())
            : maker_fee_(maker_fee), taker_fee_(taker_fee), breach_(parse_fill(fill)),
              participation_(parse_participation(participation_ratio)),
              tick_size_(tick_size), min_position_(min_position),
              max_position_(max_position)
        {
            if (min_position_ > max_position_)
                throw std::invalid_argument("min_position must not exceed max_position.");
            if (tick_size_ < 0.0)
                throw std::invalid_argument("tick_size must be non-negative.");
            reset();
        }

        void reset() override { account_.reset(); }

        ResultTuple call(const InputArray& inputs) override {
            const double bid_price = inputs[0], bid_size = inputs[1];
            const double ask_price = inputs[2], ask_size = inputs[3];
            const double open = inputs[4], high = inputs[5];
            const double low = inputs[6], close = inputs[7];
            if (isnan2(open) || isnan2(high) || isnan2(low) || isnan2(close)) {
                const double nan = std::numeric_limits<double>::quiet_NaN();
                return std::make_tuple(nan, nan, nan, nan);   // ignore the bad bar
            }

            double eq = 0, pnl = 0, position = account_.position(), cost = 0; bool did = false;

            // Buy side: resting bid at bid_price, hit when the bar trades down to it.
            if (!isnan2(bid_size) && bid_size > 0.0) {
                const double room = std::max(max_position_ - account_.position(), 0.0);
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
                const double room = std::max(account_.position() - min_position_, 0.0);
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

    private:
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
        double participation_, tick_size_, min_position_, max_position_;
        detail::PnLAccount account_;
    };

} // namespace screamer

#endif // SCREAMER_BACKTEST_OHLC_MAKER_H
