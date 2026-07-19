#ifndef SCREAMER_BACKTEST_TRADES_MAKER_H
#define SCREAMER_BACKTEST_TRADES_MAKER_H

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

    // BacktestTradesMaker: 6 -> 4. Two-sided market-making against the trade tape.
    // Each event is a print (trade_price, trade_size) and the strategy's resting bid
    // (bid_price, bid_size) and ask (ask_price, ask_size); a NaN/inf price is a
    // market order. A resting buy fills when a sell-print crosses it (touch:
    // trade_price <= bid, breach: <) for min(bid_size, participation * trade_size,
    // room) at the bid, paying maker_fee; a through-print sweeps the order. The sell
    // side is symmetric. Fills are capped to [min_position, max_position]. Marks to
    // the last trade. Outputs [equity, pnl, position, cost]. nan_policy: ignore on
    // the trade fields (a NaN trade emits an all-NaN row).
    class BacktestTradesMaker : public FunctorBase<BacktestTradesMaker, 6, 4> {
    public:
        BacktestTradesMaker(double maker_fee = 0.0, double taker_fee = 0.0,
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
            const double trade_price = inputs[4], trade_size = inputs[5];
            if (isnan2(trade_price) || isnan2(trade_size)) {
                const double nan = std::numeric_limits<double>::quiet_NaN();
                return std::make_tuple(nan, nan, nan, nan);   // ignore: need a print
            }

            double eq = 0, pnl = 0, position = account_.position(), cost = 0; bool did = false;

            // Buy side: resting bid filled by a sell-print crossing it.
            if (!isnan2(bid_size) && bid_size > 0.0) {
                const double room = std::max(max_position_ - account_.position(), 0.0);
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
                const double room = std::max(account_.position() - min_position_, 0.0);
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

#endif // SCREAMER_BACKTEST_TRADES_MAKER_H
