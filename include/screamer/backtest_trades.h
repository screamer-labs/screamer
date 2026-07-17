#ifndef SCREAMER_BACKTEST_TRADES_H
#define SCREAMER_BACKTEST_TRADES_H

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include "screamer/common/functor_base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/pnl_account.h"

namespace screamer {

    // BacktestTrades: 4 -> 4. An event-driven backtest against the trade tape. Each
    // event is a print `(trade_price, trade_size)` together with the strategy's
    // current resting limit order `(order_price, order_size)` (align your order
    // stream to the tape with combine_latest upstream). `order_size` is signed: a
    // positive size is a resting buy, negative a resting sell, and NaN or zero is
    // no order.
    //
    // A resting order fills when a print crosses it - a buy when
    // `trade_price <= order_price`, a sell when `trade_price >= order_price`
    // (`"touch"`; strict for `"breach"`) - filling min(order size, trade size) at
    // the order price and paying `maker_fee` (negative for a rebate). Fills are
    // partial up to the print size, front-of-queue (optimistic); a marketable limit
    // (priced through the tape) fills at the next print. Positions mark to the last
    // trade price. Outputs are [equity, pnl, position, cost]. nan_policy: ignore
    // for the trade fields (a NaN order price is simply "no resting order").
    class BacktestTrades : public FunctorBase<BacktestTrades, 4, 4> {
    public:
        BacktestTrades(double maker_fee = 0.0, const std::string& fill = "touch")
            : maker_fee_(maker_fee), breach_(parse_fill(fill))
        {}

        void reset() override { account_.reset(); }

        ResultTuple call(const InputArray& inputs) override {
            const double order_price = inputs[0];
            const double order_size  = inputs[1];   // signed; NaN/0 = no order
            const double trade_price = inputs[2];
            const double trade_size  = inputs[3];
            if (isnan2(trade_price) || isnan2(trade_size)) {
                const double nan = std::numeric_limits<double>::quiet_NaN();
                return std::make_tuple(nan, nan, nan, nan);   // ignore: need a print to act
            }

            double fill_dpos = 0.0, fill_price = trade_price;
            if (!isnan2(order_price) && !isnan2(order_size) && order_size != 0.0) {
                const bool buy = order_size > 0.0;
                const bool crossed = buy
                    ? (breach_ ? (trade_price < order_price) : (trade_price <= order_price))
                    : (breach_ ? (trade_price > order_price) : (trade_price >= order_price));
                if (crossed) {
                    const double filled = std::min(std::abs(order_size), trade_size);
                    fill_dpos  = buy ? filled : -filled;
                    fill_price = order_price;
                }
            }
            auto [equity, pnl, position, cost] =
                account_.step(trade_price, fill_dpos, fill_price, maker_fee_);
            return std::make_tuple(equity, pnl, position, cost);
        }

    private:
        static bool parse_fill(const std::string& fill) {
            if (fill == "touch") return false;
            if (fill == "breach") return true;
            throw std::invalid_argument("fill must be \"touch\" or \"breach\".");
        }

        double maker_fee_;
        bool breach_;
        detail::PnLAccount account_;
    };

} // namespace screamer

#endif // SCREAMER_BACKTEST_TRADES_H
