#ifndef SCREAMER_DETAIL_PNL_ACCOUNT_H
#define SCREAMER_DETAIL_PNL_ACCOUNT_H

#include <cmath>
#include <tuple>

namespace screamer {
namespace detail {

// Shared mark-to-market accounting for the backtest engines. Each bar an engine's
// fill model calls step() with the mark price `close`, the executed position
// change `dpos`, its `fill_price`, and a signed per-notional `fee_rate`, and gets
// back (equity, pnl, position, cost).
//
// The position held into the bar earns the mark move; the trade this bar costs
// dpos*(fill_price - close) - the adverse (or, for a passive maker fill,
// favorable) difference between the fill and the mark, which carries the spread
// economics with the correct sign for either side and any fill type - plus an
// explicit fee on the traded notional. `fee_rate` is signed, so a maker rebate is
// negative. Non-copyable state is trivial; reset() returns it to flat.
class PnLAccount {
public:
    PnLAccount() { reset(); }

    void reset() {
        position_ = 0.0;
        prev_close_ = 0.0;
        cum_equity_ = 0.0;
        has_prev_ = false;
    }

    double position() const { return position_; }

    // Advance the account by one bar; returns (equity, pnl, position, cost).
    std::tuple<double, double, double, double>
    step(double close, double dpos, double fill_price, double fee_rate) {
        const double mark_pnl = has_prev_ ? position_ * (close - prev_close_) : 0.0;
        const double trade_cost = dpos * (fill_price - close)
                                + std::abs(dpos) * fill_price * fee_rate;
        position_ += dpos;
        const double pnl = mark_pnl - trade_cost;
        cum_equity_ += pnl;
        prev_close_ = close;
        has_prev_ = true;
        return {cum_equity_, pnl, position_, trade_cost};
    }

private:
    double position_;
    double prev_close_;
    double cum_equity_;
    bool has_prev_;
};

}} // namespace screamer::detail
#endif // SCREAMER_DETAIL_PNL_ACCOUNT_H
