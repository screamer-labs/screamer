#ifndef SCREAMER_DETAIL_PNL_ACCOUNT_H
#define SCREAMER_DETAIL_PNL_ACCOUNT_H

#include <cmath>
#include <tuple>
#include <stdexcept>

namespace screamer {
namespace detail {

// Shared mark-to-market accounting for the backtest engines. Each bar an engine's
// fill model calls step() with the mark price `close`, the executed position
// change `dpos`, its `fill_price`, and a signed per-notional `fee_rate`, and gets
// back (equity, pnl, position, cost).
//
// The position held into the bar earns the mark move. The trade this bar costs
// the signed fill-versus-mark difference multiplied by `multiplier`, plus
// `abs(dpos) * (fee_per_contract + fill_price * multiplier * fee_rate)`.
// Both fee terms are signed, so a maker rebate can be negative. The account is
// stateful but trivially resettable; reset() returns it to flat.
class PnLAccount {
public:
    PnLAccount(double multiplier = 1.0, double fee_per_contract = 0.0)
        : multiplier_(multiplier), fee_per_contract_(fee_per_contract) {
        if (!(multiplier_ > 0.0) || !std::isfinite(multiplier_))
            throw std::invalid_argument("multiplier must be finite and positive.");
        if (!std::isfinite(fee_per_contract_))
            throw std::invalid_argument("fee_per_contract must be finite.");
        reset();
    }

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
        return step(close, dpos, fill_price, fee_rate, fee_per_contract_);
    }

    std::tuple<double, double, double, double>
    step(double close, double dpos, double fill_price, double fee_rate,
         double fee_per_contract) {
        const double mark_pnl = has_prev_
            ? position_ * (close - prev_close_) * multiplier_ : 0.0;
        const double trade_cost = dpos * (fill_price - close) * multiplier_
                                + std::abs(dpos) *
                                  (fee_per_contract + fill_price * multiplier_ * fee_rate);
        position_ += dpos;
        const double pnl = mark_pnl - trade_cost;
        cum_equity_ += pnl;
        prev_close_ = close;
        has_prev_ = true;
        return {cum_equity_, pnl, position_, trade_cost};
    }

private:
    double multiplier_ = 1.0;
    double fee_per_contract_ = 0.0;
    double position_ = 0.0;
    double prev_close_ = 0.0;
    double cum_equity_ = 0.0;
    bool has_prev_ = false;
};

}} // namespace screamer::detail
#endif // SCREAMER_DETAIL_PNL_ACCOUNT_H
