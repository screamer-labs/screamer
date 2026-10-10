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
// The position held into the bar earns the mark move, and the shares traded this
// bar earn the move from their fill price to the mark. The reported `cost` is
// the friction of the trade: the signed fill-versus-`reference` difference
// multiplied by `multiplier`, plus
// `abs(dpos) * (fee_per_contract + fill_price * multiplier * fee_rate)`.
// `reference` is the fair price at execution before any friction (the bar open
// for a next-open fill, the mid for an L1 book, the limit for a resting order);
// it defaults to the mark. The choice of reference only splits the result
// between pnl and cost: equity is the same for any reference.
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
        return step(close, dpos, fill_price, fee_rate, fee_per_contract, close);
    }

    std::tuple<double, double, double, double>
    step(double close, double dpos, double fill_price, double fee_rate,
         double fee_per_contract, double reference) {
        const double mark_pnl = has_prev_
            ? position_ * (close - prev_close_) * multiplier_ : 0.0;
        const double fees = std::abs(dpos) *
                            (fee_per_contract + fill_price * multiplier_ * fee_rate);
        const double trade_cost = dpos * (fill_price - reference) * multiplier_ + fees;
        const double fill_to_mark = dpos * (close - fill_price) * multiplier_;
        position_ += dpos;
        const double pnl = mark_pnl + fill_to_mark - fees;
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
