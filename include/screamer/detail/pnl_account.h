#ifndef SCREAMER_DETAIL_PNL_ACCOUNT_H
#define SCREAMER_DETAIL_PNL_ACCOUNT_H

#include <cmath>
#include <tuple>
#include <stdexcept>

namespace screamer {
namespace detail {

// Shared mark-to-market accounting for the backtest engines. Each bar an engine's
// fill model calls step() with the mark price `close`, the executed position
// change `dpos`, its `fill_price`, a signed per-notional `fee_rate`, a signed
// `fee_per_contract`, and the cost `reference`, and gets back
// (equity, pnl, position, cost).
//
// The position held into the bar earns the mark move, and the shares traded this
// bar earn the move from their fill price to the mark. The reported `cost` is
// the friction of the trade: the signed fill-versus-`reference` difference
// multiplied by `multiplier`, plus
// `abs(dpos) * (fee_per_contract + fill_price * multiplier * fee_rate)`.
// `reference` is the fair price at execution before any friction: the limit
// price for a resting (maker) fill, so its cost is its fee alone; for a taker
// fill the execution bar's price (OHLC), the print (trades), the mid (L1) or
// the price (Price). Every caller names it. The choice of reference only splits
// the result between pnl and cost: equity is the same for any reference.
// Both fee terms are signed, so a maker rebate can be negative. The account is
// stateful but trivially resettable; reset() returns it to flat.
class PnLAccount {
public:
    explicit PnLAccount(double multiplier = 1.0) : multiplier_(multiplier) {
        if (!(multiplier_ > 0.0) || !std::isfinite(multiplier_))
            throw std::invalid_argument("multiplier must be finite and positive.");
        reset();
    }

    void reset() {
        position_ = 0.0;
        prev_close_ = 0.0;
        cum_equity_ = 0.0;
        has_prev_ = false;
    }

    double multiplier() const { return multiplier_; }
    double position() const { return position_; }
    double equity() const { return cum_equity_; }
    // Equity marked at `price`: the last equity plus the open position's move
    // from the last mark to `price`. Used to size against current equity.
    double equity_at(double price) const {
        return has_prev_ ? cum_equity_ + position_ * (price - prev_close_) * multiplier_
                         : cum_equity_;
    }

    // Advance the account by one bar; returns (equity, pnl, position, cost).
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

    // Before its first step the account is flat.
    template <class Ar>
    void state(Ar& ar) {
        ar("position", position_);
        ar("prev_close", prev_close_);
        ar("cum_equity", cum_equity_);
        ar("has_prev", has_prev_);
        ar.check(has_prev_ || (position_ == 0.0 && cum_equity_ == 0.0),
                 "an account without a mark must be flat");
    }

private:
    double multiplier_ = 1.0;
    double position_ = 0.0;
    double prev_close_ = 0.0;
    double cum_equity_ = 0.0;
    bool has_prev_ = false;
};

}} // namespace screamer::detail
#endif // SCREAMER_DETAIL_PNL_ACCOUNT_H
