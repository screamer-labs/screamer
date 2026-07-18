#ifndef SCREAMER_BACKTEST_REPORT_H
#define SCREAMER_BACKTEST_REPORT_H

#include <cmath>
#include <limits>
#include <tuple>
#include "screamer/common/functor_base.h"
#include "screamer/common/float_info.h"

namespace screamer {

    // BacktestReport: 4 -> 6. Turns a backtest engine's per-step
    // [equity, pnl, position, cost] into the running report columns
    // [drawdown, cum_cost, turnover, trades, max_drawdown, sharpe]. Every column is
    // a causal accumulator, so a full-sample statistic is the last finite value:
    // total_pnl = last equity, max_drawdown = last max_drawdown, and so on.
    //
    //   drawdown     = equity - running peak equity   (dollar, <= 0; the equity
    //                  curve can be negative, so this is the dollar form, not the
    //                  percentage Drawdown op which is for positive price series)
    //   cum_cost     = running sum of cost
    //   turnover     = running sum of |position change| (the first step counts the
    //                  move from flat)
    //   trades       = running count of steps that changed the position
    //   max_drawdown = running minimum of drawdown (the worst so far, <= 0)
    //   sharpe       = running mean(pnl) / sample std(pnl) (Welford; NaN until two
    //                  finite pnl values with positive dispersion)
    //
    // nan_policy: ignore. A row with any NaN field is a skipped bar: the state
    // holds (no accumulator advances) and the output row is all NaN, recovering at
    // the next finite bar, as CumSum and the other accumulators do.
    class BacktestReport : public FunctorBase<BacktestReport, 4, 6> {
    public:
        BacktestReport() { reset(); }

        void reset() override {
            peak_equity_ = -std::numeric_limits<double>::infinity();
            worst_dd_ = 0.0;
            cum_cost_ = 0.0;
            cum_turnover_ = 0.0;
            trade_count_ = 0.0;
            has_prev_pos_ = false;
            prev_pos_ = 0.0;
            pnl_n_ = 0;
            pnl_mean_ = 0.0;
            pnl_m2_ = 0.0;
        }

        ResultTuple call(const InputArray& inputs) override {
            const double equity = inputs[0];
            const double pnl = inputs[1];
            const double position = inputs[2];
            const double cost = inputs[3];
            if (isnan2(equity) || isnan2(pnl) || isnan2(position) || isnan2(cost)) {
                const double nan = std::numeric_limits<double>::quiet_NaN();
                return std::make_tuple(nan, nan, nan, nan, nan, nan);   // skipped bar; state holds
            }

            if (equity > peak_equity_) peak_equity_ = equity;
            const double drawdown = equity - peak_equity_;              // dollar, <= 0
            if (drawdown < worst_dd_) worst_dd_ = drawdown;

            cum_cost_ += cost;

            const double dpos = has_prev_pos_ ? (position - prev_pos_) : position;
            const double traded = std::abs(dpos);
            cum_turnover_ += traded;
            if (traded > 0.0) trade_count_ += 1.0;
            prev_pos_ = position;
            has_prev_pos_ = true;

            // Running Sharpe over finite pnl (Welford; sample std, ddof = 1).
            pnl_n_ += 1;
            const double delta = pnl - pnl_mean_;
            pnl_mean_ += delta / static_cast<double>(pnl_n_);
            pnl_m2_ += delta * (pnl - pnl_mean_);
            double sharpe = std::numeric_limits<double>::quiet_NaN();
            if (pnl_n_ > 1) {
                const double sd = std::sqrt(pnl_m2_ / static_cast<double>(pnl_n_ - 1));
                if (sd > 0.0) sharpe = pnl_mean_ / sd;
            }

            return std::make_tuple(drawdown, cum_cost_, cum_turnover_,
                                   trade_count_, worst_dd_, sharpe);
        }

    private:
        double peak_equity_, worst_dd_;
        double cum_cost_, cum_turnover_, trade_count_;
        bool has_prev_pos_;
        double prev_pos_;
        long long pnl_n_;
        double pnl_mean_, pnl_m2_;
    };

} // namespace screamer

#endif // SCREAMER_BACKTEST_REPORT_H
