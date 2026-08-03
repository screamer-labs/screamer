#ifndef SCREAMER_PORTFOLIO_REPORT_H
#define SCREAMER_PORTFOLIO_REPORT_H

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>
#include "screamer/common/float_info.h"
#include "screamer/common/functor_base.h"

namespace screamer {

    // PortfolioReport: reduces the per-asset output of a backtest engine and
    // applies the same causal report contract as BacktestReport. The engine
    // output is one row per time step and one four-column record per asset:
    // [equity, pnl, position, cost]. The report output is
    // [drawdown, cum_cost, turnover, trades, max_drawdown, sharpe].
    //
    // Reduction happens before the running statistics are updated:
    //   equity, pnl, and cost are summed across assets;
    //   turnover is the sum of absolute per-asset position changes; and
    //   trades is the number of assets whose position changed on this row.
    // This avoids the common error of calculating turnover from net portfolio
    // position, which would hide offsetting commodity legs.
    //
    // nan_policy: ignore. If any field of any asset is NaN, the row is skipped,
    // the per-asset position state is held, and the output row is all NaN.
    // The asset count is fixed by the first row after reset; a later mismatch
    // is an error rather than a silent change in portfolio definition.
    class PortfolioReport : public FunctorBase<PortfolioReport, 4, 6> {
    public:
        PortfolioReport() { reset(); }

        void reset() override {
            peak_equity_ = -std::numeric_limits<double>::infinity();
            worst_dd_ = 0.0;
            cum_cost_ = 0.0;
            cum_turnover_ = 0.0;
            trade_count_ = 0.0;
            has_prev_pos_ = false;
            prev_pos_ = 0.0;
            prev_positions_.clear();
            pnl_n_ = 0;
            pnl_mean_ = 0.0;
            pnl_m2_ = 0.0;
        }

        // Scalar-compatible path for a pre-aggregated portfolio row. The
        // regular FunctorBase/DAG contract remains available for users that
        // already have portfolio-level [equity, pnl, position, cost].
        ResultTuple call(const InputArray& inputs) override {
            const double equity = inputs[0];
            const double pnl = inputs[1];
            const double position = inputs[2];
            const double cost = inputs[3];
            const double dpos = has_prev_pos_ ? position - prev_pos_ : position;
            prev_positions_.clear();
            return update(equity, pnl, position, cost, std::abs(dpos),
                          dpos != 0.0);
        }

        // Reduce one contiguous (assets, 4) row from a backtest engine.
        ResultTuple call_assets(const double* row, std::size_t assets) {
            if (assets == 0)
                throw std::invalid_argument("PortfolioReport requires at least one asset.");
            if (!prev_positions_.empty() && prev_positions_.size() != assets)
                throw std::invalid_argument("PortfolioReport asset count changed; call reset().");
            if (prev_positions_.empty()) prev_positions_.assign(assets, 0.0);

            double equity = 0.0, pnl = 0.0, position = 0.0, cost = 0.0;
            double turnover = 0.0, trades = 0.0;
            for (std::size_t a = 0; a < assets; ++a) {
                const double e = row[4 * a + 0];
                const double p = row[4 * a + 1];
                const double pos = row[4 * a + 2];
                const double c = row[4 * a + 3];
                if (isnan2(e) || isnan2(p) || isnan2(pos) || isnan2(c)) {
                    const double nan = std::numeric_limits<double>::quiet_NaN();
                    return std::make_tuple(nan, nan, nan, nan, nan, nan);
                }
                equity += e;
                pnl += p;
                cost += c;
                position += pos;
                const double dpos = pos - prev_positions_[a];
                turnover += std::abs(dpos);
                if (dpos != 0.0) trades += 1.0;
            }
            for (std::size_t a = 0; a < assets; ++a)
                prev_positions_[a] = row[4 * a + 2];
            return update(equity, pnl, position, cost, turnover, trades);
        }

    private:
        ResultTuple update(double equity, double pnl, double position,
                           double cost, double turnover, double trades) {
            if (isnan2(equity) || isnan2(pnl) || isnan2(position) || isnan2(cost)) {
                const double nan = std::numeric_limits<double>::quiet_NaN();
                return std::make_tuple(nan, nan, nan, nan, nan, nan);
            }

            if (equity > peak_equity_) peak_equity_ = equity;
            const double drawdown = equity - peak_equity_;
            if (drawdown < worst_dd_) worst_dd_ = drawdown;
            cum_cost_ += cost;
            cum_turnover_ += turnover;
            trade_count_ += trades;

            prev_pos_ = position;
            has_prev_pos_ = true;

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

        double peak_equity_ = -std::numeric_limits<double>::infinity();
        double worst_dd_ = 0.0;
        double cum_cost_ = 0.0, cum_turnover_ = 0.0, trade_count_ = 0.0;
        bool has_prev_pos_ = false;
        double prev_pos_ = 0.0;
        std::vector<double> prev_positions_;
        long long pnl_n_ = 0;
        double pnl_mean_ = 0.0, pnl_m2_ = 0.0;
    };

} // namespace screamer

#endif // SCREAMER_PORTFOLIO_REPORT_H
