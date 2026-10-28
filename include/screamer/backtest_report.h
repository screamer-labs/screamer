#ifndef SCREAMER_BACKTEST_REPORT_H
#define SCREAMER_BACKTEST_REPORT_H

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include "screamer/common/functor_base.h"
#include "screamer/common/float_info.h"

namespace screamer {

    namespace detail {

        // The running report of finite rows, shared by BacktestReport and the
        // book report of a portfolio (backtest_summary.h): one row's equity,
        // pnl, cost, the units it traded, and the number of position changes
        // give the six BacktestReport columns.
        struct ReportState {
            double peak_equity = -std::numeric_limits<double>::infinity();
            double worst_dd = 0.0;
            double cum_cost = 0.0, cum_turnover = 0.0, trade_count = 0.0;
            long long pnl_n = 0;
            double pnl_mean = 0.0, pnl_m2 = 0.0;

            // One finite row; writes the six columns to out.
            void step(double equity, double pnl, double cost, double traded, double changes,
                      double sqrt_ppy, double* out) {
                if (equity > peak_equity) peak_equity = equity;
                const double drawdown = equity - peak_equity;              // dollar, <= 0
                if (drawdown < worst_dd) worst_dd = drawdown;

                cum_cost += cost;
                cum_turnover += traded;
                trade_count += changes;

                // Running Sharpe over finite pnl (Welford; sample std, ddof = 1).
                pnl_n += 1;
                const double delta = pnl - pnl_mean;
                pnl_mean += delta / static_cast<double>(pnl_n);
                pnl_m2 += delta * (pnl - pnl_mean);
                double sharpe = std::numeric_limits<double>::quiet_NaN();
                if (pnl_n > 1) {
                    const double sd = std::sqrt(pnl_m2 / static_cast<double>(pnl_n - 1));
                    if (sd > 0.0) sharpe = pnl_mean / sd * sqrt_ppy;
                }
                out[0] = drawdown;
                out[1] = cum_cost;
                out[2] = cum_turnover;
                out[3] = trade_count;
                out[4] = worst_dd;
                out[5] = sharpe;
            }
        };

    } // namespace detail

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
    //                  finite pnl values with positive dispersion), times
    //                  sqrt(periods_per_year); the default 1 is the per-step ratio
    //
    // nan_policy: ignore. A row with any NaN field is a skipped bar: the state
    // holds (no accumulator advances) and the output row is all NaN, recovering at
    // the next finite bar, as CumSum and the other accumulators do.
    class BacktestReport : public FunctorBase<BacktestReport, 4, 6> {
    public:
        // The output columns, in order; the bindings label the columns with these.
        static constexpr std::array<const char*, 6> column_names{{
            "drawdown", "cum_cost", "turnover", "trades", "max_drawdown", "sharpe"}};

        explicit BacktestReport(double periods_per_year = 1.0)
            : sqrt_ppy_(std::sqrt(periods_per_year))
        {
            if (!(periods_per_year > 0.0) || !std::isfinite(periods_per_year))
                throw std::invalid_argument("periods_per_year must be finite and positive.");
            reset();
        }

        void reset() override {
            state_ = detail::ReportState{};
            has_prev_pos_ = false;
            prev_pos_ = 0.0;
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

            const double dpos = has_prev_pos_ ? (position - prev_pos_) : position;
            const double traded = std::abs(dpos);
            prev_pos_ = position;
            has_prev_pos_ = true;

            double o[6];
            state_.step(equity, pnl, cost, traded, traded > 0.0 ? 1.0 : 0.0, sqrt_ppy_, o);
            return std::make_tuple(o[0], o[1], o[2], o[3], o[4], o[5]);
        }

    private:
        detail::ReportState state_;
        bool has_prev_pos_ = false;
        double prev_pos_ = 0.0;
        double sqrt_ppy_ = 1.0;
    };

} // namespace screamer

#endif // SCREAMER_BACKTEST_REPORT_H
