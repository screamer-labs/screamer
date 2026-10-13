#ifndef SCREAMER_BACKTEST_STATS_H
#define SCREAMER_BACKTEST_STATS_H

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include "screamer/common/functor_base.h"
#include "screamer/common/float_info.h"

namespace screamer {

    // BacktestStats: 4 -> 9. Trade-level and downside statistics for a backtest
    // engine's per-step [equity, pnl, position, cost]. A sibling of
    // BacktestReport: every column is a causal running value, so a full-sample
    // statistic is the last finite value.
    //
    //   exposure       share of steps with a nonzero position
    //   round_trips    number of closed trades
    //   win_rate       share of closed trades with positive pnl
    //   profit_factor  gross profit of winning trades / gross loss of losing ones
    //   avg_win        mean pnl of winning trades
    //   avg_loss       mean pnl of losing trades (<= 0)
    //   avg_holding    mean number of steps a closed trade held a position
    //   sortino        mean(pnl) / downside deviation, times sqrt(periods_per_year)
    //   calmar         mean(pnl) * periods_per_year / |max drawdown| (dollar)
    //
    // A trade opens when the position leaves zero and closes when it returns to
    // zero; a sign flip closes one trade and opens the next on the same step.
    // Resizing without crossing zero stays one trade. A trade's pnl is the sum of
    // the step pnl from its opening step to its closing step. On a flip step the
    // whole step's pnl goes to the closing trade, because the engine reports one
    // pnl per step. The downside deviation is sqrt(mean(min(pnl, 0)^2)) over all
    // steps. Ratios with an empty denominator are NaN.
    //
    // nan_policy: ignore. A row with any NaN field is a skipped bar: the state
    // holds and the output row is all NaN.
    class BacktestStats : public FunctorBase<BacktestStats, 4, 9> {
    public:
        // The output columns, in order; the bindings label the columns with these.
        static constexpr std::array<const char*, 9> column_names{{
            "exposure", "round_trips", "win_rate", "profit_factor", "avg_win",
            "avg_loss", "avg_holding", "sortino", "calmar"}};

        explicit BacktestStats(double periods_per_year = 1.0)
            : ppy_(periods_per_year)
        {
            if (!(ppy_ > 0.0) || !std::isfinite(ppy_))
                throw std::invalid_argument("periods_per_year must be finite and positive.");
            reset();
        }

        void reset() override {
            steps_ = 0; exposed_ = 0;
            trades_ = 0; wins_ = 0;
            gross_win_ = 0.0; gross_loss_ = 0.0;
            held_total_ = 0;
            open_ = false; trade_pnl_ = 0.0; trade_len_ = 0; prev_pos_ = 0.0;
            pnl_sum_ = 0.0; down_sq_ = 0.0;
            equity_peak_ = -std::numeric_limits<double>::infinity(); worst_dd_ = 0.0;
        }

        ResultTuple call(const InputArray& in) override {
            const double equity = in[0], pnl = in[1], position = in[2], cost = in[3];
            const double nan = std::numeric_limits<double>::quiet_NaN();
            if (isnan2(equity) || isnan2(pnl) || isnan2(position) || isnan2(cost))
                return std::make_tuple(nan, nan, nan, nan, nan, nan, nan, nan, nan);

            ++steps_;
            if (position != 0.0) ++exposed_;
            pnl_sum_ += pnl;
            if (pnl < 0.0) down_sq_ += pnl * pnl;
            if (equity > equity_peak_) equity_peak_ = equity;
            if (equity - equity_peak_ < worst_dd_) worst_dd_ = equity - equity_peak_;

            // Trade bookkeeping. The step's pnl belongs to the trade that was
            // open coming into the step, or to the trade opened on this step.
            const bool was_open = open_;
            const bool flipped = prev_pos_ != 0.0 && position != 0.0 &&
                                 ((prev_pos_ > 0.0) != (position > 0.0));
            if (was_open) {
                trade_pnl_ += pnl;
                if (position != 0.0 && !flipped) ++trade_len_;   // still held
            }
            if (was_open && (position == 0.0 || flipped)) close_trade();
            if (position != 0.0 && !open_) {
                open_ = true;
                trade_pnl_ = was_open ? 0.0 : pnl;   // a flip's pnl went to the old trade
                trade_len_ = 1;
            }
            prev_pos_ = position;

            const double n = static_cast<double>(steps_);
            const double mean = pnl_sum_ / n;
            const double exposure = static_cast<double>(exposed_) / n;
            const double rt = static_cast<double>(trades_);
            const double win_rate = trades_ ? static_cast<double>(wins_) / rt : nan;
            const double pf = gross_loss_ > 0.0 ? gross_win_ / gross_loss_ : nan;
            const double avg_win = wins_ ? gross_win_ / static_cast<double>(wins_) : nan;
            const long long losses = trades_ - wins_;
            const double avg_loss = losses ? -gross_loss_ / static_cast<double>(losses) : nan;
            const double avg_hold = trades_ ? static_cast<double>(held_total_) / rt : nan;
            const double dd = std::sqrt(down_sq_ / n);
            const double sortino = dd > 0.0 ? mean / dd * std::sqrt(ppy_) : nan;
            const double calmar = worst_dd_ < 0.0 ? mean * ppy_ / -worst_dd_ : nan;
            return std::make_tuple(exposure, rt, win_rate, pf, avg_win, avg_loss,
                                   avg_hold, sortino, calmar);
        }

    private:
        void close_trade() {
            ++trades_;
            if (trade_pnl_ > 0.0) { ++wins_; gross_win_ += trade_pnl_; }
            else gross_loss_ += -trade_pnl_;
            held_total_ += trade_len_;
            open_ = false; trade_pnl_ = 0.0; trade_len_ = 0;
        }

        double ppy_;
        long long steps_ = 0, exposed_ = 0, trades_ = 0, wins_ = 0, held_total_ = 0;
        double gross_win_ = 0.0, gross_loss_ = 0.0;
        bool open_ = false;
        double trade_pnl_ = 0.0;
        long long trade_len_ = 0;
        double prev_pos_ = 0.0;
        double pnl_sum_ = 0.0, down_sq_ = 0.0;
        double equity_peak_ = -std::numeric_limits<double>::infinity();
        double worst_dd_ = 0.0;
    };

} // namespace screamer

#endif // SCREAMER_BACKTEST_STATS_H
