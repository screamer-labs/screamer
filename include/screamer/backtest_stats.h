#ifndef SCREAMER_BACKTEST_STATS_H
#define SCREAMER_BACKTEST_STATS_H

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include "screamer/common/functor_base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/round_trips.h"

namespace screamer {

    namespace detail {

        // The running trade and downside statistics of finite rows, shared by
        // BacktestStats and the book report of a portfolio
        // (backtest_summary.h). A row adds its equity and pnl (step), each
        // asset adds whether it holds a position (exposure), and each closed
        // round trip adds itself (trade). For one asset, exposure is counted
        // over the same rows as the pnl; a book counts it over its assets'
        // finite rows.
        struct StatsState {
            long long steps = 0, held_n = 0, exposed = 0, trades = 0, wins = 0, held_total = 0;
            double gross_win = 0.0, gross_loss = 0.0;
            double pnl_sum = 0.0, down_sq = 0.0;
            double equity_peak = -std::numeric_limits<double>::infinity();
            double worst_dd = 0.0;

            void step(double equity, double pnl) {
                ++steps;
                pnl_sum += pnl;
                if (pnl < 0.0) down_sq += pnl * pnl;
                if (equity > equity_peak) equity_peak = equity;
                if (equity - equity_peak < worst_dd) worst_dd = equity - equity_peak;
            }

            void exposure(double position) {
                ++held_n;
                if (position != 0.0) ++exposed;
            }

            void trade(const RoundTrip& t) {
                ++trades;
                if (t.pnl > 0.0) { ++wins; gross_win += t.pnl; }
                else gross_loss += -t.pnl;
                held_total += t.held;
            }

            // The nine BacktestStats columns, after at least one step.
            void values(double ppy, double* out) const {
                const double nan = std::numeric_limits<double>::quiet_NaN();
                const double n = static_cast<double>(steps);
                const double mean = pnl_sum / n;
                const double rt = static_cast<double>(trades);
                const long long losses = trades - wins;
                const double dd = std::sqrt(down_sq / n);
                out[0] = static_cast<double>(exposed) / static_cast<double>(held_n);
                out[1] = rt;
                out[2] = trades ? static_cast<double>(wins) / rt : nan;
                out[3] = gross_loss > 0.0 ? gross_win / gross_loss : nan;
                out[4] = wins ? gross_win / static_cast<double>(wins) : nan;
                out[5] = losses ? -gross_loss / static_cast<double>(losses) : nan;
                out[6] = trades ? static_cast<double>(held_total) / rt : nan;
                out[7] = dd > 0.0 ? mean / dd * std::sqrt(ppy) : nan;
                out[8] = worst_dd < 0.0 ? mean * ppy / -worst_dd : nan;
            }
        };

    } // namespace detail

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
            state_ = detail::StatsState{};
            tracker_.reset();
        }

        ResultTuple call(const InputArray& in) override {
            const double equity = in[0], pnl = in[1], position = in[2], cost = in[3];
            if (isnan2(equity) || isnan2(pnl) || isnan2(position) || isnan2(cost)) {
                const double nan = std::numeric_limits<double>::quiet_NaN();
                return std::make_tuple(nan, nan, nan, nan, nan, nan, nan, nan, nan);
            }

            state_.step(equity, pnl);
            state_.exposure(position);
            // Trade bookkeeping (detail::RoundTripTracker, which
            // backtest_trades shares).
            tracker_.step(0, pnl, cost, position,
                          [this](const detail::RoundTrip& t) { state_.trade(t); });

            double o[9];
            state_.values(ppy_, o);
            return std::make_tuple(o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7], o[8]);
        }

    private:
        double ppy_;
        detail::StatsState state_;
        detail::RoundTripTracker tracker_;
    };

} // namespace screamer

#endif // SCREAMER_BACKTEST_STATS_H
