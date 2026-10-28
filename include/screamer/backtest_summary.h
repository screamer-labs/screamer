#ifndef SCREAMER_BACKTEST_SUMMARY_H
#define SCREAMER_BACKTEST_SUMMARY_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "screamer/backtest_report.h"
#include "screamer/backtest_stats.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/round_trips.h"

namespace screamer {

    // The report of a backtest engine's output, shared by every binding: the
    // Python backtest_report and the JavaScript backtestReport only label what
    // these functions return, and the randomization tests score their runs with
    // the same summary (validation/scoring.h).
    //
    // An engine's output is T rows of [equity, pnl, position, cost]. The report
    // is the running BacktestReport columns (T x 6), the running BacktestStats
    // columns (T x 9), and a summary of kNumStats values. Every running column
    // is causal, so each summary value is the last finite value of one column.

    // Column names of an engine's output.
    inline constexpr std::array<const char*, 4> kEngineColumns{{
        "equity", "pnl", "position", "cost"}};

    // An engine's output must be T rows of 4 values: `ndim` and `ncols` as the
    // caller's array has them.
    inline void check_engine_rows(std::size_t ndim, std::size_t ncols) {
        if (ndim != 2 || ncols != kEngineColumns.size())
            throw std::invalid_argument(
                "values must be a (T, 4) array of [equity, pnl, position, cost]");
    }

    // backtest_report's input: one asset's (T, 4) output, or a portfolio
    // engine's (T, K, 4) with at least one asset. `d1` and `d2` are the sizes
    // of the second and third dimensions (0 where the array has none); `who`
    // names the function in the message.
    inline void check_report_rows(const std::string& who, std::size_t ndim, std::size_t d1,
                                  std::size_t d2) {
        const std::size_t nc = kEngineColumns.size();
        if ((ndim == 2 && d1 == nc) || (ndim == 3 && d1 > 0 && d2 == nc)) return;
        if (ndim == 3 && d2 == nc)
            throw std::invalid_argument(who + ": a (T, K, 4) portfolio output needs at least "
                                              "one asset");
        throw std::invalid_argument(who + ": values must be a (T, 4) or (T, K, 4) array of "
                                          "[equity, pnl, position, cost]");
    }

    // periods_per_year, as the BacktestReport and BacktestStats nodes check it.
    inline void check_periods_per_year(double periods_per_year) {
        if (!(periods_per_year > 0.0) || !std::isfinite(periods_per_year))
            throw std::invalid_argument("periods_per_year must be finite and positive.");
    }

    // Where a summary value comes from: an engine column, a BacktestReport
    // column, or a BacktestStats column.
    enum class StatSource { Engine, Report, Stats };

    // One summary value: its name, the column it reads, and its direction as a
    // test metric (`larger`: 1 larger is better, 0 smaller is better, -1 no
    // natural direction), which the randomization tests use.
    struct StatSpec { const char* name; StatSource source; int column; int larger; };

    // The summary, in the order every binding lists it. This table is the one
    // source of the names: the bindings read it instead of keeping copies.
    inline constexpr std::size_t kNumStats = 15;
    inline constexpr std::array<StatSpec, kNumStats> kStats{{
        {"total_pnl", StatSource::Engine, 0, 1},
        {"max_drawdown", StatSource::Report, 4, 1},     // <= 0, so closer to 0 is better
        {"total_cost", StatSource::Report, 1, 0},
        {"turnover", StatSource::Report, 2, -1},
        {"num_trades", StatSource::Report, 3, -1},
        {"sharpe", StatSource::Report, 5, 1},
        {"exposure", StatSource::Stats, 0, -1},
        {"round_trips", StatSource::Stats, 1, -1},
        {"win_rate", StatSource::Stats, 2, 1},
        {"profit_factor", StatSource::Stats, 3, 1},
        {"avg_win", StatSource::Stats, 4, 1},
        {"avg_loss", StatSource::Stats, 5, 1},          // <= 0, so closer to 0 is better
        {"avg_holding", StatSource::Stats, 6, -1},
        {"sortino", StatSource::Stats, 7, 1},
        {"calmar", StatSource::Stats, 8, 1},
    }};

    namespace detail {

        // Last finite value of column `col` of T rows `row_stride` apart, whose
        // columns are `col_stride` apart; NaN when there is none.
        inline double last_finite(const double* a, std::size_t T, std::ptrdiff_t row_stride,
                                  std::ptrdiff_t col_stride, std::size_t col) {
            const double* c = a + static_cast<std::ptrdiff_t>(col) * col_stride;
            for (std::size_t t = T; t-- > 0;) {
                const double v = c[static_cast<std::ptrdiff_t>(t) * row_stride];
                if (std::isfinite(v)) return v;
            }
            return std::numeric_limits<double>::quiet_NaN();
        }

        // The operator's own column loop over T engine rows, as a batch call
        // from either binding runs it.
        inline void run_engine_columns(EvalOp& op, const double* eng, std::size_t T,
                                       std::ptrdiff_t row_stride, std::ptrdiff_t col_stride,
                                       double* out, std::size_t width) {
            const double* cols[4] = {eng, eng + col_stride, eng + 2 * col_stride,
                                     eng + 3 * col_stride};
            const std::ptrdiff_t strides[4] = {row_stride, row_stride, row_stride, row_stride};
            op.eval_column(cols, strides, T, out, static_cast<std::ptrdiff_t>(width));
        }

    } // namespace detail

    // The summary of T engine rows (`row_stride` and `col_stride` apart), the
    // row-major BacktestReport rows `report` (T x 6) and BacktestStats rows
    // `stats` (T x 9): the last finite value of each kStats column, written to
    // summary[0 .. kNumStats).
    inline void summary_from(const double* eng, std::ptrdiff_t row_stride,
                             std::ptrdiff_t col_stride, const double* report,
                             const double* stats, std::size_t T, double* summary) {
        constexpr auto nr = static_cast<std::ptrdiff_t>(BacktestReport::column_names.size());
        constexpr auto ns = static_cast<std::ptrdiff_t>(BacktestStats::column_names.size());
        for (std::size_t q = 0; q < kNumStats; ++q) {
            const StatSpec& spec = kStats[q];
            const std::size_t col = static_cast<std::size_t>(spec.column);
            summary[q] = spec.source == StatSource::Engine
                             ? detail::last_finite(eng, T, row_stride, col_stride, col)
                       : spec.source == StatSource::Report
                             ? detail::last_finite(report, T, nr, 1, col)
                             : detail::last_finite(stats, T, ns, 1, col);
        }
    }

    // The whole report of T engine rows: runs `rep` and `st` over them (a
    // column run starts from a reset node), writing report (T x 6) and stats
    // (T x 9) row-major, and the summary (kNumStats values). The nodes carry
    // periods_per_year, which annualizes sharpe, sortino, and calmar.
    inline void backtest_report(BacktestReport& rep, BacktestStats& st, const double* eng,
                                std::size_t T, std::ptrdiff_t row_stride,
                                std::ptrdiff_t col_stride, double* report, double* stats,
                                double* summary) {
        detail::run_engine_columns(rep, eng, T, row_stride, col_stride, report,
                                   BacktestReport::column_names.size());
        detail::run_engine_columns(st, eng, T, row_stride, col_stride, stats,
                                   BacktestStats::column_names.size());
        summary_from(eng, row_stride, col_stride, report, stats, T, summary);
    }

    // The same with nodes built for `periods_per_year` (finite and positive,
    // or std::invalid_argument).
    inline void backtest_report(const double* eng, std::size_t T, std::ptrdiff_t row_stride,
                                std::ptrdiff_t col_stride, double periods_per_year,
                                double* report, double* stats, double* summary) {
        BacktestReport rep(periods_per_year);
        BacktestStats st(periods_per_year);
        backtest_report(rep, st, eng, T, row_stride, col_stride, report, stats, summary);
    }

    // backtest_report(by=): the statistics of each group of rows, in this
    // order.
    inline constexpr std::array<const char*, 6> kGroupStats{{
        "total_pnl", "total_cost", "turnover", "sharpe", "max_drawdown", "steps"}};

    // One group label per engine row.
    inline void check_group_codes(std::size_t n_codes, std::size_t T) {
        if (n_codes != T)
            throw std::invalid_argument("by must hold one label per row: " + std::to_string(T) +
                                        " rows, " + std::to_string(n_codes) + " labels");
    }

    namespace detail {

        // The statistics of G groups of T rows, row t in group codes[t].
        // `row(t, equity, pnl, cost, traded)` reads row t and returns false for
        // a skipped row; see backtest_groups.
        template <class Row>
        void groups_of(std::size_t T, Row&& row, const std::int64_t* codes, std::size_t G,
                       double periods_per_year, double* out) {
            check_periods_per_year(periods_per_year);
            for (std::size_t t = 0; t < T; ++t)
                if (codes[t] < 0 || static_cast<std::size_t>(codes[t]) >= G)
                    throw std::invalid_argument("group codes must lie in [0, groups)");
            struct Acc {
                double pnl = 0.0, cost = 0.0, turnover = 0.0, mean = 0.0, m2 = 0.0, cum = 0.0;
                double peak = -std::numeric_limits<double>::infinity(), worst = 0.0;
                long long n = 0;
            };
            std::vector<Acc> acc(G);
            double prev_equity = 0.0;
            for (std::size_t t = 0; t < T; ++t) {
                double equity, pnl, cost, traded;
                if (!row(t, equity, pnl, cost, traded)) continue;
                Acc& g = acc[static_cast<std::size_t>(codes[t])];
                g.pnl += equity - prev_equity;
                prev_equity = equity;
                g.cost += cost;
                g.turnover += traded;
                g.n += 1;
                const double delta = pnl - g.mean;
                g.mean += delta / static_cast<double>(g.n);
                g.m2 += delta * (pnl - g.mean);
                g.cum += pnl;
                if (g.cum > g.peak) g.peak = g.cum;
                if (g.cum - g.peak < g.worst) g.worst = g.cum - g.peak;
            }
            const double sqrt_ppy = std::sqrt(periods_per_year);
            for (std::size_t k = 0; k < G; ++k) {
                const Acc& g = acc[k];
                double sharpe = std::numeric_limits<double>::quiet_NaN();
                if (g.n > 1) {
                    const double sd = std::sqrt(g.m2 / static_cast<double>(g.n - 1));
                    if (sd > 0.0) sharpe = g.mean / sd * sqrt_ppy;
                }
                double* o = out + k * kGroupStats.size();
                o[0] = g.pnl;
                o[1] = g.cost;
                o[2] = g.turnover;
                o[3] = sharpe;
                o[4] = g.worst;
                o[5] = static_cast<double>(g.n);
            }
        }

    } // namespace detail

    // The statistics of G groups of T engine rows (`row_stride` and
    // `col_stride` apart), row t in group codes[t], written to out[g * 6 + j]
    // in kGroupStats order. Rows with a NaN field are skipped, as
    // BacktestReport skips them. turnover measures each position change from
    // the previous finite row of the whole output (the first from flat), and
    // total_pnl is the equity gained over the group's rows, each measured from
    // the previous finite row (the first from 0), so the groups add up to the
    // ungrouped total_pnl (the last equity) even where NaN rows hide a pnl.
    // For an engine's equity, the running sum of pnl, this is the sum of the
    // group's pnl. sharpe is BacktestReport's running Sharpe ratio over the
    // group's rows, so a single group reproduces the ungrouped turnover and
    // sharpe.
    inline void backtest_groups(const double* eng, std::size_t T, std::ptrdiff_t row_stride,
                                std::ptrdiff_t col_stride, const std::int64_t* codes,
                                std::size_t G, double periods_per_year, double* out) {
        bool has_prev = false;
        double prev_pos = 0.0;
        detail::groups_of(
            T,
            [&](std::size_t t, double& equity, double& pnl, double& cost, double& traded) {
                const double* r = eng + static_cast<std::ptrdiff_t>(t) * row_stride;
                equity = r[0];
                pnl = r[col_stride];
                cost = r[3 * col_stride];
                const double pos = r[2 * col_stride];
                if (std::isnan(equity) || std::isnan(pnl) || std::isnan(pos) || std::isnan(cost))
                    return false;
                traded = std::abs(has_prev ? pos - prev_pos : pos);
                prev_pos = pos;
                has_prev = true;
                return true;
            },
            codes, G, periods_per_year, out);
    }

    // ---- The book of a portfolio engine's (T, K, 4) output ----
    //
    // The book is the account the K assets share. Its row t sums the assets:
    // equity, pnl, and cost are summed over the assets, the units traded are
    // the sum of each asset's |position change|, and the position changes are
    // counted per asset, as PortfolioReport counts them. An asset whose row
    // has a NaN field is skipped on that row, as the single-asset report skips
    // it: it adds no pnl, cost, or trade, and its equity and position hold at
    // their last values (0 before its first finite row), so an asset that
    // starts late joins the book when its data starts. A book row is skipped
    // (all NaN) only when every asset is skipped.
    //
    // The book's report columns come from its rows, as for one asset: the
    // drawdown, cum_cost, max_drawdown, and sharpe of the summed equity, pnl,
    // and cost, and turnover and trades from the units traded and the
    // position changes. Its statistics count over all assets: exposure is the
    // share of the assets' finite rows that hold a position, and the round
    // trips of every asset make the trade statistics (round_trips, win_rate,
    // profit_factor, avg_win, avg_loss, avg_holding); sortino and calmar come
    // from the book's pnl and equity.

    // The book's own columns; its report and stats columns follow them.
    inline constexpr std::array<const char*, 3> kBookColumns{{"equity", "pnl", "cost"}};

    namespace detail {
        // summary_from reads the book's columns where it reads an engine's:
        // every engine column kStats reads must be the engine's equity (field
        // 0), which is also the book's column 0.
        constexpr bool engine_stats_read_equity() {
            for (const StatSpec& s : kStats)
                if (s.source == StatSource::Engine && s.column != 0) return false;
            return true;
        }
    } // namespace detail
    static_assert(detail::engine_stats_read_equity(),
                  "backtest_book_report passes the book to summary_from as engine rows; "
                  "a kStats engine column other than equity needs a book column");

    namespace detail {

        // One book row: the sums over the assets, and whether any asset had a
        // finite row.
        struct BookRow {
            double equity = 0.0, pnl = 0.0, cost = 0.0, traded = 0.0, changes = 0.0;
            bool valid = false;
        };

        // Reads the book rows of a (T, K, 4) output, holding each asset's last
        // position and equity.
        class BookReader {
        public:
            explicit BookReader(std::size_t K) : prev_pos_(K, 0.0), last_equity_(K, 0.0) {}

            // Row `r` (assets `asset_stride` apart, fields `col_stride` apart);
            // on_asset(k, pnl, cost, position) for each asset with a finite row.
            template <class OnAsset>
            BookRow read(const double* r, std::ptrdiff_t asset_stride, std::ptrdiff_t col_stride,
                         OnAsset&& on_asset) {
                BookRow b;
                for (std::size_t k = 0; k < prev_pos_.size(); ++k) {
                    const double* a = r + static_cast<std::ptrdiff_t>(k) * asset_stride;
                    const double equity = a[0], pnl = a[col_stride], pos = a[2 * col_stride],
                                 cost = a[3 * col_stride];
                    if (isnan2(equity) || isnan2(pnl) || isnan2(pos) || isnan2(cost)) {
                        b.equity += last_equity_[k];        // skipped: the asset holds
                        continue;
                    }
                    b.valid = true;
                    b.equity += equity;
                    b.pnl += pnl;
                    b.cost += cost;
                    const double traded = std::abs(pos - prev_pos_[k]);
                    b.traded += traded;
                    if (traded > 0.0) b.changes += 1.0;
                    prev_pos_[k] = pos;
                    last_equity_[k] = equity;
                    on_asset(k, pnl, cost, pos);
                }
                return b;
            }

        private:
            std::vector<double> prev_pos_, last_equity_;
        };

    } // namespace detail

    // The report of a (T, K, 4) output (rows `row_stride`, assets
    // `asset_stride`, fields `col_stride` apart): the book's columns (T x 3,
    // kBookColumns), report (T x 6), and stats (T x 9), row-major; the book's
    // summary (kNumStats values); and each asset's summary, as
    // backtest_report gives it for that asset alone (K x kNumStats).
    inline void backtest_book_report(const double* eng, std::size_t T, std::size_t K,
                                     std::ptrdiff_t row_stride, std::ptrdiff_t asset_stride,
                                     std::ptrdiff_t col_stride, double periods_per_year,
                                     double* book, double* report, double* stats,
                                     double* summary, double* asset_summary) {
        constexpr std::size_t nb = kBookColumns.size();
        constexpr std::size_t nr = BacktestReport::column_names.size();
        constexpr std::size_t ns = BacktestStats::column_names.size();
        BacktestReport rep(periods_per_year);     // checks periods_per_year
        BacktestStats st(periods_per_year);
        {
            // One asset at a time, through the single-asset report; the
            // running columns go to scratch, reused across the assets.
            std::vector<double> rs(T * nr), ss(T * ns);
            for (std::size_t k = 0; k < K; ++k)
                backtest_report(rep, st, eng + static_cast<std::ptrdiff_t>(k) * asset_stride, T,
                                row_stride, col_stride, rs.data(), ss.data(),
                                asset_summary + k * kNumStats);
        }
        const double sqrt_ppy = std::sqrt(periods_per_year);
        const double nan = std::numeric_limits<double>::quiet_NaN();
        detail::BookReader reader(K);
        detail::ReportState rstate;
        detail::StatsState sstate;
        std::vector<detail::RoundTripTracker> trackers(K);
        const auto on_trade = [&sstate](const detail::RoundTrip& tr) { sstate.trade(tr); };
        for (std::size_t t = 0; t < T; ++t) {
            const detail::BookRow b = reader.read(
                eng + static_cast<std::ptrdiff_t>(t) * row_stride, asset_stride, col_stride,
                [&](std::size_t k, double pnl, double cost, double pos) {
                    sstate.exposure(pos);
                    trackers[k].step(t, pnl, cost, pos, on_trade);
                });
            double* bo = book + t * nb;
            double* ro = report + t * nr;
            double* so = stats + t * ns;
            if (!b.valid) {
                std::fill(bo, bo + nb, nan);
                std::fill(ro, ro + nr, nan);
                std::fill(so, so + ns, nan);
                continue;
            }
            bo[0] = b.equity;
            bo[1] = b.pnl;
            bo[2] = b.cost;
            rstate.step(b.equity, b.pnl, b.cost, b.traded, b.changes, sqrt_ppy, ro);
            sstate.step(b.equity, b.pnl);
            sstate.values(periods_per_year, so);
        }
        summary_from(book, static_cast<std::ptrdiff_t>(nb), 1, report, stats, T, summary);
    }

    // backtest_groups for the book of a (T, K, 4) output: the book's rows, as
    // backtest_book_report reads them, grouped.
    inline void backtest_book_groups(const double* eng, std::size_t T, std::size_t K,
                                     std::ptrdiff_t row_stride, std::ptrdiff_t asset_stride,
                                     std::ptrdiff_t col_stride, const std::int64_t* codes,
                                     std::size_t G, double periods_per_year, double* out) {
        detail::BookReader reader(K);
        const auto none = [](std::size_t, double, double, double) {};
        detail::groups_of(
            T,
            [&](std::size_t t, double& equity, double& pnl, double& cost, double& traded) {
                const detail::BookRow b = reader.read(
                    eng + static_cast<std::ptrdiff_t>(t) * row_stride, asset_stride, col_stride,
                    none);
                equity = b.equity;
                pnl = b.pnl;
                cost = b.cost;
                traded = b.traded;
                return b.valid;
            },
            codes, G, periods_per_year, out);
    }

} // namespace screamer

#endif // SCREAMER_BACKTEST_SUMMARY_H
