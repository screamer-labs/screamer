#ifndef SCREAMER_VALIDATION_STRATEGY_TEST_H
#define SCREAMER_VALIDATION_STRATEGY_TEST_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "screamer/backtest_report.h"
#include "screamer/backtest_stats.h"
#include "screamer/portfolio_report.h"
#include "screamer/validation/surrogates.h"

namespace screamer { namespace validation {

    // strategy_test: a whole strategy against the same strategy run on
    // randomized copies of its data. This is the one implementation behind
    // Python's strategy_test and JavaScript's strategyTest. The strategy is
    // user code, so it is reached through a StrategyHost that each binding
    // implements in its own types; everything else (the checks, the
    // randomized copies, moving the other inputs with their bar, the
    // statistics, the metric, and the p-value) happens here.

    // The statistics computed for every run, named as in the summary of
    // backtest_report, in the order the bindings list them.
    //   Engine output columns: [equity, pnl, position, cost].
    //   BacktestReport columns: [drawdown, cum_cost, turnover, trades,
    //                            max_drawdown, sharpe].
    //   BacktestStats columns: [exposure, round_trips, win_rate, profit_factor,
    //                           avg_win, avg_loss, avg_holding, sortino, calmar].
    enum class StatSource { Engine, Report, Stats };
    struct StatSpec { const char* name; StatSource source; int column; };
    inline constexpr std::size_t kNumStats = 15;
    inline constexpr std::array<StatSpec, kNumStats> kStats{{
        {"total_pnl", StatSource::Engine, 0},
        {"max_drawdown", StatSource::Report, 4},
        {"total_cost", StatSource::Report, 1},
        {"turnover", StatSource::Report, 2},
        {"num_trades", StatSource::Report, 3},
        {"sharpe", StatSource::Report, 5},
        {"exposure", StatSource::Stats, 0},
        {"round_trips", StatSource::Stats, 1},
        {"win_rate", StatSource::Stats, 2},
        {"profit_factor", StatSource::Stats, 3},
        {"avg_win", StatSource::Stats, 4},
        {"avg_loss", StatSource::Stats, 5},
        {"avg_holding", StatSource::Stats, 6},
        {"sortino", StatSource::Stats, 7},
        {"calmar", StatSource::Stats, 8},
    }};

    inline std::size_t stat_index(const std::string& name) {
        for (std::size_t s = 0; s < kNumStats; ++s)
            if (name == kStats[s].name) return s;
        throw std::invalid_argument("metric \"" + name + "\" is not a backtest statistic.");
    }

    namespace detail {

        // Sum of n values a[0], a[stride], ... in NumPy's pairwise order
        // (blocks of 8 partial sums, halves above 128 values), so a summed
        // book is the same in every binding as numpy's sum along an axis.
        template <class F>
        double pairwise_sum(const double* a, std::size_t n, std::size_t stride, F f) {
            if (n < 8) {
                double res = 0.0;
                for (std::size_t i = 0; i < n; ++i) res += f(a[i * stride]);
                return res;
            }
            if (n <= 128) {
                double r[8];
                for (std::size_t q = 0; q < 8; ++q) r[q] = f(a[q * stride]);
                std::size_t i = 8;
                for (; i < n - (n % 8); i += 8)
                    for (std::size_t q = 0; q < 8; ++q) r[q] += f(a[(i + q) * stride]);
                double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
                for (; i < n; ++i) res += f(a[i * stride]);
                return res;
            }
            std::size_t n2 = n / 2;
            n2 -= n2 % 8;
            return pairwise_sum(a, n2, stride, f) + pairwise_sum(a + n2 * stride, n - n2, stride, f);
        }

        // Last finite value of column `col` of a row-major (T x ncols) block.
        inline double last_finite(const double* a, std::size_t T, std::size_t ncols,
                                  std::size_t col) {
            for (std::size_t t = T; t-- > 0;) {
                const double v = a[t * ncols + col];
                if (std::isfinite(v)) return v;
            }
            return std::numeric_limits<double>::quiet_NaN();
        }

    } // namespace detail

    // One strategy run's backtest output: (T, 4) [equity, pnl, position, cost],
    // or (T, K, 4) for a portfolio. Row-major. `ndim` and `shape` are as the
    // strategy returned them; the driver checks them.
    struct EngineOutput {
        const double* data = nullptr;
        std::size_t ndim = 0;
        std::size_t shape[3] = {0, 0, 0};
    };

    // Every statistic of one run. A (T, 4) output is one strategy. A (T, K, 4)
    // output is a portfolio, scored as one book: equity, pnl, and cost summed
    // over the assets, the gross position (sum of |position|) as its
    // position, and turnover and the trade count from PortfolioReport, which
    // counts each asset's position changes.
    class RunScorer {
    public:
        explicit RunScorer(double periods_per_year)
            : report_(periods_per_year), stats_(periods_per_year) {}

        std::array<double, kNumStats> score(const EngineOutput& out) {
            const std::size_t T = out.shape[0];
            const double* eng = out.data;
            report_buf_.resize(T * 6);
            stats_buf_.resize(T * 9);
            double* rep = report_buf_.data();
            if (out.ndim == 3) {
                const std::size_t K = out.shape[1];
                book_.resize(T * 4);
                portfolio_.reset();
                for (std::size_t t = 0; t < T; ++t) {
                    const double* row = out.data + t * K * 4;
                    double* b = book_.data() + t * 4;
                    const auto same = [](double v) { return v; };
                    b[0] = detail::pairwise_sum(row + 0, K, 4, same);
                    b[1] = detail::pairwise_sum(row + 1, K, 4, same);
                    b[2] = detail::pairwise_sum(row + 2, K, 4, [](double v) { return std::abs(v); });
                    b[3] = detail::pairwise_sum(row + 3, K, 4, same);
                }
                eng = book_.data();
                run_columns(report_, eng, T, rep, 6);
                // turnover and trades per asset, from PortfolioReport
                double pr[6];
                for (std::size_t t = 0; t < T; ++t) {
                    screamer::detail::write_tuple_to_memory(
                        pr, portfolio_.call_assets(out.data + t * K * 4, K));
                    rep[t * 6 + 2] = pr[2];
                    rep[t * 6 + 3] = pr[3];
                }
                portfolio_.reset();
            } else {
                run_columns(report_, eng, T, rep, 6);
            }
            run_columns(stats_, eng, T, stats_buf_.data(), 9);
            std::array<double, kNumStats> s;
            for (std::size_t q = 0; q < kNumStats; ++q) {
                const StatSpec& spec = kStats[q];
                const std::size_t col = static_cast<std::size_t>(spec.column);
                s[q] = spec.source == StatSource::Engine ? detail::last_finite(eng, T, 4, col)
                     : spec.source == StatSource::Report ? detail::last_finite(rep, T, 6, col)
                     : detail::last_finite(stats_buf_.data(), T, 9, col);
            }
            return s;
        }

    private:
        // The operator's own column loop over a row-major (T x 4) input, as a
        // batch call from either binding runs it.
        static void run_columns(EvalOp& op, const double* in, std::size_t T, double* out,
                                std::size_t width) {
            const double* cols[4] = {in, in + 1, in + 2, in + 3};
            const std::ptrdiff_t strides[4] = {4, 4, 4, 4};
            op.eval_column(cols, strides, T, out, static_cast<std::ptrdiff_t>(width));
        }

        BacktestReport report_;
        BacktestStats stats_;
        PortfolioReport portfolio_;
        std::vector<double> book_, report_buf_, stats_buf_;
    };

    // One strategy input: a series (T,) or a panel (T, K), row-major.
    struct StrategyInput {
        const double* data = nullptr;
        std::size_t T = 0;
        std::size_t K = 1;
        bool panel = false;
    };

    // The binding's side of the driver: it owns the strategy, the custom
    // metric, and the inputs in its own types.
    class StrategyHost {
    public:
        virtual ~StrategyHost() = default;
        // A buffer of `size` doubles for the moved copy of input i in the next
        // run; the driver fills it before run(). A host that hands the buffer
        // to the strategy as it is must return a new one on every call.
        virtual double* moved_input(std::size_t i, std::size_t size) = 0;
        // Runs the strategy. Input i is the buffer of moved_input(i) when
        // moved[i] is nonzero, and the original input otherwise. The returned
        // output stays valid until the next run.
        virtual EngineOutput run(const std::vector<unsigned char>& moved) = 0;
        // The custom metric of the last run's output (only called when the
        // options name no statistic).
        virtual double metric() = 0;
    };

    struct StrategyTestOptions {
        std::int64_t n = 500;
        std::string method = "bars";             // "bars", "returns", or "shift"
        std::vector<std::int64_t> ohlc{0, 1, 2, 3};
        std::int64_t price = 0;
        std::string mode = "log";
        std::optional<std::int64_t> volume;
        std::string volume_mode = "keep";
        std::vector<std::int64_t> columns;
        std::vector<std::int64_t> fixed;
        std::string metric = "sharpe";           // a statistic; empty: the host's metric
        bool larger_is_better = true;
        std::optional<std::int64_t> min_shift;
        std::uint64_t seed = 0;
        double periods_per_year = 1.0;
    };

    struct StrategyTestResult {
        double observed = 0.0;
        std::vector<double> null;                // the finite metrics, in run order
        double p_value = 0.0;
        std::array<double, kNumStats> observed_stats{};
        std::vector<double> null_stats;          // (kNumStats x n): row q is statistic q
    };

    namespace detail {

        enum class StrategyMethod { Bars, Returns, Shift };

        inline StrategyMethod parse_strategy_method(const std::string& m) {
            if (m == "bars") return StrategyMethod::Bars;
            if (m == "returns") return StrategyMethod::Returns;
            if (m == "shift") return StrategyMethod::Shift;
            throw std::invalid_argument("method must be \"bars\", \"returns\", or \"shift\"");
        }

        inline std::vector<std::size_t> check_positions(const std::vector<std::int64_t>& pos,
                                                        std::size_t k, const char* what) {
            std::vector<std::size_t> out;
            out.reserve(pos.size());
            for (std::int64_t i : pos) {
                if (i < 0 || static_cast<std::uint64_t>(i) >= k)
                    throw std::invalid_argument(std::string(what) +
                                                " refers to an input that does not exist");
                out.push_back(static_cast<std::size_t>(i));
            }
            return out;
        }

        inline bool same_shape(const StrategyInput& a, const StrategyInput& b) {
            return a.T == b.T && a.K == b.K && a.panel == b.panel;
        }

        inline void check_output(const EngineOutput& out) {
            const bool ok = (out.ndim == 2 && out.shape[1] == 4) ||
                            (out.ndim == 3 && out.shape[2] == 4);
            if (!ok)
                throw std::invalid_argument(
                    "strategy must return a backtest engine output of shape (T, 4) "
                    "[equity, pnl, position, cost], a portfolio output of shape (T, K, 4), "
                    "or a pipeline output holding one");
        }

    } // namespace detail

    // Runs the strategy once on the real inputs and n times on randomized
    // copies, and scores every run.
    //
    // method "bars": the inputs at `ohlc` are replaced by BarsPermutation
    // copies; "returns": the input at `price` by ReturnsPermutation copies.
    // The input at `volume` moves with its bar and, with volume_mode
    // "notional", is rescaled to keep the traded amount; a volume whose shape
    // differs from the prices moves like any other input ("keep" only). Every
    // other input moves with its bar (row t of copy j is the input's row
    // source[t]) unless it is listed in `fixed`.
    // method "shift": the inputs at `columns` are rotated in time by one
    // random offset per copy (circular_shift of the row positions, min_shift
    // defaulting to a tenth of T); the rest stay as they are.
    //
    // Copy j uses the generator seeded with (seed, j), so the copies are those
    // of the batch kernels. The p-value is (1 + hits) / (1 + finite runs),
    // where a hit is a finite run whose metric is at least as good as the
    // observed one; NaN when the observed metric is not finite.
    inline StrategyTestResult strategy_test(StrategyHost& host,
                                            const std::vector<StrategyInput>& data,
                                            const StrategyTestOptions& opt) {
        using detail::StrategyMethod;
        const StrategyMethod method = detail::parse_strategy_method(opt.method);
        if (opt.n < 1) throw std::invalid_argument("n must be at least 1");
        if (opt.seed >= (std::uint64_t(1) << 53))
            throw std::invalid_argument("seed must be an integer in [0, 2**53)");
        const std::size_t n = static_cast<std::size_t>(opt.n);
        const std::size_t k = data.size();
        if (k == 0) throw std::invalid_argument("strategy_test needs at least one input");
        const std::size_t T = data[0].T;
        for (const auto& x : data)
            if (x.T != T) throw std::invalid_argument("all inputs must have the same length");
        const std::size_t metric_stat = opt.metric.empty() ? kNumStats : stat_index(opt.metric);

        std::vector<unsigned char> fixed(k, 0), shifted(k, 0);
        for (std::size_t i : detail::check_positions(opt.fixed, k, "fixed")) fixed[i] = 1;
        const PermuteMode mode = [&] {
            if (opt.mode != "log" && opt.mode != "diff" && opt.mode != "level")
                throw std::invalid_argument("mode must be \"log\", \"diff\", or \"level\"");
            return parse_permute_mode(opt.mode);
        }();
        if (opt.volume_mode != "keep" && opt.volume_mode != "notional")
            throw std::invalid_argument("volume_mode must be \"keep\" or \"notional\"");
        const bool notional = opt.volume_mode == "notional";
        std::optional<std::size_t> vol_col;
        if (opt.volume)
            vol_col = detail::check_positions({*opt.volume}, k, "volume")[0];
        if (vol_col && method == StrategyMethod::Shift)
            throw std::invalid_argument("volume applies to method=\"bars\" or \"returns\"");
        std::vector<std::size_t> price_cols;
        std::size_t shift = 0;
        if (method == StrategyMethod::Bars) {
            price_cols = detail::check_positions(opt.ohlc, k, "ohlc");
            if (price_cols.size() != 4)
                throw std::invalid_argument("ohlc must list the positions of open, high, low, close");
            for (std::size_t i : price_cols)
                if (!detail::same_shape(data[i], data[price_cols[0]]))
                    throw std::invalid_argument("open, high, low, and close must have the same shape");
        } else if (method == StrategyMethod::Returns) {
            price_cols = detail::check_positions({opt.price}, k, "price");
        } else {
            if (opt.columns.empty())
                throw std::invalid_argument("method=\"shift\" needs the feature positions in columns");
            for (std::size_t i : detail::check_positions(opt.columns, k, "columns")) shifted[i] = 1;
            const std::int64_t m = opt.min_shift
                ? *opt.min_shift : std::max<std::int64_t>(1, static_cast<std::int64_t>(T / 10));
            if (m < 1 || static_cast<std::uint64_t>(m) > T / 2)
                throw std::invalid_argument("min_shift must be in [1, T / 2]");
            shift = static_cast<std::size_t>(m);
        }
        // A volume with the shape of the prices goes to the permutation
        // kernel. Otherwise, such as one (T,) volume next to (T, K) prices,
        // "keep" moves it with its bar like any other input, and "notional"
        // cannot rescale it.
        if (vol_col && !detail::same_shape(data[*vol_col], data[price_cols[0]])) {
            if (notional)
                throw std::invalid_argument("volume_mode=\"notional\" needs a volume input with "
                                            "the shape of the prices");
            vol_col.reset();
        }

        RunScorer scorer(opt.periods_per_year);
        StrategyTestResult res;
        std::vector<unsigned char> moved(k, 0);
        auto score = [&](std::array<double, kNumStats>& stats) {
            const EngineOutput out = host.run(moved);
            detail::check_output(out);
            stats = scorer.score(out);
            return metric_stat < kNumStats ? stats[metric_stat] : host.metric();
        };
        res.observed = score(res.observed_stats);

        // The kernels check their inputs here, after the observed run.
        std::optional<BarsPermutation> bars;
        std::optional<ReturnsPermutation> rets;
        const StrategyInput& P = data[price_cols.empty() ? 0 : price_cols[0]];
        const double* vol = vol_col ? data[*vol_col].data : nullptr;
        if (method == StrategyMethod::Bars)
            bars.emplace(data[price_cols[0]].data, data[price_cols[1]].data,
                         data[price_cols[2]].data, data[price_cols[3]].data, T, P.K, mode,
                         vol, notional);
        else if (method == StrategyMethod::Returns)
            rets.emplace(P.data, T, P.K, mode, vol, notional);
        std::vector<double> positions;
        if (method == StrategyMethod::Shift) {
            positions.resize(T);
            for (std::size_t t = 0; t < T; ++t) positions[t] = static_cast<double>(t);
        }

        // Which inputs a kernel writes: the four prices (or the one), then the
        // volume. A position listed twice keeps its last output, as a later
        // assignment would; an earlier one goes to a scratch buffer.
        std::vector<std::size_t> written = price_cols;
        if (vol_col) written.push_back(*vol_col);
        std::vector<unsigned char> replaced(k, 0);
        for (std::size_t i : written) replaced[i] = 1;
        std::vector<unsigned char> overwritten(written.size(), 0);
        for (std::size_t q = 0; q < written.size(); ++q)
            for (std::size_t r = q + 1; r < written.size(); ++r)
                overwritten[q] |= written[r] == written[q];
        const std::size_t size = T * P.K;
        std::vector<double> scratch(
            size * static_cast<std::size_t>(std::count(overwritten.begin(), overwritten.end(), 1)));
        std::vector<double> source(T);
        res.null_stats.assign(kNumStats * n, 0.0);
        res.null.reserve(n);
        std::array<double, kNumStats> stats;
        for (std::size_t j = 0; j < n; ++j) {
            std::fill(moved.begin(), moved.end(), 0);
            if (method != StrategyMethod::Shift) {
                double* dst[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
                double* spare = scratch.data();
                for (std::size_t q = 0; q < written.size(); ++q) {
                    if (overwritten[q]) {
                        dst[q] = spare;
                        spare += size;
                    } else {
                        dst[q] = host.moved_input(written[q], size);
                        moved[written[q]] = 1;
                    }
                }
                double* ov = vol_col ? dst[written.size() - 1] : nullptr;
                if (bars) bars->write(opt.seed, j, 0, 1, 1, dst[0], dst[1], dst[2], dst[3],
                                      source.data(), ov);
                else rets->write(opt.seed, j, 0, 1, 1, dst[0], source.data(), ov);
            } else {
                circular_shift(positions.data(), T, shift, opt.seed, j, 1, source.data());
            }
            for (std::size_t i = 0; i < k; ++i) {
                if (replaced[i] || fixed[i] || (method == StrategyMethod::Shift && !shifted[i]))
                    continue;
                const std::size_t K = data[i].K;
                const double* x = data[i].data;
                double* dst = host.moved_input(i, T * K);
                for (std::size_t t = 0; t < T; ++t) {
                    const double* row = x + static_cast<std::size_t>(source[t]) * K;
                    for (std::size_t a = 0; a < K; ++a) dst[t * K + a] = row[a];
                }
                moved[i] = 1;
            }
            const double v = score(stats);
            for (std::size_t q = 0; q < kNumStats; ++q) res.null_stats[q * n + j] = stats[q];
            if (std::isfinite(v)) res.null.push_back(v);
        }

        std::size_t hits = 0;
        for (double v : res.null)
            hits += opt.larger_is_better ? v >= res.observed : v <= res.observed;
        res.p_value = std::isfinite(res.observed)
            ? (1.0 + static_cast<double>(hits)) / (1.0 + static_cast<double>(res.null.size()))
            : std::numeric_limits<double>::quiet_NaN();
        return res;
    }

}} // namespace screamer::validation

#endif // SCREAMER_VALIDATION_STRATEGY_TEST_H
