#ifndef SCREAMER_VALIDATION_SCORING_H
#define SCREAMER_VALIDATION_SCORING_H

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

#include "screamer/backtest_summary.h"
#include "screamer/portfolio_report.h"


namespace screamer { namespace validation {

    // What the randomization tests share: the statistics of a backtest run,
    // the test metric, the checks of the run count and the seed, and the
    // p-value. The drivers are strategy_test (strategy_test.h) and
    // position_test and walk_forward (position_test.h); the Python and
    // JavaScript bindings call them and keep no copy of any of this.

    // The statistics computed for every run: the summary of backtest_report,
    // whose table (names, source column, and direction as a test metric) lives
    // in backtest_summary.h, the one source the bindings read.
    using screamer::StatSource;
    using screamer::StatSpec;
    using screamer::kNumStats;
    using screamer::kStats;

    inline const char* stat_source_name(StatSource s) {
        return s == StatSource::Engine ? "engine" : s == StatSource::Report ? "report" : "stats";
    }

    // The test metric: a statistic (stat < kNumStats) or the caller's own
    // function (stat == kNumStats), and its direction.
    struct Metric {
        std::size_t stat = kNumStats;
        bool larger = true;
        bool custom() const { return stat == kNumStats; }
    };

    // Resolves the metric: `custom` for the caller's function (larger is
    // better unless larger_is_better says otherwise), else `name` must be a
    // statistic, whose own direction is the default.
    inline Metric resolve_metric(const std::string& name, bool custom,
                                 std::optional<bool> larger_is_better) {
        Metric m;
        if (custom) {
            m.larger = larger_is_better.value_or(true);
            return m;
        }
        for (std::size_t s = 0; s < kNumStats; ++s)
            if (name == kStats[s].name) m.stat = s;
        if (m.custom()) {
            std::vector<std::string> names;
            for (const auto& st : kStats) names.emplace_back(st.name);
            std::sort(names.begin(), names.end());
            std::string list;
            for (const auto& q : names) list += (list.empty() ? "'" : ", '") + q + "'";
            throw std::invalid_argument("metric must be one of [" + list + "] or a callable");
        }
        const int dir = kStats[m.stat].larger;
        if (!larger_is_better && dir < 0)
            throw std::invalid_argument("metric \"" + name +
                                        "\" has no natural direction; pass larger_is_better");
        m.larger = larger_is_better ? *larger_is_better : dir == 1;
        return m;
    }

    // Seeds are integers in [0, 2^53), so they are exact JavaScript numbers
    // too: a seed gives the same randomized copies in every binding.
    inline std::uint64_t check_seed(std::int64_t seed) {
        if (seed < 0 || seed >= (std::int64_t(1) << 53))
            throw std::invalid_argument("seed must be an integer in [0, 2**53)");
        return static_cast<std::uint64_t>(seed);
    }

    // The largest number of randomized runs when each run keeps K strategies'
    // results: kNumStats + 1 doubles per strategy (its statistics and its
    // metric), which the binding holds in its own result arrays (ResultSink),
    // so nothing sized by n is held in the driver or the WebAssembly heap.
    // Natively, 2^31 - 1 runs, or fewer where the arrays could not be
    // addressed. In WebAssembly the arrays are JavaScript typed arrays of
    // n * K values each, kept within 2^31 - 1 elements.
    inline std::size_t max_runs(std::size_t K) {
        if (K == 0) K = 1;
        constexpr std::size_t kMaxRuns = 2147483647;
#if defined(__EMSCRIPTEN__)
        return std::max<std::size_t>(1, kMaxRuns / K);
#else
        const std::size_t cap = std::vector<double>().max_size();
        return std::max<std::size_t>(1, std::min(kMaxRuns, (cap - 64) / (kNumStats + 1) / K));
#endif
    }

    inline std::size_t max_strategy_runs() { return max_runs(1); }

    // n checked against [1, max_runs(K)], before anything sized by n is
    // allocated.
    inline std::size_t check_runs(std::int64_t n, std::size_t K) {
        if (n < 1) throw std::invalid_argument("n must be at least 1");
        const std::size_t limit = max_runs(K);
        if (static_cast<std::uint64_t>(n) > limit)
            throw std::invalid_argument("n is too large: at most " + std::to_string(limit) + " runs");
        return static_cast<std::size_t>(n);
    }

    // A custom metric gives one value per strategy in its input.
    inline void check_metric_values(std::size_t expected, std::size_t got) {
        if (got != expected)
            throw std::invalid_argument(
                "metric must return one value per strategy in its input: " +
                std::to_string(expected) + " expected, got " + std::to_string(got));
    }

    // What a custom metric returned, as each binding classifies it from its
    // own types: a number (not a bool), a flat array of numbers (no bools,
    // strings, or nested arrays), or anything else.
    enum class MetricKind { Number = 0, Numbers = 1, Other = 2 };

    // The error for a result of the wrong kind; the bindings raise it as
    // TypeError.
    struct MetricTypeError : std::invalid_argument {
        MetricTypeError()
            : std::invalid_argument("metric must return a number or a 1-D array of numbers") {}
    };

    // The rule for a custom metric's result, the same in every binding: the
    // right kind, then one value per strategy.
    inline void check_metric_result(MetricKind kind, std::size_t expected, std::size_t got) {
        if (kind != MetricKind::Number && kind != MetricKind::Numbers) throw MetricTypeError();
        check_metric_values(expected, got);
    }

    // Where a test's per-run results go: the binding's own arrays, filled in
    // batches, so the result is held once and the driver keeps nothing sized
    // by n.
    class ResultSink {
    public:
        virtual ~ResultSink() = default;
        // Called once n is checked, before the first run: n runs of K values
        // each (K strategies; 1 for one strategy or one book).
        virtual void begin(std::size_t n, std::size_t K) = 0;
        // Runs j0 .. j0 + runs - 1. Statistic q of run j, strategy k is
        // stats[q * stride + (j - j0) * K + k]. `metrics` holds `count`
        // values of the null distribution, appended in order: the metrics
        // in_null keeps for one strategy, and every metric (NaN kept,
        // round-major) for K strategies.
        virtual void emit(std::size_t j0, std::size_t runs, const double* stats,
                          std::size_t stride, const double* metrics, std::size_t count) = 0;
    };

    // (1 + hits) / (1 + count): the share of the null at least as good as the
    // observed value, counting the observed result itself; NaN when the
    // observed value is not finite.
    inline double p_value(std::size_t hits, std::size_t count, double observed) {
        return std::isfinite(observed)
            ? (1.0 + static_cast<double>(hits)) / (1.0 + static_cast<double>(count))
            : std::numeric_limits<double>::quiet_NaN();
    }

    // The rule of every null distribution (each strategy's, the family's max
    // statistic, and strategy_test's): a randomized run's metric belongs to
    // it unless it is NaN. An infinite metric is a real outcome (+inf beats
    // every finite observed value), so it stays.
    inline bool in_null(double v) { return !std::isnan(v); }

    // The hits and the size of a null distribution against one observed
    // value, with in_null's rule; a hit is at least as good as the observed.
    struct NullTally {
        std::size_t hits = 0, count = 0;

        void add(double v, double observed, bool larger) {
            if (!in_null(v)) return;
            ++count;
            hits += larger ? v >= observed : v <= observed;
        }

        double p(double observed) const { return p_value(hits, count, observed); }
    };

    namespace detail {

        // Collects run results and hands them to a ResultSink in batches of at
        // most kBatchValues values per statistic.
        class RunBatch {
        public:
            static constexpr std::size_t kBatchValues = 4096;

            // null_only: append only the metrics in_null keeps (one strategy);
            // otherwise every metric, NaN kept (K strategies).
            RunBatch(ResultSink& sink, std::size_t K, bool null_only)
                : sink_(sink), K_(K), cap_(std::max<std::size_t>(1, kBatchValues / std::max<std::size_t>(1, K))),
                  null_only_(null_only), stats_(kNumStats * cap_ * K), metrics_(cap_ * K) {}

            void put(std::size_t k, const std::array<double, kNumStats>& s, double v) {
                const std::size_t at = runs_ * K_ + k, stride = cap_ * K_;
                for (std::size_t q = 0; q < kNumStats; ++q) stats_[q * stride + at] = s[q];
                if (!null_only_ || in_null(v)) metrics_[count_++] = v;
            }

            void end_run() {
                if (++runs_ == cap_) flush();
            }

            void flush() {
                if (runs_ == 0) return;
                sink_.emit(j0_, runs_, stats_.data(), cap_ * K_, metrics_.data(), count_);
                total_ += count_;
                j0_ += runs_;
                runs_ = count_ = 0;
            }

            std::size_t total() const { return total_; }

        private:
            ResultSink& sink_;
            std::size_t K_, cap_;
            bool null_only_;
            std::vector<double> stats_, metrics_;
            std::size_t j0_ = 0, runs_ = 0, count_ = 0, total_ = 0;
        };

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

    } // namespace detail

    // One backtest output: (T, 4) [equity, pnl, position, cost], or
    // (T, K, 4) for K strategies or a portfolio of K assets. Row-major.
    // `ndim` and `shape` are as the caller returned them; the drivers check
    // them.
    struct EngineOutput {
        const double* data = nullptr;
        std::size_t ndim = 0;
        std::size_t shape[3] = {0, 0, 0};
    };

    // Every statistic of one run. A (T, 4) output is one strategy. A (T, K, 4)
    // output is a portfolio, scored as one book: equity, pnl, and cost summed
    // over the assets, the gross position (sum of |position|) as its
    // position, and turnover and the trade count from PortfolioReport, which
    // counts each asset's position changes. score_column scores one strategy
    // of a (T, k, 4) block of k strategies.
    class RunScorer {
    public:
        explicit RunScorer(double periods_per_year)
            : report_(periods_per_year), stats_(periods_per_year) {}

        std::array<double, kNumStats> score(const EngineOutput& out) {
            const std::size_t T = out.shape[0];
            if (out.ndim != 3) return score_series(out.data, T);
            const std::size_t K = out.shape[1];
            book_.resize(T * 4);
            report_buf_.resize(T * 6);
            double* rep = report_buf_.data();
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
            screamer::detail::run_engine_columns(report_, book_.data(), T, 4, 1, rep, 6);
            // turnover and trades per asset, from PortfolioReport
            double pr[6];
            for (std::size_t t = 0; t < T; ++t) {
                screamer::detail::write_tuple_to_memory(
                    pr, portfolio_.call_assets(out.data + t * K * 4, K));
                rep[t * 6 + 2] = pr[2];
                rep[t * 6 + 3] = pr[3];
            }
            portfolio_.reset();
            return collect(book_.data(), T);
        }

        // Strategy `k` of a (T, cols, 4) block: rows of 4 values, `cols * 4`
        // apart. It is gathered into contiguous rows first, so the three passes
        // over it read memory in order.
        std::array<double, kNumStats> score_column(const double* block, std::size_t T,
                                                   std::size_t cols, std::size_t k) {
            if (cols == 1) return score_series(block, T);
            column_.resize(T * 4);
            const double* src = block + k * 4;
            double* dst = column_.data();
            for (std::size_t t = 0; t < T; ++t, src += cols * 4, dst += 4)
                std::copy(src, src + 4, dst);
            return score_series(column_.data(), T);
        }

    private:
        // One strategy's (T, 4) rows.
        std::array<double, kNumStats> score_series(const double* eng, std::size_t T) {
            report_buf_.resize(T * 6);
            screamer::detail::run_engine_columns(report_, eng, T, 4, 1, report_buf_.data(), 6);
            return collect(eng, T);
        }

        std::array<double, kNumStats> collect(const double* eng, std::size_t T) {
            stats_buf_.resize(T * 9);
            screamer::detail::run_engine_columns(stats_, eng, T, 4, 1, stats_buf_.data(), 9);
            std::array<double, kNumStats> s;
            summary_from(eng, 4, 1, report_buf_.data(), stats_buf_.data(), T, s.data());
            return s;
        }

        BacktestReport report_;
        BacktestStats stats_;
        PortfolioReport portfolio_;
        std::vector<double> book_, column_, report_buf_, stats_buf_;
    };

}} // namespace screamer::validation

#endif // SCREAMER_VALIDATION_SCORING_H
