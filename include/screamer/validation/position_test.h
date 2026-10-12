#ifndef SCREAMER_VALIDATION_POSITION_TEST_H
#define SCREAMER_VALIDATION_POSITION_TEST_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "screamer/validation/scoring.h"
#include "screamer/validation/surrogates.h"

namespace screamer { namespace validation {

    // position_test and walk_forward: tests that run a backtest engine on
    // target position series. This is the one implementation behind Python's
    // position_test and walk_forward and JavaScript's positionTest and
    // walkForward. The engine, the market inputs, and a custom metric belong
    // to the binding and are reached through an EngineHost; everything else
    // (the checks, the randomized targets, the statistics, the metric, the
    // p-values, and the fold selection) happens here.

    // An array: (T,) when ndim is 1, (T, K) when ndim is 2. `data` is
    // row-major and only read for a target.
    struct ArrayShape {
        const double* data = nullptr;
        std::size_t ndim = 1;
        std::size_t T = 0;
        std::size_t K = 1;
    };

    // What a custom metric is applied to (EngineHost::metric).
    enum class MetricOf {
        Columns,   // each strategy of the last output: (T, 4) or (T, k, 4)
        Book,      // the last (T, K, 4) portfolio output as one book
        Window,    // each strategy of the last window_buffer, (rows, K, 4)
    };

    // The binding's side of the drivers: it owns the engine, the market
    // inputs, the caller's target, and the custom metric in its own types.
    class EngineHost {
    public:
        virtual ~EngineHost() = default;
        // Room for the target of the next run, `size` doubles that the driver
        // fills. A host that hands the buffer to the engine as it is must
        // return a new one on every call.
        virtual double* target_buffer(std::size_t size) = 0;
        // Resets the engine and runs it on a target and the market inputs.
        // `original`: the caller's target as given; otherwise the last
        // target_buffer, of shape (T,) when cols is 0 and (T, cols) otherwise.
        // The output stays valid until the next run.
        virtual EngineOutput run(bool original, std::size_t T, std::size_t cols) = 0;
        // Resets the engine after the last run.
        virtual void reset() = 0;
        // Room for a (rows, K, 4) window that the driver fills before
        // metric(MetricOf::Window, ...). The same rule as target_buffer.
        virtual double* window_buffer(std::size_t size) = 0;
        // The custom metric of `what`: `count` values, one per strategy (one
        // for a book). Checks the result with check_metric_result (the
        // binding knows how many strategies each call of the user's metric
        // sees), writes `count` values, and returns `count`. `rows` is the
        // window length.
        virtual std::size_t metric(MetricOf what, double* values, std::size_t count,
                                   std::size_t rows) = 0;
    };

    // position_test's results go to the host in batches (ResultSink), so the
    // binding holds them once in its own arrays.
    class PositionHost : public EngineHost, public ResultSink {};

    struct PositionTestOptions {
        std::int64_t n = 1000;
        std::string method = "shift";            // "shift" or "segments"
        std::string metric = "sharpe";           // a statistic, unless custom
        bool custom = false;                     // the host's metric
        std::optional<bool> larger_is_better;    // default: the metric's direction
        std::optional<std::int64_t> min_shift;
        std::int64_t seed = 0;
        double periods_per_year = 1.0;
        bool portfolio = false;                  // a portfolio engine: one book of K assets
    };

    struct PositionTestResult {
        bool family = false;                     // K strategies tested jointly
        std::size_t K = 1;                       // values per run
        std::vector<double> observed, p_value, family_p_value;   // K each (family: K)
        std::vector<double> observed_stats;      // statistic q of strategy k at q * K + k
        std::optional<std::int64_t> min_shift;   // none for "segments"
        std::size_t null_count = 0;              // values appended through emit
    };

    struct WalkForwardOptions {
        double train = 0.0;                      // checked to be an integer >= 2
        double test = 0.0;                       // checked to be an integer >= 1
        std::string metric = "sharpe";
        bool custom = false;
        std::optional<bool> larger_is_better;
        bool anchored = false;
        double periods_per_year = 1.0;
        bool portfolio = false;                  // rejected: a single-asset engine is needed
    };

    struct WalkForwardResult {
        std::size_t folds = 0, K = 0;
        std::vector<std::int64_t> fold_start, choice;
        std::unique_ptr<double[]> scores;        // (folds, K)
        std::array<double, kNumStats> stats{};
    };

    // Doubles of the (T, n) working arrays per chunk of randomized targets,
    // about 256 MB.
    inline constexpr std::size_t kChunkDoubles = 32000000;

    namespace detail {

        inline void check_engine_output(const EngineOutput& out, std::size_t T, std::size_t cols) {
            const bool ok = cols == 0
                ? out.ndim == 2 && out.shape[0] == T && out.shape[1] == 4
                : (out.ndim == 3 && out.shape[0] == T && out.shape[1] == cols && out.shape[2] == 4) ||
                  (cols == 1 && out.ndim == 2 && out.shape[0] == T && out.shape[1] == 4);
            if (!ok)
                throw std::invalid_argument(
                    "engine must return a backtest output of shape (T, 4) for a (T,) target, "
                    "or (T, k, 4) for a (T, k) target");
        }

        inline void check_book_output(const EngineOutput& out, std::size_t T, std::size_t K) {
            if (!(out.ndim == 3 && out.shape[0] == T && out.shape[1] == K && out.shape[2] == 4))
                throw std::invalid_argument(
                    "a portfolio engine must return an output of shape (T, K, 4)");
        }

        // The custom metric of `what`, checked to give `count` values.
        inline const double* custom_metric(EngineHost& host, MetricOf what, std::vector<double>& buf,
                                           std::size_t count, std::size_t rows = 0) {
            buf.assign(count, std::numeric_limits<double>::quiet_NaN());
            check_metric_values(count, host.metric(what, buf.data(), count, rows));
            return buf.data();
        }

        // The columns of a row-major (T, K) block, one after the other.
        inline std::vector<double> columns_of(const ArrayShape& x) {
            std::vector<double> cols(x.T * x.K);
            for (std::size_t t = 0; t < x.T; ++t)
                for (std::size_t k = 0; k < x.K; ++k) cols[k * x.T + t] = x.data[t * x.K + k];
            return cols;
        }

        // The common warmup and minimum shift of K columns shifted together:
        // the longest NaN prefix, and min_shift or the largest default
        // minimum shift capped at a quarter of the length after that warmup.
        struct JointShift { std::size_t start, min_shift; };
        inline JointShift joint_shift(const std::vector<double>& cols, std::size_t T, std::size_t K,
                                      const std::optional<std::int64_t>& min_shift) {
            if (K == 0) throw std::invalid_argument("target needs at least one column");
            std::size_t start = 0, def = 0;
            for (std::size_t k = 0; k < K; ++k) {
                start = std::max(start, valid_start(cols.data() + k * T, T));
                if (!min_shift) def = std::max(def, default_min_shift(cols.data() + k * T, T));
            }
            const std::int64_t m = min_shift ? *min_shift
                : static_cast<std::int64_t>(std::max<std::size_t>(1, std::min(def, (T - start) / 4)));
            if (m < 1) throw std::invalid_argument("min_shift must be at least 1");
            return {start, static_cast<std::size_t>(m)};
        }

        // Randomized runs per chunk: the (T, chunk * K) working arrays of
        // `width` doubles per value stay within kChunkDoubles.
        inline std::size_t chunk_runs(std::size_t n, std::size_t T, std::size_t width, std::size_t K) {
            const std::size_t per_run = kChunkDoubles / width / std::max<std::size_t>(1, K);
            return std::max<std::size_t>(1, std::min(n, per_run / std::max<std::size_t>(1, T)));
        }

        // Rows [a, b) of strategy k of a (T, K, 4) output, written as (b - a)
        // rows of 4 at `dst`, `dst_stride` doubles apart, and scored as a
        // backtest of its own: for a > 0 the equity is shifted so the window
        // starts from the equity the run started from.
        inline void window_rows(const double* out, std::size_t T, std::size_t K, std::size_t k,
                                std::size_t a, std::size_t b, double* dst, std::size_t dst_stride) {
            const auto eq = [&](std::size_t t) { return out[(t * K + k) * 4]; };
            for (std::size_t t = a; t < b; ++t) {
                const double* src = out + (t * K + k) * 4;
                std::copy(src, src + 4, dst + (t - a) * dst_stride);
            }
            if (a == 0) return;
            double prior = std::numeric_limits<double>::quiet_NaN();   // equity before the window
            for (std::size_t t = a; t-- > 0;)
                if (std::isfinite(eq(t))) { prior = eq(t); break; }
            std::size_t first = 0;
            while (first < T && std::isnan(eq(first))) ++first;
            if (first == T) first = 0;
            const double start = eq(first) - out[(first * K + k) * 4 + 1];   // equity the run started from
            const double shift = std::isnan(prior) ? 0.0 : prior - start;
            for (std::size_t t = a; t < b; ++t) dst[(t - a) * dst_stride] -= shift;
        }

        // The best of K scores: NaN loses, ties go to the lowest index, -1
        // when every score is NaN.
        inline std::int64_t pick(const double* s, std::size_t K, bool larger) {
            std::int64_t best = -1;
            double bv = 0.0;
            for (std::size_t k = 0; k < K; ++k) {
                if (std::isnan(s[k])) continue;
                const double v = larger ? s[k] : -s[k];
                if (best < 0 || v > bv) { best = static_cast<std::int64_t>(k); bv = v; }
            }
            return best;
        }

    } // namespace detail

    // position_test: the strategy's target against randomized copies of it,
    // each run through a fresh engine on the same market.
    //
    // One strategy, a (T,) target: method "shift" rotates the target by a
    // random offset (circular_shift; min_shift defaults to
    // default_min_shift), "segments" shuffles its runs (segment_shuffle).
    // Copies are run in chunks: the engine gets a (T, k) block of k targets.
    //
    // K strategies, a (T, K) target ("shift" only): every copy shifts the K
    // columns by one offset, after their common warmup. Each strategy gets its
    // own p-value, and the family p-value compares it with the best of the K
    // in every copy (the max statistic of White's Reality Check). Rounds
    // where every strategy's metric is NaN are left out.
    //
    // A portfolio engine, a (T, K) target and (T, K) market inputs ("shift"
    // only): every copy shifts the K assets together and the run is scored as
    // one book.
    //
    // Copy j uses the generator seeded with (seed, j). The p-value is
    // (1 + hits) / (1 + runs in the null), where the null holds every run
    // whose metric is not NaN (in_null; an infinite metric stays) and a hit
    // is a run at least as good as the observed one; NaN when the observed
    // metric is not finite.
    inline PositionTestResult position_test(PositionHost& host, const ArrayShape& target,
                                            const std::vector<ArrayShape>& market,
                                            const PositionTestOptions& opt) {
        if (opt.method != "shift" && opt.method != "segments")
            throw std::invalid_argument("method must be \"shift\" or \"segments\"");
        const bool shift = opt.method == "shift";
        const Metric metric = resolve_metric(opt.metric, opt.custom, opt.larger_is_better);
        const bool family = !opt.portfolio && target.ndim == 2;
        const std::size_t K = family ? target.K : 1;
        const std::size_t n = check_runs(opt.n, K);
        const std::uint64_t seed = check_seed(opt.seed);
        const std::size_t T = target.T;
        if (opt.portfolio) {
            if (!shift)
                throw std::invalid_argument("a portfolio is tested with method=\"shift\", "
                                            "which moves all assets together");
            if (target.ndim != 2)
                throw std::invalid_argument("a portfolio engine needs a (T, K) target");
            for (const ArrayShape& m : market)
                if (m.ndim != 2 || m.T != T || m.K != target.K)
                    throw std::invalid_argument("target and market inputs must all have shape (T, K)");
        } else {
            if (family && !shift)
                throw std::invalid_argument("several strategies need method=\"shift\", "
                                            "which shifts them jointly");
            if (!family && target.ndim != 1)
                throw std::invalid_argument("target must be a 1-D series");
            for (const ArrayShape& m : market)
                if (m.ndim != 1) throw std::invalid_argument("market input must be a 1-D series");
            for (const ArrayShape& m : market)
                if (m.T != T)
                    throw std::invalid_argument("target and market inputs must have the same length");
        }

        PositionTestResult res;
        res.family = family;
        res.K = K;
        res.observed.assign(K, 0.0);
        res.observed_stats.assign(kNumStats * K, 0.0);
        RunScorer scorer(opt.periods_per_year);
        std::vector<double> custom;
        const double sign = metric.larger ? 1.0 : -1.0;
        const auto value = [&](const std::array<double, kNumStats>& s, const double* c, std::size_t i) {
            return metric.custom() ? c[i] : s[metric.stat];
        };
        const auto observe = [&](std::size_t k, const std::array<double, kNumStats>& s, double v) {
            res.observed[k] = v;
            for (std::size_t q = 0; q < kNumStats; ++q) res.observed_stats[q * K + k] = s[q];
        };

        if (opt.portfolio) {
            const std::size_t A = target.K;
            const std::vector<double> cols = detail::columns_of(target);
            const detail::JointShift js = detail::joint_shift(cols, T, A, opt.min_shift);
            res.min_shift = static_cast<std::int64_t>(js.min_shift);
            host.begin(n, 1);
            const auto run = [&](bool original, std::array<double, kNumStats>& s) {
                const EngineOutput out = host.run(original, T, A);
                detail::check_book_output(out, T, A);
                s = scorer.score(out);
                return metric.custom() ? detail::custom_metric(host, MetricOf::Book, custom, 1)[0]
                                       : s[metric.stat];
            };
            std::array<double, kNumStats> s;
            const double obs = run(true, s);
            observe(0, s, obs);
            detail::RunBatch batch(host, 1, true);
            NullTally tally;
            for (std::size_t j = 0; j < n; ++j) {
                double* buf = host.target_buffer(T * A);
                for (std::size_t a = 0; a < A; ++a)
                    circular_shift(cols.data() + a * T, T, js.min_shift, seed, j, 1, buf + a, js.start, A);
                const double v = run(false, s);
                batch.put(0, s, v);
                batch.end_run();
                tally.add(v, obs, metric.larger);
            }
            batch.flush();
            host.reset();
            res.null_count = batch.total();
            res.p_value.assign(1, tally.p(obs));
            return res;
        }

        const std::size_t width = market.size() + 1 + 4 + 6 + 9;   // inputs, engine output, report, stats
        if (family) {
            const std::vector<double> cols = detail::columns_of(target);
            const detail::JointShift js = detail::joint_shift(cols, T, K, opt.min_shift);
            res.min_shift = static_cast<std::int64_t>(js.min_shift);
            host.begin(n, K);
            const EngineOutput out = host.run(true, T, K);
            detail::check_engine_output(out, T, K);
            const double* c = metric.custom()
                ? detail::custom_metric(host, MetricOf::Columns, custom, K) : nullptr;
            for (std::size_t k = 0; k < K; ++k) {
                const auto s = scorer.score_column(out.data, T, K, k);
                observe(k, s, value(s, c, k));
            }
            detail::RunBatch batch(host, K, false);
            // Each strategy's null, and the family's: per round, the best
            // signed metric of the K (NaN when every one is NaN, which
            // in_null leaves out), against each signed observed metric.
            std::vector<NullTally> tally(K), family(K);
            const std::size_t chunk = detail::chunk_runs(n, T, width, K);
            for (std::size_t col0 = 0; col0 < n; col0 += chunk) {
                const std::size_t m = std::min(chunk, n - col0), w = m * K;
                // (T, m * K): copy j of strategy k in column k * m + j; all K
                // strategies share each copy's offset.
                double* buf = host.target_buffer(T * w);
                for (std::size_t k = 0; k < K; ++k)
                    circular_shift(cols.data() + k * T, T, js.min_shift, seed, col0, m, buf + k * m,
                                   js.start, w);
                const EngineOutput o = host.run(false, T, w);
                detail::check_engine_output(o, T, w);
                const double* cv = metric.custom()
                    ? detail::custom_metric(host, MetricOf::Columns, custom, w) : nullptr;
                for (std::size_t j = 0; j < m; ++j) {
                    double best = std::numeric_limits<double>::quiet_NaN();
                    for (std::size_t k = 0; k < K; ++k) {
                        const auto s = scorer.score_column(o.data, T, w, k * m + j);
                        const double v = value(s, cv, k * m + j);
                        batch.put(k, s, v);
                        tally[k].add(v, res.observed[k], metric.larger);
                        const double sv = sign * v;
                        if (in_null(sv) && !(sv <= best)) best = sv;
                    }
                    batch.end_run();
                    for (std::size_t k = 0; k < K; ++k)
                        family[k].add(best, sign * res.observed[k], true);
                }
            }
            batch.flush();
            host.reset();
            res.null_count = batch.total();
            res.p_value.resize(K);
            res.family_p_value.resize(K);
            for (std::size_t k = 0; k < K; ++k) {
                res.p_value[k] = tally[k].p(res.observed[k]);
                res.family_p_value[k] = family[k].p(res.observed[k]);
            }
            return res;
        }

        host.begin(n, 1);
        const EngineOutput out = host.run(true, T, 0);
        detail::check_engine_output(out, T, 0);
        const auto s0 = scorer.score(out);
        const double obs = value(s0, metric.custom()
            ? detail::custom_metric(host, MetricOf::Columns, custom, 1) : nullptr, 0);
        observe(0, s0, obs);
        std::size_t min_shift = 0;
        if (shift) {
            const std::int64_t m = opt.min_shift
                ? *opt.min_shift : static_cast<std::int64_t>(default_min_shift(target.data, T));
            if (m < 1) throw std::invalid_argument("min_shift must be at least 1");
            res.min_shift = m;
            min_shift = static_cast<std::size_t>(m);
        }
        detail::RunBatch batch(host, 1, true);
        NullTally tally;
        const std::size_t chunk = detail::chunk_runs(n, T, width, 1);
        for (std::size_t col0 = 0; col0 < n; col0 += chunk) {
            const std::size_t k = std::min(chunk, n - col0);
            double* buf = host.target_buffer(T * k);
            if (shift) circular_shift(target.data, T, min_shift, seed, col0, k, buf);
            else segment_shuffle(target.data, T, seed, col0, k, buf);
            const EngineOutput o = host.run(false, T, k);
            detail::check_engine_output(o, T, k);
            const double* cv = metric.custom()
                ? detail::custom_metric(host, MetricOf::Columns, custom, k) : nullptr;
            for (std::size_t j = 0; j < k; ++j) {
                const auto s = scorer.score_column(o.data, T, k, j);
                const double v = value(s, cv, j);
                batch.put(0, s, v);
                batch.end_run();
                tally.add(v, obs, metric.larger);
            }
        }
        batch.flush();
        host.reset();
        res.null_count = batch.total();
        res.p_value.assign(1, tally.p(obs));
        return res;
    }

    // walk_forward: pick the best of K candidate targets on each training
    // window and trade it on the test window that follows.
    //
    // The folds start at t0 = train, train + test, ... while t0 < T. Fold i
    // tests on [t0, min(t0 + test, T)) and trains on [t0 - train, t0), or on
    // [0, t0) when anchored. Every candidate runs through the engine once
    // over the whole history; each training window is scored as a backtest of
    // its own (detail::window_rows). NaN scores lose, ties go to the lowest
    // column, and a fold where every score is NaN stays flat with choice -1.
    // The stitched target (0 before the first test window) runs through the
    // engine once, into the host's target_buffer, and the statistics cover
    // it from the first test window on.
    //
    // `targets` is (T, K); a binding passes a (T,) target as (T, 1).
    inline WalkForwardResult walk_forward(EngineHost& host, const ArrayShape& targets,
                                          const std::vector<ArrayShape>& prices,
                                          const WalkForwardOptions& opt) {
        if (opt.portfolio)
            throw std::invalid_argument("walk_forward needs a single-asset engine; portfolio "
                                        "engines are not supported");
        const Metric metric = resolve_metric(opt.metric, opt.custom, opt.larger_is_better);
        if (targets.ndim != 2) throw std::invalid_argument("targets must be a (T,) or (T, K) array");
        const std::size_t T = targets.T, K = targets.K;
        if (K == 0) throw std::invalid_argument("targets must hold at least one candidate");
        for (const ArrayShape& p : prices)
            if (p.ndim != 1) throw std::invalid_argument("price input must be a 1-D series");
        for (const ArrayShape& p : prices)
            if (p.T != T)
                throw std::invalid_argument("targets and price inputs must have the same length");
        const auto integer = [](double x) { return std::isfinite(x) && x == std::floor(x); };
        if (!integer(opt.train) || opt.train < 2)
            throw std::invalid_argument("train must be an integer of at least 2");
        if (!integer(opt.test) || opt.test < 1)
            throw std::invalid_argument("test must be an integer of at least 1");
        if (opt.train >= static_cast<double>(T))
            throw std::invalid_argument("train must be shorter than the series");
        const std::size_t train = static_cast<std::size_t>(opt.train);
        const std::size_t test = opt.test >= static_cast<double>(T) ? T : static_cast<std::size_t>(opt.test);

        WalkForwardResult res;
        res.K = K;
        res.folds = (T - train + test - 1) / test;
        const std::size_t F = res.folds;
        res.fold_start.resize(F);
        res.choice.resize(F);
        res.scores.reset(new double[F * K ? F * K : 1]);
        for (std::size_t i = 0; i < F; ++i) res.fold_start[i] = static_cast<std::int64_t>(train + i * test);

        const EngineOutput full = host.run(true, T, K);
        detail::check_engine_output(full, T, K);
        RunScorer scorer(opt.periods_per_year);
        std::vector<double> rows, custom;
        double* target = host.target_buffer(T);
        std::fill(target, target + T, 0.0);
        for (std::size_t i = 0; i < F; ++i) {
            const std::size_t t0 = train + i * test, a = opt.anchored ? 0 : t0 - train, len = t0 - a;
            double* scores = res.scores.get() + i * K;
            if (metric.custom()) {
                double* w = host.window_buffer(len * K * 4);
                for (std::size_t k = 0; k < K; ++k)
                    detail::window_rows(full.data, T, K, k, a, t0, w + k * 4, K * 4);
                const double* v = detail::custom_metric(host, MetricOf::Window, custom, K, len);
                std::copy(v, v + K, scores);
            } else {
                rows.resize(len * 4);
                for (std::size_t k = 0; k < K; ++k) {
                    detail::window_rows(full.data, T, K, k, a, t0, rows.data(), 4);
                    scores[k] = scorer.score_column(rows.data(), len, 1, 0)[metric.stat];
                }
            }
            const std::int64_t c = detail::pick(scores, K, metric.larger);
            res.choice[i] = c;
            if (c >= 0)
                for (std::size_t t = t0; t < std::min(t0 + test, T); ++t)
                    target[t] = targets.data[t * K + static_cast<std::size_t>(c)];
        }

        const EngineOutput out = host.run(false, T, 0);
        detail::check_engine_output(out, T, 0);
        host.reset();
        const std::size_t a = train;   // the first test window
        rows.resize((T - a) * 4);
        detail::window_rows(out.data, T, 1, 0, a, T, rows.data(), 4);
        res.stats = scorer.score_column(rows.data(), T - a, 1, 0);
        return res;
    }

}} // namespace screamer::validation

#endif // SCREAMER_VALIDATION_POSITION_TEST_H
