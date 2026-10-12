// Embind bindings for the batch-only surrogate generators used by the
// randomization tests (js/src/validation.ts). A separate translation unit from
// the generated point-op bindings. Buffers are WASM heap addresses from
// allocF64; the output is a row-major (T x ncols) block. Seeds travel as
// doubles (integers below 2^53) so no BigInt support is needed.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include "screamer/validation/surrogates.h"
#include "screamer/validation/position_test.h"
#include "screamer/validation/sharpe.h"
#include "screamer/validation/strategy_test.h"

namespace {

using screamer::validation::circular_shift;
using screamer::validation::default_min_shift;
using screamer::validation::permute_bars;
using screamer::validation::permute_returns;
using screamer::validation::segment_shuffle;

// A JS number as an int64: integers outside the range are clamped, so they
// fail the driver's range check with its message (a cast would be undefined).
std::int64_t to_int64(double x) {
    if (!(x == x)) return 0;
    if (x >= 9223372036854775808.0) return std::numeric_limits<std::int64_t>::max();
    if (x < -9223372036854775808.0) return std::numeric_limits<std::int64_t>::min();
    return static_cast<std::int64_t>(x);
}

// A seed for screamer::validation::check_seed: an integer in [0, 2^53)
// stays as it is, anything else becomes -1, which the check rejects.
std::int64_t seed_arg(double x) {
    constexpr double limit = 9007199254740992.0;   // 2^53
    return x >= 0.0 && x < limit && x == std::floor(x) ? static_cast<std::int64_t>(x) : -1;
}

std::uint64_t to_seed(double x) { return screamer::validation::check_seed(seed_arg(x)); }

const double* in_ptr(std::uintptr_t p) { return reinterpret_cast<const double*>(p); }
double* out_ptr(std::uintptr_t p) { return reinterpret_cast<double*>(p); }

// start < 0: use the series' own NaN warmup; otherwise a common warmup length.
void circularShiftInto(std::uintptr_t x, std::size_t T, double min_shift, double seed,
                       std::size_t col0, std::size_t ncols, std::uintptr_t out,
                       double start) {
    const std::size_t m = min_shift < 0 ? default_min_shift(in_ptr(x), T)
                                        : static_cast<std::size_t>(min_shift);
    circular_shift(in_ptr(x), T, m, to_seed(seed), col0, ncols,
                   out_ptr(out),
                   start < 0 ? static_cast<std::size_t>(-1) : static_cast<std::size_t>(start));
}

void segmentShuffleInto(std::uintptr_t x, std::size_t T, double seed,
                        std::size_t col0, std::size_t ncols, std::uintptr_t out) {
    segment_shuffle(in_ptr(x), T, to_seed(seed), col0, ncols, out_ptr(out));
}

double defaultMinShift(std::uintptr_t x, std::size_t T) {
    return static_cast<double>(default_min_shift(in_ptr(x), T));
}

// `mode` arrives as a val, not a std::string: Embind frees a std::string
// argument only on a normal return, so a rejected mode would leak its buffer.
//
// Inputs are (T x K) blocks of K assets (K = 1: one series). Outputs: open,
// high, low, close, each a (T x ncols x K) block, source (T x ncols), and with
// a volume block (vol != 0) the moved volume in ov (T x ncols x K). Each copy
// applies one permutation to every asset.
void permuteBarsInto(std::uintptr_t o, std::uintptr_t h, std::uintptr_t l, std::uintptr_t c,
                     std::size_t T, std::size_t K, emscripten::val mode, double seed, std::size_t col0,
                     std::size_t ncols, std::uintptr_t oo, std::uintptr_t oh, std::uintptr_t ol,
                     std::uintptr_t oc, std::uintptr_t src,
                     std::uintptr_t vol, bool notional, std::uintptr_t ov) {
    permute_bars(in_ptr(o), in_ptr(h), in_ptr(l), in_ptr(c), T, K,
                 screamer::validation::parse_permute_mode(mode.as<std::string>()),
                 to_seed(seed), col0, ncols,
                 out_ptr(oo), out_ptr(oh), out_ptr(ol), out_ptr(oc), out_ptr(src),
                 vol ? in_ptr(vol) : nullptr, notional, vol ? out_ptr(ov) : nullptr);
}

void permuteReturnsInto(std::uintptr_t x, std::size_t T, std::size_t K, emscripten::val mode, double seed,
                        std::size_t col0, std::size_t ncols,
                        std::uintptr_t out, std::uintptr_t src,
                        std::uintptr_t vol, bool notional, std::uintptr_t ov) {
    permute_returns(in_ptr(x), T, K, screamer::validation::parse_permute_mode(mode.as<std::string>()),
                    to_seed(seed), col0, ncols, out_ptr(out), out_ptr(src),
                    vol ? in_ptr(vol) : nullptr, notional, vol ? out_ptr(ov) : nullptr);
}

double probabilisticSharpe(std::uintptr_t pnl, std::size_t n, double benchmark) {
    return screamer::validation::probabilistic_sharpe(in_ptr(pnl), n, benchmark);
}

// Writes [probability, benchmark] to out.
void deflatedSharpeInto(std::uintptr_t pnl, std::size_t n, std::uintptr_t trials,
                        std::size_t n_trials, std::uintptr_t out) {
    double benchmark = 0.0;
    const double p = screamer::validation::deflated_sharpe(
        in_ptr(pnl), n, in_ptr(trials), n_trials, benchmark);
    out_ptr(out)[0] = p;
    out_ptr(out)[1] = benchmark;
}

// The randomization tests call JavaScript back: the strategy of strategyTest,
// and the engine and a custom metric of positionTest and walkForward
// (js/src/validation.ts). `cb` is an object of functions built by
// wasmCallbacks (js/src/errors.ts), which runs each one in callbackScope and
// never throws: it returns 0, or 1 when the user's code threw. The JS side
// keeps that error, JsCallbacks throws CallbackFailed to unwind the driver,
// and the JS wrapper rethrows the user's error unchanged. JsCallbacks::call
// is the only place C++ calls into JavaScript (js/test/validation.test.ts
// checks this).
namespace sv = screamer::validation;

struct CallbackFailed : std::exception {
    const char* what() const noexcept override { return "callback failed"; }
};

class JsCallbacks {
public:
    explicit JsCallbacks(emscripten::val cb) : cb_(std::move(cb)) {}

    template <class... Args>
    void call(const char* name, Args... args) {
        if (cb_.call<int>(name, args...) != 0) throw CallbackFailed();
    }

    // The output of the last run: `run` wrote its [ndim, d0, d1, d2] to hdr,
    // and `fetch(outPtr, count)` copies its values into `out`.
    sv::EngineOutput fetch(const double* hdr, std::vector<double>& out) {
        sv::EngineOutput o;
        o.ndim = static_cast<std::size_t>(hdr[0]);
        std::size_t count = o.ndim == 2 || o.ndim == 3 ? 1 : 0;
        for (std::size_t d = 0; d < 3; ++d) {
            o.shape[d] = static_cast<std::size_t>(hdr[1 + d]);
            if (d < o.ndim) count *= o.shape[d];
        }
        out.resize(count ? count : 1);
        if (count) call("fetch", addr(out.data()), count);
        o.data = out.data();
        return o;
    }

    // One value of a custom metric: the JS side writes [value, how many values
    // the metric gave] to hdr; the driver's rule is one per strategy.
    double metric_value(double* hdr) {
        sv::check_metric_values(1, static_cast<std::size_t>(hdr[1]));
        return hdr[0];
    }

    // sv::ResultSink: the batches are copied into JS arrays as they come, so
    // the WASM heap holds nothing sized by n.
    void begin(std::size_t n, std::size_t K) { call("begin", n, K); }

    void emit(std::size_t j0, std::size_t runs, const double* stats, std::size_t stride,
              const double* metrics, std::size_t count) {
        call("emit", j0, runs, addr(stats), stride, addr(metrics), count);
    }

    static std::uintptr_t addr(const void* p) { return reinterpret_cast<std::uintptr_t>(p); }

private:
    emscripten::val cb_;
};

// strategy_test: the driver is screamer::validation::strategy_test; `cb` has
//   run(argsPtr, hdrPtr): calls the strategy. argsPtr holds k heap addresses,
//     0 for an original input, else the moved copy (T * K doubles). Writes the
//     output's [ndim, d0, d1, d2] to hdrPtr.
//   fetch(outPtr, count): copies the last output's count values to outPtr.
//   metric(hdrPtr): writes [value, values given] of the custom metric of the
//     last output to hdrPtr.
//   begin(n, K) and emit(j0, runs, statsPtr, stride, metricsPtr, count): the
//     results (sv::ResultSink), copied into JS arrays as they come.
//   done(headPtr, nullCount): head = [observed, p_value, observed stats].
class JsStrategyHost final : public sv::StrategyHost {
public:
    JsStrategyHost(emscripten::val cb, std::size_t k)
        : js_(std::move(cb)), moved_(k), args_(k, 0.0) {}

    double* moved_input(std::size_t i, std::size_t size) override {
        moved_[i].resize(size ? size : 1);   // reused: JS copies it into a new array
        return moved_[i].data();
    }

    sv::EngineOutput run(const std::vector<unsigned char>& moved) override {
        for (std::size_t i = 0; i < args_.size(); ++i)
            args_[i] = moved[i] ? static_cast<double>(JsCallbacks::addr(moved_[i].data())) : 0.0;
        js_.call("run", JsCallbacks::addr(args_.data()), JsCallbacks::addr(hdr_));
        return js_.fetch(hdr_, out_);
    }

    std::size_t metric(double* values, std::size_t count) override {
        js_.call("metric", JsCallbacks::addr(hdr_));
        if (count) values[0] = js_.metric_value(hdr_);
        return count;
    }

    void begin(std::size_t n, std::size_t K) override { js_.begin(n, K); }

    void emit(std::size_t j0, std::size_t runs, const double* stats, std::size_t stride,
              const double* metrics, std::size_t count) override {
        js_.emit(j0, runs, stats, stride, metrics, count);
    }

    void done(const sv::StrategyTestResult& r) {
        double head[2 + sv::kNumStats] = {r.observed, r.p_value};
        std::copy(r.observed_stats.begin(), r.observed_stats.end(), head + 2);
        js_.call("done", JsCallbacks::addr(head), r.null_count);
    }

private:
    JsCallbacks js_;
    std::vector<std::vector<double>> moved_;
    std::vector<double> args_, out_;
    double hdr_[4] = {0, 0, 0, 0};
};

// position_test and walk_forward: the drivers are
// screamer::validation::position_test and walk_forward; `cb` has
//   run(original, T, cols, targetPtr, hdrPtr): resets the engine and runs it
//     on the caller's target (original) or on the (T,) or (T, cols) target at
//     targetPtr, with the market inputs. Writes [ndim, d0, d1, d2] to hdrPtr.
//   fetch(outPtr, count), as for strategy_test.
//   reset(): resets the engine.
//   metric(what, j, count, rows, windowPtr, hdrPtr): the custom metric of
//     strategy j of `count` in `what` (0: the last output, 1: the last output
//     as one book, 2: the (rows, count, 4) window at windowPtr), as [value,
//     values given] to hdrPtr.
//   begin and emit, as for strategy_test (position_test only).
//   done(...): the rest of the result (positionTestRun, walkForwardRun).
class JsEngineHost final : public sv::PositionHost {
public:
    explicit JsEngineHost(emscripten::val cb) : js_(std::move(cb)) {}

    double* target_buffer(std::size_t size) override {
        target_.resize(size ? size : 1);   // reused: JS copies it into a new array
        return target_.data();
    }

    sv::EngineOutput run(bool original, std::size_t T, std::size_t cols) override {
        std::vector<double>().swap(out_);  // the last output is not needed during the run
        js_.call("run", original, T, cols, JsCallbacks::addr(target_.data()),
                 JsCallbacks::addr(hdr_));
        return js_.fetch(hdr_, out_);
    }

    void reset() override { js_.call("reset"); }

    double* window_buffer(std::size_t size) override {
        window_.resize(size ? size : 1);
        return window_.data();
    }

    std::size_t metric(sv::MetricOf what, double* values, std::size_t count,
                       std::size_t rows) override {
        // One call per strategy: the JS metric takes one strategy's output.
        for (std::size_t j = 0; j < count; ++j) {
            js_.call("metric", static_cast<int>(what), j, count, rows,
                     JsCallbacks::addr(window_.data()), JsCallbacks::addr(hdr_));
            values[j] = js_.metric_value(hdr_);
        }
        return count;
    }

    void begin(std::size_t n, std::size_t K) override { js_.begin(n, K); }

    void emit(std::size_t j0, std::size_t runs, const double* stats, std::size_t stride,
              const double* metrics, std::size_t count) override {
        js_.emit(j0, runs, stats, stride, metrics, count);
    }

    JsCallbacks& js() { return js_; }

private:
    JsCallbacks js_;
    std::vector<double> target_, window_, out_;
    double hdr_[4] = {0, 0, 0, 0};
};

std::vector<std::int64_t> positions(const emscripten::val& v) {
    std::vector<std::int64_t> out;
    for (double x : emscripten::vecFromJSArray<double>(v)) out.push_back(to_int64(x));
    return out;
}

// `spec` holds [data address, T, K, panel] per input. The result goes to
// cb.begin and cb.emit as it is computed, and the rest to cb.done; JS
// allocates nothing sized by n before the driver has checked it. Positions
// and counts arrive as integers (the JS side maps a non-integer to -1 or 0,
// which the driver rejects); strings as vals, as in permuteBarsInto.
void strategyTestRun(emscripten::val cb, std::uintptr_t spec, std::size_t k,
                     emscripten::val method, double n, emscripten::val ohlc, double price,
                     emscripten::val mode, bool has_volume, double volume,
                     emscripten::val volume_mode, emscripten::val columns,
                     emscripten::val fixed, emscripten::val metric, bool custom,
                     bool has_larger, bool larger_is_better,
                     bool has_min_shift, double min_shift, double seed,
                     double periods_per_year) {
    const double* sp = in_ptr(spec);
    std::vector<sv::StrategyInput> data(k);
    for (std::size_t i = 0; i < k; ++i)
        data[i] = {in_ptr(static_cast<std::uintptr_t>(sp[4 * i])),
                   static_cast<std::size_t>(sp[4 * i + 1]), static_cast<std::size_t>(sp[4 * i + 2]),
                   sp[4 * i + 3] != 0.0};
    sv::StrategyTestOptions opt;
    opt.n = to_int64(n);
    opt.method = method.as<std::string>();
    opt.ohlc = positions(ohlc);
    opt.price = to_int64(price);
    opt.mode = mode.as<std::string>();
    if (has_volume) opt.volume = to_int64(volume);
    opt.volume_mode = volume_mode.as<std::string>();
    opt.columns = positions(columns);
    opt.fixed = positions(fixed);
    opt.metric = metric.as<std::string>();
    opt.custom = custom;
    if (has_larger) opt.larger_is_better = larger_is_better;
    if (has_min_shift) opt.min_shift = to_int64(min_shift);
    opt.seed = seed_arg(seed);
    opt.periods_per_year = periods_per_year;

    JsStrategyHost host(std::move(cb), k);
    const sv::StrategyTestResult r = sv::strategy_test(host, data, opt);
    host.done(r);
}

// An array spec at `spec`: [data address, ndim, T, K] per array.
std::vector<sv::ArrayShape> array_shapes(std::uintptr_t spec, std::size_t count) {
    const double* sp = in_ptr(spec);
    std::vector<sv::ArrayShape> out(count);
    for (std::size_t i = 0; i < count; ++i)
        out[i] = {in_ptr(static_cast<std::uintptr_t>(sp[4 * i])), static_cast<std::size_t>(sp[4 * i + 1]),
                  static_cast<std::size_t>(sp[4 * i + 2]), static_cast<std::size_t>(sp[4 * i + 3])};
    return out;
}

// position_test: the target and the market inputs are array specs (the
// market's data is not read). The result goes to cb.begin and cb.emit as it
// is computed, and the rest to cb.done(headPtr, K, family, minShift,
// nullCount): head = [observed (K), p_value (K), family_p_value (K),
// observed stats (kNumStats x K)], minShift -1 for "segments".
void positionTestRun(emscripten::val cb, std::uintptr_t target, std::uintptr_t market,
                     std::size_t n_market, double n, emscripten::val method, emscripten::val metric,
                     bool custom, bool has_larger, bool larger_is_better, bool has_min_shift,
                     double min_shift, double seed, double periods_per_year, bool portfolio) {
    sv::PositionTestOptions opt;
    opt.n = to_int64(n);
    opt.method = method.as<std::string>();
    opt.metric = metric.as<std::string>();
    opt.custom = custom;
    if (has_larger) opt.larger_is_better = larger_is_better;
    if (has_min_shift) opt.min_shift = to_int64(min_shift);
    opt.seed = seed_arg(seed);
    opt.periods_per_year = periods_per_year;
    opt.portfolio = portfolio;
    JsEngineHost host(std::move(cb));
    const sv::PositionTestResult r =
        sv::position_test(host, array_shapes(target, 1)[0], array_shapes(market, n_market), opt);
    const std::size_t K = r.K;
    std::vector<double> head(K * (3 + sv::kNumStats), std::numeric_limits<double>::quiet_NaN());
    std::copy(r.observed.begin(), r.observed.end(), head.begin());
    std::copy(r.p_value.begin(), r.p_value.end(), head.begin() + K);
    std::copy(r.family_p_value.begin(), r.family_p_value.end(), head.begin() + 2 * K);
    std::copy(r.observed_stats.begin(), r.observed_stats.end(), head.begin() + 3 * K);
    host.js().call("done", JsCallbacks::addr(head.data()), K, r.family,
                   r.min_shift ? static_cast<double>(*r.min_shift) : -1.0, r.null_count);
}

// walk_forward: `targets` is a (T, K) array spec (a (T,) target as (T, 1)),
// `prices` array specs. The result goes to cb.done(statsPtr, foldPtr,
// choicePtr, scoresPtr, folds, K): the statistics (kNumStats), the fold
// starts and choices (folds each, as doubles), and the (folds, K) scores.
// The stitched target and the output stay on the JS side (the last run).
void walkForwardRun(emscripten::val cb, std::uintptr_t targets, std::uintptr_t prices,
                    std::size_t n_prices, double train, double test, emscripten::val metric,
                    bool custom, bool has_larger, bool larger_is_better, bool anchored,
                    double periods_per_year, bool portfolio) {
    sv::WalkForwardOptions opt;
    opt.train = train;
    opt.test = test;
    opt.metric = metric.as<std::string>();
    opt.custom = custom;
    if (has_larger) opt.larger_is_better = larger_is_better;
    opt.anchored = anchored;
    opt.periods_per_year = periods_per_year;
    opt.portfolio = portfolio;
    JsEngineHost host(std::move(cb));
    const sv::WalkForwardResult r =
        sv::walk_forward(host, array_shapes(targets, 1)[0], array_shapes(prices, n_prices), opt);
    const std::vector<double> folds(r.fold_start.begin(), r.fold_start.end()),
        choice(r.choice.begin(), r.choice.end());
    host.js().call("done", JsCallbacks::addr(r.stats.data()), JsCallbacks::addr(folds.data()),
                   JsCallbacks::addr(choice.data()), JsCallbacks::addr(r.scores.get()), r.folds, r.K);
}

// The largest n strategyTestRun accepts (sv::max_runs).
double strategyTestMaxRuns() { return static_cast<double>(sv::max_strategy_runs()); }

// The driver's statistic table, as [name, source, column, larger] rows
// (larger: 1, 0, or -1 for no natural direction), for js/src/validation.ts.
emscripten::val strategyStats() {
    emscripten::val rows = emscripten::val::array();
    for (std::size_t q = 0; q < sv::kNumStats; ++q) {
        const auto& st = sv::kStats[q];
        emscripten::val row = emscripten::val::array();
        row.set(0, std::string(st.name));
        row.set(1, std::string(sv::stat_source_name(st.source)));
        row.set(2, st.column);
        row.set(3, st.larger);
        rows.set(q, row);
    }
    return rows;
}

}  // namespace

EMSCRIPTEN_BINDINGS(screamer_validation) {
    emscripten::function("circularShiftInto", &circularShiftInto);
    emscripten::function("segmentShuffleInto", &segmentShuffleInto);
    emscripten::function("defaultMinShift", &defaultMinShift);
    emscripten::function("permuteBarsInto", &permuteBarsInto);
    emscripten::function("permuteReturnsInto", &permuteReturnsInto);
    emscripten::function("probabilisticSharpe", &probabilisticSharpe);
    emscripten::function("deflatedSharpeInto", &deflatedSharpeInto);
    emscripten::function("strategyTestRun", &strategyTestRun);
    emscripten::function("positionTestRun", &positionTestRun);
    emscripten::function("walkForwardRun", &walkForwardRun);
    emscripten::function("strategyStats", &strategyStats);
    emscripten::function("strategyTestMaxRuns", &strategyTestMaxRuns);
}
