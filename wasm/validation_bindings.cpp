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
#include "screamer/validation/sharpe.h"
#include "screamer/validation/strategy_test.h"

namespace {

using screamer::validation::circular_shift;
using screamer::validation::default_min_shift;
using screamer::validation::permute_bars;
using screamer::validation::permute_returns;
using screamer::validation::segment_shuffle;

const double* in_ptr(std::uintptr_t p) { return reinterpret_cast<const double*>(p); }
double* out_ptr(std::uintptr_t p) { return reinterpret_cast<double*>(p); }

// start < 0: use the series' own NaN warmup; otherwise a common warmup length.
void circularShiftInto(std::uintptr_t x, std::size_t T, double min_shift, double seed,
                       std::size_t col0, std::size_t ncols, std::uintptr_t out,
                       double start) {
    const std::size_t m = min_shift < 0 ? default_min_shift(in_ptr(x), T)
                                        : static_cast<std::size_t>(min_shift);
    circular_shift(in_ptr(x), T, m, static_cast<std::uint64_t>(seed), col0, ncols,
                   out_ptr(out),
                   start < 0 ? static_cast<std::size_t>(-1) : static_cast<std::size_t>(start));
}

void segmentShuffleInto(std::uintptr_t x, std::size_t T, double seed,
                        std::size_t col0, std::size_t ncols, std::uintptr_t out) {
    segment_shuffle(in_ptr(x), T, static_cast<std::uint64_t>(seed), col0, ncols,
                    out_ptr(out));
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
                 static_cast<std::uint64_t>(seed), col0, ncols,
                 out_ptr(oo), out_ptr(oh), out_ptr(ol), out_ptr(oc), out_ptr(src),
                 vol ? in_ptr(vol) : nullptr, notional, vol ? out_ptr(ov) : nullptr);
}

void permuteReturnsInto(std::uintptr_t x, std::size_t T, std::size_t K, emscripten::val mode, double seed,
                        std::size_t col0, std::size_t ncols,
                        std::uintptr_t out, std::uintptr_t src,
                        std::uintptr_t vol, bool notional, std::uintptr_t ov) {
    permute_returns(in_ptr(x), T, K, screamer::validation::parse_permute_mode(mode.as<std::string>()),
                    static_cast<std::uint64_t>(seed), col0, ncols, out_ptr(out), out_ptr(src),
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

// strategy_test (js/src/validation.ts strategyTest): the driver is
// screamer::validation::strategy_test; this host reaches the JavaScript
// strategy through `cb`, an object of four functions built by wasmCallbacks
// (js/src/errors.ts), which runs each one in callbackScope and never throws:
//   run(argsPtr, hdrPtr): calls the strategy. argsPtr holds k heap addresses,
//     0 for an original input, else the moved copy (T * K doubles). Writes the
//     output's [ndim, d0, d1, d2] to hdrPtr.
//   fetch(outPtr, count): copies the last output's count values to outPtr.
//   metric(hdrPtr): writes the custom metric of the last output to hdrPtr[0].
//   done(headPtr, nullPtr, nullCount, statsPtr, n): copies the result out.
// Each returns 0, or 1 when the user's code threw; the JS side keeps that
// error, this host throws CallbackFailed to unwind the driver, and the JS
// wrapper rethrows the user's error unchanged. call_js is the only place C++
// calls into JavaScript (js/test/validation.test.ts checks this).
namespace sv = screamer::validation;

struct CallbackFailed : std::exception {
    const char* what() const noexcept override { return "strategyTest callback failed"; }
};

class JsStrategyHost final : public sv::StrategyHost {
public:
    JsStrategyHost(emscripten::val cb, std::size_t k)
        : cb_(std::move(cb)), moved_(k), args_(k, 0.0) {}

    double* moved_input(std::size_t i, std::size_t size) override {
        moved_[i].resize(size ? size : 1);   // reused: JS copies it into a new array
        return moved_[i].data();
    }

    sv::EngineOutput run(const std::vector<unsigned char>& moved) override {
        for (std::size_t i = 0; i < args_.size(); ++i)
            args_[i] = moved[i] ? static_cast<double>(reinterpret_cast<std::uintptr_t>(moved_[i].data()))
                                : 0.0;
        call_js("run", reinterpret_cast<std::uintptr_t>(args_.data()),
                reinterpret_cast<std::uintptr_t>(hdr_));
        sv::EngineOutput out;
        out.ndim = static_cast<std::size_t>(hdr_[0]);
        std::size_t count = out.ndim == 2 || out.ndim == 3 ? 1 : 0;
        for (std::size_t d = 0; d < 3; ++d) {
            out.shape[d] = static_cast<std::size_t>(hdr_[1 + d]);
            if (d < out.ndim) count *= out.shape[d];
        }
        out_.resize(count ? count : 1);
        if (count) call_js("fetch", reinterpret_cast<std::uintptr_t>(out_.data()), count);
        out.data = out_.data();
        return out;
    }

    double metric() override {
        call_js("metric", reinterpret_cast<std::uintptr_t>(hdr_));
        return hdr_[0];
    }

    double* null_stats(std::size_t q, std::size_t n) override {
        if (!stats_) stats_.reset(new double[sv::kNumStats * n ? sv::kNumStats * n : 1]);
        return stats_.get() + q * n;
    }

    // Hands the result to JS: head = [observed, p_value, observed stats
    // (kNumStats)], the finite null metrics, and the (kNumStats x n)
    // statistics the driver wrote.
    void done(const sv::StrategyTestResult& r, std::size_t n) {
        double head[2 + sv::kNumStats] = {r.observed, r.p_value};
        std::copy(r.observed_stats.begin(), r.observed_stats.end(), head + 2);
        call_js("done", reinterpret_cast<std::uintptr_t>(head),
                reinterpret_cast<std::uintptr_t>(r.null.data()), r.null.size(),
                reinterpret_cast<std::uintptr_t>(stats_.get()), n);
    }

private:
    template <class... Args>
    void call_js(const char* name, Args... args) {
        if (cb_.call<int>(name, args...) != 0) throw CallbackFailed();
    }

    emscripten::val cb_;
    std::vector<std::vector<double>> moved_;
    std::vector<double> args_, out_;
    double hdr_[4] = {0, 0, 0, 0};
    std::unique_ptr<double[]> stats_;   // one block, row q at q * n
};

// A JS number as an int64: integers outside the range are clamped, so they
// fail the driver's range check with its message (a cast would be undefined).
std::int64_t to_int64(double x) {
    if (!(x == x)) return 0;
    if (x >= 9223372036854775808.0) return std::numeric_limits<std::int64_t>::max();
    if (x < -9223372036854775808.0) return std::numeric_limits<std::int64_t>::min();
    return static_cast<std::int64_t>(x);
}

// A seed: an integer in [0, 2^53) (validation.ts checkSeed). Anything else
// maps to 2^53, which the driver rejects with its message.
std::uint64_t to_seed(double x) {
    constexpr double limit = 9007199254740992.0;   // 2^53
    if (!(x >= 0.0 && x < limit) || x != std::floor(x)) return std::uint64_t(1) << 53;
    return static_cast<std::uint64_t>(x);
}

std::vector<std::int64_t> positions(const emscripten::val& v) {
    std::vector<std::int64_t> out;
    for (double x : emscripten::vecFromJSArray<double>(v)) out.push_back(to_int64(x));
    return out;
}

// `spec` holds [data address, T, K, panel] per input. The result goes to
// cb.done (see JsStrategyHost::done); the statistics are written straight
// into the host's block, so the WASM heap holds the result once and JS
// allocates nothing sized by n before the driver has checked it. Positions
// and counts arrive as integers (the JS side maps a non-integer to -1 or 0,
// which the driver rejects); strings as vals, as in permuteBarsInto.
void strategyTestRun(emscripten::val cb, std::uintptr_t spec, std::size_t k,
                     emscripten::val method, double n, emscripten::val ohlc, double price,
                     emscripten::val mode, bool has_volume, double volume,
                     emscripten::val volume_mode, emscripten::val columns,
                     emscripten::val fixed, emscripten::val metric, bool larger_is_better,
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
    opt.larger_is_better = larger_is_better;
    if (has_min_shift) opt.min_shift = to_int64(min_shift);
    opt.seed = to_seed(seed);
    opt.periods_per_year = periods_per_year;

    JsStrategyHost host(std::move(cb), k);
    const sv::StrategyTestResult r = sv::strategy_test(host, data, opt);
    host.done(r, static_cast<std::size_t>(opt.n));
}

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
    emscripten::function("strategyStats", &strategyStats);
}
