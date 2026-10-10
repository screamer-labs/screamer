// Embind bindings for the batch-only surrogate generators used by the
// randomization tests (js/src/validation.ts). A separate translation unit from
// the generated point-op bindings. Buffers are WASM heap addresses from
// allocF64; the output is a row-major (T x ncols) block. Seeds travel as
// doubles (integers below 2^53) so no BigInt support is needed.
#include <cstddef>
#include <cstdint>
#include <emscripten/bind.h>
#include "screamer/validation/surrogates.h"

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

// Outputs: open, high, low, close, and source, each a (T x ncols) block.
void permuteBarsInto(std::uintptr_t o, std::uintptr_t h, std::uintptr_t l, std::uintptr_t c,
                     std::size_t T, double seed, std::size_t col0, std::size_t ncols,
                     std::uintptr_t oo, std::uintptr_t oh, std::uintptr_t ol,
                     std::uintptr_t oc, std::uintptr_t src) {
    permute_bars(in_ptr(o), in_ptr(h), in_ptr(l), in_ptr(c), T,
                 static_cast<std::uint64_t>(seed), col0, ncols,
                 out_ptr(oo), out_ptr(oh), out_ptr(ol), out_ptr(oc), out_ptr(src));
}

void permuteReturnsInto(std::uintptr_t x, std::size_t T, bool log, double seed,
                        std::size_t col0, std::size_t ncols,
                        std::uintptr_t out, std::uintptr_t src) {
    permute_returns(in_ptr(x), T, log, static_cast<std::uint64_t>(seed), col0, ncols,
                    out_ptr(out), out_ptr(src));
}

}  // namespace

EMSCRIPTEN_BINDINGS(screamer_validation) {
    emscripten::function("circularShiftInto", &circularShiftInto);
    emscripten::function("segmentShuffleInto", &segmentShuffleInto);
    emscripten::function("defaultMinShift", &defaultMinShift);
    emscripten::function("permuteBarsInto", &permuteBarsInto);
    emscripten::function("permuteReturnsInto", &permuteReturnsInto);
}
