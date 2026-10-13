// Embind binding for split (js/src/split.ts). The rules and the
// partition are include/screamer/streams/split.h, which Python's split calls
// too; JavaScript only moves the arrays in and out. Buffers are WASM heap
// addresses from allocF64.
#include <cstddef>
#include <cstdint>
#include <vector>
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include "screamer/streams/split.h"

namespace {

// Partitions the events: `rows` rows of `width` values at `v` (`valuesNdim`
// is the rank of values, time first), their sources (float64) at `s`, and
// their index at `k` when `hasIndex`. Writes the rows grouped by output, each
// group in event order, to `outV` (rows * width doubles), and the index
// likewise to `outK` when `hasIndex`. `n` is the number of outputs when
// `hasN`. The dimensions and lengths are checked first, then the sources and
// n. Returns the number of events per output, a Float64Array.
emscripten::val splitInto(std::uintptr_t v, std::size_t valuesNdim, std::size_t rows, std::size_t width,
                          std::uintptr_t s, std::size_t sourcesNdim, std::size_t sourcesLen,
                          bool hasIndex, std::size_t indexNdim, std::uintptr_t k, std::size_t indexLen,
                          bool hasN, double n, std::uintptr_t outV, std::uintptr_t outK) {
    using screamer::streams::SplitPlan;
    screamer::streams::check_split_inputs(valuesNdim, rows, sourcesNdim, sourcesLen,
                                          hasIndex, indexNdim, indexLen);
    const SplitPlan plan(reinterpret_cast<const double*>(s), rows, hasN, n);
    const std::size_t outputs = plan.outputs();
    std::vector<double*> vptr, kptr;
    std::vector<double> counts;
    screamer::streams::split_alloc(outputs, [&] {
        vptr.resize(outputs);
        kptr.resize(hasIndex ? outputs : 0);
        counts.resize(outputs);
    });
    std::size_t at = 0;
    for (std::size_t i = 0; i < outputs; ++i) {
        vptr[i] = reinterpret_cast<double*>(outV) + at * width;
        if (hasIndex) kptr[i] = reinterpret_cast<double*>(outK) + at;
        counts[i] = static_cast<double>(plan.counts()[i]);
        at += plan.counts()[i];
    }
    plan.scatter(reinterpret_cast<const double*>(v), width,
                 hasIndex ? reinterpret_cast<const double*>(k) : nullptr, vptr.data(), kptr.data());
    // new Float64Array(view) copies the counts out of the heap.
    return emscripten::val::global("Float64Array").new_(
        emscripten::val(emscripten::typed_memory_view(outputs, counts.data())));
}

}  // namespace

EMSCRIPTEN_BINDINGS(screamer_split) {
    emscripten::function("splitInto", &splitInto);
}
