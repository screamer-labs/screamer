// WASM feasibility spike: expose screamer's REAL C++ streaming kernel to JS.
//
// We wrap screamer::detail::RollingMean (and RollingSum, unmodified headers)
// with Embind. No rolling-mean logic is written here; every number comes from
// the kernel's own append().

#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <cstddef>
#include <string>

#include "screamer/detail/rolling_mean.h"
#include "screamer/detail/rolling_sum.h"

using emscripten::val;

namespace {

#ifdef SPIKE_MULTI
// Distinct C++ types that reuse the real kernel via inherited constructors.
// Each yields a separate Embind registration (see the Q6 gotcha note below).
struct MeanB : screamer::detail::RollingMean {
    using screamer::detail::RollingMean::RollingMean;
};
struct MeanC : screamer::detail::RollingMean {
    using screamer::detail::RollingMean::RollingMean;
};
#endif

// ---- Batch path, heap-owned buffer (the nanobind owner-capsule analogue) ----
//
// Allocate `n` doubles on the WASM heap, run the kernel elementwise into them,
// and hand JS a typed_memory_view aliasing that heap region. JS must copy what
// it needs and then call freeBatch(ptr) to release it. This is the explicit
// ownership model: the buffer lives until JS frees it.
uintptr_t rollingMeanBatchInto(uintptr_t inPtr, std::size_t n, std::size_t size,
                               const std::string& policy) {
    const double* in = reinterpret_cast<const double*>(inPtr);
    double* out = static_cast<double*>(std::malloc(n * sizeof(double)));
    screamer::detail::RollingMean op(size, policy);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = op.append(in[i]);
    }
    return reinterpret_cast<uintptr_t>(out);
}

// A typed_memory_view over a heap region [ptr, ptr+n). Zero-copy: JS sees the
// same bytes the WASM heap holds. Valid only until the buffer is freed or the
// heap grows (which can detach the backing ArrayBuffer), so JS must consume it
// promptly.
val viewF64(uintptr_t ptr, std::size_t n) {
    return val(emscripten::typed_memory_view(
        n, reinterpret_cast<double*>(ptr)));
}

uintptr_t allocF64(std::size_t n) {
    return reinterpret_cast<uintptr_t>(std::malloc(n * sizeof(double)));
}

void freeBuf(uintptr_t ptr) {
    std::free(reinterpret_cast<void*>(ptr));
}

#ifdef SPIKE_MULTI
// Same idea for the second distinct kernel, to measure marginal binary cost.
uintptr_t rollingSumBatchInto(uintptr_t inPtr, std::size_t n, std::size_t size,
                              const std::string& policy) {
    const double* in = reinterpret_cast<const double*>(inPtr);
    double* out = static_cast<double*>(std::malloc(n * sizeof(double)));
    screamer::detail::RollingSum op(size, policy);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = op.append(in[i]);
    }
    return reinterpret_cast<uintptr_t>(out);
}
#endif

}  // namespace

EMSCRIPTEN_BINDINGS(screamer_spike) {
    // Persistent stateful scalar object: one RollingMean per JS instance,
    // driven one append() at a time. This is what the lazy/async generators own.
    emscripten::class_<screamer::detail::RollingMean>("RollingMean")
        .constructor<std::size_t, std::string>()
        .function("append", &screamer::detail::RollingMean::append)
        .function("reset", &screamer::detail::RollingMean::reset);

#ifdef SPIKE_MULTI
    // GOTCHA: Embind keys a class registration by C++ type_id, not by the JS
    // name. Registering the SAME C++ type twice aborts at load with
    // "Cannot register type ... twice". So each extra operator must be a
    // DISTINCT C++ type. These thin subclasses reuse RollingMean's kernel but
    // are separate types, which isolates the marginal cost of one more Embind
    // class + constructor + 2 method registrations (the glue), holding the
    // kernel body constant.
    emscripten::class_<MeanB>("RollingMeanB")
        .constructor<std::size_t, std::string>()
        .function("append", &MeanB::append)
        .function("reset", &MeanB::reset);
    emscripten::class_<MeanC>("RollingMeanC")
        .constructor<std::size_t, std::string>()
        .function("append", &MeanC::append)
        .function("reset", &MeanC::reset);

    // A truly DISTINCT kernel (its own class + append body), for a marginal
    // read that includes a new kernel body, not just glue.
    emscripten::class_<screamer::detail::RollingSum>("RollingSum")
        .constructor<std::size_t, std::string>()
        .function("append", &screamer::detail::RollingSum::append)
        .function("reset", &screamer::detail::RollingSum::reset);
    emscripten::function("rollingSumBatchInto", &rollingSumBatchInto);
#endif

    emscripten::function("rollingMeanBatchInto", &rollingMeanBatchInto);
    emscripten::function("viewF64", &viewF64);
    emscripten::function("allocF64", &allocF64);
    emscripten::function("freeBuf", &freeBuf);
}
