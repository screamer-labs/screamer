#pragma once
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <optional>
#include <tuple>
#include "screamer/common/eval_op.h"
#include "screamer/common/functor_base.h"

namespace screamer_wasm {

// One event: read n_in doubles at inPtr, write n_out doubles at outPtr.
inline void evalInto(screamer::EvalOp& op, std::uintptr_t inPtr, std::uintptr_t outPtr) {
    op.eval(reinterpret_cast<const double*>(inPtr), reinterpret_cast<double*>(outPtr));
}

// Batch by the batch==stream invariant: replay eval() over `rows` events.
// Layout: inPtr = rows * n_in() doubles (row-major), outPtr = rows * n_out().
inline void evalBatchInto(screamer::EvalOp& op, std::uintptr_t inPtr,
                          std::uintptr_t outPtr, std::size_t rows) {
    const double* in = reinterpret_cast<const double*>(inPtr);
    double* out = reinterpret_cast<double*>(outPtr);
    const std::size_t ni = op.n_in(), no = op.n_out();
    op.reset();
    for (std::size_t r = 0; r < rows; ++r) op.eval(in + r * ni, out + r * no);
}

// ---- Dynamic-width reducers ------------------------------------------------
//
// A reducer op keeps the fixed n_in()/n_out() event contract above, and adds a
// second entry point that folds a variable number of groups into one event: it
// reads `groups * n_in()` doubles and writes n_out(). The group count is a
// data-shape dimension (the assets in a portfolio), not a property of the op,
// so it travels as a call argument rather than living in the op's type the way
// n_in and n_out do. PortfolioReport is the one such op today.
//
// The op supplies the reduction as call_assets(row, groups).

inline void writeResult(double* dest, double value) { dest[0] = value; }

template <class... Ts>
inline void writeResult(double* dest, const std::tuple<Ts...>& result) {
    screamer::detail::write_tuple_to_memory(dest, result);
}

// One event: reduce the (groups, n_in) block at inPtr into n_out doubles.
template <class Op>
inline void reduceInto(Op& op, std::uintptr_t inPtr, std::uintptr_t outPtr,
                       std::size_t groups) {
    writeResult(reinterpret_cast<double*>(outPtr),
                op.call_assets(reinterpret_cast<const double*>(inPtr), groups));
}

// Batch by the batch==stream invariant: replay the reduction over `rows`
// events. Layout: inPtr = rows * groups * n_in() doubles (row-major),
// outPtr = rows * n_out(). Brackets the run with reset() at both ends, so a
// batch call is independent of what came before and leaves no state behind.
template <class Op>
inline void reduceBatchInto(Op& op, std::uintptr_t inPtr, std::uintptr_t outPtr,
                            std::size_t rows, std::size_t groups) {
    const double* in = reinterpret_cast<const double*>(inPtr);
    double* out = reinterpret_cast<double*>(outPtr);
    const std::size_t ni = op.n_in(), no = op.n_out();
    op.reset();
    for (std::size_t r = 0; r < rows; ++r)
        writeResult(out + r * no, op.call_assets(in + r * groups * ni, groups));
    op.reset();
}

inline std::uintptr_t allocF64(std::size_t n) {
    return reinterpret_cast<std::uintptr_t>(std::malloc(n * sizeof(double)));
}
inline void freeBuf(std::uintptr_t p) { std::free(reinterpret_cast<void*>(p)); }
inline emscripten::val viewF64(std::uintptr_t p, std::size_t n) {
    return emscripten::val(emscripten::typed_memory_view(n, reinterpret_cast<double*>(p)));
}

// A NaN sentinel maps "argument not provided" to std::nullopt for the EW ops
// whose Python ctors take std::optional<double> decay parameters.
inline std::optional<double> opt(double v) {
    return std::isnan(v) ? std::nullopt : std::optional<double>(v);
}

// Raw address of an op instance as an integer. The JS wrapper holds the C++
// EvalOp, so this hands JS the raw EvalOp* it passes to GraphBuilder.addFunctor
// (and addResample's reducer). Mirrors the uintptr_t convention of evalInto.
inline std::uintptr_t opPtr(screamer::EvalOp& op) {
    return reinterpret_cast<std::uintptr_t>(&op);
}

}  // namespace screamer_wasm

// Register the shared runtime on the EvalOp base class ONCE. Every op class is
// registered with base<EvalOp>, so all ops inherit evalInto/evalBatchInto/etc.
#define SCREAMER_REGISTER_EVAL_OP_RUNTIME()                                    \
    emscripten::class_<screamer::EvalOp>("EvalOp")                             \
        .function("evalInto", &screamer_wasm::evalInto, emscripten::allow_raw_pointers()) \
        .function("evalBatchInto", &screamer_wasm::evalBatchInto, emscripten::allow_raw_pointers()) \
        .function("reset", &screamer::EvalOp::reset)                           \
        .function("nIn", &screamer::EvalOp::n_in)                              \
        .function("nOut", &screamer::EvalOp::n_out);                           \
    emscripten::function("allocF64", &screamer_wasm::allocF64);               \
    emscripten::function("freeBuf", &screamer_wasm::freeBuf);                 \
    emscripten::function("viewF64", &screamer_wasm::viewF64);                 \
    emscripten::function("opPtr", &screamer_wasm::opPtr,                       \
                         emscripten::allow_raw_pointers())
