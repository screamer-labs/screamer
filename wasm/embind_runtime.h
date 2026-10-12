#pragma once
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <emscripten/emmalloc.h>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <optional>
#include <array>
#include <stdexcept>
#include <tuple>
#include "screamer/common/column_driver.h"
#include "screamer/common/eval_op.h"
#include "screamer/common/functor_base.h"
#include "screamer/common/row_op.h"

namespace screamer_wasm {

// One event: read n_in doubles at inPtr, write n_out doubles at outPtr.
inline void evalInto(screamer::EvalOp& op, std::uintptr_t inPtr, std::uintptr_t outPtr) {
    op.eval(reinterpret_cast<const double*>(inPtr), reinterpret_cast<double*>(outPtr));
}

// A batch of `rows` row-major events as one column: inPtr = rows * n_in()
// doubles, outPtr = rows * n_out(). Input i starts at inPtr + i and advances by
// n_in() per event.
inline void evalBatchInto(screamer::EvalOp& op, std::uintptr_t inPtr,
                          std::uintptr_t outPtr, std::size_t rows) {
    const double* in = reinterpret_cast<const double*>(inPtr);
    const std::size_t ni = op.n_in(), no = op.n_out();
    if (ni > screamer::kMaxColumnInputs)
        throw std::invalid_argument("evalBatchInto: too many inputs");
    std::array<const double*, screamer::kMaxColumnInputs> ptr{};
    std::array<std::ptrdiff_t, screamer::kMaxColumnInputs> stride{};
    for (std::size_t i = 0; i < ni; ++i) {
        ptr[i] = in + i;
        stride[i] = static_cast<std::ptrdiff_t>(ni);
    }
    op.eval_column(ptr.data(), stride.data(), rows, reinterpret_cast<double*>(outPtr),
                   static_cast<std::ptrdiff_t>(no));
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

// The same two entry points with one price per group: pricePtr holds `groups`
// doubles for one event, `rows * groups` (row-major) for a batch. The op
// supplies call_assets(row, prices, groups).
template <class Op>
inline void reducePricedInto(Op& op, std::uintptr_t inPtr, std::uintptr_t pricePtr,
                             std::uintptr_t outPtr, std::size_t groups) {
    writeResult(reinterpret_cast<double*>(outPtr),
                op.call_assets(reinterpret_cast<const double*>(inPtr),
                               reinterpret_cast<const double*>(pricePtr), groups));
}

template <class Op>
inline void reducePricedBatchInto(Op& op, std::uintptr_t inPtr, std::uintptr_t pricePtr,
                                  std::uintptr_t outPtr, std::size_t rows, std::size_t groups) {
    const double* in = reinterpret_cast<const double*>(inPtr);
    const double* prices = reinterpret_cast<const double*>(pricePtr);
    double* out = reinterpret_cast<double*>(outPtr);
    const std::size_t ni = op.n_in(), no = op.n_out();
    op.reset();
    for (std::size_t r = 0; r < rows; ++r)
        writeResult(out + r * no,
                    op.call_assets(in + r * groups * ni, prices + r * groups, groups));
    op.reset();
}

// ---- Row ops (cross-sectional operators, portfolio engines) ---------------
//
// A RowOp event is one row of k assets: fieldsIn() * k values in (field-major),
// k * fieldsOut() values out (asset-major); k is a data dimension, so it travels
// per call. mapRowsInto runs `rows` such events laid end to end and brackets
// the run with reset(), like a batch call in Python.

template <class Op>
inline void mapRowInto(Op& op, std::uintptr_t inPtr, std::uintptr_t outPtr, std::size_t k) {
    op.call_row(reinterpret_cast<const double*>(inPtr), reinterpret_cast<double*>(outPtr), k);
}

// The field counts as methods of the concrete class: embind does not attach a
// base-class member pointer to a derived registration.
template <class Op> inline std::size_t fieldsIn(Op& op) { return op.fields_in(); }
template <class Op> inline std::size_t fieldsOut(Op& op) { return op.fields_out(); }

template <class Op>
inline void mapRowsInto(Op& op, std::uintptr_t inPtr, std::uintptr_t outPtr,
                        std::size_t rows, std::size_t k) {
    const double* in = reinterpret_cast<const double*>(inPtr);
    double* out = reinterpret_cast<double*>(outPtr);
    const std::size_t ni = op.fields_in() * k, no = op.fields_out() * k;
    op.reset();
    for (std::size_t r = 0; r < rows; ++r) op.call_row(in + r * ni, out + r * no, k);
    op.reset();
}

inline std::uintptr_t allocF64(std::size_t n) {
    return reinterpret_cast<std::uintptr_t>(std::malloc(n * sizeof(double)));
}
inline void freeBuf(std::uintptr_t p) { std::free(reinterpret_cast<void*>(p)); }

// Bytes the allocator has handed out and not taken back, for leak tests: the
// heap it manages less the part of it that is free.
inline double heapInUse() {
    return static_cast<double>(emmalloc_dynamic_heap_size() - emmalloc_free_dynamic_memory());
}
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
    emscripten::function("heapInUse", &screamer_wasm::heapInUse);             \
    emscripten::function("viewF64", &screamer_wasm::viewF64);                 \
    emscripten::function("opPtr", &screamer_wasm::opPtr,                       \
                         emscripten::allow_raw_pointers())
