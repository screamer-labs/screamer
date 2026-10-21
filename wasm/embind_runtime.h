#pragma once
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <emscripten/emmalloc.h>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <new>
#include <memory>
#include <optional>
#include <array>
#include <stdexcept>
#include <tuple>
#include "screamer/common/column_driver.h"
#include "screamer/common/finite_settings.h"
#include "screamer/common/int_settings.h"
#include <string>
#include <vector>
#include "screamer/common/eval_op.h"
#include "screamer/common/functor_base.h"
#include "screamer/common/op_arg.h"
#include "screamer/common/row_op.h"

namespace screamer_wasm {

// One event: read n_in doubles at inPtr, write n_out doubles at outPtr.
inline void evalInto(screamer::EvalOp& op, std::uintptr_t inPtr, std::uintptr_t outPtr) {
    op.eval(reinterpret_cast<const double*>(inPtr), reinterpret_cast<double*>(outPtr));
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
// k * fieldsOut() values out (asset-major), or fieldsOut() for a row reduction;
// k is a data dimension, so it travels per call. mapRowsInto runs `rows` such events laid end to end and brackets
// the run with reset(), like a batch call in Python.

template <class Op>
inline void mapRowInto(Op& op, std::uintptr_t inPtr, std::uintptr_t outPtr, std::size_t k) {
    op.call_row(reinterpret_cast<const double*>(inPtr), reinterpret_cast<double*>(outPtr), k);
}

// The field counts as methods of the concrete class: embind does not attach a
// base-class member pointer to a derived registration.
template <class Op> inline std::size_t fieldsIn(Op& op) { return op.fields_in(); }
template <class Op> inline std::size_t fieldsOut(Op& op) { return op.fields_out(); }
// False for a row reduction (CrossStd): one event writes fieldsOut() values
// for the whole row instead of k * fieldsOut().
template <class Op> inline bool perAssetOut(Op& op) { return op.per_asset_out(); }

// `packed` reads each row as k assets of fieldsIn() values (a packed
// (T, K, F) array) instead of fieldsIn() blocks of k values (RowOp::call_rows).
template <class Op>
inline void mapRowsInto(Op& op, std::uintptr_t inPtr, std::uintptr_t outPtr,
                        std::size_t rows, std::size_t k, bool packed) {
    op.reset();
    try {
        op.call_rows(reinterpret_cast<const double*>(inPtr), reinterpret_cast<double*>(outPtr),
                     rows, k, packed);
    } catch (...) {
        op.reset();
        throw;
    }
    op.reset();
}

// n doubles of WASM heap, or std::bad_alloc (JavaScript's out-of-memory
// RangeError, errors.ts) when the heap cannot grow to them: no caller ever
// writes through a null pointer. JavaScript checks n before the call (an
// Embind size_t would wrap a number past 2**32).
inline std::uintptr_t allocF64(std::size_t n) {
    if (n > SIZE_MAX / sizeof(double)) throw std::bad_alloc();
    void* p = std::malloc(n * sizeof(double));
    if (!p && n) throw std::bad_alloc();
    return reinterpret_cast<std::uintptr_t>(p);
}
inline void freeBuf(std::uintptr_t p) { std::free(reinterpret_cast<void*>(p)); }

// Bytes the allocator has handed out and not taken back, for leak tests: the
// heap it manages less the part of it that is free. Finding the free part
// walks emmalloc's free lists, O(free blocks), so keep it off hot paths.
inline double heapInUse() {
    return static_cast<double>(emmalloc_dynamic_heap_size() - emmalloc_free_dynamic_memory());
}

// A NaN sentinel maps "argument not provided" to std::nullopt for the EW ops
// whose Python ctors take std::optional<double> decay parameters.
inline std::optional<double> opt(double v) {
    return std::isnan(v) ? std::nullopt : std::optional<double>(v);
}

// An operator setting (op_arg.h) from JavaScript. `op` is a raw Embind handle
// to a fresh copy of the operator, made by the generated factory for this
// call; the OpArg owns it from here, so the handle is deleted with the last
// keep (when the op that takes it is deleted, or when its constructor throws).
inline screamer::OpArg op_arg(emscripten::val op) {
    screamer::OpArg a;
    a.op = op.as<screamer::EvalOp*>(emscripten::allow_raw_pointers());
    a.keep = std::shared_ptr<void>(new emscripten::val(std::move(op)), [](void* p) {
        auto* held = static_cast<emscripten::val*>(p);
        emscripten::val& v = *held;
        v.call<void>("delete");   // the Embind handle's own delete
        delete held;
    });
    return a;
}

// Raw address of an op instance as an integer. The JS wrapper holds the C++
// EvalOp, so this hands JS the raw EvalOp* it passes to GraphBuilder.addFunctor
// (and addResample's reducer). Mirrors the uintptr_t convention of evalInto.
inline std::uintptr_t opPtr(screamer::EvalOp& op) {
    return reinterpret_cast<std::uintptr_t>(&op);
}

// The message for a float setting that is not finite (finite_settings.h),
// or an integer setting below its smallest value (int_settings.h), asked by
// a generated factory after its constructor threw: `names` is an
// Array of the camelCase setting names, `values` an Array of numbers or
// Arrays of numbers. "" when no setting is at fault.
inline std::string finiteSettingError(std::string op, emscripten::val names, emscripten::val values,
                                      emscripten::val texts, bool allocFailed) {
    const auto n = names["length"].as<unsigned>();
    std::vector<std::string> ns;
    std::vector<std::vector<double>> vs;
    for (unsigned i = 0; i < n; ++i) {
        ns.push_back(names[i].as<std::string>());
        emscripten::val v = values[i];
        std::vector<double> xs;
        if (v.isArray()) {
            const auto m = v["length"].as<unsigned>();
            for (unsigned j = 0; j < m; ++j) xs.push_back(v[j].as<double>());
        } else {
            xs.push_back(v.as<double>());
        }
        vs.push_back(std::move(xs));
    }
    std::string msg = screamer::finite_setting_error(op, ns, vs, true);
    if (!msg.empty()) return msg;
    // An integer setting outside its range, or too large to allocate
    // (int_settings.h); `texts` holds String(value) for the message.
    std::vector<double> firsts;
    std::vector<std::string> ts;
    for (const auto& x : vs) firsts.push_back(x.empty() ? 0.0 : x[0]);
    const auto nt = texts["length"].as<unsigned>();
    for (unsigned i = 0; i < nt; ++i) ts.push_back(texts[i].as<std::string>());
    return screamer::int_setting_error(op, ns, firsts, ts, allocFailed);
}

}  // namespace screamer_wasm

// Register the shared runtime on the EvalOp base class ONCE. Every op class is
// registered with base<EvalOp>, so all ops inherit evalInto/reset/etc.
#define SCREAMER_REGISTER_EVAL_OP_RUNTIME()                                    \
    emscripten::class_<screamer::EvalOp>("EvalOp")                             \
        .function("evalInto", &screamer_wasm::evalInto, emscripten::allow_raw_pointers()) \
        .function("reset", &screamer::EvalOp::reset)                           \
        .function("nIn", &screamer::EvalOp::n_in)                              \
        .function("nOut", &screamer::EvalOp::n_out);                           \
    emscripten::function("allocF64", &screamer_wasm::allocF64);               \
    emscripten::function("freeBuf", &screamer_wasm::freeBuf);                 \
    emscripten::function("heapInUse", &screamer_wasm::heapInUse);             \
    emscripten::function("finiteSettingError", &screamer_wasm::finiteSettingError); \
    emscripten::function("opPtr", &screamer_wasm::opPtr,                       \
                         emscripten::allow_raw_pointers())
