#pragma once
// Embind layer over the pure header-only dag/ engine (GraphBuilder +
// CompiledGraph). This mirrors bindings/bindings_dag.cpp (the nanobind wrapper
// of the SAME engine), swapping nb::object marshalling for uintptr_t pointers
// and flat heap buffers the JS side reads via viewF64 and frees via freeBuf.
//
// Pointer / integer conventions (all crossing the JS boundary):
//   * An EvalOp* is passed as a uintptr_t (reinterpret_cast), exactly like
//     evalInto. JS obtains it from an op instance via the free function opPtr().
//   * Engine indices are std::int64_t. To avoid BigInt on the JS boundary, every
//     index/duration/int64 parameter is accepted as a `double` and cast to
//     int64_t here, and output indices are emitted index-as-double. Indices with
//     magnitude above 2^53 lose precision under this convention.
//   * Node ids are std::size_t (wasm32: 32-bit). VectorSizeT == vector<size_t>.

#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "embind_runtime.h"
#include "screamer/common/eval_op.h"
#include "screamer/dag/graph.h"
#include "screamer/dag/compiled_graph.h"
#include "screamer/dag/resample_params.h"

namespace screamer_wasm {

// Flat descriptor for one gathered output stream. The JS side reads the two heap
// buffers with viewF64(indexPtr, rows) and viewF64(valuePtr, rows*width), then
// frees both with freeBuf. Returned by value as an Embind value_object.
struct OutBufFlat {
    std::uintptr_t indexPtr = 0;  // heap double[rows]: index-as-double
    std::uintptr_t valuePtr = 0;  // heap double[rows*width]: row-major values
    std::size_t    rows     = 0;
    std::size_t    width    = 1;
};

// Copy one OutputBuffer into fresh malloc'd double[] buffers (index-as-double).
// Mirrors marshal_gather in bindings_dag.cpp; the caller frees via freeBuf.
inline OutBufFlat marshal_flat(const screamer::dag::OutputBuffer& b) {
    const std::size_t rows  = b.indices.size();
    const std::size_t width = b.width;
    const std::size_t nv    = rows * width;
    double* idx = static_cast<double*>(std::malloc((rows ? rows : 1) * sizeof(double)));
    double* val = static_cast<double*>(std::malloc((nv ? nv : 1) * sizeof(double)));
    for (std::size_t r = 0; r < rows; ++r)
        idx[r] = static_cast<double>(b.indices[r]);
    if (nv) std::memcpy(val, b.values.data(), nv * sizeof(double));
    OutBufFlat out;
    out.indexPtr = reinterpret_cast<std::uintptr_t>(idx);
    out.valuePtr = reinterpret_cast<std::uintptr_t>(val);
    out.rows     = rows;
    out.width    = width;
    return out;
}

// Pick output `outIdx` from a drained/batched buffer set, or an empty buffer if
// the index is out of range.
inline OutBufFlat marshal_flat_at(const std::vector<screamer::dag::OutputBuffer>& outs,
                                  std::size_t outIdx) {
    if (outIdx >= outs.size()) return OutBufFlat{};
    return marshal_flat(outs[outIdx]);
}

// Wraps a compiled graph (non-copyable engine type held by unique_ptr).
class CompiledGraphWrap {
public:
    explicit CompiledGraphWrap(const screamer::dag::GraphSpec& spec)
        : cg_(std::make_unique<screamer::dag::CompiledGraph>(spec)) {}

    void reset() { cg_->reset(); }
    void flush() { cg_->flush(); }
    void advance(double now) { cg_->advance(static_cast<std::int64_t>(now)); }

    void pushEvent(std::size_t inputIdx, double index, double value) {
        cg_->push_event(inputIdx, static_cast<std::int64_t>(index), value);
    }

    // valuesPtr points at `width` contiguous doubles (heap or wasm memory) that
    // outlive the call. index-as-double, cast to int64.
    void pushEventWide(std::size_t inputIdx, double index,
                       std::uintptr_t valuesPtr, std::size_t width) {
        cg_->push_event_wide(inputIdx, static_cast<std::int64_t>(index),
                             reinterpret_cast<const double*>(valuesPtr), width);
    }

    // Drain every frame emitted since the last drain/reset and marshal output
    // `outIdx` into flat heap buffers.
    OutBufFlat drainFlat(std::size_t outIdx) {
        return marshal_flat_at(cg_->drain(), outIdx);
    }

    // Batch drive: feed `numInputs` input streams then marshal output `outIdx`.
    // All array-of-pointers arguments are wasm addresses (uint32 each):
    //   idxPtrsPtr  -> numInputs double*  : per-input index-as-double arrays
    //   valPtrsPtr  -> numInputs double*  : per-input row-major value arrays
    //   lensPtr     -> numInputs uint32   : per-input event counts
    //   widthsPtr   -> numInputs uint32   : per-input column widths
    // Indices are converted to int64 here (VectorSource wants const int64_t*).
    OutBufFlat runBatchFlat(std::uintptr_t idxPtrsPtr, std::uintptr_t valPtrsPtr,
                            std::uintptr_t lensPtr, std::uintptr_t widthsPtr,
                            std::size_t numInputs, std::size_t outIdx) {
        const auto* idxPtrs = reinterpret_cast<const std::uint32_t*>(idxPtrsPtr);
        const auto* valPtrs = reinterpret_cast<const std::uint32_t*>(valPtrsPtr);
        const auto* lens    = reinterpret_cast<const std::uint32_t*>(lensPtr);
        const auto* widths  = reinterpret_cast<const std::uint32_t*>(widthsPtr);

        std::vector<std::vector<std::int64_t>> idx_i64(numInputs);
        std::vector<const std::int64_t*> in_indices(numInputs);
        std::vector<const double*>       in_vals(numInputs);
        std::vector<std::size_t>         in_lens(numInputs);
        std::vector<std::size_t>         in_widths(numInputs);
        for (std::size_t i = 0; i < numInputs; ++i) {
            const std::size_t len = lens[i];
            const auto* idx_d = reinterpret_cast<const double*>(idxPtrs[i]);
            idx_i64[i].resize(len);
            for (std::size_t j = 0; j < len; ++j)
                idx_i64[i][j] = static_cast<std::int64_t>(idx_d[j]);
            in_indices[i] = idx_i64[i].data();
            in_vals[i]    = reinterpret_cast<const double*>(valPtrs[i]);
            in_lens[i]    = len;
            in_widths[i]  = widths[i];
        }
        return marshal_flat_at(
            cg_->run_batch(in_indices, in_vals, in_lens, in_widths), outIdx);
    }

private:
    std::unique_ptr<screamer::dag::CompiledGraph> cg_;
};

// Wraps dag::GraphBuilder. Node ids are size_t; edge lists are VectorSizeT.
// EvalOp*s arrive as uintptr_t (from opPtr). No op keepalive is held here: the
// JS side owns op instances and must keep them alive for the compiled graph's
// lifetime (the raw EvalOp* points into those JS-owned C++ objects).
class GraphBuilderWrap {
public:
    std::size_t addInput() { return b_.add_input(); }

    std::size_t addFunctor(std::uintptr_t opPtr, std::vector<std::size_t> inputs) {
        auto* op = reinterpret_cast<screamer::EvalOp*>(opPtr);
        return b_.add_functor(op, std::move(inputs));
    }

    std::size_t addCombineLatest(std::vector<std::size_t> inputs, bool when_all,
                                 double max_pending) {
        return b_.add_combine_latest(std::move(inputs), when_all,
                                     static_cast<std::size_t>(max_pending));
    }

    std::size_t addDropna(std::vector<std::size_t> inputs, bool how_all) {
        return b_.add_dropna(std::move(inputs), how_all);
    }

    std::size_t addSelect(std::vector<std::size_t> inputs,
                          std::vector<std::size_t> columns) {
        return b_.add_select(std::move(inputs), std::move(columns));
    }

    std::size_t addFilter(std::vector<std::size_t> inputs) {
        return b_.add_filter(std::move(inputs));
    }

    std::size_t addDelay(std::vector<std::size_t> inputs, double duration) {
        return b_.add_delay(std::move(inputs), static_cast<std::int64_t>(duration));
    }

    // planPtr points at planLen int32 pairs [agg0, col0, agg1, col1, ...] (may be
    // 0/empty). reducerPtr is an EvalOp* or 0 for none. int64 params arrive as
    // doubles (see the header note on the 2^53 caveat).
    std::size_t addResample(std::vector<std::size_t> inputs, int mode, int agg,
                            int label, int fill, double width, double origin,
                            double count, double threshold, double max_age,
                            std::uintptr_t planPtr, std::size_t planLen,
                            std::uintptr_t reducerPtr) {
        screamer::dag::ResampleParams rp;
        rp.mode      = static_cast<screamer::dag::ResampleMode>(mode);
        rp.agg       = static_cast<screamer::dag::ResampleAgg>(agg);
        rp.label     = static_cast<screamer::dag::ResampleLabel>(label);
        rp.fill      = static_cast<screamer::dag::ResampleFill>(fill);
        rp.width     = static_cast<std::int64_t>(width);
        rp.origin    = static_cast<std::int64_t>(origin);
        rp.count     = static_cast<std::int64_t>(count);
        rp.threshold = threshold;
        rp.max_age   = static_cast<std::int64_t>(max_age);
        if (planLen && planPtr) {
            const auto* p = reinterpret_cast<const std::int32_t*>(planPtr);
            rp.plan.reserve(planLen);
            for (std::size_t i = 0; i < planLen; ++i) {
                screamer::dag::ResamplePlanEntry e{};
                e.agg       = static_cast<screamer::dag::ResampleAgg>(p[2 * i]);
                e.input_col = static_cast<std::size_t>(p[2 * i + 1]);
                rp.plan.push_back(e);
            }
        }
        if (reducerPtr)
            rp.reducer = reinterpret_cast<screamer::EvalOp*>(reducerPtr);
        return b_.add_resample(std::move(inputs), rp);
    }

    void setOutputs(std::vector<std::size_t> outs) {
        b_.set_outputs(std::move(outs));
    }

    // Compile the accumulated spec into a fresh CompiledGraphWrap; JS takes
    // ownership (return_value_policy::take_ownership) and must .delete() it.
    CompiledGraphWrap* compile() {
        return new CompiledGraphWrap(b_.spec());
    }

private:
    screamer::dag::GraphBuilder b_;
};

}  // namespace screamer_wasm

// Register the DAG classes, the VectorSizeT vector, and the flat output value
// object. Call inside EMSCRIPTEN_BINDINGS(screamer_dag).
#define SCREAMER_REGISTER_DAG()                                                 \
    emscripten::register_vector<std::size_t>("VectorSizeT");                    \
    emscripten::value_object<screamer_wasm::OutBufFlat>("OutBufFlat")           \
        .field("indexPtr", &screamer_wasm::OutBufFlat::indexPtr)               \
        .field("valuePtr", &screamer_wasm::OutBufFlat::valuePtr)               \
        .field("rows",     &screamer_wasm::OutBufFlat::rows)                   \
        .field("width",    &screamer_wasm::OutBufFlat::width);                 \
    emscripten::class_<screamer_wasm::CompiledGraphWrap>("CompiledGraph")       \
        .function("reset",        &screamer_wasm::CompiledGraphWrap::reset)     \
        .function("flush",        &screamer_wasm::CompiledGraphWrap::flush)     \
        .function("advance",      &screamer_wasm::CompiledGraphWrap::advance)   \
        .function("pushEvent",    &screamer_wasm::CompiledGraphWrap::pushEvent) \
        .function("pushEventWide",                                             \
                  &screamer_wasm::CompiledGraphWrap::pushEventWide)            \
        .function("drainFlat",    &screamer_wasm::CompiledGraphWrap::drainFlat) \
        .function("runBatchFlat", &screamer_wasm::CompiledGraphWrap::runBatchFlat); \
    emscripten::class_<screamer_wasm::GraphBuilderWrap>("GraphBuilder")         \
        .constructor<>()                                                       \
        .function("addInput",         &screamer_wasm::GraphBuilderWrap::addInput) \
        .function("addFunctor",       &screamer_wasm::GraphBuilderWrap::addFunctor) \
        .function("addCombineLatest", &screamer_wasm::GraphBuilderWrap::addCombineLatest) \
        .function("addDropna",        &screamer_wasm::GraphBuilderWrap::addDropna) \
        .function("addSelect",        &screamer_wasm::GraphBuilderWrap::addSelect) \
        .function("addFilter",        &screamer_wasm::GraphBuilderWrap::addFilter) \
        .function("addDelay",         &screamer_wasm::GraphBuilderWrap::addDelay) \
        .function("addResample",      &screamer_wasm::GraphBuilderWrap::addResample) \
        .function("setOutputs",       &screamer_wasm::GraphBuilderWrap::setOutputs) \
        .function("compile",          &screamer_wasm::GraphBuilderWrap::compile, \
                  emscripten::allow_raw_pointers(),                            \
                  emscripten::return_value_policy::take_ownership())
