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
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "embind_runtime.h"
#include "screamer/common/eval_op.h"
#include "screamer/dag/graph.h"
#include "screamer/dag/compiled_graph.h"
#include "screamer/dag/output_join.h"
#include "screamer/dag/max_pending.h"
#include "screamer/dag/checked_index.h"
#include "screamer/dag/resample_params.h"
#include "screamer/dag/forecast_pairs.h"

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

// The message of a rule's error, or "" when `f` does not throw. The bindings
// below that take a string return the message and JavaScript throws it
// (kinds.ts throwMessage): a throw out of an Embind function leaks its string
// arguments' buffers. The first character gives the error's class: 'T' for
// a kind error (TypeError), 'R' for a value error (RangeError).
template <class F>
inline std::string message_of(F f) {
    try {
        f();
    } catch (const screamer::dag::IndexTypeError& e) {
        return std::string("T") + e.what();
    } catch (const screamer::dag::SettingTypeError& e) {
        return std::string("T") + e.what();
    } catch (const std::exception& e) {
        return std::string("R") + e.what();
    }
    return std::string();
}

// The engine's kind error (a TypeError in JavaScript) for an index that is
// not a number; `kind` names its type. The message has one source
// (checked_index.h).
inline std::string indexKindMessage(std::string kind) {
    return message_of([&] { screamer::dag::throw_index_kind(kind); });
}
// The feed-length rule (checked_index.h): one index value per row of values.
// Returns the message, or "" (JavaScript throws the RangeError: a throw out
// of an Embind function would leak the string argument's buffer).
inline std::string feedLengthMessage(std::string who, int input, double rows, double len) {
    return message_of([&] {
        screamer::dag::check_feed_length(who, input, static_cast<std::size_t>(rows),
                                         static_cast<std::size_t>(len));
    });
}

// maxPending under the engine's rule (max_pending.h): maxPendingMessage is
// the range error of a number (RangeError), maxPendingKindMessage the error
// for a value that is not a number (TypeError); "" when there is none.
// defaultMaxPending is the engine's default. `name` is "<op>: <setting>".
inline std::string maxPendingMessage(double v, std::string name) {
    return message_of([&] { screamer::dag::checked_max_pending(v, name); });
}
inline std::string maxPendingKindMessage(std::string kind, std::string name) {
    return message_of([&] { screamer::dag::throw_max_pending_kind(kind, name); });
}
// maxFill of a fill resample, under the same rule (resample_params.h).
inline std::string maxFillMessage(double v, std::string name) {
    return message_of([&] { screamer::dag::checked_max_fill(v, name); });
}
inline double defaultMaxFill() { return static_cast<double>(screamer::dag::kDefaultMaxFill); }
// The settings rule of a resample (dag/resample_check.h), shared with Python:
// the message of its first error ('T' TypeError, 'R' RangeError), or "".
inline std::string resampleSettingsMessage(bool hasEvery, double every, bool hasCount,
                                           double count, bool hasThreshold, double threshold,
                                           bool clock, bool aggFunctor, std::string agg,
                                           std::string label, std::string fill, bool hasOrigin,
                                           double origin, bool hasMaxAge, bool maxAgeWhole,
                                           double maxAge) {
    screamer::dag::ResampleSettings s;
    s.has_every = hasEvery;         s.every = every;
    s.has_count = hasCount;         s.count = count;
    s.has_threshold = hasThreshold; s.threshold = threshold;
    s.clock = clock;
    s.agg_functor = aggFunctor;
    s.agg = std::move(agg); s.label = std::move(label); s.fill = std::move(fill);
    s.has_origin = hasOrigin;       s.origin = origin;
    s.has_max_age = hasMaxAge;      s.max_age_whole = maxAgeWhole; s.max_age = maxAge;
    return message_of([&] { screamer::dag::check_resample_settings(s); });
}
// forecastPairs: the input rule of an eager call (dag/forecast_pairs.h).
inline void checkForecastPairsInputs(bool countMode, double xNdim, double xRows, double xCols,
                                     double yNdim, double yRows, double yCols) {
    auto n = [](double v) { return static_cast<std::size_t>(v); };
    screamer::dag::check_forecast_pairs_inputs(countMode, n(xNdim), n(xRows), n(xCols),
                                               n(yNdim), n(yRows), n(yCols));
}

// A forecastPairs count or duration (JavaScript truncates it): a number in the
// int64 range, else the rule's range error.
inline std::optional<std::int64_t> horizon(bool has, double v, const char* name) {
    if (!has) return std::nullopt;
    if (!(v >= -9223372036854775808.0 && v < 9223372036854775808.0))
        screamer::dag::throw_forecast_pairs_range(name, screamer::dag::number_text(v));
    return static_cast<std::int64_t>(v);
}

// The forecastPairs horizon rule (dag/forecast_pairs.h), checked at the call.
inline void checkForecastPairsHorizon(bool hasCount, double count, bool hasDuration,
                                      double duration) {
    screamer::dag::check_forecast_pairs_horizon(horizon(hasCount, count, "count"),
                                                horizon(hasDuration, duration, "duration"));
}

// The forecastPairs index rule (dag/forecast_pairs.h): "" for "features" or
// "target", else the message.
inline std::string forecastPairsIndexMessage(const std::string& index) {
    return message_of([&] { screamer::dag::forecast_pairs_features_index(index); });
}

inline double defaultMaxPending() { return static_cast<double>(screamer::dag::kDefaultMaxPending); }

// Wraps a compiled graph (non-copyable engine type held by unique_ptr).
class CompiledGraphWrap {
public:
    explicit CompiledGraphWrap(screamer::dag::GraphSpec spec)
        : cg_(std::make_unique<screamer::dag::CompiledGraph>(std::move(spec))) {}

    void reset() { cg_->reset(); }
    // A lazy run (js/src/pipeline.ts) feeds the inputs merged in index order,
    // as Python's lazy driver does: reset(), then setMergeOrder(true).
    void setMergeOrder(bool on) { cg_->set_merge_order(on); }
    void flush() { cg_->flush(); }
    // This object's address, for the plain exports of dag_bindings.cpp.
    std::uintptr_t address() { return reinterpret_cast<std::uintptr_t>(this); }
    void advance(double now) { cg_->advance(screamer::dag::checked_index(now)); }

    void pushEvent(std::size_t inputIdx, double index, double value) {
        cg_->push_event(inputIdx, screamer::dag::checked_index(index), value);
    }

    // valuesPtr points at `width` contiguous doubles (heap or wasm memory) that
    // outlive the call. index-as-double, cast to int64.
    void pushEventWide(std::size_t inputIdx, double index,
                       std::uintptr_t valuesPtr, std::size_t width) {
        cg_->push_event_wide(inputIdx, screamer::dag::checked_index(index),
                             reinterpret_cast<const double*>(valuesPtr), width);
    }

    // The inputs of a batch drive, read from wasm addresses (uint32 each):
    //   idxPtrsPtr  -> numInputs double*  : per-input index-as-double arrays
    //   valPtrsPtr  -> numInputs double*  : per-input row-major value arrays
    //   lensPtr     -> numInputs uint32   : per-input index lengths
    //   rowsPtr     -> numInputs uint32   : per-input rows of values
    //   widthsPtr   -> numInputs uint32   : per-input column widths
    // Each index is converted to int64 under the engine's index rule, and
    // must have one value per row (checked_index.h check_feed_length) before
    // the engine reads the values.
    struct BatchInputs {
        std::vector<std::vector<std::int64_t>> idx_i64;
        std::vector<const std::int64_t*> indices;
        std::vector<const double*>       vals;
        std::vector<std::size_t>         lens;
        std::vector<std::size_t>         widths;
    };
    static BatchInputs batch_inputs(std::uintptr_t idxPtrsPtr, std::uintptr_t valPtrsPtr,
                                    std::uintptr_t lensPtr, std::uintptr_t rowsPtr,
                                    std::uintptr_t widthsPtr, std::size_t numInputs) {
        const auto* idxPtrs = reinterpret_cast<const std::uint32_t*>(idxPtrsPtr);
        const auto* valPtrs = reinterpret_cast<const std::uint32_t*>(valPtrsPtr);
        const auto* lens    = reinterpret_cast<const std::uint32_t*>(lensPtr);
        const auto* rows    = reinterpret_cast<const std::uint32_t*>(rowsPtr);
        const auto* widths  = reinterpret_cast<const std::uint32_t*>(widthsPtr);
        BatchInputs in;
        in.idx_i64.resize(numInputs);
        in.indices.resize(numInputs);
        in.vals.resize(numInputs);
        in.lens.resize(numInputs);
        in.widths.resize(numInputs);
        for (std::size_t i = 0; i < numInputs; ++i) {
            const std::size_t len = lens[i];
            const auto* idx_d = reinterpret_cast<const double*>(idxPtrs[i]);
            in.idx_i64[i].resize(len);
            for (std::size_t j = 0; j < len; ++j)
                in.idx_i64[i][j] = screamer::dag::checked_index(idx_d[j]);
            // One index value per row, as Python's feeds (marshal_feeds).
            screamer::dag::check_feed_length("Pipeline", static_cast<std::ptrdiff_t>(i),
                                             rows[i], len);
            in.indices[i] = in.idx_i64[i].data();
            in.vals[i]    = reinterpret_cast<const double*>(valPtrs[i]);
            in.lens[i]    = len;
            in.widths[i]  = widths[i];
        }
        return in;
    }

    // Batch drive: feed `numInputs` input streams then marshal output `outIdx`.
    OutBufFlat runBatchFlat(std::uintptr_t idxPtrsPtr, std::uintptr_t valPtrsPtr,
                            std::uintptr_t lensPtr, std::uintptr_t rowsPtr,
                            std::uintptr_t widthsPtr, std::size_t numInputs,
                            std::size_t outIdx) {
        BatchInputs in = batch_inputs(idxPtrsPtr, valPtrsPtr, lensPtr, rowsPtr, widthsPtr, numInputs);
        return marshal_flat_at(cg_->run_batch(in.indices, in.vals, in.lens, in.widths), outIdx);
    }

    // Batch drive ONCE, caching every OutputBuffer so a multi-output pipeline
    // reads each output without recomputing the graph. JS calls this once, then
    // outputCount()/output(i) per output. Same pointer conventions as
    // runBatchFlat above. Overwrites any previously stored buffers.
    //
    // With `align` and more than one output, the outputs are stored aligned on
    // one index (dag::align_outputs, the rule Python uses): alignedIndex()
    // gives the index and output(i) the values of output i on it.
    void runBatchStore(std::uintptr_t idxPtrsPtr, std::uintptr_t valPtrsPtr,
                       std::uintptr_t lensPtr, std::uintptr_t rowsPtr,
                       std::uintptr_t widthsPtr, std::size_t numInputs, bool align) {
        BatchInputs in = batch_inputs(idxPtrsPtr, valPtrsPtr, lensPtr, rowsPtr, widthsPtr, numInputs);
        cg_->run_batch_ref(in.indices, in.vals, in.lens, in.widths);
        store(align);   // a run ends with a flush: the join settles every row
    }

    // Drain every frame emitted since the last drain/reset (a live session's
    // result) and store it as runBatchStore does. Aligned rows carry the
    // output join across drains (CompiledGraph::drain_aligned).
    void drainStore(bool align) { store(align); }

    // Number of cached outputs from the last runBatchStore call.
    std::size_t outputCount() const { return stored_.size(); }

    // Whether the stored outputs are aligned on one index.
    bool storedAligned() const { return aligned_mode_; }

    // The i-th stored output, in place: valuePtr points into the stored rows
    // and indexPtr at its index as doubles, both valid until the next output()
    // or clearStored() (JS copies them and frees neither). Out of range: empty.
    OutBufFlat output(std::size_t outIdx) {
        OutBufFlat out;
        if (outIdx >= stored_.size()) return out;
        const screamer::dag::OutputBuffer& b = stored_[outIdx];
        out.rows  = b.indices.size();
        out.width = b.width;
        index_d_.resize(out.rows);
        for (std::size_t r = 0; r < out.rows; ++r) index_d_[r] = static_cast<double>(b.indices[r]);
        out.indexPtr = reinterpret_cast<std::uintptr_t>(index_d_.data());
        out.valuePtr = reinterpret_cast<std::uintptr_t>(b.values.data());
        return out;
    }

    // The values of aligned stored output i, in place: valuePtr points into
    // the stored rows, which stay until clearStored() (JS copies them and does
    // not free the pointer). indexPtr is 0: the index is alignedIndex().
    OutBufFlat alignedOutput(std::size_t outIdx) const {
        OutBufFlat out;
        if (outIdx >= aligned_.values.size()) return out;
        out.rows     = aligned_.rows();
        out.width    = aligned_.widths[outIdx];
        out.valuePtr = reinterpret_cast<std::uintptr_t>(aligned_.values[outIdx].data());
        return out;
    }

    // The index of the aligned stored outputs as doubles, in place (valuePtr 0,
    // width 0), valid until clearStored(); JS copies it once for every output.
    OutBufFlat alignedIndex() {
        index_d_.resize(aligned_.rows());
        for (std::size_t r = 0; r < index_d_.size(); ++r)
            index_d_[r] = static_cast<double>(aligned_.index[r]);
        OutBufFlat out;
        out.rows     = aligned_.rows();
        out.width    = 0;
        out.indexPtr = reinterpret_cast<std::uintptr_t>(index_d_.data());
        return out;
    }

    // Empties the stored outputs once JS has read them. Their storage stays
    // (cleared) for the graph's next run, which takes it back through
    // drain_into, so a repeated call does not grow its buffers again. Rows
    // that align_outputs moved out are given back first.
    void clearStored() {
        if (aligned_mode_) {
            for (std::size_t j = 0; j < stored_.size() && j < aligned_.values.size(); ++j)
                if (stored_[j].values.capacity() == 0) stored_[j].values.swap(aligned_.values[j]);
            if (!stored_.empty() && stored_[0].indices.capacity() == 0)
                stored_[0].indices.swap(aligned_.index);
        }
        for (auto& b : stored_) { b.indices.clear(); b.values.clear(); }
        aligned_ = screamer::dag::AlignedOutputs{};
        index_d_.clear();
        aligned_mode_ = false;
    }

private:
    // Moves the graph's output rows into stored_ (no copy), or, aligned when
    // asked and there is more than one output, into aligned_.
    void store(bool align) {
        aligned_mode_ = align && cg_->output_count() > 1;
        if (aligned_mode_) aligned_ = cg_->drain_aligned(stored_);
        else cg_->drain_into(stored_);
    }

    std::unique_ptr<screamer::dag::CompiledGraph> cg_;
    std::vector<screamer::dag::OutputBuffer>      stored_;  // last runBatchStore / drainStore
    screamer::dag::AlignedOutputs                 aligned_; // stored_, aligned (align mode)
    std::vector<double>                           index_d_;  // an output's index as doubles
    bool                                          aligned_mode_ = false;
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

    // The graph's join cap (GraphSpec::max_pending), checked; call it first.
    void setMaxPending(double max_pending) {
        b_.set_max_pending(screamer::dag::checked_max_pending(max_pending, "Pipeline: maxPending"));
    }

    std::size_t addCombineLatest(std::vector<std::size_t> inputs, bool when_all,
                                 double max_pending) {
        return b_.add_combine_latest(std::move(inputs), when_all,
                                     screamer::dag::checked_max_pending(max_pending, "combineLatest: maxPending"));
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
                            std::uintptr_t reducerPtr, double max_pending,
                            double max_fill) {
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
        rp.max_fill  = screamer::dag::checked_max_fill(max_fill, "resample: maxFill");
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
        // A cumulative resample with [value, driver] as two inputs: the builder
        // joins them first (GraphBuilder::add_resample).
        return b_.add_resample(std::move(inputs), rp,
                               screamer::dag::checked_max_pending(max_pending, "Pipeline: maxPending"));
    }

    // The forecast_pairs nodes (dag/forecast_pairs.h). A horizon that is not
    // given is passed as has* = false; featuresIndex stamps the rows at the
    // features (index "features"), else at the target.
    std::size_t addForecastPairs(std::size_t x, std::size_t y, bool hasCount, double count,
                                 bool hasDuration, double duration, bool dropna,
                                 bool featuresIndex) {
        return screamer::dag::add_forecast_pairs(
            b_, x, y, horizon(hasCount, count, "count"),
            horizon(hasDuration, duration, "duration"), dropna, featuresIndex);
    }

    void setOutputs(std::vector<std::size_t> outs) {
        b_.set_outputs(std::move(outs));
    }

    // Compile the accumulated spec into a fresh CompiledGraphWrap; JS takes
    // ownership (return_value_policy::take_ownership) and must .delete() it.
    // The spec moves into the graph, so a builder compiles once: two graphs
    // over one builder would share its operators' state.
    CompiledGraphWrap* compile() {
        if (compiled_)
            throw std::runtime_error("this graph builder was compiled already; build a new one");
        compiled_ = true;
        return new CompiledGraphWrap(b_.take());
    }

private:
    screamer::dag::GraphBuilder b_;
    bool compiled_ = false;
};

}  // namespace screamer_wasm

// Register the DAG classes, the VectorSizeT vector, and the flat output value
// object. Call inside EMSCRIPTEN_BINDINGS(screamer_dag).
#define SCREAMER_REGISTER_DAG()                                                 \
    emscripten::function("indexKindMessage", &screamer_wasm::indexKindMessage); \
    emscripten::function("feedLengthMessage", &screamer_wasm::feedLengthMessage); \
    emscripten::function("maxPendingMessage", &screamer_wasm::maxPendingMessage); \
    emscripten::function("maxPendingKindMessage", &screamer_wasm::maxPendingKindMessage); \
    emscripten::function("defaultMaxPending", &screamer_wasm::defaultMaxPending); \
    emscripten::function("maxFillMessage", &screamer_wasm::maxFillMessage); \
    emscripten::function("defaultMaxFill", &screamer_wasm::defaultMaxFill);     \
    emscripten::function("resampleSettingsMessage",                            \
                         &screamer_wasm::resampleSettingsMessage);             \
    emscripten::function("checkForecastPairsInputs",                           \
                         &screamer_wasm::checkForecastPairsInputs);            \
    emscripten::function("checkForecastPairsHorizon",                          \
                         &screamer_wasm::checkForecastPairsHorizon);           \
    emscripten::function("forecastPairsIndexMessage",                          \
                         &screamer_wasm::forecastPairsIndexMessage);           \
    emscripten::register_vector<std::size_t>("VectorSizeT");                    \
    emscripten::value_object<screamer_wasm::OutBufFlat>("OutBufFlat")           \
        .field("indexPtr", &screamer_wasm::OutBufFlat::indexPtr)               \
        .field("valuePtr", &screamer_wasm::OutBufFlat::valuePtr)               \
        .field("rows",     &screamer_wasm::OutBufFlat::rows)                   \
        .field("width",    &screamer_wasm::OutBufFlat::width);                 \
    emscripten::class_<screamer_wasm::CompiledGraphWrap>("CompiledGraph")       \
        .function("reset",        &screamer_wasm::CompiledGraphWrap::reset)     \
        .function("flush",        &screamer_wasm::CompiledGraphWrap::flush)     \
        .function("setMergeOrder",&screamer_wasm::CompiledGraphWrap::setMergeOrder) \
        .function("advance",      &screamer_wasm::CompiledGraphWrap::advance)   \
        .function("pushEvent",    &screamer_wasm::CompiledGraphWrap::pushEvent) \
        .function("address",      &screamer_wasm::CompiledGraphWrap::address)   \
        .function("pushEventWide",                                             \
                  &screamer_wasm::CompiledGraphWrap::pushEventWide)            \
        .function("runBatchFlat", &screamer_wasm::CompiledGraphWrap::runBatchFlat) \
        .function("runBatchStore",&screamer_wasm::CompiledGraphWrap::runBatchStore) \
        .function("drainStore",   &screamer_wasm::CompiledGraphWrap::drainStore) \
        .function("outputCount",  &screamer_wasm::CompiledGraphWrap::outputCount) \
        .function("storedAligned",&screamer_wasm::CompiledGraphWrap::storedAligned) \
        .function("output",       &screamer_wasm::CompiledGraphWrap::output)    \
        .function("alignedOutput",&screamer_wasm::CompiledGraphWrap::alignedOutput) \
        .function("alignedIndex", &screamer_wasm::CompiledGraphWrap::alignedIndex) \
        .function("clearStored",  &screamer_wasm::CompiledGraphWrap::clearStored); \
    emscripten::class_<screamer_wasm::GraphBuilderWrap>("GraphBuilder")         \
        .constructor<>()                                                       \
        .function("addInput",         &screamer_wasm::GraphBuilderWrap::addInput) \
        .function("addFunctor",       &screamer_wasm::GraphBuilderWrap::addFunctor) \
        .function("setMaxPending",    &screamer_wasm::GraphBuilderWrap::setMaxPending) \
        .function("addCombineLatest", &screamer_wasm::GraphBuilderWrap::addCombineLatest) \
        .function("addDropna",        &screamer_wasm::GraphBuilderWrap::addDropna) \
        .function("addSelect",        &screamer_wasm::GraphBuilderWrap::addSelect) \
        .function("addFilter",        &screamer_wasm::GraphBuilderWrap::addFilter) \
        .function("addDelay",         &screamer_wasm::GraphBuilderWrap::addDelay) \
        .function("addResample",      &screamer_wasm::GraphBuilderWrap::addResample) \
        .function("addForecastPairs", &screamer_wasm::GraphBuilderWrap::addForecastPairs) \
        .function("setOutputs",       &screamer_wasm::GraphBuilderWrap::setOutputs) \
        .function("compile",          &screamer_wasm::GraphBuilderWrap::compile, \
                  emscripten::allow_raw_pointers(),                            \
                  emscripten::return_value_policy::take_ownership())
