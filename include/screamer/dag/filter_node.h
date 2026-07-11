#ifndef SCREAMER_DAG_FILTER_NODE_H
#define SCREAMER_DAG_FILTER_NODE_H

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>
#include "screamer/common/float_info.h"
#include "screamer/dag/frame.h"
#include "screamer/dag/resettable.h"
#include "screamer/streams/combine_latest.h"

namespace screamer { namespace dag {

// Mask-gate fan-in node. Two fixed input ports: port 0 = data, port 1 = mask.
// Uses CombineLatest(2, when_all=true) so it only fires once both ports have
// produced at least one value. Coalescing: one (data-only) frame is emitted per
// DISTINCT index, but only when the settled mask is nonzero and not NaN.
//   mask == 0.0  -> row is dropped
//   mask is NaN  -> row is dropped
//   any other mask value (incl. negative, incl. 1.0) -> row is kept
// The DATA value passes through unchanged even when it is NaN (the mask gates,
// not the data). Flush semantics mirror CombineLatestNode exactly: we wait until
// BOTH ports have flushed before emitting the settled final row (once) with the
// gate applied, then propagate flush() downstream. Non-copyable/movable: the
// Port structs hold a back-reference to this node.
template <class Index>
class FilterNode : public Resettable {
public:
    explicit FilterNode(Sink<Index>& downstream)
        : cl_(2, true), downstream_(downstream), flushed_(2, false) {
        ports_.reserve(2);
        for (std::size_t i = 0; i < 2; ++i) ports_.emplace_back(*this, i);
    }

    // Non-movable/copyable: the ports hold a reference back to this node.
    FilterNode(const FilterNode&) = delete;
    FilterNode& operator=(const FilterNode&) = delete;
    FilterNode(FilterNode&&) = delete;
    FilterNode& operator=(FilterNode&&) = delete;

    Sink<Index>& port(std::size_t i) { return ports_[i]; }

    void reset() override {
        cl_.reset();
        has_buffered_ = false;
        std::fill(flushed_.begin(), flushed_.end(), false);
        flushed_count_ = 0;
    }

private:
    void on_port(std::size_t i, const Frame<Index>& f) {
        assert(f.width == 1);
        if (cl_.on_event(static_cast<std::uint32_t>(i), f.values[0])) {
            const std::vector<double>& row = cl_.latest();
            if (has_buffered_ && f.index != buffered_index_) {
                // Index advanced: emit the buffered (settled) frame with gate.
                emit_buffered();
            }
            // Buffer the latest row at this index (overwrites if same index).
            buffered_index_ = f.index;
            buffered_data_  = row[0];
            buffered_mask_  = row[1];
            has_buffered_   = true;
        }
    }

    // Emit the buffered row if it passes the mask gate, then clear the buffer.
    // The DATA value is forwarded unchanged; only the MASK gates.
    void emit_buffered() {
        if (has_buffered_ && buffered_mask_ != 0.0 &&
                !screamer::isnan2(buffered_mask_)) {
            downstream_.push(Frame<Index>{buffered_index_, &buffered_data_, 1});
        }
        has_buffered_ = false;
    }

    // Called once per input port at end-of-input. Mirror of CombineLatestNode:
    // wait until BOTH ports have flushed before emitting the settled final row
    // (gated) once, then propagate the flush downstream. Re-arm afterwards so a
    // second flush cycle works correctly (idempotent: has_buffered_ is false).
    void flush_downstream(std::size_t i) {
        if (!flushed_[i]) { flushed_[i] = true; ++flushed_count_; }
        if (flushed_count_ < 2) return;  // not every port has flushed yet

        // Both ports have delivered their final event: emit the settled row once.
        emit_buffered();
        downstream_.flush();

        // Re-arm for a subsequent flush cycle.
        std::fill(flushed_.begin(), flushed_.end(), false);
        flushed_count_ = 0;
    }

    // A single input port: routes an event to its owning node with its index.
    struct Port : Sink<Index> {
        FilterNode& node;
        std::size_t idx;
        Port(FilterNode& n, std::size_t i) : node(n), idx(i) {}
        void push(const Frame<Index>& f) override { node.on_port(idx, f); }
        void flush() override { node.flush_downstream(idx); }
        // Each port accepts one value per event; output is owned by the parent node.
        std::size_t n_in()  const override { return 1; }
        std::size_t n_out() const override { return 0; }
    };
    friend struct Port;

    screamer::streams::CombineLatest cl_;  // reused operator (no re-derivation)
    Sink<Index>& downstream_;
    std::vector<Port> ports_;

    // Coalescing buffer: holds the latest aligned row at the current index.
    bool has_buffered_ = false;
    Index buffered_index_{};
    double buffered_data_ = 0.0;  // stored as member so &buffered_data_ is stable
    double buffered_mask_ = 0.0;

    // End-of-input coalescing: which ports have flushed in the current cycle.
    std::vector<bool> flushed_;
    std::size_t flushed_count_ = 0;
};

}} // namespace screamer::dag
#endif
