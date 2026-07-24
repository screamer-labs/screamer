#ifndef SCREAMER_DAG_FILTER_NODE_H
#define SCREAMER_DAG_FILTER_NODE_H

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
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
// not the data).
//
// Like CombineLatestNode, a delayed sibling on one port makes its frames
// future-dated relative to a sibling on the other port; applying events in
// arrival order would gate the data against a stale mask (or vice versa) at the
// wrong index. FilterNode therefore uses the SAME reorder buffer: a per-port
// watermark and a min-heap of pending frames, applying a frame only once the
// merge can prove it safe (its index <= the minimum per-port watermark), in
// global index order. The gated emit runs exactly like CombineLatestNode's
// coalescing emit, but through the mask gate.
//
// Flush semantics mirror CombineLatestNode exactly: we wait until BOTH ports have
// flushed before emitting the settled final row (once) with the gate applied,
// then propagate flush() downstream. Non-copyable/movable: the Port structs hold
// a back-reference to this node.
template <class Index>
class FilterNode : public Resettable {
public:
    static constexpr std::size_t DEFAULT_MAX_PENDING = 1'000'000;

    explicit FilterNode(Sink<Index>& downstream,
                        std::size_t max_pending = DEFAULT_MAX_PENDING)
        : cl_(2, true), downstream_(downstream), max_pending_(max_pending),
          wm_(2, std::numeric_limits<Index>::min()), flushed_(2, false) {
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
        std::fill(wm_.begin(), wm_.end(), std::numeric_limits<Index>::min());
        while (!pending_.empty()) pending_.pop();
        next_seq_ = 0;
        last_forwarded_wm_ = std::numeric_limits<Index>::min();
    }

private:
    // Unchanged as-of + coalescing (gated) logic, now fed in guaranteed global
    // index order by release(). Takes an explicit index and value.
    void apply_ordered(std::size_t i, Index ev_index, double value) {
        if (cl_.on_event(static_cast<std::uint32_t>(i), value)) {
            const std::vector<double>& row = cl_.latest();
            if (has_buffered_ && ev_index != buffered_index_) {
                // Index advanced: emit the buffered (settled) frame with gate.
                emit_buffered();
            }
            // Buffer the latest row at this index (overwrites if same index).
            buffered_index_ = ev_index;
            buffered_data_  = row[0];
            buffered_mask_  = row[1];
            has_buffered_   = true;
        }
    }

    void on_port(std::size_t i, const Frame<Index>& f) {
        assert(f.width == 1);
        if (wm_[i] < f.index) wm_[i] = f.index;   // a data frame advances the port watermark
        pending_.push(Pending{f.index, static_cast<std::uint32_t>(i), next_seq_++, f.values[0]});
        if (pending_.size() > max_pending_)
            throw std::runtime_error(
                "Filter reorder buffer overflow: an input has not advanced; "
                "a stream is stalled. Call advance(now) or check the feed.");
        release();
    }

    void on_port_watermark(std::size_t i, Index w) {
        if (wm_[i] < w) wm_[i] = w;
        release();
    }

    // Apply every pending frame the merge can now prove is safe (index <= the
    // minimum per-port watermark), in global index order, then flush any
    // coalescing buffer whose index is strictly below the settled watermark
    // (the next event must arrive at a higher index, so the buffer is done),
    // then forward the settled watermark downstream (gated per last_forwarded_wm_).
    void release() {
        Index low = std::min(wm_[0], wm_[1]);
        while (!pending_.empty() && pending_.top().index <= low) {
            Pending p = pending_.top(); pending_.pop();
            apply_ordered(p.port, p.index, p.value);
        }
        // The coalescing buffer holds the settled row for the latest index that
        // has been applied. If the watermark is now strictly above that index,
        // no future event can share the same index, so it is safe to emit.
        if (has_buffered_ && buffered_index_ < low) emit_buffered();
        if (low != std::numeric_limits<Index>::min() && low > last_forwarded_wm_) {
            last_forwarded_wm_ = low;
            downstream_.on_watermark(low);
        }
    }

    // Emit the buffered row if it passes the mask gate, then clear the buffer.
    // The DATA value is forwarded unchanged; only the MASK gates.
    // When the gate drops the row, emit on_watermark so downstream time advances.
    void emit_buffered() {
        if (has_buffered_) {
            if (buffered_mask_ != 0.0 && !screamer::isnan2(buffered_mask_))
                downstream_.push(Frame<Index>{buffered_index_, &buffered_data_, 1});
            else if (buffered_index_ > last_forwarded_wm_) {
                last_forwarded_wm_ = buffered_index_;
                downstream_.on_watermark(buffered_index_);   // dropped: advance time only
            }
        }
        has_buffered_ = false;
    }

    // Called once per input port at end-of-input. Mirror of CombineLatestNode:
    // set the port watermark to MAX (this port will send no more), drain what is
    // now safe, then wait until BOTH ports have flushed before emitting the
    // settled final row (gated) once and propagating flush downstream. Re-arm
    // afterwards so a second flush cycle works (idempotent: has_buffered_ is
    // false, so re-completing forwards flush without re-emitting).
    void flush_downstream(std::size_t i) {
        wm_[i] = std::numeric_limits<Index>::max();   // this port will send no more
        if (!flushed_[i]) { flushed_[i] = true; ++flushed_count_; }
        release();                                    // drain what is now safe
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
        void on_watermark(Index w) override { node.on_port_watermark(idx, w); }
        // Each port accepts one value per event; output is owned by the parent node.
        std::size_t n_in()  const override { return 1; }
        std::size_t n_out() const override { return 0; }
    };
    friend struct Port;

    screamer::streams::CombineLatest cl_;  // reused operator (no re-derivation)
    Sink<Index>& downstream_;
    std::size_t max_pending_;
    std::vector<Port> ports_;

    // Coalescing buffer: holds the latest aligned row at the current index.
    bool has_buffered_ = false;
    Index buffered_index_{};
    double buffered_data_ = 0.0;  // stored as member so &buffered_data_ is stable
    double buffered_mask_ = 0.0;

    // Per-port watermark (INT64_MIN = port not yet seen) and a min-heap of frames
    // waiting until the merge is safe to apply them in global index order. The seq
    // field is a monotonic arrival counter that preserves FIFO order among frames
    // sharing the same (index, port) - a single port can carry two events at the
    // same index, and last-write-wins requires arrival order.
    std::vector<Index> wm_;
    struct Pending { Index index; std::uint32_t port; std::uint64_t seq; double value; };
    struct PendGreater {
        bool operator()(const Pending& a, const Pending& b) const {
            if (a.index != b.index) return a.index > b.index;  // min-heap on index
            if (a.port  != b.port)  return a.port  > b.port;   // tie-break by port
            return a.seq > b.seq;                              // then FIFO by arrival
        }
    };
    std::priority_queue<Pending, std::vector<Pending>, PendGreater> pending_;
    std::uint64_t next_seq_ = 0;   // monotonic arrival counter for pending_

    // Last watermark forwarded downstream: skip a redundant virtual call when the
    // settled watermark has not advanced.
    Index last_forwarded_wm_ = std::numeric_limits<Index>::min();

    // End-of-input coalescing: which ports have flushed in the current cycle.
    std::vector<bool> flushed_;
    std::size_t flushed_count_ = 0;
};

}} // namespace screamer::dag
#endif
