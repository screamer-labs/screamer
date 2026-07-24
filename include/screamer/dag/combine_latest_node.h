#ifndef SCREAMER_DAG_COMBINE_LATEST_NODE_H
#define SCREAMER_DAG_COMBINE_LATEST_NODE_H

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <vector>
#include "screamer/dag/frame.h"
#include "screamer/dag/resettable.h"
#include "screamer/streams/combine_latest.h"

namespace screamer { namespace dag {

// Aligning fan-in node. Exposes N single-value input ports; on any port event it
// updates the reused CombineLatest operator and, when it fires (per when_all),
// buffers the latest aligned row. Coalescing: one frame is emitted per DISTINCT
// index (same-index events update the buffer; the frame is pushed when the index
// advances). At end-of-input each port is flushed once; because a producer pushes
// its final event just before flushing its own port, flush() must wait until EVERY
// port has flushed before emitting the settled final row (once) - otherwise the
// shared final index would be emitted once per port. flush() is idempotent.
// Derives from Resettable so CompiledGraph can reset it via a single polymorphic
// list without knowing its concrete type (CombineLatestNode is not a Sink).
template <class Index>
class CombineLatestNode : public Resettable {
public:
    CombineLatestNode(std::size_t n, bool when_all, Sink<Index>& downstream)
        : cl_(n, when_all), downstream_(downstream), n_(n),
          buffered_row_(n, 0.0),
          wm_(n, std::numeric_limits<Index>::min()),
          flushed_(n, false) {
        ports_.reserve(n);
        for (std::size_t i = 0; i < n; ++i) ports_.emplace_back(*this, i);
    }

    // Non-movable/copyable: the ports hold a reference back to this node.
    CombineLatestNode(const CombineLatestNode&) = delete;
    CombineLatestNode& operator=(const CombineLatestNode&) = delete;
    CombineLatestNode(CombineLatestNode&&) = delete;
    CombineLatestNode& operator=(CombineLatestNode&&) = delete;

    Sink<Index>& port(std::size_t i) { return ports_[i]; }

    void reset() override {
        cl_.reset();
        has_buffered_ = false;
        std::fill(flushed_.begin(), flushed_.end(), false);
        flushed_count_ = 0;
        std::fill(wm_.begin(), wm_.end(), std::numeric_limits<Index>::min());
        while (!pending_.empty()) pending_.pop();
        next_seq_ = 0;
    }

private:
    // Unchanged as-of + coalescing logic, now fed in guaranteed global index order
    // by release(). Takes an explicit index and value instead of a Frame.
    void apply_ordered(std::size_t i, Index ev_index, double value) {
        if (cl_.on_event(static_cast<std::uint32_t>(i), value)) {
            const std::vector<double>& row = cl_.latest();
            if (has_buffered_ && ev_index != buffered_index_) {
                // Index advanced: emit the buffered (settled) frame.
                downstream_.push(Frame<Index>{buffered_index_,
                                              buffered_row_.data(), n_});
            }
            // Buffer the latest row at this index (overwrites if same index).
            buffered_index_ = ev_index;
            std::copy(row.begin(), row.end(), buffered_row_.begin());
            has_buffered_ = true;
        }
    }

    void on_port(std::size_t i, const Frame<Index>& f) {
        assert(f.width == 1);
        if (wm_[i] < f.index) wm_[i] = f.index;   // a data frame advances the port watermark
        pending_.push(Pending{f.index, static_cast<std::uint32_t>(i), next_seq_++, f.values[0]});
        release();
    }

    void on_port_watermark(std::size_t i, Index w) {
        if (wm_[i] < w) wm_[i] = w;
        release();
    }

    // Apply every pending frame the merge can now prove is safe (index <= the
    // minimum per-port watermark), in global index order, then forward the
    // settled watermark downstream.
    void release() {
        Index low = wm_[0];
        for (std::size_t j = 1; j < n_; ++j) low = std::min(low, wm_[j]);
        while (!pending_.empty() && pending_.top().index <= low) {
            Pending p = pending_.top(); pending_.pop();
            apply_ordered(p.port, p.index, p.value);
        }
        if (low != std::numeric_limits<Index>::min()) downstream_.on_watermark(low);
    }

    // Called once per input port at end-of-input. Each producer pushes its final
    // (same-index) event just BEFORE flushing its own port, so flushing eagerly on
    // the first port would emit the final row with the other ports' values still
    // stale (and again on the next port) - duplicating the shared final index. To
    // coalesce (mirroring the mid-stream "emit on index advance" logic), we wait
    // until EVERY port has flushed before emitting the settled final row once and
    // propagating the flush downstream. A per-port bitmask dedups repeat flushes of
    // one port (a producer with multiple upstreams flushes it more than once).
    void flush_downstream(std::size_t i) {
        wm_[i] = std::numeric_limits<Index>::max();   // this port will send no more
        if (!flushed_[i]) { flushed_[i] = true; ++flushed_count_; }
        release();                                    // drain what is now safe
        if (flushed_count_ < n_) return;   // not every port has flushed yet

        // Every port has now delivered its final event: emit the settled row once.
        if (has_buffered_) {
            downstream_.push(Frame<Index>{buffered_index_,
                                          buffered_row_.data(), n_});
            has_buffered_ = false;
        }
        downstream_.flush();

        // Re-arm for a subsequent flush cycle (idempotent: has_buffered_ is now
        // false, so re-completing the mask forwards flush without re-emitting).
        std::fill(flushed_.begin(), flushed_.end(), false);
        flushed_count_ = 0;
    }

    // A single input port: routes an event to its owning node with its index.
    struct Port : Sink<Index> {
        CombineLatestNode& node;
        std::size_t idx;
        Port(CombineLatestNode& n, std::size_t i) : node(n), idx(i) {}
        void push(const Frame<Index>& f) override { node.on_port(idx, f); }
        void flush() override { node.flush_downstream(idx); }
        void on_watermark(Index w) override { node.on_port_watermark(idx, w); }
        // Each port accepts one value per event; output is owned by the parent node.
        std::size_t n_in()  const override { return 1; }
        std::size_t n_out() const override { return 0; }
    };
    friend struct Port;

    screamer::streams::CombineLatest cl_;   // reused operator (no re-derivation)
    Sink<Index>& downstream_;
    std::size_t n_;
    std::vector<Port> ports_;

    // Coalescing buffer: holds the latest aligned row at the current index.
    bool has_buffered_ = false;
    Index buffered_index_{};
    std::vector<double> buffered_row_;

    // Per-port watermark (INT64_MIN = port not yet seen) and a min-heap of frames
    // waiting until the merge is safe to apply them in global index order. The seq
    // field is a monotonic arrival counter: it preserves FIFO order among frames
    // that share the same (index, port) - a single port can carry two events at the
    // same index (duplicate indices), and last-write-wins requires arrival order.
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

    // End-of-input coalescing: which ports have flushed in the current cycle.
    std::vector<bool> flushed_;
    std::size_t flushed_count_ = 0;
};

}} // namespace screamer::dag
#endif
