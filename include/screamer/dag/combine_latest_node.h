#ifndef SCREAMER_DAG_COMBINE_LATEST_NODE_H
#define SCREAMER_DAG_COMBINE_LATEST_NODE_H

#include <algorithm>
#include <cassert>
#include <cstddef>
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
          buffered_row_(n, 0.0), flushed_(n, false) {
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
    }

private:
    void on_port(std::size_t i, const Frame<Index>& f) {
        assert(f.width == 1);
        if (cl_.on_event(static_cast<std::uint32_t>(i), f.values[0])) {
            const Index ev_index = f.index;
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

    // Called once per input port at end-of-input. Each producer pushes its final
    // (same-index) event just BEFORE flushing its own port, so flushing eagerly on
    // the first port would emit the final row with the other ports' values still
    // stale (and again on the next port) — duplicating the shared final index. To
    // coalesce (mirroring the mid-stream "emit on index advance" logic), we wait
    // until EVERY port has flushed before emitting the settled final row once and
    // propagating the flush downstream. A per-port bitmask dedups repeat flushes of
    // one port (a producer with multiple upstreams flushes it more than once).
    void flush_downstream(std::size_t i) {
        if (!flushed_[i]) { flushed_[i] = true; ++flushed_count_; }
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

    // End-of-input coalescing: which ports have flushed in the current cycle.
    std::vector<bool> flushed_;
    std::size_t flushed_count_ = 0;
};

}} // namespace screamer::dag
#endif
