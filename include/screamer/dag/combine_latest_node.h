#ifndef SCREAMER_DAG_COMBINE_LATEST_NODE_H
#define SCREAMER_DAG_COMBINE_LATEST_NODE_H

#include <cassert>
#include <cstddef>
#include <vector>
#include "screamer/dag/frame.h"
#include "screamer/streams/combine_latest.h"

namespace screamer { namespace dag {

// Aligning fan-in node. Exposes N single-value input ports; on any port event it
// updates the reused CombineLatest operator and, when it fires (per when_all),
// buffers the latest aligned row. Coalescing: one frame is emitted per DISTINCT
// index (same-index events update the buffer; the frame is pushed when the index
// advances). flush() emits the final buffered frame (if any) and is idempotent
// (clears buffer after emitting so repeat calls are no-ops).
template <class Index>
class CombineLatestNode {
public:
    CombineLatestNode(std::size_t n, bool when_all, Sink<Index>& downstream)
        : cl_(n, when_all), downstream_(downstream), n_(n),
          buffered_row_(n, 0.0) {
        ports_.reserve(n);
        for (std::size_t i = 0; i < n; ++i) ports_.emplace_back(*this, i);
    }

    // Non-movable/copyable: the ports hold a reference back to this node.
    CombineLatestNode(const CombineLatestNode&) = delete;
    CombineLatestNode& operator=(const CombineLatestNode&) = delete;
    CombineLatestNode(CombineLatestNode&&) = delete;
    CombineLatestNode& operator=(CombineLatestNode&&) = delete;

    Sink<Index>& port(std::size_t i) { return ports_[i]; }

    void reset() {
        cl_.reset();
        has_buffered_ = false;
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

    void flush_downstream() {
        // Emit the buffered final frame exactly once (idempotent).
        if (has_buffered_) {
            downstream_.push(Frame<Index>{buffered_index_,
                                          buffered_row_.data(), n_});
            has_buffered_ = false;
        }
        downstream_.flush();
    }

    // A single input port: routes an event to its owning node with its index.
    struct Port : Sink<Index> {
        CombineLatestNode& node;
        std::size_t idx;
        Port(CombineLatestNode& n, std::size_t i) : node(n), idx(i) {}
        void push(const Frame<Index>& f) override { node.on_port(idx, f); }
        void flush() override { node.flush_downstream(); }
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
};

}} // namespace screamer::dag
#endif
