#ifndef SCREAMER_DAG_COMBINE_LATEST_NODE_H
#define SCREAMER_DAG_COMBINE_LATEST_NODE_H

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>
#include "screamer/dag/frame.h"
#include "screamer/dag/max_pending.h"
#include "screamer/dag/reorder_heap.h"
#include "screamer/dag/resettable.h"
#include "screamer/streams/combine_latest.h"
#include <stdexcept>
#include <string>

namespace screamer { namespace dag {

// Aligning fan-in node. Exposes N input ports, each carrying its own number of
// columns (learned from its first frame); the aligned row concatenates them in
// port order. On any port event it
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
    static constexpr std::size_t DEFAULT_MAX_PENDING = kDefaultMaxPending;

    // widths[i] is the compile-time estimate of port i's columns. A port's real
    // width is fixed by its first frame (see on_port).
    CombineLatestNode(std::vector<std::size_t> widths, bool when_all, Sink<Index>& downstream,
                      std::size_t max_pending = DEFAULT_MAX_PENDING)
        : cl_(widths, when_all), declared_(widths), when_all_(when_all),
          downstream_(downstream), n_(widths.size()),
          max_pending_(max_pending),
          learned_(widths.size(), false),
          pool_(widths.size()), free_slots_(widths.size()),
          wm_(widths.size(), std::numeric_limits<Index>::min()),
          flushed_(widths.size(), false) {
        ports_.reserve(n_);
        for (std::size_t i = 0; i < n_; ++i) ports_.emplace_back(*this, i);
    }

    CombineLatestNode(std::size_t n, bool when_all, Sink<Index>& downstream,
                      std::size_t max_pending = DEFAULT_MAX_PENDING)
        : CombineLatestNode(std::vector<std::size_t>(n, 1), when_all, downstream, max_pending) {}

    // Non-movable/copyable: the ports hold a reference back to this node.
    CombineLatestNode(const CombineLatestNode&) = delete;
    CombineLatestNode& operator=(const CombineLatestNode&) = delete;
    CombineLatestNode(CombineLatestNode&&) = delete;
    CombineLatestNode& operator=(CombineLatestNode&&) = delete;

    Sink<Index>& port(std::size_t i) { return ports_[i]; }

    void reset() override {
        cl_.reset(declared_);                       // the declared widths, no allocation
        std::fill(learned_.begin(), learned_.end(), false);
        emitted_any_ = false;
        for (std::size_t p = 0; p < n_; ++p) { pool_[p].clear(); free_slots_[p].clear(); }
        has_buffered_ = false;
        std::fill(flushed_.begin(), flushed_.end(), false);
        flushed_count_ = 0;
        std::fill(wm_.begin(), wm_.end(), std::numeric_limits<Index>::min());
        pending_.clear();
        next_seq_ = 0;
        last_forwarded_wm_ = std::numeric_limits<Index>::min();
    }

private:
    // Unchanged as-of + coalescing logic, now fed in guaranteed global index order
    // by release(). Takes an explicit index and the port's values instead of a Frame.
    //
    // The buffered row is the operator's latest row itself: once a row is due,
    // every later event makes one due too (when_all has seen every input), so
    // while a row is buffered no event changes the latest row without
    // re-buffering it. An event at a new index therefore emits the buffered
    // row before it updates the latest one, and no copy is kept.
    void apply_ordered(std::size_t i, Index ev_index, const double* values) {
        if (has_buffered_ && ev_index != buffered_index_) emit_buffered();
        if (cl_.on_event(static_cast<std::uint32_t>(i), values)) {
            emitted_any_ = true;
            // Buffer the latest row at this index (overwrites if same index).
            buffered_index_ = ev_index;
            has_buffered_ = true;
        }
    }

    // Emits the buffered (settled) row: the latest row (see apply_ordered).
    void emit_buffered() {
        const std::vector<double>& row = cl_.latest();
        downstream_.push(Frame<Index>{buffered_index_, row.data(), row.size()});
        has_buffered_ = false;
    }

    void on_port(std::size_t i, const Frame<Index>& f) {
        if (!learned_[i]) {
            // A port's width is fixed by its first event. Before that, a row
            // emitted under emit="on_any" used the compile-time estimate.
            learned_[i] = true;
            if (f.width != cl_.width(i)) {
                if (emitted_any_)
                    throw std::invalid_argument(
                        "CombineLatest: input " + std::to_string(i) + " has " +
                        std::to_string(f.width) + "-column events, but rows were already "
                        "emitted with " + std::to_string(cl_.width(i)) + " column(s) for it. "
                        "With emit=\"on_any\" inside a Pipeline, a row can be due before every "
                        "input has arrived; use emit=\"when_all\".");
                cl_.set_width(i, f.width);
            }
        } else if (f.width != cl_.width(i)) {
            throw std::invalid_argument(
                "CombineLatest: input " + std::to_string(i) + " changed from " +
                std::to_string(cl_.width(i)) + " to " + std::to_string(f.width) + " columns");
        }
        if (wm_[i] < f.index) wm_[i] = f.index;   // a data frame advances the port watermark
        Pending p{f.index, static_cast<std::uint32_t>(i), next_seq_++, {}};
        // A one-column frame keeps its value in the entry; a wider one in its
        // port's pool.
        if (f.width == 1) p.value = f.values[0];
        else p.slot = store(i, f.values, f.width);
        pending_.push(p);
        release();   // first: the event that releases a full buffer does not raise
        if (pending_.size() > max_pending_)
            throw std::runtime_error(
                "CombineLatest reorder buffer overflow: an input has not advanced; "
                "a stream is stalled. Call advance(now) or check the feed.");
    }

    // Copy a pending frame's values into a free slot of its port's pool.
    // A zero-width port holds no values, so it needs no slot.
    std::size_t store(std::size_t port, const double* values, std::size_t w) {
        if (w == 0) return 0;
        std::size_t slot;
        if (!free_slots_[port].empty()) { slot = free_slots_[port].back(); free_slots_[port].pop_back(); }
        else { slot = pool_[port].size() / w; pool_[port].resize(pool_[port].size() + w); }
        std::copy(values, values + w, pool_[port].begin() + slot * w);
        return slot;
    }

    void on_port_watermark(std::size_t i, Index w) {
        if (wm_[i] < w) wm_[i] = w;
        release();
    }

    // Apply every pending frame the merge can now prove is safe (index <= the
    // minimum per-port watermark), in global index order, then flush any
    // coalescing buffer whose index is strictly below the settled watermark
    // (the next event must arrive at a higher index, so the buffer is done),
    // then forward the settled watermark downstream.
    void release() {
        Index low = wm_[0];
        for (std::size_t j = 1; j < n_; ++j) low = std::min(low, wm_[j]);
        while (!pending_.empty() && pending_.top().index <= low) {
            Pending p = pending_.top(); pending_.pop();
            const std::size_t w = cl_.width(p.port);
            if (w == 1) {
                apply_ordered(p.port, p.index, &p.value);
                continue;
            }
            apply_ordered(p.port, p.index, pool_[p.port].data() + p.slot * w);
            if (w) free_slots_[p.port].push_back(p.slot);
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
        if (has_buffered_) emit_buffered();
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
        // n_in() is nominal: a port takes any width, fixed by its first frame.
        std::size_t n_in()  const override { return 1; }
        std::size_t n_out() const override { return 0; }
    };
    friend struct Port;

    screamer::streams::CombineLatest cl_;   // reused operator (no re-derivation)
    std::vector<std::size_t> declared_;     // compile-time width estimates, restored by reset()
    bool when_all_;
    Sink<Index>& downstream_;
    std::size_t n_;
    std::size_t max_pending_;
    std::vector<Port> ports_;

    // Coalescing buffer: a row is due at buffered_index_; the row is the
    // operator's latest row (see apply_ordered).
    bool has_buffered_ = false;
    Index buffered_index_{};

    // Width learning: learned_[i] once port i's first frame fixed its width;
    // emitted_any_ once a row has gone out (after which widths cannot change).
    std::vector<bool> learned_;
    bool emitted_any_ = false;

    // Per-port value pool for pending frames wider than one column: a frame's
    // values live in slot `slot` of its port's pool (width cl_.width(port));
    // applied slots are recycled through free_slots_, so steady state
    // allocates nothing. A one-column frame holds its value in its Pending.
    std::vector<std::vector<double>> pool_;
    std::vector<std::vector<std::size_t>> free_slots_;

    // Per-port watermark (INT64_MIN = port not yet seen) and a min-heap of frames
    // waiting until the merge is safe to apply them in global index order. The seq
    // field is a monotonic arrival counter: it preserves FIFO order among frames
    // that share the same (index, port) - a single port can carry two events at the
    // same index (duplicate indices), and last-write-wins requires arrival order.
    std::vector<Index> wm_;
    struct Pending {
        Index index;
        std::uint32_t port;
        std::uint64_t seq;
        union {
            double value;       // a one-column frame's value
            std::size_t slot;   // a wider frame's slot in pool_[port]
        };
    };
    struct PendGreater {
        bool operator()(const Pending& a, const Pending& b) const {
            if (a.index != b.index) return a.index > b.index;  // min-heap on index
            if (a.port  != b.port)  return a.port  > b.port;   // tie-break by port
            return a.seq > b.seq;                              // then FIFO by arrival
        }
    };
    ReorderHeap<Pending, PendGreater> pending_;
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
