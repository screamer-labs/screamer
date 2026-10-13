#ifndef SCREAMER_DAG_ORDERED_MERGE_NODE_H
#define SCREAMER_DAG_ORDERED_MERGE_NODE_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>
#include "screamer/dag/frame.h"
#include "screamer/dag/resettable.h"

namespace screamer { namespace dag {

// The input of a node with several producers (an operator or Dropna given
// more than one stream). Each producer is a separate stream of events; the
// node takes them as one stream in index order, events at one index in port
// order (the order of the node's inputs), and on one port in arrival order.
// That order does not depend on how the producers' events interleave, so a
// batch run, the lazy driver and a live session (where inputs may run ahead
// of each other) feed the node the same events in the same order.
//
// An event at index t on port p goes on once no event can still come before
// it: every port q is known to send nothing below t, and nothing at t when
// q < p. A port knows this from its own events and watermarks (a stream's
// index never decreases) and from flush(); while one merge feeds the graph in
// index order (set_merge_order: a batch run, the lazy driver), an event at t
// on a port that is never ahead of the merge also bounds every port that is
// in order with it (as for a clock Resample, resample_node.h). Events wait in
// a min-heap until then, at most max_pending of them.
//
// The node forwards the lowest index it may still emit as its watermark.
// Non-copyable and non-movable: the ports hold a reference to the node.
template <class Index>
class OrderedMergeNode : public Resettable {
public:
    OrderedMergeNode(std::size_t n, Sink<Index>& downstream, std::size_t max_pending)
        : downstream_(downstream), n_(n), max_pending_(max_pending),
          front_(n, kMin), in_order_(n, 0), never_ahead_(n, 0), flushed_(n, 0),
          width_(n, kUnset), pool_(n), free_slots_(n) {
        ports_.reserve(n);
        for (std::size_t i = 0; i < n; ++i) ports_.emplace_back(*this, i);
    }

    OrderedMergeNode(const OrderedMergeNode&) = delete;
    OrderedMergeNode& operator=(const OrderedMergeNode&) = delete;
    OrderedMergeNode(OrderedMergeNode&&) = delete;
    OrderedMergeNode& operator=(OrderedMergeNode&&) = delete;

    Sink<Index>& port(std::size_t i) { return ports_[i]; }

    // How port i's producer relates to the merge of the graph's inputs
    // (CompiledGraph::merge_relation): in order, and never ahead.
    void set_port_relation(std::size_t i, bool in_order, bool never_ahead) {
        in_order_[i] = in_order;
        never_ahead_[i] = never_ahead;
    }

    // Whether one merge feeds the graph's inputs in index order until the
    // next reset().
    void set_merge_order(bool on) { merge_order_ = on; }

    void reset() override {
        std::fill(front_.begin(), front_.end(), kMin);
        std::fill(flushed_.begin(), flushed_.end(), 0);
        std::fill(width_.begin(), width_.end(), kUnset);
        for (std::size_t p = 0; p < n_; ++p) { pool_[p].clear(); free_slots_[p].clear(); }
        flushed_count_ = 0;
        has_gfront_ = false;
        gfront_ = kMin;
        pending_ = Heap();
        next_seq_ = 0;
        last_forwarded_ = kMin;
    }

private:
    static constexpr Index kMin = std::numeric_limits<Index>::min();
    static constexpr Index kMax = std::numeric_limits<Index>::max();
    static constexpr std::size_t kUnset = static_cast<std::size_t>(-1);

    // Every later event on port q has an index at or above this.
    Index front(std::size_t q) const {
        const Index f = front_[q];
        if (merge_order_ && in_order_[q] && has_gfront_ && gfront_ > f) return gfront_;
        return f;
    }

    // No event can still come before (t, port p).
    bool ready(Index t, std::size_t p) const {
        for (std::size_t q = 0; q < n_; ++q) {
            if (q == p) continue;
            const Index f = front(q);
            if (q < p ? !(f > t) : f < t) return false;
        }
        return true;
    }

    void note(std::size_t i, Index k) {
        if (front_[i] < k) front_[i] = k;
        if (merge_order_ && never_ahead_[i] && (!has_gfront_ || k > gfront_)) {
            gfront_ = k;
            has_gfront_ = true;
        }
    }

    void on_port(std::size_t i, const Frame<Index>& f) {
        note(i, f.index);
        if (pending_.empty() && ready(f.index, i)) {
            downstream_.push(f);                 // nothing waits: no copy
            forward_watermark();
            return;
        }
        if (width_[i] == kUnset) width_[i] = f.width;
        else if (width_[i] != f.width)
            throw std::invalid_argument(
                "input " + std::to_string(i) + " of an operator changed from " +
                std::to_string(width_[i]) + " to " + std::to_string(f.width) + " columns");
        Pending p{f.index, static_cast<std::uint32_t>(i), next_seq_++, {}};
        if (f.width == 1) p.value = f.values[0];       // one column: kept in the entry
        else p.slot = store(i, f.values, f.width);
        pending_.push(p);
        if (pending_.size() > max_pending_)
            throw std::runtime_error(
                "an operator's reorder buffer overflowed: one of its inputs has not "
                "advanced; a stream is stalled. Call advance(now) or check the feed.");
        release();
    }

    // Copies a waiting event's values into its port's pool (one width per
    // port); applied slots are reused, so steady state allocates nothing.
    std::size_t store(std::size_t port, const double* values, std::size_t w) {
        if (w == 0) return 0;
        std::size_t slot;
        if (!free_slots_[port].empty()) { slot = free_slots_[port].back(); free_slots_[port].pop_back(); }
        else { slot = pool_[port].size() / w; pool_[port].resize(pool_[port].size() + w); }
        std::copy(values, values + w, pool_[port].begin() + static_cast<std::ptrdiff_t>(slot * w));
        return slot;
    }

    void on_port_watermark(std::size_t i, Index w) {
        if (front_[i] < w) front_[i] = w;
        release();
    }

    // Sends every waiting event that is ready, in order, then the watermark.
    void release() {
        while (!pending_.empty()) {
            const Pending p = pending_.top();
            if (!ready(p.index, p.port)) break;
            pending_.pop();
            const std::size_t w = width_[p.port];
            if (w == 1) {
                downstream_.push(Frame<Index>{p.index, &p.value, 1});
                continue;
            }
            const double* row = w ? pool_[p.port].data() + p.slot * w : nullptr;
            downstream_.push(Frame<Index>{p.index, row, w});
            if (w) free_slots_[p.port].push_back(p.slot);
        }
        forward_watermark();
    }

    // The lowest index the node may still emit: the first waiting event, or
    // the lowest port front.
    void forward_watermark() {
        Index low = front(0);
        for (std::size_t q = 1; q < n_; ++q) low = std::min(low, front(q));
        if (!pending_.empty() && pending_.top().index < low) low = pending_.top().index;
        if (low != kMin && low > last_forwarded_) {
            last_forwarded_ = low;
            downstream_.on_watermark(low);
        }
    }

    // A port ends: it sends nothing more. Once every port has ended, every
    // waiting event goes on and the flush goes downstream, once.
    void flush_port(std::size_t i) {
        front_[i] = kMax;
        if (!flushed_[i]) { flushed_[i] = 1; ++flushed_count_; }
        release();
        if (flushed_count_ < n_) return;
        downstream_.flush();
        std::fill(flushed_.begin(), flushed_.end(), 0);
        flushed_count_ = 0;
    }

    struct Port : Sink<Index> {
        OrderedMergeNode& node;
        std::size_t idx;
        Port(OrderedMergeNode& n, std::size_t i) : node(n), idx(i) {}
        void push(const Frame<Index>& f) override { node.on_port(idx, f); }
        void flush() override { node.flush_port(idx); }
        void on_watermark(Index w) override { node.on_port_watermark(idx, w); }
    };
    friend struct Port;

    struct Pending {
        Index index;
        std::uint32_t port;
        std::uint64_t seq;
        union {
            double value;       // a one-column event's value
            std::size_t slot;   // a wider event's slot in pool_[port]
        };
    };
    struct PendGreater {
        bool operator()(const Pending& a, const Pending& b) const {
            if (a.index != b.index) return a.index > b.index;
            if (a.port != b.port) return a.port > b.port;
            return a.seq > b.seq;
        }
    };
    using Heap = std::priority_queue<Pending, std::vector<Pending>, PendGreater>;

    Sink<Index>& downstream_;
    std::size_t n_;
    std::size_t max_pending_;
    std::vector<Port> ports_;
    std::vector<Index> front_;                  // per port: every later event is at or above it
    std::vector<unsigned char> in_order_, never_ahead_;
    std::vector<unsigned char> flushed_;
    std::size_t flushed_count_ = 0;
    bool merge_order_ = false;
    bool has_gfront_ = false;
    Index gfront_ = kMin;                       // highest index on a never-ahead port
    std::vector<std::size_t> width_;            // per port, kUnset until an event waits
    std::vector<std::vector<double>> pool_;     // per port: waiting events' values
    std::vector<std::vector<std::size_t>> free_slots_;
    Heap pending_;
    std::uint64_t next_seq_ = 0;
    Index last_forwarded_ = kMin;
};

}} // namespace screamer::dag
#endif
