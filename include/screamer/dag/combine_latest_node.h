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
// emits ONE width-N frame carrying the aligned latest values. The emitted frame
// points at the operator's own latest() buffer (stable during the push).
template <class Key>
class CombineLatestNode {
public:
    CombineLatestNode(std::size_t n, bool when_all, Sink<Key>& downstream)
        : cl_(n, when_all), downstream_(downstream), n_(n) {
        ports_.reserve(n);
        for (std::size_t i = 0; i < n; ++i) ports_.emplace_back(*this, i);
    }

    // Non-movable/copyable: the ports hold a reference back to this node.
    CombineLatestNode(const CombineLatestNode&) = delete;
    CombineLatestNode& operator=(const CombineLatestNode&) = delete;
    CombineLatestNode(CombineLatestNode&&) = delete;
    CombineLatestNode& operator=(CombineLatestNode&&) = delete;

    Sink<Key>& port(std::size_t i) { return ports_[i]; }
    void reset() { cl_.reset(); }

private:
    void on_port(std::size_t i, const Frame<Key>& f) {
        assert(f.width == 1);
        if (cl_.on_event(static_cast<std::uint32_t>(i), f.values[0])) {
            const std::vector<double>& row = cl_.latest();
            downstream_.push(Frame<Key>{f.key, row.data(), n_});
        }
    }

    // A single input port: routes an event to its owning node with its index.
    struct Port : Sink<Key> {
        CombineLatestNode& node;
        std::size_t idx;
        Port(CombineLatestNode& n, std::size_t i) : node(n), idx(i) {}
        void push(const Frame<Key>& f) override { node.on_port(idx, f); }
    };
    friend struct Port;

    screamer::streams::CombineLatest cl_;   // reused operator (no re-derivation)
    Sink<Key>& downstream_;
    std::size_t n_;
    std::vector<Port> ports_;
};

}} // namespace screamer::dag
#endif
