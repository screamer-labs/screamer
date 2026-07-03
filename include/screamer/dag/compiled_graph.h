#ifndef SCREAMER_DAG_COMPILED_GRAPH_H
#define SCREAMER_DAG_COMPILED_GRAPH_H

// Design: CompiledGraph wires the push-graph ONCE in the constructor (compile
// time). run_batch() resets all stateful ops/combine nodes, drives the merged
// input stream, then returns a copy of the persistent output buffers. Wiring
// cost is paid once; per-batch cost is pure event processing. Zero per-event
// allocation is preserved.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>
#include "screamer/dag/broadcast.h"
#include "screamer/dag/combine_latest_node.h"
#include "screamer/dag/frame.h"
#include "screamer/dag/functor_node.h"
#include "screamer/dag/graph.h"
#include "screamer/streams/merge_source.h"
#include "screamer/streams/vector_source.h"

namespace screamer { namespace dag {

// One output stream gathered during a batch run.
struct OutputBuffer {
    std::vector<std::int64_t> keys;
    std::vector<double> values;   // row-major, width columns per row
    std::size_t width = 1;
};

// Terminal sink: appends every received frame into an OutputBuffer.
class GatherSink : public Sink<std::int64_t> {
public:
    explicit GatherSink(OutputBuffer& buf) : buf_(buf) {}
    void push(const Frame<std::int64_t>& f) override {
        buf_.width = f.width;
        buf_.keys.push_back(f.key);
        buf_.values.insert(buf_.values.end(), f.values, f.values + f.width);
    }
private:
    OutputBuffer& buf_;
};

// Owns the GraphSpec; builds the wired push-graph once in the constructor.
// run_batch() resets state, drives the event loop, then returns the gathered
// outputs. Batch output is byte-identical to the previous rebuild-per-run
// implementation.
class CompiledGraph {
public:
    // Non-copyable: nodes/gather-sinks hold raw pointers into this object's own
    // members (outputs_, owned_). Move IS safe — std::vector move keeps element
    // addresses (heap) unchanged and shared_ptr move doesn't relocate the pointee;
    // compile() returns a prvalue so C++17 elides the move entirely.
    CompiledGraph(const CompiledGraph&) = delete;
    CompiledGraph& operator=(const CompiledGraph&) = delete;

    explicit CompiledGraph(GraphSpec spec) : spec_(std::move(spec)) {
        const GraphSpec& s = spec_;
        std::size_t n       = s.nodes.size();
        std::size_t num_out = s.output_ids.size();
        num_in_             = s.input_ids.size();

        // Pre-allocate output buffers before creating GatherSinks so that
        // the vector does not reallocate while sinks hold references into it.
        outputs_.resize(num_out);

        // Create persistent GatherSinks pointing at outputs_.
        std::vector<GatherSink*> gather_ptrs;
        gather_ptrs.reserve(num_out);
        for (std::size_t o = 0; o < num_out; ++o) {
            auto g = std::make_shared<GatherSink>(outputs_[o]);
            gather_ptrs.push_back(g.get());
            owned_.push_back(g);
        }

        // Build adjacency: consumers[i] = (consumer_id, slot) pairs for node i.
        // Edge-aware: one pair per edge so a producer at K slots of one consumer
        // contributes K pairs (one per slot), not K² entries.
        std::vector<std::vector<std::pair<std::size_t,std::size_t>>> consumers(n);
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t k = 0; k < s.nodes[j].inputs.size(); ++k)
                consumers[s.nodes[j].inputs[k]].push_back({j, k});

        // Reverse-topological order via Kahn's (producers first → reverse).
        std::vector<int> in_deg(n, 0);
        for (std::size_t j = 0; j < n; ++j)
            in_deg[j] = static_cast<int>(s.nodes[j].inputs.size());

        std::vector<std::size_t> topo;
        topo.reserve(n);
        std::queue<std::size_t> q;
        for (std::size_t i = 0; i < n; ++i) if (in_deg[i] == 0) q.push(i);
        while (!q.empty()) {
            auto id = q.front(); q.pop();
            topo.push_back(id);
            for (auto [c, slot] : consumers[id]) if (--in_deg[c] == 0) q.push(c);
        }
        std::reverse(topo.begin(), topo.end()); // consumers first

        // Cycle detection — Kahn's sort omits nodes involved in cycles.
        if (topo.size() != s.nodes.size())
            throw std::runtime_error("compile: graph has a cycle");

        // Map output_id → which output indices it serves.
        std::vector<std::vector<std::size_t>> node_out_idx(n);
        for (std::size_t o = 0; o < num_out; ++o)
            node_out_idx[s.output_ids[o]].push_back(o);

        // Map input node id → its signature index.
        std::vector<std::size_t> input_sig(n, static_cast<std::size_t>(-1));
        for (std::size_t idx = 0; idx < s.input_ids.size(); ++idx)
            input_sig[s.input_ids[idx]] = idx;

        // node_input_sink[i](slot) = Sink entry-point for slot `slot` of node i.
        // Functor: returns the same FunctorNode for any slot (wide edge).
        // CombineLatest: returns &n->port(slot) (each producer to its own port).
        // Input nodes have no entry (they are sources, not consumers).
        std::vector<std::function<Sink<std::int64_t>*(std::size_t)>> node_input_sink(n);

        // input_sinks_[sig_idx] = downstream Sink for that input.
        input_sinks_.assign(num_in_, nullptr);

        // Wire in reverse-topological order (consumers first).
        for (auto id : topo) {
            const auto& ns = s.nodes[id];

            // Collect all immediate downstream sinks for this node.
            std::vector<Sink<std::int64_t>*> ds;
            for (auto [c, slot] : consumers[id]) {
                if (!node_input_sink[c]) {
                    // Input nodes are the only legitimate case for an empty
                    // sink resolver (they are sources, not consumers).
                    if (s.nodes[c].kind != NodeKind::Input)
                        throw std::runtime_error(
                            "compile: internal error, unresolved consumer sink");
                    continue;
                }
                ds.push_back(node_input_sink[c](slot));
            }
            for (auto o : node_out_idx[id])
                ds.push_back(gather_ptrs[o]);

            if (ds.empty())
                throw std::runtime_error(
                    "compile: node " + std::to_string(id) + " has no downstream");

            // Fan-out via Broadcast when >1 downstream; direct otherwise.
            Sink<std::int64_t>* downstream;
            if (ds.size() == 1) {
                downstream = ds[0];
            } else {
                auto bcast = std::make_shared<Broadcast<std::int64_t>>();
                for (auto* sink : ds) bcast->add(*sink);
                downstream = bcast.get();
                owned_.push_back(bcast);
            }

            switch (ns.kind) {
            case NodeKind::Input:
                input_sinks_[input_sig[id]] = downstream;
                break;
            case NodeKind::Functor: {
                // Record op for reset(); do NOT reset here — reset() handles it.
                reset_ops_.push_back(ns.op);
                auto fn = std::make_shared<FunctorNode<std::int64_t>>(*ns.op, *downstream);
                node_input_sink[id] = [ptr = fn.get()](std::size_t) -> Sink<std::int64_t>* {
                    return ptr;
                };
                owned_.push_back(fn);
                break;
            }
            case NodeKind::CombineLatest: {
                auto cn = std::make_shared<CombineLatestNode<std::int64_t>>(
                    ns.inputs.size(), ns.when_all, *downstream);
                reset_combines_.push_back(cn.get());
                node_input_sink[id] = [ptr = cn.get()](std::size_t slot) -> Sink<std::int64_t>* {
                    return &ptr->port(slot);
                };
                owned_.push_back(cn);
                break;
            }
            }
        }
    }

    // Resets all stateful ops, combine nodes, and output buffers. Called at the
    // start of every run_batch so each batch sees a clean initial state.
    void reset() {
        for (auto* op : reset_ops_)      op->reset();
        for (auto* c  : reset_combines_) c->reset();
        for (auto& b  : outputs_)        { b.keys.clear(); b.values.clear(); b.width = 1; }
    }

    // in_* are per-input arrays (one entry per input, in signature order).
    // Returns one OutputBuffer per output, in output_ids order.
    std::vector<OutputBuffer> run_batch(
            const std::vector<const std::int64_t*>& in_keys,
            const std::vector<const double*>& in_vals,
            const std::vector<std::size_t>& in_lens) {

        // Validate caller-supplied input-array counts.
        if (in_keys.size() != num_in_)
            throw std::runtime_error(
                "run_batch: expected " + std::to_string(num_in_) +
                " input arrays, got " + std::to_string(in_keys.size()));
        if (in_vals.size() != num_in_)
            throw std::runtime_error(
                "run_batch: expected " + std::to_string(num_in_) +
                " value arrays, got " + std::to_string(in_vals.size()));
        if (in_lens.size() != num_in_)
            throw std::runtime_error(
                "run_batch: expected " + std::to_string(num_in_) +
                " length arrays, got " + std::to_string(in_lens.size()));

        // Reset all stateful state and clear output buffers.
        reset();

        // Build VectorSources and drive via MergeSource.
        std::vector<std::unique_ptr<streams::VectorSource<std::int64_t>>> srcs;
        std::vector<streams::Source<std::int64_t>*> child_ptrs;
        srcs.reserve(num_in_);
        child_ptrs.reserve(num_in_);
        for (std::size_t i = 0; i < num_in_; ++i) {
            srcs.push_back(std::make_unique<streams::VectorSource<std::int64_t>>(
                in_keys[i], in_vals[i], in_lens[i]));
            child_ptrs.push_back(srcs.back().get());
        }
        streams::MergeSource<std::int64_t> merge(child_ptrs);

        double one;
        while (auto e = merge.next()) {
            one = e->value;
            Frame<std::int64_t> f{e->key, &one, 1};
            input_sinks_[e->source]->push(f);
        }
        for (auto* s : input_sinks_) if (s) s->flush();

        return outputs_;
    }

private:
    GraphSpec spec_;

    // Wiring (built once in constructor, persistent across all run_batch calls).
    std::size_t num_in_ = 0;
    std::vector<std::shared_ptr<void>>            owned_;          // all heap nodes/broadcasts
    std::vector<Sink<std::int64_t>*>              input_sinks_;    // per input signature index
    std::vector<EvalOp*>                          reset_ops_;      // functor ops to reset
    std::vector<CombineLatestNode<std::int64_t>*> reset_combines_; // combine nodes to reset
    std::vector<OutputBuffer>                     outputs_;        // persistent output buffers
};

// compile() wraps the spec into a CompiledGraph (wiring happens in constructor).
inline CompiledGraph compile(const GraphSpec& spec) {
    return CompiledGraph{spec};
}

}} // namespace screamer::dag
#endif
