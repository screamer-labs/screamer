#ifndef SCREAMER_DAG_COMPILED_GRAPH_H
#define SCREAMER_DAG_COMPILED_GRAPH_H

// Design: CompiledGraph stores the GraphSpec and rebuilds the wired push-graph
// fresh inside each run_batch() call. This keeps all wiring logic in one place
// and avoids gather-reattachment machinery. The graph is small; rebuild cost is
// negligible versus the event loop. Chosen for clarity over micro-optimisation.

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

// Owns the GraphSpec; builds and drives a fresh wired push-graph on each
// run_batch() call (rebuild-per-run simplification — see file header).
class CompiledGraph {
public:
    explicit CompiledGraph(GraphSpec spec) : spec_(std::move(spec)) {}

    // in_* are per-input arrays (one entry per input, in signature order).
    // Returns one OutputBuffer per output, in output_ids order.
    std::vector<OutputBuffer> run_batch(
            const std::vector<const std::int64_t*>& in_keys,
            const std::vector<const double*>& in_vals,
            const std::vector<std::size_t>& in_lens) {

        const GraphSpec& spec = spec_;
        std::size_t n          = spec.nodes.size();
        std::size_t num_out    = spec.output_ids.size();
        std::size_t num_in     = spec.input_ids.size();

        // Fix 1: validate caller-supplied input-array counts.
        if (in_keys.size() != num_in)
            throw std::runtime_error(
                "run_batch: expected " + std::to_string(num_in) +
                " input arrays, got " + std::to_string(in_keys.size()));
        if (in_vals.size() != num_in)
            throw std::runtime_error(
                "run_batch: expected " + std::to_string(num_in) +
                " value arrays, got " + std::to_string(in_vals.size()));
        if (in_lens.size() != num_in)
            throw std::runtime_error(
                "run_batch: expected " + std::to_string(num_in) +
                " length arrays, got " + std::to_string(in_lens.size()));

        // --- per-run output buffers -----------------------------------------
        std::vector<OutputBuffer> outputs(num_out);
        std::vector<std::unique_ptr<GatherSink>> gathers;
        gathers.reserve(num_out);
        for (auto& out : outputs)
            gathers.push_back(std::make_unique<GatherSink>(out));

        // --- build adjacency: consumers[i] = (consumer_id, slot) pairs for node i -------
        // Edge-aware: one pair per edge, so a producer appearing at K slots of one
        // consumer correctly contributes K pairs (one per slot), not K² entries.
        std::vector<std::vector<std::pair<std::size_t,std::size_t>>> consumers(n);
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t k = 0; k < spec.nodes[j].inputs.size(); ++k)
                consumers[spec.nodes[j].inputs[k]].push_back({j, k});

        // --- reverse-topological order via Kahn's (producers first → reverse) -
        std::vector<int> in_deg(n, 0);
        for (std::size_t j = 0; j < n; ++j)
            in_deg[j] = static_cast<int>(spec.nodes[j].inputs.size());

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

        // Fix 2: cycle detection — Kahn's sort omits nodes involved in cycles.
        if (topo.size() != spec.nodes.size())
            throw std::runtime_error("compile: graph has a cycle");

        // --- map output_id → which output indices it serves ------------------
        std::vector<std::vector<std::size_t>> node_out_idx(n);
        for (std::size_t o = 0; o < num_out; ++o)
            node_out_idx[spec.output_ids[o]].push_back(o);

        // --- map input node id → its signature index -------------------------
        std::vector<std::size_t> input_sig(n, static_cast<std::size_t>(-1));
        for (std::size_t idx = 0; idx < spec.input_ids.size(); ++idx)
            input_sig[spec.input_ids[idx]] = idx;

        // --- node_input_sink[i](slot) = Sink entry-point for slot `slot` of node i ---
        // Functor: returns the same FunctorNode for any slot (wide edge, slot ignored).
        // CombineLatest: returns &n->port(slot) (each producer wires to its own port).
        // Input nodes have no node_input_sink entry (they are sources, not consumers).
        std::vector<std::function<Sink<std::int64_t>*(std::size_t)>> node_input_sink(n);

        // combine_ptrs[i] is non-null iff node i is a CombineLatestNode.
        // Used to call make_multi_port_sink when one producer feeds multiple slots.
        std::vector<CombineLatestNode<std::int64_t>*> combine_ptrs(n, nullptr);

        // input_sinks[sig_idx] = downstream Sink for input with that index
        std::vector<Sink<std::int64_t>*> input_sinks(num_in, nullptr);

        // ownership of heap-allocated wiring objects (FunctorNodes, CombineLatestNodes,
        // Broadcasts); shared_ptr<void> is used so every type is stored uniformly.
        std::vector<std::shared_ptr<void>> owned;

        // --- wire in reverse-topological order (consumers first) -------------
        for (auto id : topo) {
            const auto& ns = spec.nodes[id];

            // Collect all immediate downstream sinks for this node (edge-aware).
            // Group (c, slot) pairs by consumer ID first so that when one producer
            // feeds the same CombineLatestNode at multiple slots, we use the atomic
            // multi-port sink (emits exactly once per event) rather than separate
            // port pushes (which would emit once per port push after warm-up).
            std::vector<Sink<std::int64_t>*> ds;
            {
                // Build ordered list of (consumer_id, [slots]) preserving insertion order.
                std::vector<std::pair<std::size_t, std::vector<std::size_t>>> groups;
                for (auto [c, slot] : consumers[id]) {
                    if (!node_input_sink[c]) {
                        // Fix 3: Input nodes are the only legitimate case for an
                        // empty sink resolver (they are sources, not consumers).
                        // A non-Input consumer with no resolver is a compiler bug.
                        if (spec.nodes[c].kind != NodeKind::Input)
                            throw std::runtime_error(
                                "compile: internal error, unresolved consumer sink");
                        continue;
                    }
                    bool found = false;
                    for (auto& [gc, gs] : groups) {
                        if (gc == c) { gs.push_back(slot); found = true; break; }
                    }
                    if (!found) groups.push_back({c, {slot}});
                }
                for (auto& [c, slots] : groups) {
                    if (slots.size() == 1) {
                        // Normal single-edge case.
                        ds.push_back(node_input_sink[c](slots[0]));
                    } else if (combine_ptrs[c]) {
                        // Same producer → multiple slots of one CombineLatest:
                        // use atomic multi-port sink to emit exactly once per event.
                        auto ms = std::shared_ptr<Sink<std::int64_t>>(
                            combine_ptrs[c]->make_multi_port_sink(slots));
                        ds.push_back(ms.get());
                        owned.push_back(ms);
                    } else {
                        // Same producer → multiple slots of a Functor (wide edge):
                        // push once per slot (degenerate; Functor ignores slot).
                        for (auto s : slots)
                            ds.push_back(node_input_sink[c](s));
                    }
                }
            }
            for (auto o : node_out_idx[id])
                ds.push_back(gathers[o].get());

            if (ds.empty())
                throw std::runtime_error(
                    "compile: node " + std::to_string(id) + " has no downstream");

            // Fan-out via Broadcast when >1 downstream; direct otherwise.
            Sink<std::int64_t>* downstream;
            if (ds.size() == 1) {
                downstream = ds[0];
            } else {
                auto bcast = std::make_shared<Broadcast<std::int64_t>>();
                for (auto* s : ds) bcast->add(*s);
                downstream = bcast.get();
                owned.push_back(bcast); // keep alive
            }

            switch (ns.kind) {
            case NodeKind::Input:
                input_sinks[input_sig[id]] = downstream;
                break;
            case NodeKind::Functor: {
                ns.op->reset();
                auto fn = std::make_shared<FunctorNode<std::int64_t>>(*ns.op, *downstream);
                // Functor accepts a single wide edge: return the node for any slot.
                node_input_sink[id] = [ptr = fn.get()](std::size_t) -> Sink<std::int64_t>* {
                    return ptr;
                };
                owned.push_back(fn); // keep alive
                break;
            }
            case NodeKind::CombineLatest: {
                auto cn = std::make_shared<CombineLatestNode<std::int64_t>>(
                    ns.inputs.size(), ns.when_all, *downstream);
                // Each producer wires to its own port: return port(slot).
                node_input_sink[id] = [ptr = cn.get()](std::size_t slot) -> Sink<std::int64_t>* {
                    return &ptr->port(slot);
                };
                combine_ptrs[id] = cn.get(); // for multi-slot sink creation
                owned.push_back(cn); // keep alive (ports hold back-reference to node)
                break;
            }
            }
        }

        // --- drive: merge all input streams, route events to their sinks -----
        std::vector<std::unique_ptr<streams::VectorSource<std::int64_t>>> srcs;
        std::vector<streams::Source<std::int64_t>*> child_ptrs;
        srcs.reserve(num_in);
        child_ptrs.reserve(num_in);
        for (std::size_t i = 0; i < num_in; ++i) {
            srcs.push_back(std::make_unique<streams::VectorSource<std::int64_t>>(
                in_keys[i], in_vals[i], in_lens[i]));
            child_ptrs.push_back(srcs.back().get());
        }
        streams::MergeSource<std::int64_t> merge(child_ptrs);

        double one;
        while (auto e = merge.next()) {
            one = e->value;
            Frame<std::int64_t> f{e->key, &one, 1};
            input_sinks[e->source]->push(f);
        }
        for (auto* s : input_sinks) if (s) s->flush();

        return outputs;
    }

private:
    GraphSpec spec_;
};

// compile() is trivial: just wraps the spec. All wiring happens inside run_batch().
inline CompiledGraph compile(const GraphSpec& spec) {
    return CompiledGraph{spec};
}

}} // namespace screamer::dag
#endif
