#ifndef SCREAMER_DAG_COMPILED_GRAPH_H
#define SCREAMER_DAG_COMPILED_GRAPH_H

// Design: CompiledGraph wires the push-graph ONCE in the constructor (compile
// time). run_batch() resets all stateful nodes via a single polymorphic pass,
// drives the merged input stream, then returns a copy of the persistent output
// buffers. Wiring cost is paid once; per-batch cost is pure event processing.
// Zero per-event allocation is preserved.

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
#include "screamer/dag/checked_index.h"
#include "screamer/dag/combine_latest_node.h"
#include "screamer/dag/dropna_node.h"
#include "screamer/dag/filter_node.h"
#include "screamer/dag/select_node.h"
#include "screamer/dag/delay_node.h"
#include "screamer/dag/forward_asof_node.h"
#include "screamer/dag/lag_rows_node.h"
#include "screamer/dag/lead_rows_node.h"
#include "screamer/dag/ordered_merge_node.h"
#include "screamer/dag/resample_node.h"
#include "screamer/dag/resample_generic_node.h"
#include "screamer/dag/frame.h"
#include "screamer/dag/functor_node.h"
#include "screamer/dag/graph.h"
#include "screamer/dag/output_buffer.h"
#include "screamer/dag/output_join.h"
#include "screamer/dag/resettable.h"
#include "screamer/streams/merge_source.h"
#include "screamer/streams/vector_source.h"
#include <utility>

namespace screamer { namespace dag {


// Terminal sink: appends every received frame into an OutputBuffer.
class GatherSink : public Sink<std::int64_t> {
public:
    explicit GatherSink(OutputBuffer& buf) : buf_(buf) {}
    void push(const Frame<std::int64_t>& f) override {
        // One width per stream: a frame of another width after rows are
        // stored would leave the rows and the width disagreeing (a later
        // read of rows * width values would run past the buffer).
        if (!buf_.indices.empty() && buf_.width != f.width)
            throw std::invalid_argument(
                "an output received a frame of " + std::to_string(f.width) +
                " values after frames of " + std::to_string(buf_.width));
        buf_.width = f.width;
        buf_.indices.push_back(f.index);
        if (f.width == 1) buf_.values.push_back(f.values[0]);   // no range insert per value
        else buf_.values.insert(buf_.values.end(), f.values, f.values + f.width);
    }
    // The output's watermark: the output join settles on it (output_join.h).
    void on_watermark(std::int64_t w) override {
        if (!buf_.has_watermark || w > buf_.watermark) {
            buf_.watermark = w;
            buf_.has_watermark = true;
        }
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
    // A batch run of at most this many inputs merges them by a scan
    // (run_batch_ref), more by streams::MergeSource's heap.
    static constexpr std::size_t kScanMergeInputs = 8;

    // Non-copyable: nodes/gather-sinks hold raw pointers into this object's own
    // members (outputs_, owned_). Move IS safe - std::vector move keeps element
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
        topo_producers_first_ = topo;           // producers first, kept for width checks
        std::reverse(topo.begin(), topo.end()); // consumers first

        // Cycle detection - Kahn's sort omits nodes involved in cycles.
        if (topo.size() != s.nodes.size())
            throw std::runtime_error("compile: graph has a cycle");

        // Width of every node's emitted frame. Producers-first (reverse of the
        // consumers-first topo) so a pass-through node can read its input's width.
        std::vector<std::size_t> node_width(n, 1);
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            std::size_t id = *it;
            const auto& nd = s.nodes[id];
            switch (nd.kind) {
            case NodeKind::Input:         node_width[id] = 1; break;
            case NodeKind::Functor:       node_width[id] = nd.op->n_out(); break;
            case NodeKind::CombineLatest: {
                std::size_t w = 0;
                for (auto in : nd.inputs) w += node_width[in];
                node_width[id] = w;
                break;
            }
            case NodeKind::DropNa:        node_width[id] = node_width[nd.inputs[0]]; break;
            case NodeKind::Select:        node_width[id] = nd.columns.size(); break;
            case NodeKind::Resample: {
                if (nd.resample.mode == ResampleMode::ByCumulative) {
                    // ByCumulative: output width is determined by the agg (single-col
                    // plan or builtin agg). The input is always width-2 (value+driver)
                    // but the output width is from the value reducer only.
                    node_width[id] = resample_output_width(nd.resample);
                } else if (nd.resample.mode == ResampleMode::ByClock) {
                    // ByClock: output width is determined by the agg. The input is
                    // width-(N+1) [value_cols..., is_clock] but output is the reduced
                    // value width. For OhlcvBars, the value width is input_width-1.
                    if (nd.resample.agg == ResampleAgg::OhlcvBars && nd.resample.plan.empty()) {
                        // Dynamic: output width = input width - 1 (subtract the clock column).
                        std::size_t inp_w = nd.inputs.empty() ? 2u : node_width[nd.inputs[0]];
                        node_width[id] = inp_w > 1 ? inp_w - 1 : 1;
                    } else {
                        node_width[id] = resample_output_width(nd.resample);
                    }
                } else if (nd.resample.agg == ResampleAgg::OhlcvBars && nd.resample.plan.empty()) {
                    // Dynamic ohlcv_bars: output width equals input width.
                    std::size_t inp_w = nd.inputs.empty() ? 1u : node_width[nd.inputs[0]];
                    node_width[id] = inp_w;
                } else {
                    node_width[id] = resample_output_width(nd.resample);
                }
                break;
            }
            case NodeKind::Filter:        node_width[id] = 1; break;
            case NodeKind::Delay:
            case NodeKind::LagRows:       node_width[id] = node_width[nd.inputs[0]]; break;
            case NodeKind::LeadRows:
            case NodeKind::ForwardAsOf:   node_width[id] = node_width[nd.inputs[0]] + 1; break;
            }
        }
        join_ = OutputJoin(num_out);
        output_widths_.resize(num_out);
        for (std::size_t o = 0; o < num_out; ++o)
            output_widths_[o] = node_width[s.output_ids[o]];
        result_w_ = output_widths_;

        // Map output_id → which output indices it serves.
        std::vector<std::vector<std::size_t>> node_out_idx(n);
        for (std::size_t o = 0; o < num_out; ++o)
            node_out_idx[s.output_ids[o]].push_back(o);

        // Map input node id → its signature index.
        std::vector<std::size_t> input_sig(n, static_cast<std::size_t>(-1));
        for (std::size_t idx = 0; idx < s.input_ids.size(); ++idx)
            input_sig[s.input_ids[idx]] = idx;
        input_sig_ = input_sig;

        // node_input_sink[i](slot) = Sink entry-point for slot `slot` of node i.
        // Functor: returns the same FunctorNode for any slot (wide edge).
        // CombineLatest: returns &n->port(slot) (each producer to its own port).
        // Input nodes have no entry (they are sources, not consumers).
        std::vector<std::function<Sink<std::int64_t>*(std::size_t)>> node_input_sink(n);

        // input_sinks_[sig_idx] = downstream Sink for that input.
        input_sinks_.assign(num_in_, nullptr);
        in_width_.assign(num_in_, 0);
        in_last_.assign(num_in_, 0);
        in_count_.assign(num_in_, 0);

        // Wire in reverse-topological order (consumers first).
        std::vector<unsigned char> order_memo;   // merge_relation, for clock resamples
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
                // Push FunctorNode (not the bare EvalOp) into reset_nodes_: its
                // reset() forwards to op_.reset(), so the old reset_ops_ EvalOp
                // path is preserved via one polymorphic hop.
                auto fn = std::make_shared<FunctorNode<std::int64_t>>(*ns.op, *downstream);
                reset_nodes_.push_back(fn.get());
                node_input_sink[id] = ordered_input(s, id, *fn, order_memo);
                owned_.push_back(fn);
                break;
            }
            case NodeKind::CombineLatest: {
                std::vector<std::size_t> widths;
                for (auto in : ns.inputs) widths.push_back(node_width[in]);
                auto cn = std::make_shared<CombineLatestNode<std::int64_t>>(
                    widths, ns.when_all, *downstream, ns.max_pending);
                reset_nodes_.push_back(cn.get());
                node_input_sink[id] = [ptr = cn.get()](std::size_t slot) -> Sink<std::int64_t>* {
                    return &ptr->port(slot);
                };
                owned_.push_back(cn);
                break;
            }
            case NodeKind::DropNa: {
                auto dn = std::make_shared<DropNaNode<std::int64_t>>(ns.how_all, *downstream);
                node_input_sink[id] = ordered_input(s, id, *dn, order_memo);
                owned_.push_back(dn);
                break;
            }
            case NodeKind::Select: {
                auto sn = std::make_shared<SelectNode<std::int64_t>>(ns.columns, *downstream);
                node_input_sink[id] = [ptr = sn.get()](std::size_t) -> Sink<std::int64_t>* {
                    return ptr;
                };
                owned_.push_back(sn);
                break;
            }
            case NodeKind::Resample: {
                if (ns.resample.reducer) {
                    // Functor-reducer bucketing (GenericResampleNode).
                    auto rn = std::make_shared<GenericResampleNode<std::int64_t>>(
                        ns.resample, *downstream);
                    reset_nodes_.push_back(rn.get());
                    node_input_sink[id] = [ptr = rn.get()](std::size_t) -> Sink<std::int64_t>* {
                        return ptr;
                    };
                    owned_.push_back(rn);
                } else if (ns.resample.mode == ResampleMode::ByClock) {
                    // ByClock: 2-port node (value port 0, clock port 1).
                    // Each producer routes into the appropriate port. The node
                    // learns how each port relates to the input merge, so it
                    // proves a bar complete from the ports' frontiers.
                    auto rn = std::make_shared<ResampleNode<std::int64_t>>(ns.resample, *downstream);
                    clock_nodes_.push_back(rn.get());
                    if (ns.inputs.size() == 2) {
                        if (order_memo.empty()) order_memo.assign(n, 0);
                        rn->set_clock_ports(merge_relation(s, ns.inputs[0], order_memo) & 1,
                                            merge_relation(s, ns.inputs[0], order_memo) & 2,
                                            merge_relation(s, ns.inputs[1], order_memo) & 1,
                                            merge_relation(s, ns.inputs[1], order_memo) & 2);
                    }
                    reset_nodes_.push_back(rn.get());
                    // Initialize ports eagerly here so the lambda captures are stable.
                    rn->port(0);  // creates clock_ports_ on the node
                    node_input_sink[id] = [ptr = rn.get()](std::size_t slot) -> Sink<std::int64_t>* {
                        return &ptr->port(slot);
                    };
                    owned_.push_back(rn);
                } else {
                    // OhlcvBars with no plan: the ResampleNode defers plan creation
                    // to its first push() when the actual frame width is known.
                    // Do NOT try to build make_ohlcv_bars_plan here - the input-node
                    // width is always 1 at static inference time (unknown at compile time).
                    auto rn = std::make_shared<ResampleNode<std::int64_t>>(ns.resample, *downstream);
                    reset_nodes_.push_back(rn.get());
                    node_input_sink[id] = [ptr = rn.get()](std::size_t) -> Sink<std::int64_t>* {
                        return ptr;
                    };
                    owned_.push_back(rn);
                }
                break;
            }
            case NodeKind::Filter: {
                // inputs[0] = data (port 0), inputs[1] = mask (port 1).
                // node_input_sink returns &fn->port(slot) so data wires to port 0
                // and mask wires to port 1, matching the input order exactly.
                auto fn = std::make_shared<FilterNode<std::int64_t>>(*downstream, ns.max_pending);
                reset_nodes_.push_back(fn.get());
                node_input_sink[id] = [ptr = fn.get()](std::size_t slot) -> Sink<std::int64_t>* {
                    return &ptr->port(slot);
                };
                owned_.push_back(fn);
                break;
            }
            case NodeKind::LagRows: {
                auto ln = std::make_shared<LagRowsNode<std::int64_t>>(ns.lag_count, *downstream);
                reset_nodes_.push_back(ln.get());
                node_input_sink[id] = [ptr = ln.get()](std::size_t) -> Sink<std::int64_t>* { return ptr; };
                owned_.push_back(ln);
                break;
            }
            case NodeKind::LeadRows: {
                // inputs[0] = X (port 0), inputs[1] = y (port 1); the node
                // learns how each port relates to the input merge.
                auto ln = std::make_shared<LeadRowsNode<std::int64_t>>(ns.lag_count, *downstream,
                                                                       ns.max_pending);
                if (order_memo.empty()) order_memo.assign(n, 0);
                const unsigned rx = merge_relation(s, ns.inputs[0], order_memo);
                const unsigned ry = merge_relation(s, ns.inputs[1], order_memo);
                ln->set_ports(rx & 1, rx & 2, ry & 1, ry & 2);
                reset_nodes_.push_back(ln.get());
                lead_nodes_.push_back(ln.get());
                node_input_sink[id] = [ptr = ln.get()](std::size_t slot) -> Sink<std::int64_t>* {
                    return &ptr->port(slot);
                };
                owned_.push_back(ln);
                break;
            }
            case NodeKind::ForwardAsOf: {
                // inputs[0] = X (port 0), inputs[1] = y (port 1). The node
                // learns how each port relates to the input merge, as a clock
                // resample does.
                auto an = std::make_shared<ForwardAsOfNode<std::int64_t>>(*downstream,
                                                                          ns.max_pending);
                if (order_memo.empty()) order_memo.assign(n, 0);
                an->set_ports(merge_relation(s, ns.inputs[0], order_memo) & 1,
                              merge_relation(s, ns.inputs[1], order_memo) & 2);
                reset_nodes_.push_back(an.get());
                asof_nodes_.push_back(an.get());
                node_input_sink[id] = [ptr = an.get()](std::size_t slot) -> Sink<std::int64_t>* {
                    return &ptr->port(slot);
                };
                owned_.push_back(an);
                break;
            }
            case NodeKind::Delay: {
                auto dn = std::make_shared<DelayNode<std::int64_t>>(ns.delay_duration, *downstream);
                node_input_sink[id] = [ptr = dn.get()](std::size_t) -> Sink<std::int64_t>* { return ptr; };
                owned_.push_back(dn);
                break;                                        // stateless: not added to reset_nodes_
            }
            }
        }
        set_result_widths(in_width_, {});
        unknown_result_w_ = result_w_;
    }

    // Resets all stateful nodes and output buffers via a single polymorphic pass.
    // Called at the start of every run_batch so each batch sees a clean initial state.
    void reset() {
        set_merge_order(false);
        poisoned_ = false;
        flushed_ = false;
        join_.reset();
        std::fill(in_width_.begin(), in_width_.end(), std::size_t(0));
        std::fill(in_count_.begin(), in_count_.end(), std::size_t(0));
        has_floor_ = false;
        for (auto* n : reset_nodes_) n->reset();
        for (std::size_t o = 0; o < outputs_.size(); ++o) {
            outputs_[o].indices.clear();
            outputs_[o].values.clear();
            outputs_[o].has_watermark = false;
        }
        // No input width is known after a reset: an output's width is the one
        // the graph alone fixes (an operator's outputs, a Select), if any.
        result_w_ = unknown_result_w_;
        for (std::size_t o = 0; o < outputs_.size(); ++o) outputs_[o].width = result_w_[o];
    }

    // Ends the input: every node emits what it holds (a resample its partial
    // bucket, a join its last row). After it, push and advance raise until
    // reset(): the joins have released every row, so a later event could
    // only come out of order. A repeated flush does nothing.
    void flush() {
        check_live();
        if (flushed_) return;
        for (auto* s : input_sinks_) if (s) s->flush();
        flushed_ = true;
    }

    // Throws once flush() ended the input (until reset()).
    void check_open() const {
        if (flushed_)
            throw std::invalid_argument(
                "flush() ended this session's input; start a new live() session "
                "to push more events");
    }

    // Whether one index-ordered merge feeds the inputs from here until the
    // next reset() (a batch run, the lazy driver, a combine_latest stream):
    // clock resamples may then prove a bar from the merge order. A live
    // session leaves it off, as reset() does.
    void set_merge_order(bool on) {
        for (auto* n : clock_nodes_) n->set_merge_order(on);
        for (auto* n : merge_nodes_) n->set_merge_order(on);
        for (auto* n : asof_nodes_) n->set_merge_order(on);
        for (auto* n : lead_nodes_) n->set_merge_order(on);
    }

    // Time-driven finalization: inject a watermark at every input sink so the
    // signal flows through the full graph topology. Resample nodes close their
    // open windows on the watermark (via on_watermark -> advance), and
    // combinators (CombineLatest, Filter) unblock buffered rows whose index is
    // now safe to release. Safe to call repeatedly with non-decreasing `now`.
    // A later event below `now` raises (check_order).
    //
    // A fill Resample whose gap to `now` passes max_fill raises FillLimitError
    // before it changes any state, and the session goes on: the watermark is
    // a promise (no event below `now`, kept by the floor), so the nodes that
    // took it before the rejection hold valid state, and the nodes that did
    // not take it only release their rows later. Any other rejection poisons
    // the session.
    void advance(std::int64_t now) {
        check_live();
        check_open();
        if (!has_floor_ || now > floor_) { floor_ = now; has_floor_ = true; }
        try { for (auto* s : input_sinks_) if (s) s->on_watermark(now); }
        catch (const FillLimitError&) { throw; }
        catch (...) { poisoned_ = true; throw; }
    }

    // Routes a single width-1 event into the graph without resetting state.
    // Bounds-checked: throws if input_idx >= num_in_.
    void push_event(std::size_t input_idx, std::int64_t index, double value) {
        if (input_idx >= num_in_)
            throw std::runtime_error("push_event: input index out of range");
        check_live();
        check_open();
        check_order(input_idx, index);
        if (in_width_[input_idx] != 1) set_width(input_idx, 1);
        note_order(input_idx, index);
        double v = value;
        Frame<std::int64_t> f{index, &v, 1};
        try { input_sinks_[input_idx]->push(f); }
        catch (...) { poisoned_ = true; throw; }
    }

    // Routes a single wide event into the graph without resetting state.
    // The values span [values, values + width) must be contiguous and outlive the call.
    void push_event_wide(std::size_t input_idx, std::int64_t index,
                         const double* values, std::size_t width) {
        if (input_idx >= num_in_)
            throw std::runtime_error("push_event_wide: input index out of range");
        if (width == 0)
            throw std::invalid_argument("a wide event needs at least one value, got none");
        check_live();
        check_open();
        check_order(input_idx, index);
        if (in_width_[input_idx] != width) set_width(input_idx, width);
        note_order(input_idx, index);
        Frame<std::int64_t> f{index, values, width};
        try { input_sinks_[input_idx]->push(f); }
        catch (...) { poisoned_ = true; throw; }
    }

    // Returns all OutputBuffers accumulated since the last drain()/reset(), then
    // clears the buffers in-place so GatherSink references remain valid (the
    // outputs_ vector is never reallocated; only the per-element indices/values are
    // cleared). A poisoned session throws, as push, advance and flush do: its
    // buffers may hold rows the rejected event produced.
    std::vector<OutputBuffer> drain() {
        std::vector<OutputBuffer> out;
        drain_into(out);
        return out;
    }

    // drain() into `dst` without a copy: each output's rows are swapped into
    // dst (one buffer per output), and the output keeps dst's old storage,
    // emptied, so a caller that drains into the same vector allocates nothing
    // in steady state.
    void drain_into(std::vector<OutputBuffer>& dst) {
        check_live();
        dst.resize(outputs_.size());
        for (std::size_t o = 0; o < outputs_.size(); ++o) {
            OutputBuffer& d = dst[o];
            d.indices.clear();
            d.values.clear();
            std::swap(d.indices, outputs_[o].indices);
            std::swap(d.values, outputs_[o].values);
            d.width = outputs_[o].width;
            d.watermark = outputs_[o].watermark;          // monotone: the output keeps it
            d.has_watermark = outputs_[o].has_watermark;
            outputs_[o].width = result_w_[o];
        }
    }

    // drain(), with the outputs aligned on one index (output_join.h): the rows
    // the output join settles with this drain. The join carries each output's
    // latest value across drains, so the rows of repeated drains, concatenated,
    // are the rows of one drain at the end; after a flush (the end of input)
    // every row is settled. `scratch` holds the drained buffers (reuse it).
    AlignedOutputs drain_aligned(std::vector<OutputBuffer>& scratch) {
        drain_into(scratch);
        try { return join_.take(scratch, flushed_); }
        catch (...) { poisoned_ = true; throw; }   // an output changed width
    }

    std::size_t output_count() const { return outputs_.size(); }

    // Throws when a rejected event poisoned the session (until reset()).
    void check_live() const {
        if (poisoned_)
            throw std::runtime_error(
                "this session rejected an event after part of the graph took it; "
                "call reset() or start a new live() session");
    }

    // in_* are per-input arrays (one entry per input, in signature order).
    // in_widths: column count per input (empty = all width-1 / backward-compat).
    //   Width-1 inputs emit scalar Frames as before.
    //   Width-W inputs (W>1) emit W-wide Frames: the row-major in_vals pointer
    //   is strided so event j from source i points to in_vals[i] + j*W.
    // Returns one OutputBuffer per output, in output_ids order.
    std::vector<OutputBuffer> run_batch(
            const std::vector<const std::int64_t*>& in_indices,
            const std::vector<const double*>& in_vals,
            const std::vector<std::size_t>& in_lens,
            const std::vector<std::size_t>& in_widths = {}) {
        return run_batch_ref(in_indices, in_vals, in_lens, in_widths);
    }

    // run_batch() without the copy: the outputs stay in the graph and the
    // reference is valid until the next run, push, drain or reset.
    const std::vector<OutputBuffer>& run_batch_ref(
            const std::vector<const std::int64_t*>& in_indices,
            const std::vector<const double*>& in_vals,
            const std::vector<std::size_t>& in_lens,
            const std::vector<std::size_t>& in_widths = {}) {

        // Validate caller-supplied input-array counts.
        if (in_indices.size() != num_in_)
            throw std::invalid_argument(
                "run_batch: expected " + std::to_string(num_in_) +
                " input arrays, got " + std::to_string(in_indices.size()));
        if (in_vals.size() != num_in_)
            throw std::invalid_argument(
                "run_batch: expected " + std::to_string(num_in_) +
                " value arrays, got " + std::to_string(in_vals.size()));
        if (in_lens.size() != num_in_)
            throw std::invalid_argument(
                "run_batch: expected " + std::to_string(num_in_) +
                " length arrays, got " + std::to_string(in_lens.size()));
        if (!in_widths.empty() && in_widths.size() != num_in_)
            throw std::invalid_argument(
                "run_batch: in_widths size must be 0 (use defaults) or equal to num inputs");

        // Resolve widths: default to 1 per input when not specified.
        std::vector<std::size_t> widths(num_in_, 1u);
        if (!in_widths.empty()) widths = in_widths;

        // Reset all stateful state and clear output buffers. The merge below
        // feeds the inputs in index order.
        reset();
        set_merge_order(true);

        // Check every node against the input widths before any event is fed.
        {
            std::vector<std::size_t> known(num_in_, 0);
            std::vector<bool> present(num_in_, false);
            for (std::size_t i = 0; i < num_in_; ++i)
                if (in_lens[i] > 0) { known[i] = widths[i]; present[i] = true; }
            validate_widths(known, present);
        }
        set_batch_result_widths(widths);

        // One input needs no merge: its events go in array order, as the
        // merge of one source gives them, without a heap step per event.
        if (num_in_ == 1) {
            Sink<std::int64_t>* sink = input_sinks_[0];
            const std::size_t w = widths[0];
            const std::int64_t* k = in_indices[0];
            const double* v = in_vals[0];
            for (std::size_t r = 0, n = in_lens[0]; r < n; ++r) {
                if (r != 0 && k[r] < k[r - 1]) throw_index_decreases(std::size_t(0), r, k[r - 1], k[r]);
                Frame<std::int64_t> f{k[r], v + r * w, w};
                sink->push(f);
            }
            sink->flush();
            flushed_ = true;
            return outputs_;
        }

        // A few inputs merge by a scan of their next rows: the smallest index
        // goes first, the lower input on a tie, as streams::MergeSource orders
        // them. As there, an input's next row is read (and its index order
        // checked) only after its previous row went through the graph.
        if (num_in_ <= kScanMergeInputs) {
            std::size_t pos[kScanMergeInputs] = {};
            std::size_t last = num_in_;           // the input whose row went last
            for (;;) {
                if (last != num_in_) {
                    const std::size_t r = pos[last];
                    const std::int64_t* k = in_indices[last];
                    if (r < in_lens[last] && k[r] < k[r - 1])
                        throw_index_decreases(last, r, k[r - 1], k[r]);
                }
                std::size_t src = num_in_;
                std::int64_t best = 0;
                for (std::size_t i = 0; i < num_in_; ++i) {
                    if (pos[i] == in_lens[i]) continue;
                    const std::int64_t t = in_indices[i][pos[i]];
                    if (src == num_in_ || t < best) { src = i; best = t; }
                }
                if (src == num_in_) break;
                const std::size_t w = widths[src];
                Frame<std::int64_t> f{best, in_vals[src] + pos[src] * w, w};
                input_sinks_[src]->push(f);
                ++pos[src];
                last = src;
            }
            for (auto* s : input_sinks_) if (s) s->flush();
            flushed_ = true;
            return outputs_;
        }

        // Build VectorSources for index-merge ordering (always width-1 view;
        // the actual values for wide inputs are fetched below via row counters).
        std::vector<std::unique_ptr<streams::VectorSource<std::int64_t>>> srcs;
        std::vector<streams::Source<std::int64_t>*> child_ptrs;
        srcs.reserve(num_in_);
        child_ptrs.reserve(num_in_);
        for (std::size_t i = 0; i < num_in_; ++i) {
            srcs.push_back(std::make_unique<streams::VectorSource<std::int64_t>>(
                in_indices[i], in_vals[i], in_lens[i], i));
            child_ptrs.push_back(srcs.back().get());
        }
        streams::MergeSource<std::int64_t> merge(child_ptrs);

        // Per-source row counters: track how many events each source has emitted
        // so we can address the correct row in wide (multi-column) value arrays.
        std::vector<std::size_t> row_idx(num_in_, 0u);

        double one;
        while (auto e = merge.next()) {
            std::size_t src = e->source;
            std::size_t w   = widths[src];
            if (w == 1) {
                // Scalar (width-1) path: use the event's value directly.
                one = e->value;
                Frame<std::int64_t> f{e->index, &one, 1};
                input_sinks_[src]->push(f);
            } else {
                // Wide (width-W) path: point into the row-major value array.
                const double* row_ptr = in_vals[src] + row_idx[src] * w;
                Frame<std::int64_t> f{e->index, row_ptr, w};
                input_sinks_[src]->push(f);
            }
            ++row_idx[src];
        }
        for (auto* s : input_sinks_) if (s) s->flush();
        flushed_ = true;

        return outputs_;
    }

private:
    GraphSpec spec_;

    // The input of node `id`, which takes its frames at `sink`: the sink
    // itself for one producer, else the ports of an OrderedMergeNode in front
    // of it, so the node takes its producers' events in index order whatever
    // their arrival order (ordered_merge_node.h).
    std::function<Sink<std::int64_t>*(std::size_t)> ordered_input(
            const GraphSpec& s, std::size_t id, Sink<std::int64_t>& sink,
            std::vector<unsigned char>& order_memo) {
        const NodeSpec& ns = s.nodes[id];
        if (ns.inputs.size() <= 1)
            return [ptr = &sink](std::size_t) -> Sink<std::int64_t>* { return ptr; };
        auto mn = std::make_shared<OrderedMergeNode<std::int64_t>>(
            ns.inputs.size(), sink, ns.max_pending);
        if (order_memo.empty()) order_memo.assign(s.nodes.size(), 0);
        for (std::size_t k = 0; k < ns.inputs.size(); ++k) {
            const unsigned r = merge_relation(s, ns.inputs[k], order_memo);
            mn->set_port_relation(k, r & 1u, r & 2u);
        }
        reset_nodes_.push_back(mn.get());
        merge_nodes_.push_back(mn.get());
        owned_.push_back(mn);
        return [ptr = mn.get()](std::size_t slot) -> Sink<std::int64_t>* { return &ptr->port(slot); };
    }

    // How node `id`'s output relates to the merge of the graph's inputs, as
    // bits (memo[id] holds them plus 4 once computed):
    // 1, in order: it never emits an index below one the merge has passed (an
    //    Input, or functors, Select, Dropna, a row lag or a forward Delay that
    //    emit each event as they take it, over such nodes);
    // 2, never ahead: it never emits an index or watermark above the merge's
    //    (every node keeps or lowers its inputs' indexes, but a forward Delay).
    static unsigned merge_relation(const GraphSpec& s, std::size_t id,
                                   std::vector<unsigned char>& memo) {
        if (memo[id] & 4) return memo[id] & 3u;
        const NodeSpec& nd = s.nodes[id];
        bool in_order = false, not_ahead = true;
        switch (nd.kind) {
        case NodeKind::Input: in_order = true; break;
        case NodeKind::Delay:
            in_order = nd.delay_duration >= 0;
            not_ahead = nd.delay_duration <= 0;
            break;
        case NodeKind::Functor: case NodeKind::DropNa:
            // Several producers go through an OrderedMergeNode, which holds an
            // event until the merge has passed its index.
            in_order = nd.inputs.size() <= 1;
            break;
        case NodeKind::Select: case NodeKind::LagRows:
            in_order = true;
            break;
        default: break;
        }
        for (std::size_t in : nd.inputs) {
            const unsigned r = merge_relation(s, in, memo);
            in_order = in_order && (r & 1u);
            not_ahead = not_ahead && (r & 2u);
        }
        memo[id] = static_cast<unsigned char>(4u | (in_order ? 1u : 0u) | (not_ahead ? 2u : 0u));
        return memo[id] & 3u;
    }

    // A live event's index: not below the input's last event (the index of a
    // stream must not decrease) nor below an earlier advance(now). Checked
    // before the event reaches a node; note_order records it once the event
    // passed the width check, so a rejected event leaves no trace.
    void check_order(std::size_t i, std::int64_t index) const {
        if (in_count_[i] != 0 && index < in_last_[i])
            throw_index_decreases(i, in_count_[i], in_last_[i], index);
        if (has_floor_ && index < floor_)
            throw std::invalid_argument(
                "the index of input " + std::to_string(i) + " is " + std::to_string(index) +
                " at event " + std::to_string(in_count_[i]) + ", below an earlier advance(" +
                std::to_string(floor_) + "); an event cannot come before the time the "
                "session has advanced to");
    }
    void note_order(std::size_t i, std::int64_t index) {
        in_last_[i] = index;
        ++in_count_[i];
    }

    // Wiring (built once in constructor, persistent across all run_batch calls).
    //
    // Input widths. The first event after reset() sets an input's width (a
    // scalar is width 1); an event of another width then throws
    // std::invalid_argument. Before the width is stored, the whole graph is
    // checked against it (validate_widths), so a node that cannot take the
    // width rejects the event while no node has seen it: nothing is pushed and
    // no width is locked. A rejection that only a node can decide (a Filter
    // mask, a Resample reducer, a CombineLatest port that changes width after
    // rows were emitted) happens after upstream nodes took the event, so it
    // poisons the session: every later push, advance and flush throws until
    // reset(), and so does drain(), since the buffers may hold rows that the
    // rejected event produced.
    void set_width(std::size_t input_idx, std::size_t width) {
        const std::size_t w = in_width_[input_idx];
        if (w != 0)
            throw std::invalid_argument(
                "input " + std::to_string(input_idx) + " received an event of " +
                std::to_string(width) + " values after events of " + std::to_string(w));
        std::vector<std::size_t> candidate = in_width_;
        candidate[input_idx] = width;
        validate_widths(candidate);
        in_width_[input_idx] = width;
        set_result_widths(in_width_, {});
    }

    // An output without rows keeps the width that a batch run's arrays give
    // it, as an empty array does: (0, w). A run with the widths of the last
    // one reuses its result.
    void set_batch_result_widths(const std::vector<std::size_t>& widths) {
        if (widths != batch_in_w_) {
            set_result_widths(widths, std::vector<bool>(num_in_, true));
            batch_in_w_ = widths;
            batch_result_w_ = result_w_;
            return;
        }
        result_w_ = batch_result_w_;
        for (std::size_t o = 0; o < outputs_.size(); ++o) outputs_[o].width = result_w_[o];
    }

    // Sets the width that each output reports while it has no rows (an empty
    // batch, an empty drain): the width the graph gives it for these input
    // widths, or, when that depends on an input width that is not known yet,
    // the estimate for width-1 inputs.
    void set_result_widths(const std::vector<std::size_t>& in_w,
                           const std::vector<bool>& present) {
        const NodeWidths nw = propagate_widths(in_w, present, false);
        for (std::size_t o = 0; o < result_w_.size(); ++o) {
            const std::size_t id = spec_.output_ids[o];
            result_w_[o] = (nw.w[id] != 0 || nw.zero[id]) ? nw.w[id] : output_widths_[o];
            if (outputs_[o].indices.empty()) outputs_[o].width = result_w_[o];
        }
    }

    // Propagates the known input widths (0 = not known yet) through the graph
    // and throws std::invalid_argument for a node that cannot take its width.
    // Each rule is the one the node itself applies, so a graph passes here
    // exactly when its nodes accept the widths; the node checks stay as a
    // backstop for widths that are not known here.
    //
    // `present` marks the inputs that have events (a batch run): such an input
    // of width 0 is a known zero width, not an unknown one. The zero follows
    // the nodes that pass rows through (Dropna, Delay, a lag), so a width rule
    // with a minimum sees it.
    void validate_widths(const std::vector<std::size_t>& in_w,
                         const std::vector<bool>& present = {}) const {
        propagate_widths(in_w, present, true);
    }

    // Each node's width for the input widths in_w (0 = unknown), and whether a
    // 0 is a known width of 0. With `check`, the rules of validate_widths
    // throw for a node that cannot take its width. Without, nothing throws,
    // and a node whose width the graph alone fixes (a reducer or plan
    // Resample) has it even when its input width is unknown.
    struct NodeWidths { std::vector<std::size_t> w; std::vector<bool> zero; };
    NodeWidths propagate_widths(const std::vector<std::size_t>& in_w,
                                const std::vector<bool>& present, bool check) const {
        const std::size_t n = spec_.nodes.size();
        NodeWidths out{std::vector<std::size_t>(n, 0), std::vector<bool>(n, false)};
        std::vector<std::size_t>& w = out.w;      // 0 = unknown
        std::vector<bool>& zero = out.zero;       // a known width of 0
        auto in_of = [&](const NodeSpec& nd) -> std::size_t {
            return nd.inputs.size() == 1 ? w[nd.inputs[0]] : 0;
        };
        for (std::size_t id : topo_producers_first_) {
            const NodeSpec& nd = spec_.nodes[id];
            if (check) for (const WidthRule& r : nd.width_rules) {
                const std::size_t in = nd.inputs[r.slot];
                const std::size_t iw = w[in];
                if ((iw != 0 && ((r.exact && iw != r.exact) || iw < r.min)) ||
                        (zero[in] && r.min > 0))
                    throw std::invalid_argument(width_rule_text(r, iw));
            }
            switch (nd.kind) {
            case NodeKind::Input:
                w[id] = in_w[input_sig_[id]];
                zero[id] = w[id] == 0 && input_sig_[id] < present.size() &&
                           present[input_sig_[id]];
                break;
            case NodeKind::Functor: {
                // Every producer pushes its own frames into the operator, so each
                // one must carry the operator's columns. Two producers do not
                // form one row; CombineLatest does.
                const std::size_t need = nd.op->n_in();
                for (std::size_t k = 0; check && k < nd.inputs.size(); ++k) {
                    const std::size_t iw = w[nd.inputs[k]];
                    if (iw == 0 || iw == need) continue;
                    if (nd.inputs.size() == 1)
                        throw std::invalid_argument(
                            "an operator takes " + std::to_string(need) +
                            " column(s), but its input has " + std::to_string(iw));
                    throw std::invalid_argument(
                        "an operator takes " + std::to_string(need) +
                        " column(s), but its input " + std::to_string(k) + " has " +
                        std::to_string(iw) + "; each input of an operator is a "
                        "separate stream of " + std::to_string(need) +
                        "-column events. To join streams into one row, use "
                        "CombineLatest");
                }
                w[id] = nd.op->n_out();
                break;
            }
            case NodeKind::CombineLatest: {
                // The sum of the input widths once every one is known; a known
                // zero counts, so a row of zero values reaches the checks.
                std::size_t sum = 0;
                bool known = true;
                for (auto in : nd.inputs) {
                    if (w[in] == 0 && !zero[in]) { known = false; break; }
                    sum += w[in];
                }
                w[id] = known ? sum : 0;
                zero[id] = known && sum == 0;
                break;
            }
            case NodeKind::DropNa:
            case NodeKind::Delay:
            case NodeKind::LagRows:
                w[id] = in_of(nd);
                zero[id] = nd.inputs.size() == 1 && zero[nd.inputs[0]];
                break;
            case NodeKind::LeadRows:
            case NodeKind::ForwardAsOf: {
                // X of any width, y one value; the row is [X..., y].
                const std::size_t yw = w[nd.inputs[1]];
                if (check && yw > 1)
                    throw std::invalid_argument(
                        std::string(nd.kind == NodeKind::LeadRows ? "a lead"
                                                                  : "a forward as-of join") +
                        " takes y events of one value, got " + std::to_string(yw));
                w[id] = w[nd.inputs[0]] != 0 ? w[nd.inputs[0]] + 1 : 0;
                break;
            }
            case NodeKind::Select: {
                const std::size_t iw = in_of(nd);
                if (check && iw != 0)
                    for (std::size_t c : nd.columns)
                        if (c >= iw)
                            throw std::invalid_argument(
                                "select needs column " + std::to_string(c) +
                                ", but its input has " + std::to_string(iw) + " column(s)");
                w[id] = nd.columns.size();
                break;
            }
            case NodeKind::Resample:
                w[id] = resample_width_check(nd, w, check);
                break;
            case NodeKind::Filter:
                for (std::size_t k = 0; check && k < nd.inputs.size(); ++k) {
                    const std::size_t iw = w[nd.inputs[k]];
                    if (iw > 1)
                        throw std::invalid_argument(
                            FilterNode<std::int64_t>::width_message(k, iw));
                }
                w[id] = 1;
                break;
            }
        }
        return out;
    }

    // The Resample rules of validate_widths: throws for a value width the node
    // cannot take and returns the node's output width (0 = unknown). Without
    // `check` nothing throws, and a width the node fixes on its own (a reducer,
    // a plan, a threshold= bar) is returned for an unknown input width too.
    static std::size_t resample_width_check(const NodeSpec& nd,
                                            const std::vector<std::size_t>& w,
                                            bool check = true) {
        const ResampleParams& p = nd.resample;
        const bool clock = p.mode == ResampleMode::ByClock;
        const bool bars = p.agg == ResampleAgg::OhlcvBars && p.plan.empty();
        // The value stream: port 0 of a clock resample, else the one input.
        std::size_t iw = 0;
        if (clock) { if (!nd.inputs.empty()) iw = w[nd.inputs[0]]; }
        else if (nd.inputs.size() == 1) iw = w[nd.inputs[0]];
        if (iw == 0) {
            if (check) return 0;
            if (p.reducer) return p.reducer->n_out();
            if (!p.plan.empty()) return p.plan.size();
            if (p.mode == ResampleMode::ByCumulative && !bars) return resample_output_width(p);
            return 0;
        }
        if (p.reducer) {
            if (check && iw != 1)
                throw std::invalid_argument(
                    "a reducer resample takes 1 column, but its input has " +
                    std::to_string(iw));
            return p.reducer->n_out();
        }
        if (p.mode == ResampleMode::ByCumulative) {
            if (check && iw != 2)
                throw std::invalid_argument(
                    "a cumulative resample takes [value, driver] (2 columns), "
                    "but its input has " + std::to_string(iw));
            if (bars) {
                if (!check) return 0;
                check_ohlcv_bars_width(iw - 1);   // one value column: always too few
            }
            return resample_output_width(p);
        }
        if (check && clock && p.max_age >= 0 && iw != 1)
            throw std::invalid_argument(
                "a clock resample with max_age takes 1 value column, but its "
                "input has " + std::to_string(iw));
        if (!p.plan.empty()) {               // ohlc_bars, ohlcv, ohlcv2
            const std::size_t need = resample_plan_input_cols(p.plan);
            if (check && iw != need)
                throw std::invalid_argument(
                    "a resample plan takes " + std::to_string(need) +
                    " column(s), but its input has " + std::to_string(iw));
            return p.plan.size();
        }
        if (bars) {
            if (check) check_ohlcv_bars_width(iw);
            return iw;
        }
        // A builtin agg aggregates every column on its own; ohlc gives four
        // outputs per column.
        return iw == 1 ? resample_width(p.agg)
                       : (p.agg == ResampleAgg::Ohlc ? 4 * iw : iw);
    }

    std::vector<std::size_t> topo_producers_first_;
    std::vector<std::size_t> input_sig_;   // node id -> input signature index
    bool poisoned_ = false;
    bool flushed_ = false;                 // flush() ended the input (until reset())
    OutputJoin join_;                      // the output alignment across drains
    std::vector<std::size_t> in_width_;   // per input, 0 = not set yet
    std::vector<std::int64_t> in_last_;    // per input, the last live event's index
    std::vector<std::size_t> in_count_;    // per input, live events taken since reset()
    std::int64_t floor_ = 0;               // the highest advance(now) since reset()
    bool has_floor_ = false;
    std::size_t num_in_ = 0;
    std::vector<std::shared_ptr<void>>            owned_;           // all heap nodes/broadcasts
    std::vector<Sink<std::int64_t>*>              input_sinks_;     // per input signature index
    // Polymorphic reset: every stateful node (FunctorNode, CombineLatestNode,
    // FilterNode, ResampleNode, GenericResampleNode) is pushed here once.
    // reset() does one polymorphic pass instead of multiple typed loops.
    std::vector<Resettable*>                      reset_nodes_;
    std::vector<OutputBuffer>                     outputs_;         // persistent output buffers
    std::vector<ResampleNode<std::int64_t>*>      clock_nodes_;     // clock resamples (set_merge_order)
    std::vector<OrderedMergeNode<std::int64_t>*>  merge_nodes_;     // multi-producer inputs (set_merge_order)
    std::vector<ForwardAsOfNode<std::int64_t>*>   asof_nodes_;      // forward as-of joins (set_merge_order)
    std::vector<LeadRowsNode<std::int64_t>*>      lead_nodes_;      // row leads (set_merge_order)
    std::vector<std::size_t>                      output_widths_;   // expected width for each output
    std::vector<std::size_t>                      result_w_;        // each output's width while it has no rows
    std::vector<std::size_t>                      unknown_result_w_; // result_w_ with no input width known
    std::vector<std::size_t>                      batch_in_w_;      // the input widths of the last batch run
    std::vector<std::size_t>                      batch_result_w_;  // result_w_ for batch_in_w_
};

// compile() wraps the spec into a CompiledGraph (wiring happens in constructor).
inline CompiledGraph compile(const GraphSpec& spec) {
    return CompiledGraph{spec};
}

}} // namespace screamer::dag
#endif
