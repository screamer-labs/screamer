#ifndef SCREAMER_DAG_GRAPH_H
#define SCREAMER_DAG_GRAPH_H

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>
#include "screamer/common/eval_op.h"
#include "screamer/dag/resample_params.h"
#include "screamer/dag/max_pending.h"
#include <cstdint>
#include <memory>
#include <utility>

namespace screamer { namespace dag {

enum class NodeKind { Input, Functor, CombineLatest, DropNa, Select, Resample, Filter, Delay };

// Pure data: one node of a graph definition.
struct NodeSpec {
    NodeKind kind = NodeKind::Input;
    EvalOp* op = nullptr;                 // Functor only
    bool when_all = true;                 // CombineLatest only
    bool how_all = false;                 // DropNa only
    std::vector<std::size_t> columns;     // Select only
    ResampleParams resample;              // Resample only
    std::int64_t delay_duration = 0;     // Delay only
    std::size_t max_pending = kDefaultMaxPending;  // CombineLatest, Filter: reorder buffer cap
    std::vector<std::size_t> inputs;      // producer node ids (edges into this node)
};

struct GraphSpec {
    std::vector<NodeSpec> nodes;
    std::vector<std::size_t> input_ids;   // Input nodes, in signature order
    std::vector<std::size_t> output_ids;  // output nodes, in order
    // The join cap of the graph: the default of its CombineLatest and Filter
    // nodes, and the cap of its output alignment (OutputJoin).
    std::size_t max_pending = kDefaultMaxPending;
    // Operators the graph owns: the ones a C++ graph definition creates (the
    // Lag of forecast_pairs). A binding's own operators stay with the binding.
    // A copy of the spec shares them, so compile a spec once.
    std::vector<std::shared_ptr<EvalOp>> owned_ops;
};

// Accumulates a GraphSpec; returns node ids.
class GraphBuilder {
public:
    std::size_t add_input() {
        NodeSpec ns;
        ns.kind = NodeKind::Input;
        spec_.nodes.push_back(std::move(ns));
        std::size_t id = spec_.nodes.size() - 1;
        spec_.input_ids.push_back(id);
        return id;
    }
    std::size_t add_functor(EvalOp* op, std::vector<std::size_t> inputs) {
        NodeSpec ns;
        ns.kind   = NodeKind::Functor;
        ns.op     = op;
        ns.inputs = std::move(inputs);
        spec_.nodes.push_back(std::move(ns));
        return spec_.nodes.size() - 1;
    }
    // A functor node over an operator the graph owns (GraphSpec::owned_ops).
    std::size_t add_owned_functor(std::shared_ptr<EvalOp> op, std::vector<std::size_t> inputs) {
        EvalOp* raw = op.get();
        spec_.owned_ops.push_back(std::move(op));
        return add_functor(raw, std::move(inputs));
    }
    // The graph's join cap (GraphSpec::max_pending); set it before the nodes
    // that take it as their default.
    void set_max_pending(std::size_t n) { spec_.max_pending = n; }

    std::size_t add_combine_latest(std::vector<std::size_t> inputs, bool when_all) {
        return add_combine_latest(std::move(inputs), when_all, spec_.max_pending);
    }
    std::size_t add_combine_latest(std::vector<std::size_t> inputs, bool when_all,
                                   std::size_t max_pending) {
        NodeSpec ns;
        ns.kind        = NodeKind::CombineLatest;
        ns.when_all    = when_all;
        ns.max_pending = max_pending;
        ns.inputs      = std::move(inputs);
        spec_.nodes.push_back(std::move(ns));
        return spec_.nodes.size() - 1;
    }
    std::size_t add_dropna(std::vector<std::size_t> inputs, bool how_all) {
        NodeSpec ns;
        ns.kind    = NodeKind::DropNa;
        ns.how_all = how_all;
        ns.inputs  = std::move(inputs);
        spec_.nodes.push_back(std::move(ns));
        return spec_.nodes.size() - 1;
    }
    std::size_t add_select(std::vector<std::size_t> inputs,
                           std::vector<std::size_t> columns) {
        NodeSpec ns;
        ns.kind    = NodeKind::Select;
        ns.columns = std::move(columns);
        ns.inputs  = std::move(inputs);
        spec_.nodes.push_back(std::move(ns));
        return spec_.nodes.size() - 1;
    }
    // A cumulative resample given [value, driver] as two inputs reads them as
    // one [value, driver] row: the builder joins them first with a
    // CombineLatest (emit="when_all", reorder cap `max_pending`) and resamples
    // the joined stream. One input is that joined stream already.
    std::size_t add_resample(std::vector<std::size_t> inputs, ResampleParams rp) {
        return add_resample(std::move(inputs), std::move(rp), spec_.max_pending);
    }
    std::size_t add_resample(std::vector<std::size_t> inputs, ResampleParams rp,
                             std::size_t max_pending) {
        if (rp.mode == ResampleMode::ByCumulative && inputs.size() == 2)
            inputs = {add_combine_latest(std::move(inputs), true, max_pending)};
        NodeSpec ns;
        ns.kind     = NodeKind::Resample;
        ns.resample = rp;
        ns.inputs   = std::move(inputs);
        spec_.nodes.push_back(std::move(ns));
        return spec_.nodes.size() - 1;
    }
    std::size_t add_delay(std::vector<std::size_t> inputs, std::int64_t duration) {
        NodeSpec ns;
        ns.kind           = NodeKind::Delay;
        ns.delay_duration = duration;
        ns.inputs         = std::move(inputs);
        spec_.nodes.push_back(std::move(ns));
        return spec_.nodes.size() - 1;
    }
    std::size_t add_filter(std::vector<std::size_t> inputs) {
        // Filter is a fixed 2-input gate (data, mask); fail early on wrong arity
        // rather than half-wire the node and misbehave at run time.
        if (inputs.size() != 2)
            throw std::invalid_argument("Filter takes 2 inputs (data, mask), got " +
                                        std::to_string(inputs.size()));
        NodeSpec ns;
        ns.kind        = NodeKind::Filter;
        ns.max_pending = spec_.max_pending;
        ns.inputs      = std::move(inputs);
        spec_.nodes.push_back(std::move(ns));
        return spec_.nodes.size() - 1;
    }
    void set_outputs(std::vector<std::size_t> outs) { spec_.output_ids = std::move(outs); }
    const GraphSpec& spec() const { return spec_; }
    // The spec, moved out; the builder is empty afterwards.
    GraphSpec take() { GraphSpec s = std::move(spec_); spec_ = GraphSpec{}; return s; }

private:
    GraphSpec spec_;
};

}} // namespace screamer::dag
#endif
