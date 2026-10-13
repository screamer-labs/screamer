#ifndef SCREAMER_DAG_GRAPH_H
#define SCREAMER_DAG_GRAPH_H

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>
#include "screamer/common/eval_op.h"
#include "screamer/dag/resample_params.h"
#include "screamer/dag/max_pending.h"
#include "screamer/dag/resample_check.h"
#include <cstdint>
#include <utility>

namespace screamer { namespace dag {

enum class NodeKind { Input, Functor, CombineLatest, DropNa, Select, Resample, Filter, Delay,
                      LagRows };

// A rule on the width of input `slot` of a node: exactly `exact` columns
// (0 = any), and at least `min` columns. A graph given another width raises
// "<what> holds N columns, expected E" or "<what> has N columns, expected at
// least M".
struct WidthRule {
    std::size_t slot = 0;
    std::size_t exact = 0;
    std::size_t min = 0;
    std::string what;
};

// The text of a width that breaks a rule.
inline std::string width_rule_text(const WidthRule& r, std::size_t width) {
    if (r.exact && width != r.exact)
        return r.what + " holds " + std::to_string(width) + " columns, expected " +
               std::to_string(r.exact);
    return r.what + " has " + std::to_string(width) + " columns, expected at least " +
           std::to_string(r.min);
}

// Pure data: one node of a graph definition.
struct NodeSpec {
    NodeKind kind = NodeKind::Input;
    EvalOp* op = nullptr;                 // Functor only
    bool when_all = true;                 // CombineLatest only
    bool how_all = false;                 // DropNa only
    std::vector<std::size_t> columns;     // Select only
    ResampleParams resample;              // Resample only
    std::int64_t delay_duration = 0;     // Delay only
    std::size_t lag_count = 1;            // LagRows only: events to lag by
    // Width rules on the node's inputs, checked before any event is fed
    // (CompiledGraph::validate_widths).
    std::vector<WidthRule> width_rules;
    std::size_t max_pending = kDefaultMaxPending;  // reorder buffer cap: CombineLatest, Filter,
                                                   // a Functor or DropNa with several inputs
    std::vector<std::size_t> inputs;      // producer node ids (edges into this node)
};

struct GraphSpec {
    std::vector<NodeSpec> nodes;
    std::vector<std::size_t> input_ids;   // Input nodes, in signature order
    std::vector<std::size_t> output_ids;  // output nodes, in order
    // The join cap of the graph: the default of its CombineLatest and Filter
    // nodes and of the input of a node with several producers. The output
    // alignment (OutputJoin) has no count cap: watermarks bound it.
    std::size_t max_pending = kDefaultMaxPending;
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
        ns.max_pending = spec_.max_pending;   // several producers: their reorder cap
        ns.inputs = std::move(inputs);
        spec_.nodes.push_back(std::move(ns));
        return spec_.nodes.size() - 1;
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
        ns.max_pending = spec_.max_pending;   // several producers: their reorder cap
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
        check_resample_params_numbers(static_cast<int>(rp.mode), rp.width, rp.count,
                                      rp.threshold);
        // fill= fills empty bars, which only freq= (every=) and clock= have:
        // a count= or threshold= bar always holds events.
        if (rp.fill != ResampleFill::Skip &&
            (rp.mode == ResampleMode::ByCount || rp.mode == ResampleMode::ByCumulative))
            throw std::invalid_argument(
                std::string("resample: fill= applies to freq= (every=) and clock= bars; a ") +
                (rp.mode == ResampleMode::ByCount ? "count=" : "threshold=") +
                " bar is never empty, so pass fill=\"skip\" (the default)");
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
    // Lags rows of any width by `count` events (LagRowsNode).
    std::size_t add_lag_rows(std::vector<std::size_t> inputs, std::size_t count) {
        if (inputs.size() != 1 || count < 1)
            throw std::invalid_argument("a lag takes one input and a count of at least 1");
        NodeSpec ns;
        ns.kind      = NodeKind::LagRows;
        ns.lag_count = count;
        ns.inputs    = std::move(inputs);
        spec_.nodes.push_back(std::move(ns));
        return spec_.nodes.size() - 1;
    }
    // Adds a width rule on input `slot` of node `id` (NodeSpec::width_rules).
    void require_width(std::size_t id, WidthRule rule) {
        NodeSpec& ns = spec_.nodes.at(id);
        if (rule.slot >= ns.inputs.size())
            throw std::invalid_argument("require_width: the node has no input " +
                                        std::to_string(rule.slot));
        ns.width_rules.push_back(std::move(rule));
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
