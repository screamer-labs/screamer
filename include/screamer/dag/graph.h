#ifndef SCREAMER_DAG_GRAPH_H
#define SCREAMER_DAG_GRAPH_H

#include <cstddef>
#include <stdexcept>
#include <vector>
#include "screamer/common/eval_op.h"
#include "screamer/dag/resample_params.h"

namespace screamer { namespace dag {

enum class NodeKind { Input, Functor, CombineLatest, DropNa, Select, Resample,
                      MultiResample, Filter };

// Pure data: one node of a graph definition.
struct NodeSpec {
    NodeKind kind;
    EvalOp* op = nullptr;                 // Functor only
    bool when_all = true;                 // CombineLatest only
    bool how_all = false;                 // DropNa only
    std::vector<std::size_t> columns;     // Select only
    ResampleParams resample;              // Resample / MultiResample (the clock)
    std::vector<std::size_t> inputs;      // producer node ids (edges into this node)
    std::vector<EvalOp*> reducers;        // MultiResample only (one per input port)
};

struct GraphSpec {
    std::vector<NodeSpec> nodes;
    std::vector<std::size_t> input_ids;   // Input nodes, in signature order
    std::vector<std::size_t> output_ids;  // output nodes, in order
};

// Accumulates a GraphSpec; returns node ids.
class GraphBuilder {
public:
    std::size_t add_input() {
        spec_.nodes.push_back(NodeSpec{NodeKind::Input, nullptr, true, false, {}, {}, {}});
        std::size_t id = spec_.nodes.size() - 1;
        spec_.input_ids.push_back(id);
        return id;
    }
    std::size_t add_functor(EvalOp* op, std::vector<std::size_t> inputs) {
        spec_.nodes.push_back(NodeSpec{NodeKind::Functor, op, true, false, {}, {}, std::move(inputs)});
        return spec_.nodes.size() - 1;
    }
    std::size_t add_combine_latest(std::vector<std::size_t> inputs, bool when_all) {
        spec_.nodes.push_back(NodeSpec{NodeKind::CombineLatest, nullptr, when_all, false,
                                       {}, {}, std::move(inputs)});
        return spec_.nodes.size() - 1;
    }
    std::size_t add_dropna(std::vector<std::size_t> inputs, bool how_all) {
        spec_.nodes.push_back(NodeSpec{NodeKind::DropNa, nullptr, true, how_all,
                                       {}, {}, std::move(inputs)});
        return spec_.nodes.size() - 1;
    }
    std::size_t add_select(std::vector<std::size_t> inputs,
                           std::vector<std::size_t> columns) {
        NodeSpec ns{NodeKind::Select, nullptr, true, false, std::move(columns), {}, std::move(inputs)};
        spec_.nodes.push_back(std::move(ns));
        return spec_.nodes.size() - 1;
    }
    std::size_t add_resample(std::vector<std::size_t> inputs, ResampleParams rp) {
        NodeSpec ns{NodeKind::Resample, nullptr, true, false, {}, rp, std::move(inputs)};
        spec_.nodes.push_back(std::move(ns));
        return spec_.nodes.size() - 1;
    }
    std::size_t add_filter(std::vector<std::size_t> inputs) {
        // Filter is a fixed 2-input gate (data, mask); fail early on wrong arity
        // rather than half-wire the node and misbehave at run time.
        if (inputs.size() != 2)
            throw std::runtime_error("add_filter: Filter needs exactly 2 inputs (data, mask)");
        spec_.nodes.push_back(NodeSpec{NodeKind::Filter, nullptr, true, false,
                                       {}, {}, std::move(inputs)});
        return spec_.nodes.size() - 1;
    }
    std::size_t add_multicolumn_resample(std::vector<std::size_t> inputs,
                                         ResampleParams clock,
                                         std::vector<EvalOp*> reducers) {
        NodeSpec ns{NodeKind::MultiResample, nullptr, true, false, {}, clock,
                    std::move(inputs)};
        ns.reducers = std::move(reducers);
        spec_.nodes.push_back(std::move(ns));
        return spec_.nodes.size() - 1;
    }
    void set_outputs(std::vector<std::size_t> outs) { spec_.output_ids = std::move(outs); }
    const GraphSpec& spec() const { return spec_; }

private:
    GraphSpec spec_;
};

}} // namespace screamer::dag
#endif
