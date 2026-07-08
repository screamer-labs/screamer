#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "screamer/common/eval_op.h"
#include "screamer/dag/functor_node.h"
#include "screamer/dag/collector.h"
#include "screamer/dag/driver.h"
#include "screamer/dag/graph.h"
#include "screamer/dag/compiled_graph.h"

namespace py = pybind11;
using namespace screamer;

// Hand-wire source -> FunctorNode(op) -> collector and run it in batch.
// `values` is (T,) [width 1] or (T, W) [width W]; returns (T, op.n_out()).
static py::array_t<double> run_functor_batch(
        EvalOp& op,
        py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> index,
        py::array_t<double, py::array::c_style | py::array::forcecast> values) {
    auto vinfo = values.request();
    std::size_t T = static_cast<std::size_t>(vinfo.shape[0]);
    std::size_t width = (vinfo.ndim == 1)
        ? 1u : static_cast<std::size_t>(vinfo.shape[1]);
    if (width != op.n_in()) {
        throw std::runtime_error(
            "run_functor_batch: input width does not match op num_inputs");
    }
    std::size_t out_w = op.n_out();

    py::array_t<double> out({static_cast<py::ssize_t>(T),
                             static_cast<py::ssize_t>(out_w)});

    dag::Collector<std::int64_t> collector(
        static_cast<double*>(out.request().ptr), out_w);
    dag::FunctorNode<std::int64_t> node(op, collector);
    dag::replay_batch<std::int64_t>(
        static_cast<const std::int64_t*>(index.request().ptr),
        static_cast<const double*>(vinfo.ptr), T, width, node);
    return out;
}

// Marshal a gathered index/value buffer into a Python tuple (index_1d, values_2d).
static py::tuple marshal_gather(const std::vector<std::int64_t>& out_k,
                                const std::vector<double>& out_v,
                                std::size_t width) {
    std::size_t m = out_k.size();
    py::array_t<std::int64_t> rk(static_cast<py::ssize_t>(m));
    if (m) std::memcpy(rk.request().ptr, out_k.data(), m * sizeof(std::int64_t));
    py::array_t<double> rv({static_cast<py::ssize_t>(m),
                            static_cast<py::ssize_t>(width)});
    if (m) std::memcpy(rv.request().ptr, out_v.data(), m * width * sizeof(double));
    return py::make_tuple(rk, rv);
}

// Marshal a vector of OutputBuffers into a Python list of (index_1d, values_2d).
// Shared by _CompiledGraph.run_batch and _CompiledGraph.drain.
static py::list marshal_output_buffers(const std::vector<dag::OutputBuffer>& outs) {
    py::list result;
    for (const auto& o : outs)
        result.append(marshal_gather(o.indices, o.values, o.width));
    return result;
}

// Helper: marshal a list of (index, values) feed tuples into raw C++ spans.
// Fills ks/vs (keep-alive arrays), kp/vp (raw pointers), lens (lengths).
static void marshal_feeds(
        py::list feeds,
        std::vector<py::array_t<std::int64_t>>& ks,
        std::vector<py::array_t<double>>& vs,
        std::vector<const std::int64_t*>& kp,
        std::vector<const double*>& vp,
        std::vector<std::size_t>& lens) {
    for (auto item : feeds) {
        auto t = py::cast<py::tuple>(item);
        ks.push_back(py::cast<py::array_t<std::int64_t,
                     py::array::c_style | py::array::forcecast>>(t[0]));
        vs.push_back(py::cast<py::array_t<double,
                     py::array::c_style | py::array::forcecast>>(t[1]));
        kp.push_back(static_cast<const std::int64_t*>(ks.back().request().ptr));
        vp.push_back(static_cast<const double*>(vs.back().request().ptr));
        lens.push_back(static_cast<std::size_t>(vs.back().request().shape[0]));
    }
}

void init_bindings_dag(py::module& m) {
    m.def("_run_functor_batch", &run_functor_batch,
          py::arg("op"), py::arg("index"), py::arg("values"));

    // Compiled graph wrapper: holds a persistent CompiledGraph plus op_refs so
    // functor Python objects stay alive for the compiled graph's lifetime.
    struct PyCompiledGraph {
        std::vector<py::object> op_refs;  // destroyed AFTER cg (declared first)
        std::unique_ptr<dag::CompiledGraph> cg;  // destroyed FIRST (declared last)

        void reset() { cg->reset(); }

        void push_event(std::size_t input_idx, std::int64_t index, double value) {
            cg->push_event(input_idx, index, value);
        }

        void flush() { cg->flush(); }

        py::list drain() {
            return marshal_output_buffers(cg->drain());
        }

        py::list run_batch(py::list feeds) {
            std::vector<py::array_t<std::int64_t>> ks;
            std::vector<py::array_t<double>> vs;
            std::vector<const std::int64_t*> kp;
            std::vector<const double*> vp;
            std::vector<std::size_t> lens;
            marshal_feeds(feeds, ks, vs, kp, vp, lens);
            return marshal_output_buffers(cg->run_batch(kp, vp, lens));
        }
    };

    // Register _CompiledGraph before _GraphBuilder so compile() return type is known.
    py::class_<PyCompiledGraph>(m, "_CompiledGraph")
        .def("reset",       &PyCompiledGraph::reset)
        .def("push_event",  &PyCompiledGraph::push_event,
             py::arg("input_idx"), py::arg("index"), py::arg("value"))
        .def("flush",       &PyCompiledGraph::flush)
        .def("drain",       &PyCompiledGraph::drain)
        .def("run_batch",   &PyCompiledGraph::run_batch, py::arg("feeds"));

    // Python-facing GraphBuilder wrapper that keeps functor Python objects alive
    // for the lifetime of the builder (raw EvalOp* point into Python objects;
    // if the caller passes temporaries they'd be GC'd without this ref-holding).
    struct PyGraphBuilder {
        dag::GraphBuilder builder;
        std::vector<py::object> op_refs;  // keeps Python functor objects alive

        std::size_t add_input() { return builder.add_input(); }

        std::size_t add_functor(py::object op_obj, std::vector<std::size_t> inputs) {
            EvalOp* op = py::cast<EvalOp*>(op_obj);
            op_refs.push_back(op_obj);
            return builder.add_functor(op, std::move(inputs));
        }

        std::size_t add_combine_latest(std::vector<std::size_t> inputs, bool when_all) {
            return builder.add_combine_latest(std::move(inputs), when_all);
        }

        std::size_t add_dropna(std::vector<std::size_t> inputs, bool how_all) {
            return builder.add_dropna(std::move(inputs), how_all);
        }

        std::size_t add_select(std::vector<std::size_t> inputs,
                               std::vector<std::size_t> columns) {
            return builder.add_select(std::move(inputs), std::move(columns));
        }

        std::size_t add_resample(std::vector<std::size_t> inputs, int mode, int agg,
                                 int label, std::int64_t width, std::int64_t origin,
                                 std::int64_t count, py::object reducer) {
            dag::ResampleParams rp;
            rp.mode   = static_cast<dag::ResampleMode>(mode);    // 0=ByIndex, 1=ByCount
            rp.agg    = static_cast<dag::ResampleAgg>(agg);      // 0..7 First..Ohlc
            rp.label  = static_cast<dag::ResampleLabel>(label);  // 0=Left, 1=Right
            rp.width  = width;
            rp.origin = origin;
            rp.count  = count;
            // Optional functor reducer: extract the base EvalOp* and keep the Python
            // object alive for the compiled graph's lifetime (op_refs is copied into
            // the _CompiledGraph at compile()). Raw pointer would else dangle on GC.
            if (!reducer.is_none()) {
                EvalOp* op = py::cast<EvalOp*>(reducer);
                rp.reducer = op;
                op_refs.push_back(reducer);
            }
            return builder.add_resample(std::move(inputs), rp);
        }

        void set_outputs(std::vector<std::size_t> outs) {
            builder.set_outputs(std::move(outs));
        }

        const dag::GraphSpec& spec() const { return builder.spec(); }

        // Compile the accumulated spec into a persistent CompiledGraph.
        // The returned _CompiledGraph also holds op_refs so functors stay alive.
        PyCompiledGraph compile() {
            PyCompiledGraph pcg;
            pcg.cg = std::make_unique<dag::CompiledGraph>(builder.spec());
            pcg.op_refs = op_refs;
            return pcg;
        }
    };

    // DAG compiler: _GraphBuilder accumulates a GraphSpec.
    // run_batch compiles and drives it fresh each call (rebuild-per-run).
    // compile() returns a persistent _CompiledGraph for streaming use.
    py::class_<PyGraphBuilder>(m, "_GraphBuilder")
        .def(py::init<>())
        .def("add_input", &PyGraphBuilder::add_input)
        .def("add_functor", [](PyGraphBuilder& b, py::object op,
                               std::vector<std::size_t> inputs) {
            return b.add_functor(op, std::move(inputs));
        }, py::arg("op"), py::arg("inputs"))
        .def("add_combine_latest", [](PyGraphBuilder& b,
                                      std::vector<std::size_t> inputs, bool when_all) {
            return b.add_combine_latest(std::move(inputs), when_all);
        }, py::arg("inputs"), py::arg("when_all") = true)
        .def("add_dropna", [](PyGraphBuilder& b,
                              std::vector<std::size_t> inputs, bool how_all) {
            return b.add_dropna(std::move(inputs), how_all);
        }, py::arg("inputs"), py::arg("how_all") = false)
        .def("add_select", [](PyGraphBuilder& b,
                              std::vector<std::size_t> inputs,
                              std::vector<std::size_t> columns) {
            return b.add_select(std::move(inputs), std::move(columns));
        }, py::arg("inputs"), py::arg("columns"))
        .def("add_resample", [](PyGraphBuilder& b, std::vector<std::size_t> inputs,
                                int mode, int agg, int label,
                                std::int64_t width, std::int64_t origin, std::int64_t count,
                                py::object reducer) {
            return b.add_resample(std::move(inputs), mode, agg, label, width, origin,
                                  count, reducer);
        }, py::arg("inputs"), py::arg("mode"), py::arg("agg"), py::arg("label"),
           py::arg("width"), py::arg("origin"), py::arg("count"),
           py::arg("reducer") = py::none())
        .def("set_outputs", &PyGraphBuilder::set_outputs, py::arg("output_ids"))
        .def("compile", [](PyGraphBuilder& b) { return b.compile(); })
        .def("run_batch", [](PyGraphBuilder& b, py::list feeds) {
            // Marshal feeds (list of (index, values) tuples) -> raw spans.
            std::vector<py::array_t<std::int64_t>> ks;
            std::vector<py::array_t<double>> vs;
            std::vector<const std::int64_t*> kp;
            std::vector<const double*> vp;
            std::vector<std::size_t> lens;
            marshal_feeds(feeds, ks, vs, kp, vp, lens);
            // Compile (stores spec) then run (builds + drives push-graph).
            dag::CompiledGraph g(b.spec());
            return marshal_output_buffers(g.run_batch(kp, vp, lens));
        }, py::arg("feeds"));
}
