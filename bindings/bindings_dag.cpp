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
#include "screamer/streams/event.h"
#include "screamer/streams/merge_source.h"
#include "screamer/streams/py_source.h"
#include <algorithm>
#include <cassert>
#include <deque>
#include <tuple>

namespace py = pybind11;
using namespace screamer;

// Build the Python value for row r of an output buffer: a scalar for a width-1
// output, a tuple for a wider one.
static py::object make_output_value(const dag::OutputBuffer& b, std::size_t r,
                                    std::size_t w) {
    if (w == 1) return py::float_(b.values[r]);
    py::tuple t(w);
    for (std::size_t j = 0; j < w; ++j) t[j] = py::float_(b.values[r * w + j]);
    return std::move(t);
}

// ---------------------------------------------------------------------------
// LazyDriver: run a compiled graph over Python-iterator feeds, event by event,
// in C++. It merges the per-input (value, index) iterators by index
// (streams::MergeSource, ascending, same-index ties by input order), pushes each
// merged event into the CompiledGraph, drains the frames the graph emitted, and
// yields output rows - the same rows the Python _LazyDag produced. Holds a
// keepalive to the owning _CompiledGraph object.
//
// One output (n_out == 1): yield (value, index) rows, value scalar or tuple.
// M>1 co-indexed outputs: run the same watermark as-of join _LazyDag used. The M
// outputs drain at independent rates, so their merged index stream is NOT
// globally sorted; forward-filling naively would misalign. Instead buffer drained
// events and finalize an index only once every output has drained strictly past
// it (its as-of value is then settled), emitting one row per index with every
// output's latest value. Rows are (col0, ..., col_{M-1}, index).
// ---------------------------------------------------------------------------
class LazyDriver {
public:
    LazyDriver(py::object cg_keepalive, dag::CompiledGraph& cg, py::list iterators,
               std::size_t n_out)
        : cg_keepalive_(std::move(cg_keepalive)), cg_(cg), n_out_(n_out),
          latest_(n_out, py::none()), wm_val_(n_out, 0), wm_set_(n_out, 0) {
        for (py::handle h : iterators) {
            sources_.push_back(std::make_unique<streams::PySource<std::int64_t>>(
                py::reinterpret_borrow<py::object>(h), /*positional=*/false));
            child_ptrs_.push_back(sources_.back().get());
        }
        merge_ = std::make_unique<streams::MergeSource<std::int64_t>>(child_ptrs_);
        cg_.reset();
    }

    py::object next() {
        while (pending_.empty() && !done_) {
            if (auto e = merge_->next()) {
                cg_.push_event(static_cast<std::size_t>(e->source), e->index, e->value);
                collect();
            } else {
                cg_.flush();
                collect();
                if (n_out_ > 1) settle(0, /*has_bound=*/false);  // emit the tail
                done_ = true;
            }
        }
        if (pending_.empty()) throw py::stop_iteration();
        py::object row = std::move(pending_.front());
        pending_.pop_front();
        return row;
    }

private:
    void collect() { (n_out_ == 1) ? collect_single() : collect_multi(); }

    // One output: buffer each drained frame directly as a (value, index) row.
    void collect_single() {
        std::vector<dag::OutputBuffer> bufs = cg_.drain();
        assert(bufs.size() == 1 && "collect_single: graph must have one output");
        const dag::OutputBuffer& b = bufs[0];
        const std::size_t w = b.width;
        const std::size_t rows = b.indices.size();
        for (std::size_t r = 0; r < rows; ++r) {
            pending_.push_back(py::make_tuple(make_output_value(b, r, w),
                                              py::int_(b.indices[r])));
        }
    }

    // M>1 outputs: buffer this drain's events, advance each output's watermark,
    // then settle every index the join can now finalize (strictly below the
    // lowest watermark - no output can still emit below it).
    void collect_multi() {
        std::vector<dag::OutputBuffer> bufs = cg_.drain();
        assert(bufs.size() == n_out_ && "collect_multi: drain count must equal n_out");
        for (std::size_t out_pos = 0; out_pos < bufs.size(); ++out_pos) {
            const dag::OutputBuffer& b = bufs[out_pos];
            const std::size_t w = b.width;
            const std::size_t rows = b.indices.size();
            for (std::size_t r = 0; r < rows; ++r) {
                const std::int64_t k = b.indices[r];
                buf_.emplace_back(k, out_pos, make_output_value(b, r, w));
                if (!wm_set_[out_pos] || k > wm_val_[out_pos]) {
                    wm_val_[out_pos] = k;
                    wm_set_[out_pos] = 1;
                }
            }
        }
        for (unsigned char s : wm_set_) if (!s) return;   // an output has not fired
        std::int64_t bound = wm_val_[0];
        for (std::size_t j = 1; j < n_out_; ++j) bound = std::min(bound, wm_val_[j]);
        settle(bound, /*has_bound=*/true);
    }

    // Emit one row per distinct buffered index < bound (all buffered when
    // !has_bound), forward-filling each output's as-of value. Suppress until every
    // output has a value (when_all). Stable order: the last drained value wins per
    // (index, output).
    void settle(std::int64_t bound, bool has_bound) {
        if (buf_.empty()) return;
        std::vector<Buffered> keep, ready;
        for (const Buffered& e : buf_) {
            if (!has_bound || std::get<0>(e) < bound) ready.push_back(e);
            else keep.push_back(e);
        }
        if (ready.empty()) return;                        // buf_ left intact
        buf_ = std::move(keep);
        std::stable_sort(ready.begin(), ready.end(),
            [](const Buffered& a, const Buffered& b) {
                return std::get<0>(a) < std::get<0>(b);
            });
        std::size_t i = 0;
        const std::size_t n = ready.size();
        while (i < n) {
            const std::int64_t k = std::get<0>(ready[i]);
            while (i < n && std::get<0>(ready[i]) == k) {
                latest_[std::get<1>(ready[i])] = std::get<2>(ready[i]);
                ++i;
            }
            bool all_set = true;
            for (const py::object& o : latest_) if (o.is_none()) { all_set = false; break; }
            if (all_set) {
                py::tuple row(n_out_ + 1);
                for (std::size_t j = 0; j < n_out_; ++j) row[j] = latest_[j];
                row[n_out_] = py::int_(k);
                pending_.push_back(std::move(row));
            }
        }
    }

    using Buffered = std::tuple<std::int64_t, std::size_t, py::object>;  // (index, out, value)

    py::object cg_keepalive_;                 // keep the _CompiledGraph alive
    dag::CompiledGraph& cg_;
    std::size_t n_out_;
    std::vector<std::unique_ptr<streams::PySource<std::int64_t>>> sources_;
    std::vector<streams::Source<std::int64_t>*> child_ptrs_;
    std::unique_ptr<streams::MergeSource<std::int64_t>> merge_;
    std::deque<py::object> pending_;
    // Watermark as-of join state (M>1 outputs only).
    std::vector<py::object> latest_;          // each output's as-of value (None = unset)
    std::vector<std::int64_t> wm_val_;        // highest index drained per output
    std::vector<unsigned char> wm_set_;       // whether each output has drained yet
    std::vector<Buffered> buf_;               // drained-but-unsettled events
    bool done_ = false;
};

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

        void advance(std::int64_t now) { cg->advance(now); }

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
        .def("advance",     &PyCompiledGraph::advance, py::arg("now"))
        .def("drain",       &PyCompiledGraph::drain)
        .def("run_batch",   &PyCompiledGraph::run_batch, py::arg("feeds"));

    // Lazy driver over a single-output compiled graph and Python-iterator feeds.
    py::class_<LazyDriver>(m, "_LazyDriver")
        .def(py::init([](py::object cg_obj, py::list iterators, std::size_t n_out) {
                 PyCompiledGraph& pcg = cg_obj.cast<PyCompiledGraph&>();
                 return std::make_unique<LazyDriver>(cg_obj, *pcg.cg, iterators, n_out);
             }),
             py::arg("cg"), py::arg("iterators"), py::arg("n_out"))
        .def("__iter__", [](py::object self) { return self; })
        .def("__next__", &LazyDriver::next);

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

        std::size_t add_filter(std::vector<std::size_t> inputs) {
            return builder.add_filter(std::move(inputs));
        }

        std::size_t add_dropna(std::vector<std::size_t> inputs, bool how_all) {
            return builder.add_dropna(std::move(inputs), how_all);
        }

        std::size_t add_select(std::vector<std::size_t> inputs,
                               std::vector<std::size_t> columns) {
            return builder.add_select(std::move(inputs), std::move(columns));
        }

        std::size_t add_delay(std::vector<std::size_t> inputs, std::int64_t duration) {
            return builder.add_delay(std::move(inputs), duration);
        }

        std::size_t add_resample(std::vector<std::size_t> inputs, int mode, int agg,
                                 int label, std::int64_t width, std::int64_t origin,
                                 std::int64_t count, py::object reducer, int fill) {
            dag::ResampleParams rp;
            rp.mode   = static_cast<dag::ResampleMode>(mode);    // 0=ByIndex, 1=ByCount
            rp.agg    = static_cast<dag::ResampleAgg>(agg);      // 0..7 First..Ohlc
            rp.label  = static_cast<dag::ResampleLabel>(label);  // 0=Left, 1=Right
            rp.fill   = static_cast<dag::ResampleFill>(fill);    // 0=Skip, 1=Nan, 2=Carry
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
        .def("add_filter", [](PyGraphBuilder& b, std::vector<std::size_t> inputs) {
            return b.add_filter(std::move(inputs));
        }, py::arg("inputs"))
        .def("add_dropna", [](PyGraphBuilder& b,
                              std::vector<std::size_t> inputs, bool how_all) {
            return b.add_dropna(std::move(inputs), how_all);
        }, py::arg("inputs"), py::arg("how_all") = false)
        .def("add_select", [](PyGraphBuilder& b,
                              std::vector<std::size_t> inputs,
                              std::vector<std::size_t> columns) {
            return b.add_select(std::move(inputs), std::move(columns));
        }, py::arg("inputs"), py::arg("columns"))
        .def("add_delay", [](PyGraphBuilder& b, std::vector<std::size_t> inputs,
                             std::int64_t duration) {
            return b.add_delay(std::move(inputs), duration);
        }, py::arg("inputs"), py::arg("duration"))
        .def("add_resample", [](PyGraphBuilder& b, std::vector<std::size_t> inputs,
                                int mode, int agg, int label,
                                std::int64_t width, std::int64_t origin, std::int64_t count,
                                py::object reducer, int fill) {
            return b.add_resample(std::move(inputs), mode, agg, label, width, origin,
                                  count, reducer, fill);
        }, py::arg("inputs"), py::arg("mode"), py::arg("agg"), py::arg("label"),
           py::arg("width"), py::arg("origin"), py::arg("count"),
           py::arg("reducer") = py::none(), py::arg("fill") = 0)
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
