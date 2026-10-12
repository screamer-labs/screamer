#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <string_view>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include "screamer/common/eval_op.h"
#include "screamer/dag/functor_node.h"
#include "screamer/dag/collector.h"
#include "screamer/dag/driver.h"
#include "screamer/dag/graph.h"
#include "screamer/dag/compiled_graph.h"
#include "screamer/streams/event.h"
#include "screamer/streams/merge_source.h"
#include "screamer/streams/py_source.h"
#include "screamer/common/gc.h"
#include <algorithm>
#include <cassert>
#include <deque>
#include <tuple>
#include <cstddef>
#include <utility>

namespace nb = nanobind;
using namespace nb::literals;
using namespace screamer;

namespace {

// Allocate a NEW numpy array that OWNS `data` (a `new T[...]` C-contiguous
// buffer) via an owner capsule with a delete[] deleter. Generic over T.
template <class T>
nb::object owned_array(T* data, const std::vector<size_t>& shape) {
    nb::capsule owner(data, [](void* p) noexcept { delete[] (T*) p; });
    return nb::cast(nb::ndarray<nb::numpy, T>(
        data, shape.size(), shape.data(), owner));
}

// Typed, C-contiguous 1-D input array (nanobind converts dtype/order if needed,
// mirroring pybind's `py::array_t<T, forcecast|c_style>` behavior).
template <class T>
using In1D = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

// Typed, C-contiguous input array of any rank (values may be 1-D or 2-D).
using InValues = nb::ndarray<const double, nb::c_contig, nb::device::cpu>;

}  // namespace

// Build the Python value for row r of an output buffer: a scalar for a width-1
// output, a tuple for a wider one.
static nb::object make_output_value(const dag::OutputBuffer& b, std::size_t r,
                                    std::size_t w) {
    if (w == 1) return nb::float_(b.values[r]);
    nb::list lst;
    for (std::size_t j = 0; j < w; ++j) lst.append(b.values[r * w + j]);
    return nb::steal(PySequence_Tuple(lst.ptr()));
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
    LazyDriver(nb::object cg_keepalive, dag::CompiledGraph& cg, nb::list iterators,
               std::size_t n_out)
        : cg_keepalive_(std::move(cg_keepalive)), cg_(cg), n_out_(n_out),
          latest_(n_out, nb::none()), wm_val_(n_out, 0), wm_set_(n_out, 0) {
        for (nb::handle h : iterators) {
            sources_.push_back(std::make_unique<streams::PySource<std::int64_t>>(
                nb::borrow<nb::object>(h), /*positional=*/false, sources_.size(), "Pipeline"));
            child_ptrs_.push_back(sources_.back().get());
        }
        merge_ = std::make_unique<streams::MergeSource<std::int64_t>>(child_ptrs_);
        cg_.reset();
    }

    nb::object next() {
        while (pending_.empty() && !done_) {
            if (auto e = merge_->next()) {
                const std::size_t src = static_cast<std::size_t>(e->source);
                const auto* py_src = sources_[src].get();
                if (py_src->is_wide()) {
                    const auto& wv = py_src->wide_values();
                    cg_.push_event_wide(src, e->index, wv.data(), wv.size());
                } else {
                    cg_.push_event(src, e->index, e->value);
                }
                collect();
            } else {
                cg_.flush();
                collect();
                if (n_out_ > 1) settle(0, /*has_bound=*/false);  // emit the tail
                done_ = true;
            }
        }
        if (pending_.empty()) throw nb::stop_iteration();
        nb::object row = std::move(pending_.front());
        pending_.pop_front();
        return row;
    }

    // Garbage-collector support (gc.h): the graph, the source iterators and the
    // buffered rows and values.
    int traverse(visitproc visit, void* arg) const {
        if (int r = gc_visit(cg_keepalive_, visit, arg)) return r;
        for (const auto& s : sources_)
            if (int r = s->traverse(visit, arg)) return r;
        if (int r = gc_visit_all(pending_, visit, arg)) return r;
        if (int r = gc_visit_all(latest_, visit, arg)) return r;
        for (const Buffered& e : buf_)
            if (int r = gc_visit(std::get<2>(e), visit, arg)) return r;
        return 0;
    }

    // Drops the sources and the buffered values; the iterator then ends. The
    // graph stays: cg_ points into it. The members are emptied before the
    // objects are released, so code a release runs sees an ended iterator.
    void clear() {
        done_ = true;
        auto pending = std::move(pending_);
        auto latest = std::move(latest_);
        auto buf = std::move(buf_);
        pending_.clear();
        latest_.clear();
        buf_.clear();
        for (auto& s : sources_) s->clear();
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
            pending_.push_back(nb::make_tuple(make_output_value(b, r, w),
                                              nb::int_(b.indices[r])));
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
            for (const nb::object& o : latest_) if (o.is_none()) { all_set = false; break; }
            if (all_set) {
                nb::list row;
                for (std::size_t j = 0; j < n_out_; ++j) row.append(latest_[j]);
                row.append(nb::int_(k));
                pending_.push_back(nb::steal(PySequence_Tuple(row.ptr())));
            }
        }
    }

    using Buffered = std::tuple<std::int64_t, std::size_t, nb::object>;  // (index, out, value)

    nb::object cg_keepalive_;                 // keep the _CompiledGraph alive
    dag::CompiledGraph& cg_;
    std::size_t n_out_;
    std::vector<std::unique_ptr<streams::PySource<std::int64_t>>> sources_;
    std::vector<streams::Source<std::int64_t>*> child_ptrs_;
    std::unique_ptr<streams::MergeSource<std::int64_t>> merge_;
    std::deque<nb::object> pending_;
    // Watermark as-of join state (M>1 outputs only).
    std::vector<nb::object> latest_;          // each output's as-of value (None = unset)
    std::vector<std::int64_t> wm_val_;        // highest index drained per output
    std::vector<unsigned char> wm_set_;       // whether each output has drained yet
    std::vector<Buffered> buf_;               // drained-but-unsettled events
    bool done_ = false;
};

// Hand-wire source -> FunctorNode(op) -> collector and run it in batch.
// `values` is (T,) [width 1] or (T, W) [width W]; returns (T, op.n_out()).
static nb::object run_functor_batch(
        EvalOp& op,
        In1D<std::int64_t> index,
        InValues values) {
    std::size_t T = static_cast<std::size_t>(values.shape(0));
    std::size_t width = (values.ndim() == 1)
        ? 1u : static_cast<std::size_t>(values.shape(1));
    if (width != op.n_in()) {
        throw std::runtime_error(
            "run_functor_batch: input width does not match op num_inputs");
    }
    std::size_t out_w = op.n_out();

    double* obuf = new double[(T * out_w) ? (T * out_w) : 1];

    dag::Collector<std::int64_t> collector(obuf, out_w);
    dag::FunctorNode<std::int64_t> node(op, collector);
    dag::replay_batch<std::int64_t>(index.data(), values.data(), T, width, node);
    return owned_array(obuf, {T, out_w});
}

// Marshal a gathered index/value buffer into a Python tuple (index_1d, values_2d).
static nb::object marshal_gather(const std::vector<std::int64_t>& out_k,
                                 const std::vector<double>& out_v,
                                 std::size_t width) {
    std::size_t m = out_k.size();
    std::int64_t* rk = new std::int64_t[m ? m : 1];
    if (m) std::memcpy(rk, out_k.data(), m * sizeof(std::int64_t));
    double* rv = new double[(m * width) ? (m * width) : 1];
    if (m) std::memcpy(rv, out_v.data(), m * width * sizeof(double));
    return nb::make_tuple(owned_array(rk, {m}), owned_array(rv, {m, width}));
}

// Marshal a vector of OutputBuffers into a Python list of (index_1d, values_2d).
// Shared by _CompiledGraph.run_batch and _CompiledGraph.drain.
static nb::list marshal_output_buffers(const std::vector<dag::OutputBuffer>& outs) {
    nb::list result;
    for (const auto& o : outs)
        result.append(marshal_gather(o.indices, o.values, o.width));
    return result;
}

// Helper: marshal a list of (index, values) feed tuples into raw C++ spans.
// Fills ks/vs (keep-alive ndarrays), kp/vp (raw pointers), lens (lengths).
// Also fills widths: 1 for a 1-D values array, W for a 2-D (N, W) values array.
static void marshal_feeds(
        nb::list feeds,
        std::vector<In1D<std::int64_t>>& ks,
        std::vector<InValues>& vs,
        std::vector<const std::int64_t*>& kp,
        std::vector<const double*>& vp,
        std::vector<std::size_t>& lens,
        std::vector<std::size_t>& widths) {
    for (auto item : feeds) {
        nb::tuple t = nb::cast<nb::tuple>(item);
        ks.push_back(nb::cast<In1D<std::int64_t>>(t[0]));
        vs.push_back(nb::cast<InValues>(t[1]));
        kp.push_back(ks.back().data());
        vp.push_back(vs.back().data());
        lens.push_back(static_cast<std::size_t>(vs.back().shape(0)));
        // Detect column width: 1 for 1-D arrays, shape[1] for 2-D arrays.
        widths.push_back(vs.back().ndim() >= 2
            ? static_cast<std::size_t>(vs.back().shape(1)) : 1u);
    }
}

void init_bindings_dag(nb::module_& m) {
    m.def("_run_functor_batch", &run_functor_batch,
          "op"_a, "index"_a, "values"_a);

    // Compiled graph wrapper: holds a persistent CompiledGraph plus op_refs so
    // functor Python objects stay alive for the compiled graph's lifetime.
    struct PyCompiledGraph {
        std::vector<nb::object> op_refs;  // destroyed AFTER cg (declared first)
        std::unique_ptr<dag::CompiledGraph> cg;  // destroyed FIRST (declared last)

        // The graph points into op_refs, so the collector only reads them (gc.h).
        int traverse(visitproc visit, void* arg) const {
            return gc_visit_all(op_refs, visit, arg);
        }

        void reset() { cg->reset(); }

        void push_event(std::size_t input_idx, std::int64_t index, double value) {
            cg->push_event(input_idx, index, value);
        }

        void push_event_wide(std::size_t input_idx, std::int64_t index,
                             InValues vals) {
            const double* ptr = vals.data();
            std::size_t w = static_cast<std::size_t>(vals.size());
            cg->push_event_wide(input_idx, index, ptr, w);
        }

        void flush() { cg->flush(); }

        void advance(std::int64_t now) { cg->advance(now); }

        nb::list drain() {
            return marshal_output_buffers(cg->drain());
        }

        nb::list run_batch(nb::list feeds) {
            std::vector<In1D<std::int64_t>> ks;
            std::vector<InValues> vs;
            std::vector<const std::int64_t*> kp;
            std::vector<const double*> vp;
            std::vector<std::size_t> lens;
            std::vector<std::size_t> widths;
            marshal_feeds(feeds, ks, vs, kp, vp, lens, widths);
            return marshal_output_buffers(cg->run_batch(kp, vp, lens, widths));
        }
    };

    // Register _CompiledGraph before _GraphBuilder so compile() return type is known.
    nb::class_<PyCompiledGraph>(m, "_CompiledGraph",
                                nb::type_slots(gc_traverse_slots<PyCompiledGraph>))
        .def("reset",            &PyCompiledGraph::reset)
        .def("push_event",       &PyCompiledGraph::push_event,
             "input_idx"_a, "index"_a, "value"_a)
        .def("push_event_wide",  &PyCompiledGraph::push_event_wide,
             "input_idx"_a, "index"_a, "values"_a)
        .def("flush",            &PyCompiledGraph::flush)
        .def("advance",          &PyCompiledGraph::advance, "now"_a)
        .def("drain",            &PyCompiledGraph::drain)
        .def("run_batch",        &PyCompiledGraph::run_batch, "feeds"_a);

    // Live session: the per-event push, in C++ (name lookup, one conversion).
    // The Python _LiveDag subclasses this and adds result().
    struct PyLiveSession {
        nb::object cg_obj;               // the _CompiledGraph (keeps its ops alive)
        PyCompiledGraph* pcg;
        nb::object asarray, float64;     // numpy, for the sequence path
        std::map<std::string, std::size_t, std::less<>> names;
        std::vector<double> scratch;     // the plain-sequence conversion buffer

        PyLiveSession(nb::object cg, nb::list input_names)
            : cg_obj(std::move(cg)), pcg(&nb::cast<PyCompiledGraph&>(cg_obj)) {
            nb::module_ np = nb::module_::import_("numpy");
            asarray = np.attr("asarray");
            float64 = np.attr("float64");
            std::size_t i = 0;
            for (nb::handle h : input_names) names[nb::cast<std::string>(h)] = i++;
        }

        int traverse(visitproc visit, void* arg) const {
            if (int r = gc_visit(cg_obj, visit, arg)) return r;
            if (int r = gc_visit(asarray, visit, arg)) return r;
            return gc_visit(float64, visit, arg);
        }

        std::size_t source(nb::handle input) const {
            PyObject* p = input.ptr();
            if (PyUnicode_Check(p)) {
                Py_ssize_t n = 0;
                const char* c = PyUnicode_AsUTF8AndSize(p, &n);
                if (!c) throw nb::python_error();
                auto it = names.find(std::string_view(c, static_cast<std::size_t>(n)));
                if (it == names.end()) throw nb::key_error(c);
                return it->second;
            }
            PyObject* ix = PyNumber_Index(p);
            if (!ix) throw nb::python_error();
            Py_ssize_t k = PyNumber_AsSsize_t(ix, PyExc_OverflowError);
            Py_DECREF(ix);
            if (k == -1 && PyErr_Occurred()) throw nb::python_error();
            return static_cast<std::size_t>(k);   // negative wraps, the engine range-checks
        }

        static std::int64_t whole_index(double d) {
            if (!(d == std::floor(d)) || std::fabs(d) >= 9.2e18)
                throw nb::type_error(
                    "index must be integer-valued; a fractional index was seen. The "
                    "streaming engine is int64-indexed (float indices are not yet "
                    "supported). Pass integer indices.");
            return static_cast<std::int64_t>(d);
        }

        static std::int64_t index_of(nb::handle index) {
            PyObject* p = index.ptr();
            if (PyLong_Check(p)) {
                long long v = PyLong_AsLongLong(p);
                if (v == -1 && PyErr_Occurred()) throw nb::python_error();
                return v;
            }
            if (PyFloat_Check(p)) return whole_index(PyFloat_AsDouble(p));
            if (PyObject* ix = PyNumber_Index(p)) {   // NumPy integers
                long long v = PyLong_AsLongLong(ix);
                Py_DECREF(ix);
                if (v == -1 && PyErr_Occurred()) throw nb::python_error();
                return v;
            }
            PyErr_Clear();
            PyObject* f = PyNumber_Float(p);          // np.float32, Fraction ...
            if (!f) throw nb::python_error();
            double d = PyFloat_AsDouble(f);
            Py_DECREF(f);
            return whole_index(d);
        }

        // list / tuple of plain numbers, converted in one pass into a reused buffer.
        // Returns false (nothing pushed) for any other item, which takes the general path.
        bool push_plain_sequence(std::size_t src, std::int64_t t, PyObject* p) {
            const bool is_list = PyList_Check(p);
            const Py_ssize_t n = is_list ? PyList_Size(p) : PyTuple_Size(p);
            if (n <= 0) return false;
            scratch.resize(static_cast<std::size_t>(n));
            for (Py_ssize_t i = 0; i < n; ++i) {
                PyObject* x = is_list ? PyList_GetItem(p, i) : PyTuple_GetItem(p, i);
                if (PyFloat_Check(x)) scratch[i] = PyFloat_AsDouble(x);
                else if (PyLong_Check(x)) {
                    double d = PyLong_AsDouble(x);
                    if (d == -1.0 && PyErr_Occurred()) throw nb::python_error();
                    scratch[i] = d;
                } else return false;
            }
            pcg->cg->push_event_wide(src, t, scratch.data(), scratch.size());
            return true;
        }

        void push_sequence(std::size_t src, std::int64_t t, nb::handle value) {
            PyObject* p = value.ptr();
            if ((PyList_Check(p) || PyTuple_Check(p)) && push_plain_sequence(src, t, p)) return;
            InValues fast;   // a C-contiguous float64 array is used as it is, no conversion
            if (nb::try_cast<InValues>(value, fast, false) && fast.ndim() == 1 && fast.size() > 0) {
                pcg->cg->push_event_wide(src, t, fast.data(), static_cast<std::size_t>(fast.size()));
                return;
            }
            nb::object arr = asarray(value, nb::arg("dtype") = float64);
            InValues v = nb::cast<InValues>(arr);
            if (v.ndim() == 0) {                   // a 0-d array is a scalar
                pcg->cg->push_event(src, t, v.data()[0]);
                return;
            }
            if (v.ndim() != 1)
                throw nb::value_error(
                    ("a wide event is a 1-D sequence of numbers, got " +
                     std::to_string(v.ndim()) + " dimensions").c_str());
            pcg->cg->push_event_wide(src, t, v.data(), static_cast<std::size_t>(v.size()));
        }

        void push(nb::handle input, nb::handle index, nb::handle value) {
            const std::size_t src = source(input);
            const std::int64_t t = index_of(index);
            PyObject* p = value.ptr();
            if (PyFloat_Check(p)) {                // float, np.float64
                pcg->cg->push_event(src, t, PyFloat_AsDouble(p));
            } else if (PyLong_Check(p)) {          // int, bool
                double d = PyLong_AsDouble(p);
                if (d == -1.0 && PyErr_Occurred()) throw nb::python_error();
                pcg->cg->push_event(src, t, d);
            } else if (p == Py_None) {             // a missing value, NaN (A9)
                pcg->cg->push_event(src, t, std::numeric_limits<double>::quiet_NaN());
            } else if (PyBytes_Check(p) || PyByteArray_Check(p)) {
                throw nb::type_error("a value must be a number or a sequence of numbers, not bytes");
            } else if (!PyUnicode_Check(p) && PySequence_Check(p)) {
                push_sequence(src, t, value);      // list, tuple, ndarray, Series ...
            } else {                               // NumPy scalars, numeric strings
                double d = PyFloat_AsDouble(p);    // via __float__
                if (d == -1.0 && PyErr_Occurred()) {
                    PyErr_Clear();
                    PyObject* f = PyNumber_Float(p);   // strings, with the usual message
                    if (!f) throw nb::python_error();
                    d = PyFloat_AsDouble(f);
                    Py_DECREF(f);
                }
                pcg->cg->push_event(src, t, d);
            }
        }
    };

    nb::class_<PyLiveSession>(m, "_LiveSession", nb::dynamic_attr(),
                              nb::type_slots(gc_traverse_slots<PyLiveSession>))
        .def(nb::init<nb::object, nb::list>(), "cg"_a, "input_names"_a)
        .def("push", [](nb::handle self, nb::handle input, nb::handle index,
                        nb::handle value) {
                 nb::inst_ptr<PyLiveSession>(self)->push(input, index, value);
                 return nb::borrow(self);
             }, "input"_a, "index"_a, "value"_a.none());

    // Lazy driver over a single-output compiled graph and Python-iterator feeds.
    nb::class_<LazyDriver>(m, "_LazyDriver", nb::type_slots(gc_slots<LazyDriver>))
        .def("__init__", [](LazyDriver* self, nb::object cg_obj, nb::list iterators,
                            std::size_t n_out) {
                 PyCompiledGraph& pcg = nb::cast<PyCompiledGraph&>(cg_obj);
                 new (self) LazyDriver(cg_obj, *pcg.cg, iterators, n_out);
             },
             "cg"_a, "iterators"_a, "n_out"_a)
        .def("__iter__", [](nb::object self) { return self; })
        .def("__next__", &LazyDriver::next);

    // Python-facing GraphBuilder wrapper that keeps functor Python objects alive
    // for the lifetime of the builder (raw EvalOp* point into Python objects;
    // if the caller passes temporaries they'd be GC'd without this ref-holding).
    struct PyGraphBuilder {
        dag::GraphBuilder builder;
        std::vector<nb::object> op_refs;  // keeps Python functor objects alive

        // The builder points into op_refs, so the collector only reads them (gc.h).
        int traverse(visitproc visit, void* arg) const {
            return gc_visit_all(op_refs, visit, arg);
        }

        std::size_t add_input() { return builder.add_input(); }

        std::size_t add_functor(nb::object op_obj, std::vector<std::size_t> inputs) {
            EvalOp* op = nb::cast<EvalOp*>(op_obj);
            op_refs.push_back(op_obj);
            return builder.add_functor(op, std::move(inputs));
        }

        std::size_t add_combine_latest(std::vector<std::size_t> inputs, bool when_all,
                                       std::size_t max_pending = 1'000'000) {
            return builder.add_combine_latest(std::move(inputs), when_all, max_pending);
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
                                 std::int64_t count, nb::object reducer, int fill,
                                 nb::list plan, double threshold, std::int64_t max_age) {
            dag::ResampleParams rp;
            rp.mode      = static_cast<dag::ResampleMode>(mode);    // 0=ByIndex, 1=ByCount, 2=ByCumulative
            rp.agg       = static_cast<dag::ResampleAgg>(agg);      // 0..9 First..SumNeg
            rp.label     = static_cast<dag::ResampleLabel>(label);  // 0=Left, 1=Right
            rp.fill      = static_cast<dag::ResampleFill>(fill);    // 0=Skip, 1=Nan, 2=Carry
            rp.width     = width;
            rp.origin    = origin;
            rp.count     = count;
            rp.threshold = threshold;
            rp.max_age   = max_age;
            // Optional per-column reducer plan (multi-column bar aggs).
            // Each plan entry is a (agg_code, input_col) tuple.
            for (auto item : plan) {
                nb::tuple t = nb::cast<nb::tuple>(item);
                dag::ResamplePlanEntry e{};
                e.agg       = static_cast<dag::ResampleAgg>(nb::cast<int>(t[0]));
                e.input_col = nb::cast<std::size_t>(t[1]);
                rp.plan.push_back(e);
            }
            // Optional functor reducer: extract the base EvalOp* and keep the Python
            // object alive for the compiled graph's lifetime (op_refs is copied into
            // the _CompiledGraph at compile()). Raw pointer would else dangle on GC.
            if (!reducer.is_none()) {
                EvalOp* op = nb::cast<EvalOp*>(reducer);
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
    nb::class_<PyGraphBuilder>(m, "_GraphBuilder",
                               nb::type_slots(gc_traverse_slots<PyGraphBuilder>))
        .def(nb::init<>())
        .def("add_input", &PyGraphBuilder::add_input)
        .def("add_functor", [](PyGraphBuilder& b, nb::object op,
                               std::vector<std::size_t> inputs) {
            return b.add_functor(op, std::move(inputs));
        }, "op"_a, "inputs"_a)
        .def("add_combine_latest", [](PyGraphBuilder& b,
                                      std::vector<std::size_t> inputs, bool when_all,
                                      std::size_t max_pending) {
            return b.add_combine_latest(std::move(inputs), when_all, max_pending);
        }, "inputs"_a, "when_all"_a = true,
           "max_pending"_a = static_cast<std::size_t>(1'000'000))
        .def("add_filter", [](PyGraphBuilder& b, std::vector<std::size_t> inputs) {
            return b.add_filter(std::move(inputs));
        }, "inputs"_a)
        .def("add_dropna", [](PyGraphBuilder& b,
                              std::vector<std::size_t> inputs, bool how_all) {
            return b.add_dropna(std::move(inputs), how_all);
        }, "inputs"_a, "how_all"_a = false)
        .def("add_select", [](PyGraphBuilder& b,
                              std::vector<std::size_t> inputs,
                              std::vector<std::size_t> columns) {
            return b.add_select(std::move(inputs), std::move(columns));
        }, "inputs"_a, "columns"_a)
        .def("add_delay", [](PyGraphBuilder& b, std::vector<std::size_t> inputs,
                             std::int64_t duration) {
            return b.add_delay(std::move(inputs), duration);
        }, "inputs"_a, "duration"_a)
        .def("add_resample", [](PyGraphBuilder& b, std::vector<std::size_t> inputs,
                                int mode, int agg, int label,
                                std::int64_t width, std::int64_t origin, std::int64_t count,
                                nb::object reducer, int fill, nb::list plan, double threshold,
                                std::int64_t max_age) {
            return b.add_resample(std::move(inputs), mode, agg, label, width, origin,
                                  count, reducer, fill, plan, threshold, max_age);
        }, "inputs"_a, "mode"_a, "agg"_a, "label"_a,
           "width"_a, "origin"_a, "count"_a,
           "reducer"_a = nb::none(), "fill"_a = 0,
           "plan"_a = nb::list(), "threshold"_a = 0.0,
           "max_age"_a = -1)
        .def("set_outputs", &PyGraphBuilder::set_outputs, "output_ids"_a)
        .def("compile", [](PyGraphBuilder& b) { return b.compile(); })
        .def("run_batch", [](PyGraphBuilder& b, nb::list feeds) {
            // Marshal feeds (list of (index, values) tuples) -> raw spans.
            std::vector<In1D<std::int64_t>> ks;
            std::vector<InValues> vs;
            std::vector<const std::int64_t*> kp;
            std::vector<const double*> vp;
            std::vector<std::size_t> lens;
            std::vector<std::size_t> widths;
            marshal_feeds(feeds, ks, vs, kp, vp, lens, widths);
            // Compile (stores spec) then run (builds + drives push-graph).
            dag::CompiledGraph g(b.spec());
            return marshal_output_buffers(g.run_batch(kp, vp, lens, widths));
        }, "feeds"_a);
}
