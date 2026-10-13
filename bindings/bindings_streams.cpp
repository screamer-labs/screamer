#include <cstdint>
#include <cstring>
#include <memory>
#include <queue>
#include <vector>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include "screamer/common/base.h"
#include "screamer/streams/event.h"
#include "screamer/streams/vector_source.h"
#include "screamer/streams/collector_sink.h"
#include "screamer/streams/functor_node.h"
#include "screamer/streams/driver.h"
#include "screamer/streams/merge_source.h"
#include "screamer/streams/combine_latest.h"
#include "screamer/streams/py_source.h"
#include "screamer/streams/split.h"
#include "screamer/streams/join_plan.h"
#include "screamer/common/gc.h"
#include <algorithm>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>

namespace nb = nanobind;
using namespace nb::literals;
using namespace screamer;
using namespace screamer::streams;

namespace {

// Allocate a NEW numpy array that OWNS `data` (a `new T[...]` C-contiguous
// buffer) via an owner capsule with a delete[] deleter. Generic over T so it
// serves int64 / double / uint32 outputs. Mirrors detail::make_owned_array.
template <class T>
nb::object owned_array(T* data, const std::vector<size_t>& shape) {
    nb::capsule owner(data, [](void* p) noexcept { delete[] (T*) p; });
    return nb::cast(nb::ndarray<nb::numpy, T>(
        data, shape.size(), shape.data(), owner));
}

// A NEW numpy array over the storage of `v`, which it takes without a copy.
template <class T>
nb::object vector_array(std::vector<T>&& v, const std::vector<size_t>& shape) {
    auto* owned = new std::vector<T>(std::move(v));
    if (owned->empty()) owned->resize(1);   // a non-null data pointer for an empty array
    nb::capsule owner(owned, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
    return nb::cast(nb::ndarray<nb::numpy, T>(owned->data(), shape.size(), shape.data(), owner));
}

// Typed, C-contiguous 1-D input array (nanobind converts dtype/order if needed,
// mirroring pybind's `py::array_t<T>` forcecast-on-arg behavior).
template <class T>
using In1D = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

}  // namespace

// ---------------------------------------------------------------------------
// MergeLazyPuller: k-way merge of Python-iterator sources through a C++ heap.
//
// Uses deferred refill: after popping the winning event, the winning source is
// NOT refilled until the NEXT call to next(). This mirrors the Python generator
// "prime -> yield -> refill" pattern so that the number of Python iterator
// advances per next() call is identical to the old _merge_lazy implementation.
//
// Tuple order: (value, index_or_None, source) - identical to _merge_lazy yield.
// Positional sources: index emitted as nb::none(); internal counter drives order.
// ---------------------------------------------------------------------------
template <class Index>
class MergeLazyPuller {
public:
    MergeLazyPuller(nb::list iter_list, bool positional)
        : positional_(positional), pending_source_(-1),
          unit_(std::make_unique<screamer::TimeUnit>()) {
        std::size_t n = iter_list.size();
        sources_.reserve(n);
        child_ptrs_.reserve(n);
        for (nb::handle h : iter_list) {
            sources_.push_back(
                std::make_unique<PySource<Index>>(nb::borrow<nb::object>(h), positional,
                                                  sources_.size(), "merge", unit_.get()));
            child_ptrs_.push_back(sources_.back().get());
        }
        // Prime the heap with the first event from each child.
        for (std::size_t i = 0; i < n; ++i) {
            prime_child(i);
        }
    }

    // Non-copyable: holds unique_ptr sources. Explicitly deleting the copy ctor
    // makes std::is_copy_constructible report false, so nanobind does not try to
    // synthesize a copy (the vector<unique_ptr> member's copy ctor is declared
    // but uncompilable, which the builtin trait would otherwise treat as copyable).
    MergeLazyPuller(const MergeLazyPuller&) = delete;
    MergeLazyPuller& operator=(const MergeLazyPuller&) = delete;

    // Garbage-collector support (gc.h): the source iterators. After clear(),
    // next() returns None.
    int traverse(visitproc visit, void* arg) const {
        for (const auto& s : sources_)
            if (int r = s->traverse(visit, arg)) return r;
        return 0;
    }
    void clear() {
        heap_ = {};
        pending_source_ = -1;
        for (auto& s : sources_) s->clear();
    }

    nb::object next() {
        // Deferred refill: advance the winning source from the previous call.
        if (pending_source_ >= 0) {
            prime_child(static_cast<std::size_t>(pending_source_));
            pending_source_ = -1;
        }
        if (heap_.empty()) return nb::none();
        Node top = heap_.top();
        heap_.pop();
        pending_source_ = static_cast<int>(top.source);
        nb::object idx = positional_ ? nb::none() : nb::cast(top.index);
        return nb::make_tuple(top.value, idx, top.source);
    }

private:
    struct Node {
        Index index;
        std::uint32_t source;
        double value;
    };
    // Min-heap: smaller index first; ties -> smaller source index (stable order).
    struct Greater {
        bool operator()(const Node& a, const Node& b) const {
            if (a.index != b.index) return a.index > b.index;
            return a.source > b.source;
        }
    };

    void prime_child(std::size_t i) {
        if (auto e = child_ptrs_[i]->next()) {
            heap_.push(Node{e->index, static_cast<std::uint32_t>(i), e->value});
        }
    }

    bool positional_;
    int pending_source_;
    std::unique_ptr<screamer::TimeUnit> unit_;   // the index kind of every source
    std::vector<std::unique_ptr<PySource<Index>>> sources_;
    std::vector<Source<Index>*> child_ptrs_;
    std::priority_queue<Node, std::vector<Node>, Greater> heap_;
};

template <class Index>
static nb::object run_chain(std::vector<ScreamerBase*> fns,
                            In1D<Index> index,
                            In1D<double> values,
                            bool return_index) {
    if (index.shape(0) < values.shape(0)) {
        throw std::runtime_error("run_chain: index array is shorter than values array");
    }
    std::size_t n = static_cast<std::size_t>(values.shape(0));
    const Index* kptr = index.data();
    const double* vptr = values.data();

    double* ov = new double[n ? n : 1];

    // Wire the functor chain in front of the chosen terminal sink.
    auto drive = [&](Sink<Index>& terminal) {
        Sink<Index>* downstream = &terminal;
        std::vector<std::unique_ptr<FunctorNode<Index>>> nodes;
        for (auto it = fns.rbegin(); it != fns.rend(); ++it) {
            (*it)->reset();
            nodes.push_back(std::make_unique<FunctorNode<Index>>(**it, *downstream));
            downstream = nodes.back().get();
        }
        VectorSource<Index> src(kptr, vptr, n);
        run_batch<Index>(src, *downstream);
        for (auto* f : fns) f->reset();
    };

    if (return_index) {
        Index* ok = new Index[n ? n : 1];
        CollectorSink<Index> collector(ok, ov);
        drive(collector);
        return nb::make_tuple(owned_array(ok, {n}), owned_array(ov, {n}));
    }
    ValueCollectorSink<Index> collector(ov);
    drive(collector);
    return owned_array(ov, {n});
}

// Shared setup: cast N (index, values) numpy arrays, validate per-child length
// agreement, build a VectorSource per child, and collect non-owning child
// pointers. Returns the total event count (sum of child lengths). The caller
// owns `indices`/`vals` (ndarray refs keep buffers alive) and `sources`.
template <class Index>
static std::size_t build_vector_sources(
        nb::list index_arrays, nb::list value_arrays,
        std::vector<In1D<Index>>& indices,
        std::vector<In1D<double>>& vals,
        std::vector<std::unique_ptr<VectorSource<Index>>>& sources,
        std::vector<Source<Index>*>& child_ptrs) {
    std::size_t n = index_arrays.size();
    if (value_arrays.size() != n) {
        throw std::runtime_error("streams: index/values list length mismatch");
    }
    indices.reserve(n);
    vals.reserve(n);
    sources.reserve(n);
    child_ptrs.reserve(n);
    std::size_t total = 0;
    for (std::size_t i = 0; i < n; ++i) {
        indices.push_back(nb::cast<In1D<Index>>(index_arrays[i]));
        vals.push_back(nb::cast<In1D<double>>(value_arrays[i]));
        if (indices[i].shape(0) != vals[i].shape(0)) {
            throw std::runtime_error("streams: a child's index/values length differ");
        }
        std::size_t len = static_cast<std::size_t>(indices[i].shape(0));
        total += len;
        sources.push_back(std::make_unique<VectorSource<Index>>(
            indices[i].data(), vals[i].data(), len, i));
        child_ptrs.push_back(sources.back().get());
    }
    return total;
}

template <class Index>
static nb::object merge_batch(nb::list index_arrays, nb::list value_arrays) {
    std::vector<In1D<Index>> indices;
    std::vector<In1D<double>> vals;
    std::vector<std::unique_ptr<VectorSource<Index>>> sources;
    std::vector<Source<Index>*> child_ptrs;
    std::size_t total = build_vector_sources<Index>(index_arrays, value_arrays,
                                                    indices, vals, sources, child_ptrs);

    Index* ok = new Index[total ? total : 1];
    double* ov = new double[total ? total : 1];
    std::uint32_t* os = new std::uint32_t[total ? total : 1];

    MergeSource<Index> merge(child_ptrs);
    std::size_t i = 0;
    while (auto e = merge.next()) {
        ok[i] = e->index;
        ov[i] = e->value;
        os[i] = e->source;
        ++i;
    }
    return nb::make_tuple(owned_array(ok, {total}),
                          owned_array(ov, {total}),
                          owned_array(os, {total}));
}

// split: the partition and its rules are include/screamer/streams/split.h,
// shared with the WebAssembly binding. Python adapts the arguments
// (screamer/streams.py split): values as a float64 array of any rank with
// time first, sources as a list or tuple, or an integer, bool or float64
// array, the index as int64 or float64 or None. Returns
// [(values, index_or_None), ...], one pair per output; each output's values
// have the shape (rows, *values.shape[1:]).
using AnyArray = nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu>;
using AnyF64 = nb::ndarray<const double, nb::c_contig, nb::device::cpu>;

template <class Index>
static nb::list split_outputs(const SplitPlan& plan, const AnyF64& values, const Index* k) {
    const std::size_t n = plan.outputs();
    std::vector<std::size_t> shape(values.ndim());
    std::size_t width = 1;
    for (std::size_t d = 1; d < values.ndim(); ++d) {
        shape[d] = values.shape(d);
        width *= shape[d];
    }
    std::vector<double*> vptr(n);
    std::vector<Index*> kptr(k ? n : 0);
    nb::list out;
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t c = plan.counts()[i];
        shape[0] = c;
        vptr[i] = new double[c * width ? c * width : 1];
        nb::object ov = owned_array(vptr[i], shape);      // owns the buffer from here
        if (k) {
            kptr[i] = new Index[c ? c : 1];
            out.append(nb::make_tuple(ov, owned_array(kptr[i], {c})));
        } else {
            out.append(nb::make_tuple(ov, nb::none()));
        }
    }
    plan.scatter(values.data(), width, k, vptr.data(), kptr.data());
    return out;
}

// The text of a Python object for an error message: its repr.
static std::string repr_text(nb::handle x) { return nb::cast<std::string>(nb::repr(x)); }

// numpy.bool_, looked up once per call when NumPy is loaded (else null).
static PyObject* numpy_bool_type() {
    PyObject* np = PyImport_GetModule(nb::str("numpy").ptr());
    if (!np) { PyErr_Clear(); return nullptr; }
    PyObject* t = PyObject_GetAttrString(np, "bool_");
    Py_DECREF(np);
    if (!t) { PyErr_Clear(); return nullptr; }
    Py_DECREF(t);                                      // NumPy keeps the type alive
    return t;
}

// One source from a list item: an int (or anything with __index__) or a
// float under split_tag; a bool, None, a string or another kind is a wrong
// kind, and a nested sequence a shape error.
static std::uint32_t split_tag_py(nb::handle e, PyObject* np_bool) {
    PyObject* o = e.ptr();
    if (PyBool_Check(o) || (np_bool && Py_TYPE(o) == reinterpret_cast<PyTypeObject*>(np_bool)))
        throw_split_source_kind("a bool");
    if (PyFloat_Check(o)) return split_tag(PyFloat_AsDouble(o));
    if (PyObject* ix = PyNumber_Index(o)) {             // int, NumPy integer
        int overflow = 0;
        const long long v = PyLong_AsLongLongAndOverflow(ix, &overflow);
        Py_DECREF(ix);
        if (overflow) throw_split_source_range(repr_text(e));
        if (v == -1 && PyErr_Occurred()) throw nb::python_error();
        return split_tag(static_cast<std::int64_t>(v));
    }
    PyErr_Clear();
    if (PyList_Check(o) || PyTuple_Check(o) ||
        (nb::hasattr(e, "ndim") && nb::cast<int>(e.attr("ndim")) > 0))
        throw std::invalid_argument("split: sources must be 1-D (one source per event), got 2 dimensions");
    if (!is_text(e) && !e.is_none() && PyNumber_Check(o) && !PyComplex_Check(o)) {
        const double d = PyFloat_AsDouble(o);           // NumPy floating
        if (!(d == -1.0 && PyErr_Occurred())) return split_tag(d);
        PyErr_Clear();
    }
    throw_split_source_kind(e.is_none() ? "None" : "a " + nb::cast<std::string>(nb::type_name(e.type())));
}

static SplitPlan split_plan(nb::handle sources, std::size_t& count, std::size_t& ndim,
                            bool has_n, double n, std::vector<std::uint32_t>& tags) {
    PyObject* o = sources.ptr();
    if (PyList_Check(o) || PyTuple_Check(o)) {
        const bool list = PyList_Check(o);
        count = static_cast<std::size_t>(list ? PyList_Size(o) : PyTuple_Size(o));
        ndim = 1;
        tags.resize(count);
        PyObject* np_bool = numpy_bool_type();
        for (std::size_t j = 0; j < count; ++j)
            tags[j] = split_tag_py(nb::handle(list ? PyList_GetItem(o, j) : PyTuple_GetItem(o, j)), np_bool);
        return SplitPlan(std::move(tags), has_n, n);
    }
    AnyArray s = nb::cast<AnyArray>(sources, /*convert=*/false);
    ndim = s.ndim();
    count = ndim ? s.shape(0) : 1;
    const std::size_t total = ndim == 1 ? count : 0;  // the caller rejects ndim != 1 first
    using C = nb::dlpack::dtype_code;
    const nb::dlpack::dtype dt = s.dtype();
    const void* p = s.data();
    if (dt.code == static_cast<std::uint8_t>(C::Bool))
        return SplitPlan(static_cast<const bool*>(p), total, has_n, n);
    if (dt.code == static_cast<std::uint8_t>(C::UInt)) {
        if (dt.bits == 8)  return SplitPlan(static_cast<const std::uint8_t*>(p), total, has_n, n);
        if (dt.bits == 16) return SplitPlan(static_cast<const std::uint16_t*>(p), total, has_n, n);
        if (dt.bits == 32) return SplitPlan(static_cast<const std::uint32_t*>(p), total, has_n, n);
        if (dt.bits == 64) return SplitPlan(static_cast<const std::uint64_t*>(p), total, has_n, n);
    } else if (dt.code == static_cast<std::uint8_t>(C::Int)) {
        if (dt.bits == 8)  return SplitPlan(static_cast<const std::int8_t*>(p), total, has_n, n);
        if (dt.bits == 16) return SplitPlan(static_cast<const std::int16_t*>(p), total, has_n, n);
        if (dt.bits == 32) return SplitPlan(static_cast<const std::int32_t*>(p), total, has_n, n);
        if (dt.bits == 64) return SplitPlan(static_cast<const std::int64_t*>(p), total, has_n, n);
    } else if (dt.code == static_cast<std::uint8_t>(C::Float) && dt.bits == 64) {
        return SplitPlan(static_cast<const double*>(p), total, has_n, n);
    }
    throw nb::type_error("split: sources must be an integer or float64 array");
}

// n: None, or a whole number under the setting rule (an int, a NumPy integer,
// a bool). A float is a wrong kind, as for every integer setting, with the
// message the shared rule gives; Python has already refused a string.
static bool split_n(nb::handle n, double& out) {
    if (n.is_none()) return false;
    PyObject* o = n.ptr();
    if (!PyFloat_Check(o)) {
        if (PyObject* ix = PyNumber_Index(o)) {
            int overflow = 0;
            const long long v = PyLong_AsLongLongAndOverflow(ix, &overflow);
            Py_DECREF(ix);
            if (v == -1 && !overflow && PyErr_Occurred()) throw nb::python_error();
            // Beyond int64 is beyond the cap; the shared rule reports it.
            out = overflow ? (overflow > 0 ? 1e300 : -1e300) : static_cast<double>(v);
            if (overflow > 0)
                throw std::invalid_argument("split: n must be at most 2**31 - 1, got " + repr_text(n));
            return true;
        }
        PyErr_Clear();
    }
    throw_split_n_kind(PyFloat_Check(o) || nb::hasattr(n, "dtype") ? repr_text(n)
                       : "a " + nb::cast<std::string>(nb::type_name(n.type())));
}

static nb::list split_py(AnyF64 values, nb::handle sources, nb::handle index, nb::handle n) {
    const bool has_index = !index.is_none();
    AnyArray idx;
    if (has_index && !nb::try_cast(index, idx, /*convert=*/false))
        throw nb::type_error("split: index must be an int64 or float64 array");
    // The shapes first, so a bad call raises the same error as in JavaScript.
    std::size_t src_count = 0, src_ndim = 1;
    PyObject* so = sources.ptr();
    if (PyList_Check(so) || PyTuple_Check(so)) {
        src_count = static_cast<std::size_t>(PyList_Check(so) ? PyList_Size(so) : PyTuple_Size(so));
    } else {
        AnyArray s;
        if (!nb::try_cast(sources, s, /*convert=*/false))
            throw nb::type_error("split: sources must be an integer or float64 array");
        src_ndim = s.ndim();
        src_count = src_ndim ? s.shape(0) : 1;
    }
    check_split_inputs(values.ndim(), values.ndim() ? values.shape(0) : 0, src_ndim, src_count,
                       has_index, has_index ? idx.ndim() : 0,
                       has_index && idx.ndim() ? idx.shape(0) : 0);
    double nd = 0;
    const bool has_n = split_n(n, nd);
    std::vector<std::uint32_t> tags;
    const SplitPlan plan = split_plan(sources, src_count, src_ndim, has_n, nd, tags);
    if (!has_index) return split_outputs<double>(plan, values, nullptr);
    const nb::dlpack::dtype dt = idx.dtype();
    using C = nb::dlpack::dtype_code;
    if (dt.code == static_cast<std::uint8_t>(C::Int) && dt.bits == 64)
        return split_outputs(plan, values, static_cast<const std::int64_t*>(idx.data()));
    if (dt.code == static_cast<std::uint8_t>(C::Float) && dt.bits == 64)
        return split_outputs(plan, values, static_cast<const double*>(idx.data()));
    throw nb::type_error("split: index must be an int64 or float64 array");
}

// ---------------------------------------------------------------------------
// combine_latest: the layout join (streams/join_plan.h) and its Python forms.
// ---------------------------------------------------------------------------

// The layouts of an n-input join: None for every input, one string for all,
// or one entry (a string or None) per input.
static std::vector<std::optional<std::string>> join_layouts(nb::handle layout, std::size_t n) {
    if (layout.is_none()) return std::vector<std::optional<std::string>>(n);
    if (PyUnicode_Check(layout.ptr()))
        return std::vector<std::optional<std::string>>(n, nb::cast<std::string>(layout));
    std::vector<std::optional<std::string>> out;
    nb::list items = nb::list(layout);
    check_layout_count(items.size(), n);
    for (nb::handle h : items) {
        if (h.is_none()) out.emplace_back();
        else if (PyUnicode_Check(h.ptr())) out.emplace_back(nb::cast<std::string>(h));
        else throw nb::type_error(("combine_latest: a layout is a string or None, got " +
                                   std::string(type_short_name(h.type()))).c_str());
    }
    return out;
}

// A joined-rows result as a NumPy array: flat rows (m, width), or (m, *A, F)
// with asset axes. The array takes the storage, no copy, unless the rows
// are assembled.
static nb::object joined_values(const JoinPlan& plan, std::vector<double>&& flat, std::size_t m) {
    std::vector<std::size_t> shape{m};
    if (plan.assets.empty()) {
        shape.push_back(plan.width);
        return vector_array(std::move(flat), shape);
    }
    std::unique_ptr<double[]> out(new double[std::max<std::size_t>(1, m * plan.row_size())]);
    plan.assemble(flat.data(), m, out.get());
    for (std::size_t d : plan.row_shape()) shape.push_back(d);
    return owned_array(out.release(), shape);
}

// The batch combine_latest: value arrays of any rank, each read by its
// layout, and index arrays (sorted, one per input), or None for positional
// inputs (paired by position along the T axis their layouts name). Returns
// (index, values).
template <class Index>
static nb::object combine_latest_batch(nb::handle index_arrays, nb::list value_arrays,
                                       bool when_all, nb::handle layout) {
    using InAny = nb::ndarray<const double, nb::c_contig, nb::device::cpu>;
    const std::size_t n = value_arrays.size();
    std::vector<In1D<Index>> indices;
    std::vector<InAny> vals;
    std::vector<std::optional<Shape>> shapes;
    for (std::size_t i = 0; i < n; ++i) {
        vals.push_back(nb::cast<InAny>(value_arrays[i]));
        Shape sh;
        for (std::size_t d = 0; d < vals[i].ndim(); ++d) sh.push_back(vals[i].shape(d));
        shapes.push_back(std::move(sh));
    }
    const JoinPlan plan = plan_join(shapes, join_layouts(layout, n));
    std::vector<const Index*> ip;
    std::vector<std::size_t> lens;
    if (!index_arrays.is_none()) {
        nb::list idx = nb::cast<nb::list>(index_arrays);
        if (idx.size() != n) throw std::runtime_error("streams: index/values list length mismatch");
        for (std::size_t i = 0; i < n; ++i) {
            indices.push_back(nb::cast<In1D<Index>>(idx[i]));
            ip.push_back(indices[i].data());
            lens.push_back(indices[i].shape(0));
        }
    }
    std::vector<const double*> dp;
    for (const InAny& v : vals) dp.push_back(v.data());
    JoinedRows<Index> rows = combine_latest_join<Index>(plan, std::move(ip), std::move(lens), dp, when_all);
    const std::size_t m = rows.index.size();
    nb::object index = vector_array(std::move(rows.index), {m});
    return nb::make_tuple(index, joined_values(plan, std::move(rows.values), m));
}

// The plan of an iterator combine_latest, built from one event per source
// (None for a source with no events); the Python lazy forms use it to read
// events and assemble rows.
struct PyJoinPlan {
    JoinPlan plan;

    PyJoinPlan(nb::list shapes, nb::handle layout) {
        std::vector<std::optional<Shape>> heads;
        for (nb::handle h : shapes) {
            if (h.is_none()) { heads.emplace_back(); continue; }
            heads.push_back(nb::cast<Shape>(h));
        }
        plan = plan_join(heads, join_layouts(layout, heads.size()), /*events=*/true);
    }

    // One event of input i (float64, any rank) as its flat values.
    nb::object flat(std::size_t i, nb::ndarray<const double, nb::c_contig, nb::device::cpu> e) const {
        const JoinInput& in = plan.inputs.at(i);
        plan.check_event(i, e.size());
        std::vector<double> out(in.width);
        plan.read(i, e.data(), out.data());
        return vector_array(std::move(out), {in.width});
    }

    // One flat row (width values, any rank) as a joined (*A, F) row.
    nb::object row(nb::ndarray<const double, nb::c_contig, nb::device::cpu> r) const {
        if (r.size() != plan.width)
            throw nb::value_error("combine_latest: a joined row has the wrong width");
        std::vector<double> out(plan.row_size());
        plan.assemble(r.data(), 1, out.data());
        return vector_array(std::move(out), plan.row_shape());
    }

};

// The positional (no-index) iterator combine_latest over Python iterators:
// each step reads one item from every source as doubles and hands the step to
// the shared PositionalZip (join_plan.h), which checks it and joins the row.
// One field per source yields (tuple_of_floats, None); a wide join yields
// (row_array, None), the row (width,) or (*A, F).
class CombineZip {
public:
    CombineZip(nb::list sources, const PyJoinPlan& plan, nb::object as_float64)
        : zip_(plan.plan), as_float64_(std::move(as_float64)) {
        for (nb::handle h : sources) its_.push_back(nb::borrow<nb::object>(h));
        const std::size_t n = its_.size();
        items_.resize(n);
        keep_.resize(n);
        slot_.resize(n);
        events_.resize(n);
        sizes_.resize(n);
        out_.resize(plan.plan.row_size());
    }

    nb::object next() {
        const JoinPlan& plan = zip_.plan();
        const std::size_t n = its_.size();
        for (std::size_t i = 0; i < n; ++i) {
            PyObject* x = its_[i].is_valid() ? PyIter_Next(its_[i].ptr()) : nullptr;
            if (!x && PyErr_Occurred()) throw nb::python_error();
            items_[i] = x ? nb::steal(x) : nb::object();
        }
        for (std::size_t i = 0; i < n; ++i) {
            keep_[i] = nb::object();
            if (!items_[i].is_valid()) { events_[i] = nullptr; continue; }
            read_event(i, plan.wide);
        }
        if (!zip_.step(events_.data(), sizes_.data(), out_.data())) throw nb::stop_iteration();
        if (!plan.wide) {
            nb::object t = nb::steal(PyTuple_New(static_cast<Py_ssize_t>(n)));
            if (!t.is_valid()) throw nb::python_error();
            for (std::size_t i = 0; i < n; ++i) {
                PyObject* f = PyFloat_FromDouble(out_[i]);
                if (!f) throw nb::python_error();
                if (PyTuple_SetItem(t.ptr(), static_cast<Py_ssize_t>(i), f) != 0) throw nb::python_error();
            }
            return nb::make_tuple(t, nb::none());
        }
        return nb::make_tuple(vector_array(std::vector<double>(out_), plan.row_shape()), nb::none());
    }

    int traverse(visitproc visit, void* arg) const {
        if (int r = gc_visit_all(its_, visit, arg)) return r;
        if (int r = gc_visit_all(items_, visit, arg)) return r;
        if (int r = gc_visit_all(keep_, visit, arg)) return r;
        return gc_visit(as_float64_, visit, arg);
    }
    void clear() {
        auto its = std::move(its_);
        auto items = std::move(items_);
        auto keep = std::move(keep_);
        nb::object f = std::move(as_float64_);
        its_.clear();
        items_.clear();
        keep_.clear();
    }

private:
    // Item i of this step as doubles: a number (None is NaN) in its slot, or
    // an array (float64 as as_float64 reads it, any rank) held in keep_.
    void read_event(std::size_t i, bool wide) {
        using InAny = nb::ndarray<const double, nb::c_contig, nb::device::cpu>;
        const nb::object& x = items_[i];
        if (PyFloat_Check(x.ptr())) {                     // a float: no array
            slot_[i] = PyFloat_AsDouble(x.ptr());
            events_[i] = &slot_[i];
            sizes_[i] = 1;
            return;
        }
        if (is_text(x)) throw_text_value("", x);
        // One field: a number, the value rule of a scalar (None is NaN).
        if (!wide && !nb::hasattr(x, "__len__")) {
            if (!read_double(x, slot_[i])) throw_text_value("", x);
            events_[i] = &slot_[i];
            sizes_[i] = 1;
            return;
        }
        InAny e;
        keep_[i] = x;
        if (!nb::try_cast(keep_[i], e, /*convert=*/false)) {
            keep_[i] = as_float64_(x);
            e = nb::cast<InAny>(keep_[i]);
        }
        events_[i] = e.data();
        sizes_[i] = e.size();
    }

    PositionalZip zip_;
    std::vector<nb::object> its_, items_, keep_;
    nb::object as_float64_;
    std::vector<double> slot_, out_;
    std::vector<const double*> events_;
    std::vector<std::size_t> sizes_;
};

template <class Index>
class CombineLatestPuller {
public:
    CombineLatestPuller(nb::list index_arrays, nb::list value_arrays, bool when_all)
        : n_(index_arrays.size()), cl_(index_arrays.size(), when_all) {
        if (n_ == 0) {
            throw std::runtime_error("combine_latest: needs at least one stream");
        }
        std::vector<Source<Index>*> child_ptrs;
        build_vector_sources<Index>(index_arrays, value_arrays, indices_, vals_, sources_, child_ptrs);
        merge_ = std::make_unique<MergeSource<Index>>(child_ptrs);
    }

    nb::object next() {
        while (auto e = merge_->next()) {
            if (cl_.on_event(e->source, e->value)) {
                const std::vector<double>& row = cl_.latest();
                nb::list lst;
                for (std::size_t j = 0; j < row.size(); ++j) lst.append(row[j]);
                nb::object t = nb::steal(PySequence_Tuple(lst.ptr()));
                return nb::make_tuple(e->index, t);
            }
        }
        return nb::none();
    }

private:
    std::size_t n_;
    std::vector<In1D<Index>> indices_;
    std::vector<In1D<double>> vals_;
    std::vector<std::unique_ptr<VectorSource<Index>>> sources_;
    std::unique_ptr<MergeSource<Index>> merge_;
    CombineLatest cl_;
};

void init_bindings_streams(nb::module_& m) {
    m.def("_run_chain_i64", &run_chain<std::int64_t>,
          "functors"_a, "index"_a, "values"_a,
          "return_index"_a = false);
    m.def("_run_chain_f64", &run_chain<double>,
          "functors"_a, "index"_a, "values"_a,
          "return_index"_a = false);
    m.def("_merge_i64", &merge_batch<std::int64_t>,
          "index_arrays"_a, "value_arrays"_a);
    m.def("_merge_f64", &merge_batch<double>,
          "index_arrays"_a, "value_arrays"_a);
    m.def("_split", &split_py, "values"_a, "sources"_a, "index"_a.none(), "n"_a.none());
    m.def("_combine_latest_i64", &combine_latest_batch<std::int64_t>,
          "index_arrays"_a.none(), "value_arrays"_a, "when_all"_a, "layout"_a.none() = nb::none());
    m.def("_combine_latest_f64", &combine_latest_batch<double>,
          "index_arrays"_a.none(), "value_arrays"_a, "when_all"_a, "layout"_a.none() = nb::none());
    nb::class_<PyJoinPlan>(m, "_JoinPlan")
        .def(nb::init<nb::list, nb::handle>(), "shapes"_a, "layout"_a.none())
        .def_prop_ro("wide", [](const PyJoinPlan& p) { return p.plan.wide; })
        .def_prop_ro("assets", [](const PyJoinPlan& p) { return !p.plan.assets.empty(); })
        .def("in_order", [](const PyJoinPlan& p, std::size_t i) { return p.plan.inputs.at(i).in_order; })
        .def("flat", &PyJoinPlan::flat, "i"_a, "event"_a)
        .def("row", &PyJoinPlan::row, "row"_a);
    nb::class_<CombineZip>(m, "_CombineZip", nb::type_slots(gc_slots<CombineZip>))
        .def(nb::init<nb::list, const PyJoinPlan&, nb::object>(), "sources"_a, "plan"_a, "as_float64"_a)
        .def("__iter__", [](nb::handle self) { return self; })
        .def("__next__", &CombineZip::next);
    nb::class_<CombineLatestPuller<std::int64_t>>(m, "_CombineLatestPuller_i64")
        .def(nb::init<nb::list, nb::list, bool>())
        .def("next", &CombineLatestPuller<std::int64_t>::next);
    nb::class_<CombineLatestPuller<double>>(m, "_CombineLatestPuller_f64")
        .def(nb::init<nb::list, nb::list, bool>())
        .def("next", &CombineLatestPuller<double>::next);
    nb::class_<MergeLazyPuller<std::int64_t>>(m, "_MergeLazyPuller_i64",
                                         nb::type_slots(gc_slots<MergeLazyPuller<std::int64_t>>))
        .def(nb::init<nb::list, bool>())
        .def("next", &MergeLazyPuller<std::int64_t>::next);
    nb::class_<MergeLazyPuller<double>>(m, "_MergeLazyPuller_f64",
                                         nb::type_slots(gc_slots<MergeLazyPuller<double>>))
        .def(nb::init<nb::list, bool>())
        .def("next", &MergeLazyPuller<double>::next);
}
