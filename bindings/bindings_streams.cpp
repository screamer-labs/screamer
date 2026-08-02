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
#include <algorithm>
#include <cstddef>
#include <stdexcept>

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
        : positional_(positional), pending_source_(-1) {
        std::size_t n = iter_list.size();
        sources_.reserve(n);
        child_ptrs_.reserve(n);
        for (nb::handle h : iter_list) {
            sources_.push_back(
                std::make_unique<PySource<Index>>(nb::borrow<nb::object>(h), positional));
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
            indices[i].data(), vals[i].data(), len));
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

// Partition a tagged (values, sources, index) stream back into n per-source
// (values, index) pairs - the inverse of merge_batch. One counting pass sizes
// each output, a second scatters into it, O(N) total. Source order within each
// output is preserved (stable).
template <class Index>
static nb::object split_batch(In1D<double> values,
                              In1D<std::uint32_t> sources,
                              In1D<Index> index, int n) {
    const std::size_t total = static_cast<std::size_t>(values.shape(0));
    if (static_cast<std::size_t>(sources.shape(0)) != total ||
        static_cast<std::size_t>(index.shape(0)) != total) {
        throw std::runtime_error("split: values/sources/index length differ");
    }
    if (n < 0) {
        throw std::runtime_error("split: n must be non-negative");
    }
    const double* v = values.data();
    const std::uint32_t* s = sources.data();
    const Index* k = index.data();

    std::vector<std::size_t> counts(static_cast<std::size_t>(n), 0);
    for (std::size_t j = 0; j < total; ++j) {
        if (s[j] >= static_cast<std::uint32_t>(n)) {
            throw std::runtime_error("split: a source tag is >= n; events would be dropped");
        }
        counts[s[j]]++;
    }

    nb::list out;
    std::vector<double*> vptr(static_cast<std::size_t>(n));
    std::vector<Index*> kptr(static_cast<std::size_t>(n));
    std::vector<std::size_t> pos(static_cast<std::size_t>(n), 0);
    for (int i = 0; i < n; ++i) {
        double* ov = new double[counts[i] ? counts[i] : 1];
        Index* ok = new Index[counts[i] ? counts[i] : 1];
        vptr[i] = ov;
        kptr[i] = ok;
        out.append(nb::make_tuple(owned_array(ov, {counts[i]}),
                                  owned_array(ok, {counts[i]})));  // holds the buffers alive
    }
    for (std::size_t j = 0; j < total; ++j) {
        const std::uint32_t src = s[j];
        vptr[src][pos[src]] = v[j];
        kptr[src][pos[src]] = k[j];
        pos[src]++;
    }
    return out;
}

template <class Index>
static nb::object combine_latest_batch(nb::list index_arrays,
                                       nb::list value_arrays,
                                       bool when_all) {
    std::size_t n = index_arrays.size();
    if (n == 0) {
        throw std::runtime_error("combine_latest: needs at least one stream");
    }

    std::vector<In1D<Index>> indices;
    std::vector<In1D<double>> vals;
    std::vector<std::unique_ptr<VectorSource<Index>>> sources;
    std::vector<Source<Index>*> child_ptrs;
    std::size_t total = build_vector_sources<Index>(index_arrays, value_arrays,
                                                    indices, vals, sources, child_ptrs);

    std::vector<Index> out_k;
    std::vector<double> out_v;
    out_k.reserve(total);
    out_v.reserve(total * n);

    // Coalesce to one row per distinct index, exactly as the graph
    // CombineLatestNode does: same-index events update a buffered row; the row is
    // emitted when the index advances, and the final buffered row is emitted at
    // the end. MergeSource yields index-sorted events, so equal indices are
    // consecutive and this keeps the last row per index.
    CombineLatest cl(n, when_all);
    MergeSource<Index> merge(child_ptrs);
    bool has_buffered = false;
    Index buffered_index{};
    std::vector<double> buffered_row(n, 0.0);
    while (auto e = merge.next()) {
        if (cl.on_event(e->source, e->value)) {
            const std::vector<double>& row = cl.latest();
            if (has_buffered && e->index != buffered_index) {
                out_k.push_back(buffered_index);
                out_v.insert(out_v.end(), buffered_row.begin(), buffered_row.end());
            }
            buffered_index = e->index;
            std::copy(row.begin(), row.end(), buffered_row.begin());
            has_buffered = true;
        }
    }
    if (has_buffered) {
        out_k.push_back(buffered_index);
        out_v.insert(out_v.end(), buffered_row.begin(), buffered_row.end());
    }

    std::size_t m = out_k.size();
    Index* rk = new Index[m ? m : 1];
    if (m) std::memcpy(rk, out_k.data(), m * sizeof(Index));
    double* rv = new double[(m * n) ? (m * n) : 1];
    if (m) std::memcpy(rv, out_v.data(), m * n * sizeof(double));
    return nb::make_tuple(owned_array(rk, {m}), owned_array(rv, {m, n}));
}

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
    m.def("_split_i64", &split_batch<std::int64_t>,
          "values"_a, "sources"_a, "index"_a, "n"_a);
    m.def("_split_f64", &split_batch<double>,
          "values"_a, "sources"_a, "index"_a, "n"_a);
    m.def("_combine_latest_i64", &combine_latest_batch<std::int64_t>,
          "index_arrays"_a, "value_arrays"_a, "when_all"_a);
    m.def("_combine_latest_f64", &combine_latest_batch<double>,
          "index_arrays"_a, "value_arrays"_a, "when_all"_a);
    nb::class_<CombineLatestPuller<std::int64_t>>(m, "_CombineLatestPuller_i64")
        .def(nb::init<nb::list, nb::list, bool>())
        .def("next", &CombineLatestPuller<std::int64_t>::next);
    nb::class_<CombineLatestPuller<double>>(m, "_CombineLatestPuller_f64")
        .def(nb::init<nb::list, nb::list, bool>())
        .def("next", &CombineLatestPuller<double>::next);
    nb::class_<MergeLazyPuller<std::int64_t>>(m, "_MergeLazyPuller_i64")
        .def(nb::init<nb::list, bool>())
        .def("next", &MergeLazyPuller<std::int64_t>::next);
    nb::class_<MergeLazyPuller<double>>(m, "_MergeLazyPuller_f64")
        .def(nb::init<nb::list, bool>())
        .def("next", &MergeLazyPuller<double>::next);
}
