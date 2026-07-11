#include <cstdint>
#include <cstring>
#include <memory>
#include <queue>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "screamer/common/base.h"
#include "screamer/streams/event.h"
#include "screamer/streams/vector_source.h"
#include "screamer/streams/collector_sink.h"
#include "screamer/streams/functor_node.h"
#include "screamer/streams/driver.h"
#include "screamer/streams/merge_source.h"
#include "screamer/streams/combine_latest.h"

namespace py = pybind11;
using namespace screamer;
using namespace screamer::streams;

// ---------------------------------------------------------------------------
// PySource: a Source<Index> that pulls events from a Python iterator.
//
// positional=true  - item is a bare scalar; index = arrival counter (int64).
// positional=false - item is a (value, index) tuple; index extracted as Index.
//
// GIL note: next() is called from C++ while Python drives the puller, so the
// GIL is held. No acquire/release needed.
// ---------------------------------------------------------------------------
template <class Index>
class PySource : public Source<Index> {
public:
    PySource(py::object it, bool positional)
        : it_(std::move(it)), positional_(positional), counter_(0) {}

    std::optional<Event<Index>> next() override {
        py::object item;
        try {
            item = it_.attr("__next__")();
        } catch (py::error_already_set& e) {
            if (e.matches(PyExc_StopIteration)) return std::nullopt;
            throw;
        }
        Event<Index> ev;
        ev.source = 0;
        if (positional_) {
            ev.index = static_cast<Index>(counter_++);
            ev.value = item.cast<double>();
        } else {
            py::tuple tup = item.cast<py::tuple>();
            ev.value = tup[0].cast<double>();
            ev.index = tup[1].cast<Index>();
        }
        return ev;
    }

private:
    py::object it_;
    bool positional_;
    std::int64_t counter_;
};

// ---------------------------------------------------------------------------
// MergeLazyPuller: k-way merge of Python-iterator sources through a C++ heap.
//
// Uses deferred refill: after popping the winning event, the winning source is
// NOT refilled until the NEXT call to next(). This mirrors the Python generator
// "prime -> yield -> refill" pattern so that the number of Python iterator
// advances per next() call is identical to the old _merge_lazy implementation.
//
// Tuple order: (value, index_or_None, source) - identical to _merge_lazy yield.
// Positional sources: index emitted as py::none(); internal counter drives order.
// ---------------------------------------------------------------------------
template <class Index>
class MergeLazyPuller {
public:
    MergeLazyPuller(py::list iter_list, bool positional)
        : positional_(positional), pending_source_(-1) {
        std::size_t n = iter_list.size();
        sources_.reserve(n);
        child_ptrs_.reserve(n);
        for (py::handle h : iter_list) {
            sources_.push_back(
                std::make_unique<PySource<Index>>(h.cast<py::object>(), positional));
            child_ptrs_.push_back(sources_.back().get());
        }
        // Prime the heap with the first event from each child.
        for (std::size_t i = 0; i < n; ++i) {
            prime_child(i);
        }
    }

    py::object next() {
        // Deferred refill: advance the winning source from the previous call.
        if (pending_source_ >= 0) {
            prime_child(static_cast<std::size_t>(pending_source_));
            pending_source_ = -1;
        }
        if (heap_.empty()) return py::none();
        Node top = heap_.top();
        heap_.pop();
        pending_source_ = static_cast<int>(top.source);
        py::object idx = positional_ ? py::none() : py::cast(top.index);
        return py::make_tuple(top.value, idx, top.source);
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
static py::object run_chain(std::vector<ScreamerBase*> fns,
                           py::array_t<Index> index,
                           py::array_t<double> values,
                           bool return_index) {
    auto vinfo = values.request();
    auto kinfo = index.request();
    if (kinfo.shape[0] < vinfo.shape[0]) {
        throw std::runtime_error("run_chain: index array is shorter than values array");
    }
    std::size_t n = static_cast<std::size_t>(vinfo.shape[0]);
    const Index* kptr = static_cast<const Index*>(kinfo.ptr);
    const double* vptr = static_cast<const double*>(vinfo.ptr);

    py::array_t<double> out_v(n);
    double* ov = static_cast<double*>(out_v.request().ptr);

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
        py::array_t<Index> out_k(n);
        Index* ok = static_cast<Index*>(out_k.request().ptr);
        CollectorSink<Index> collector(ok, ov);
        drive(collector);
        return py::make_tuple(out_k, out_v);
    }
    ValueCollectorSink<Index> collector(ov);
    drive(collector);
    return out_v;
}

// Shared setup: cast N (index, values) numpy arrays, validate per-child length
// agreement, build a VectorSource per child, and collect non-owning child
// pointers. Returns the total event count (sum of child lengths). The caller
// owns `indices`/`vals` (Python refs keep buffers alive) and `sources`.
template <class Index>
static std::size_t build_vector_sources(
        py::list index_arrays, py::list value_arrays,
        std::vector<py::array_t<Index>>& indices,
        std::vector<py::array_t<double>>& vals,
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
        indices.push_back(py::cast<py::array_t<Index>>(index_arrays[i]));
        vals.push_back(py::cast<py::array_t<double>>(value_arrays[i]));
        auto kinfo = indices[i].request();
        auto vinfo = vals[i].request();
        if (kinfo.shape[0] != vinfo.shape[0]) {
            throw std::runtime_error("streams: a child's index/values length differ");
        }
        std::size_t len = static_cast<std::size_t>(kinfo.shape[0]);
        total += len;
        sources.push_back(std::make_unique<VectorSource<Index>>(
            static_cast<const Index*>(kinfo.ptr),
            static_cast<const double*>(vinfo.ptr), len));
        child_ptrs.push_back(sources.back().get());
    }
    return total;
}

template <class Index>
static py::tuple merge_batch(py::list index_arrays, py::list value_arrays) {
    std::vector<py::array_t<Index>> indices;
    std::vector<py::array_t<double>> vals;
    std::vector<std::unique_ptr<VectorSource<Index>>> sources;
    std::vector<Source<Index>*> child_ptrs;
    std::size_t total = build_vector_sources<Index>(index_arrays, value_arrays,
                                                    indices, vals, sources, child_ptrs);

    py::array_t<Index> out_k(total);
    py::array_t<double> out_v(total);
    py::array_t<std::uint32_t> out_s(total);
    Index* ok = static_cast<Index*>(out_k.request().ptr);
    double* ov = static_cast<double*>(out_v.request().ptr);
    std::uint32_t* os = static_cast<std::uint32_t*>(out_s.request().ptr);

    MergeSource<Index> merge(child_ptrs);
    std::size_t i = 0;
    while (auto e = merge.next()) {
        ok[i] = e->index;
        ov[i] = e->value;
        os[i] = e->source;
        ++i;
    }
    return py::make_tuple(out_k, out_v, out_s);
}

template <class Index>
class MergePuller {
public:
    MergePuller(py::list index_arrays, py::list value_arrays) {
        std::vector<Source<Index>*> child_ptrs;
        build_vector_sources<Index>(index_arrays, value_arrays, indices_, vals_, sources_, child_ptrs);
        merge_ = std::make_unique<MergeSource<Index>>(child_ptrs);
    }

    py::object next() {
        if (auto e = merge_->next()) {
            return py::make_tuple(e->index, e->value, e->source);
        }
        return py::none();
    }

private:
    std::vector<py::array_t<Index>> indices_;
    std::vector<py::array_t<double>> vals_;
    std::vector<std::unique_ptr<VectorSource<Index>>> sources_;
    std::unique_ptr<MergeSource<Index>> merge_;
};

template <class Index>
static py::tuple combine_latest_batch(py::list index_arrays,
                                      py::list value_arrays,
                                      bool when_all) {
    std::size_t n = index_arrays.size();
    if (n == 0) {
        throw std::runtime_error("combine_latest: needs at least one stream");
    }

    std::vector<py::array_t<Index>> indices;
    std::vector<py::array_t<double>> vals;
    std::vector<std::unique_ptr<VectorSource<Index>>> sources;
    std::vector<Source<Index>*> child_ptrs;
    std::size_t total = build_vector_sources<Index>(index_arrays, value_arrays,
                                                    indices, vals, sources, child_ptrs);

    std::vector<Index> out_k;
    std::vector<double> out_v;
    out_k.reserve(total);
    out_v.reserve(total * n);

    CombineLatest cl(n, when_all);
    MergeSource<Index> merge(child_ptrs);
    while (auto e = merge.next()) {
        if (cl.on_event(e->source, e->value)) {
            out_k.push_back(e->index);
            const std::vector<double>& row = cl.latest();
            out_v.insert(out_v.end(), row.begin(), row.end());
        }
    }

    std::size_t m = out_k.size();
    py::array_t<Index> rk(static_cast<py::ssize_t>(m));
    if (m) std::memcpy(rk.request().ptr, out_k.data(), m * sizeof(Index));
    py::array_t<double> rv({static_cast<py::ssize_t>(m), static_cast<py::ssize_t>(n)});
    if (m) std::memcpy(rv.request().ptr, out_v.data(), m * n * sizeof(double));
    return py::make_tuple(rk, rv);
}

template <class Index>
class CombineLatestPuller {
public:
    CombineLatestPuller(py::list index_arrays, py::list value_arrays, bool when_all)
        : n_(index_arrays.size()), cl_(index_arrays.size(), when_all) {
        if (n_ == 0) {
            throw std::runtime_error("combine_latest: needs at least one stream");
        }
        std::vector<Source<Index>*> child_ptrs;
        build_vector_sources<Index>(index_arrays, value_arrays, indices_, vals_, sources_, child_ptrs);
        merge_ = std::make_unique<MergeSource<Index>>(child_ptrs);
    }

    py::object next() {
        while (auto e = merge_->next()) {
            if (cl_.on_event(e->source, e->value)) {
                const std::vector<double>& row = cl_.latest();
                py::tuple t(row.size());
                for (std::size_t j = 0; j < row.size(); ++j) t[j] = row[j];
                return py::make_tuple(e->index, t);
            }
        }
        return py::none();
    }

private:
    std::size_t n_;
    std::vector<py::array_t<Index>> indices_;
    std::vector<py::array_t<double>> vals_;
    std::vector<std::unique_ptr<VectorSource<Index>>> sources_;
    std::unique_ptr<MergeSource<Index>> merge_;
    CombineLatest cl_;
};

void init_bindings_streams(py::module& m) {
    m.def("_run_chain_i64", &run_chain<std::int64_t>,
          py::arg("functors"), py::arg("index"), py::arg("values"),
          py::arg("return_index") = false);
    m.def("_run_chain_f64", &run_chain<double>,
          py::arg("functors"), py::arg("index"), py::arg("values"),
          py::arg("return_index") = false);
    m.def("_merge_i64", &merge_batch<std::int64_t>,
          py::arg("index_arrays"), py::arg("value_arrays"));
    m.def("_merge_f64", &merge_batch<double>,
          py::arg("index_arrays"), py::arg("value_arrays"));
    py::class_<MergePuller<std::int64_t>>(m, "_MergePuller_i64")
        .def(py::init<py::list, py::list>())
        .def("next", &MergePuller<std::int64_t>::next);
    py::class_<MergePuller<double>>(m, "_MergePuller_f64")
        .def(py::init<py::list, py::list>())
        .def("next", &MergePuller<double>::next);
    m.def("_combine_latest_i64", &combine_latest_batch<std::int64_t>,
          py::arg("index_arrays"), py::arg("value_arrays"), py::arg("when_all"));
    m.def("_combine_latest_f64", &combine_latest_batch<double>,
          py::arg("index_arrays"), py::arg("value_arrays"), py::arg("when_all"));
    py::class_<CombineLatestPuller<std::int64_t>>(m, "_CombineLatestPuller_i64")
        .def(py::init<py::list, py::list, bool>())
        .def("next", &CombineLatestPuller<std::int64_t>::next);
    py::class_<CombineLatestPuller<double>>(m, "_CombineLatestPuller_f64")
        .def(py::init<py::list, py::list, bool>())
        .def("next", &CombineLatestPuller<double>::next);
    py::class_<MergeLazyPuller<std::int64_t>>(m, "_MergeLazyPuller_i64")
        .def(py::init<py::list, bool>())
        .def("next", &MergeLazyPuller<std::int64_t>::next);
    py::class_<MergeLazyPuller<double>>(m, "_MergeLazyPuller_f64")
        .def(py::init<py::list, bool>())
        .def("next", &MergeLazyPuller<double>::next);
}
