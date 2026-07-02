#include <cstring>
#include <memory>
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

template <class Key>
static py::object run_chain(std::vector<ScreamerBase*> fns,
                           py::array_t<Key> keys,
                           py::array_t<double> values,
                           bool return_keys) {
    auto vinfo = values.request();
    auto kinfo = keys.request();
    if (kinfo.shape[0] < vinfo.shape[0]) {
        throw std::runtime_error("run_chain: keys array is shorter than values array");
    }
    std::size_t n = static_cast<std::size_t>(vinfo.shape[0]);
    const Key* kptr = static_cast<const Key*>(kinfo.ptr);
    const double* vptr = static_cast<const double*>(vinfo.ptr);

    py::array_t<double> out_v(n);
    double* ov = static_cast<double*>(out_v.request().ptr);

    // Wire the functor chain in front of the chosen terminal sink.
    auto drive = [&](Sink<Key>& terminal) {
        Sink<Key>* downstream = &terminal;
        std::vector<std::unique_ptr<FunctorNode<Key>>> nodes;
        for (auto it = fns.rbegin(); it != fns.rend(); ++it) {
            (*it)->reset();
            nodes.push_back(std::make_unique<FunctorNode<Key>>(**it, *downstream));
            downstream = nodes.back().get();
        }
        VectorSource<Key> src(kptr, vptr, n);
        run_batch<Key>(src, *downstream);
        for (auto* f : fns) f->reset();
    };

    if (return_keys) {
        py::array_t<Key> out_k(n);
        Key* ok = static_cast<Key*>(out_k.request().ptr);
        CollectorSink<Key> collector(ok, ov);
        drive(collector);
        return py::make_tuple(out_k, out_v);
    }
    ValueCollectorSink<Key> collector(ov);
    drive(collector);
    return out_v;
}

// Shared setup: cast N (keys, values) numpy arrays, validate per-child length
// agreement, build a VectorSource per child, and collect non-owning child
// pointers. Returns the total event count (sum of child lengths). The caller
// owns `keys`/`vals` (Python refs keep buffers alive) and `sources`.
template <class Key>
static std::size_t build_vector_sources(
        py::list key_arrays, py::list value_arrays,
        std::vector<py::array_t<Key>>& keys,
        std::vector<py::array_t<double>>& vals,
        std::vector<std::unique_ptr<VectorSource<Key>>>& sources,
        std::vector<Source<Key>*>& child_ptrs) {
    std::size_t n = key_arrays.size();
    if (value_arrays.size() != n) {
        throw std::runtime_error("streams: keys/values list length mismatch");
    }
    keys.reserve(n);
    vals.reserve(n);
    sources.reserve(n);
    child_ptrs.reserve(n);
    std::size_t total = 0;
    for (std::size_t i = 0; i < n; ++i) {
        keys.push_back(py::cast<py::array_t<Key>>(key_arrays[i]));
        vals.push_back(py::cast<py::array_t<double>>(value_arrays[i]));
        auto kinfo = keys[i].request();
        auto vinfo = vals[i].request();
        if (kinfo.shape[0] != vinfo.shape[0]) {
            throw std::runtime_error("streams: a child's keys/values length differ");
        }
        std::size_t len = static_cast<std::size_t>(kinfo.shape[0]);
        total += len;
        sources.push_back(std::make_unique<VectorSource<Key>>(
            static_cast<const Key*>(kinfo.ptr),
            static_cast<const double*>(vinfo.ptr), len));
        child_ptrs.push_back(sources.back().get());
    }
    return total;
}

template <class Key>
static py::tuple merge_batch(py::list key_arrays, py::list value_arrays) {
    std::vector<py::array_t<Key>> keys;
    std::vector<py::array_t<double>> vals;
    std::vector<std::unique_ptr<VectorSource<Key>>> sources;
    std::vector<Source<Key>*> child_ptrs;
    std::size_t total = build_vector_sources<Key>(key_arrays, value_arrays,
                                                  keys, vals, sources, child_ptrs);

    py::array_t<Key> out_k(total);
    py::array_t<double> out_v(total);
    py::array_t<std::uint32_t> out_s(total);
    Key* ok = static_cast<Key*>(out_k.request().ptr);
    double* ov = static_cast<double*>(out_v.request().ptr);
    std::uint32_t* os = static_cast<std::uint32_t*>(out_s.request().ptr);

    MergeSource<Key> merge(child_ptrs);
    std::size_t i = 0;
    while (auto e = merge.next()) {
        ok[i] = e->key;
        ov[i] = e->value;
        os[i] = e->source;
        ++i;
    }
    return py::make_tuple(out_k, out_v, out_s);
}

template <class Key>
class MergePuller {
public:
    MergePuller(py::list key_arrays, py::list value_arrays) {
        std::vector<Source<Key>*> child_ptrs;
        build_vector_sources<Key>(key_arrays, value_arrays, keys_, vals_, sources_, child_ptrs);
        merge_ = std::make_unique<MergeSource<Key>>(child_ptrs);
    }

    py::object next() {
        if (auto e = merge_->next()) {
            return py::make_tuple(e->key, e->value, e->source);
        }
        return py::none();
    }

private:
    std::vector<py::array_t<Key>> keys_;
    std::vector<py::array_t<double>> vals_;
    std::vector<std::unique_ptr<VectorSource<Key>>> sources_;
    std::unique_ptr<MergeSource<Key>> merge_;
};

template <class Key>
static py::tuple combine_latest_batch(py::list key_arrays,
                                      py::list value_arrays,
                                      bool when_all) {
    std::size_t n = key_arrays.size();
    if (n == 0) {
        throw std::runtime_error("combine_latest: needs at least one series");
    }

    std::vector<py::array_t<Key>> keys;
    std::vector<py::array_t<double>> vals;
    std::vector<std::unique_ptr<VectorSource<Key>>> sources;
    std::vector<Source<Key>*> child_ptrs;
    std::size_t total = build_vector_sources<Key>(key_arrays, value_arrays,
                                                  keys, vals, sources, child_ptrs);

    std::vector<Key> out_k;
    std::vector<double> out_v;
    out_k.reserve(total);
    out_v.reserve(total * n);

    CombineLatest cl(n, when_all);
    MergeSource<Key> merge(child_ptrs);
    while (auto e = merge.next()) {
        if (cl.on_event(e->source, e->value)) {
            out_k.push_back(e->key);
            const std::vector<double>& row = cl.latest();
            out_v.insert(out_v.end(), row.begin(), row.end());
        }
    }

    std::size_t m = out_k.size();
    py::array_t<Key> rk(static_cast<py::ssize_t>(m));
    if (m) std::memcpy(rk.request().ptr, out_k.data(), m * sizeof(Key));
    py::array_t<double> rv({static_cast<py::ssize_t>(m), static_cast<py::ssize_t>(n)});
    if (m) std::memcpy(rv.request().ptr, out_v.data(), m * n * sizeof(double));
    return py::make_tuple(rk, rv);
}

template <class Key>
class CombineLatestPuller {
public:
    CombineLatestPuller(py::list key_arrays, py::list value_arrays, bool when_all)
        : n_(key_arrays.size()), cl_(key_arrays.size(), when_all) {
        if (n_ == 0) {
            throw std::runtime_error("combine_latest: needs at least one series");
        }
        std::vector<Source<Key>*> child_ptrs;
        build_vector_sources<Key>(key_arrays, value_arrays, keys_, vals_, sources_, child_ptrs);
        merge_ = std::make_unique<MergeSource<Key>>(child_ptrs);
    }

    py::object next() {
        while (auto e = merge_->next()) {
            if (cl_.on_event(e->source, e->value)) {
                const std::vector<double>& row = cl_.latest();
                py::tuple t(row.size());
                for (std::size_t j = 0; j < row.size(); ++j) t[j] = row[j];
                return py::make_tuple(e->key, t);
            }
        }
        return py::none();
    }

private:
    std::size_t n_;
    std::vector<py::array_t<Key>> keys_;
    std::vector<py::array_t<double>> vals_;
    std::vector<std::unique_ptr<VectorSource<Key>>> sources_;
    std::unique_ptr<MergeSource<Key>> merge_;
    CombineLatest cl_;
};

void init_bindings_streams(py::module& m) {
    m.def("_run_chain_i64", &run_chain<std::int64_t>,
          py::arg("functors"), py::arg("keys"), py::arg("values"),
          py::arg("return_keys") = false);
    m.def("_run_chain_f64", &run_chain<double>,
          py::arg("functors"), py::arg("keys"), py::arg("values"),
          py::arg("return_keys") = false);
    m.def("_merge_i64", &merge_batch<std::int64_t>,
          py::arg("key_arrays"), py::arg("value_arrays"));
    m.def("_merge_f64", &merge_batch<double>,
          py::arg("key_arrays"), py::arg("value_arrays"));
    py::class_<MergePuller<std::int64_t>>(m, "_MergePuller_i64")
        .def(py::init<py::list, py::list>())
        .def("next", &MergePuller<std::int64_t>::next);
    py::class_<MergePuller<double>>(m, "_MergePuller_f64")
        .def(py::init<py::list, py::list>())
        .def("next", &MergePuller<double>::next);
    m.def("_combine_latest_i64", &combine_latest_batch<std::int64_t>,
          py::arg("key_arrays"), py::arg("value_arrays"), py::arg("when_all"));
    m.def("_combine_latest_f64", &combine_latest_batch<double>,
          py::arg("key_arrays"), py::arg("value_arrays"), py::arg("when_all"));
    py::class_<CombineLatestPuller<std::int64_t>>(m, "_CombineLatestPuller_i64")
        .def(py::init<py::list, py::list, bool>())
        .def("next", &CombineLatestPuller<std::int64_t>::next);
    py::class_<CombineLatestPuller<double>>(m, "_CombineLatestPuller_f64")
        .def(py::init<py::list, py::list, bool>())
        .def("next", &CombineLatestPuller<double>::next);
}
