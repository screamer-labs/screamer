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

namespace py = pybind11;
using namespace screamer;
using namespace screamer::streams;

// Build source -> fns[0] -> fns[1] -> ... -> collector, run it in batch, and
// return a tuple (out_keys, out_values). Keys pass through unchanged (1->1).
// reset() is called before and after to match the existing array path's batch
// semantics.
template <class Key>
static py::tuple run_chain(std::vector<ScreamerBase*> fns,
                            py::array_t<Key> keys,
                            py::array_t<double> values) {
    auto vinfo = values.request();
    auto kinfo = keys.request();
    std::size_t n = static_cast<std::size_t>(vinfo.shape[0]);

    if (kinfo.shape[0] < vinfo.shape[0]) {
        throw std::runtime_error("run_chain: keys array is shorter than values array");
    }

    py::array_t<Key> out_k(n);
    py::array_t<double> out_v(n);
    CollectorSink<Key> collector(static_cast<Key*>(out_k.request().ptr),
                                 static_cast<double*>(out_v.request().ptr));

    Sink<Key>* downstream = &collector;
    std::vector<std::unique_ptr<FunctorNode<Key>>> nodes;
    // Wire from the last functor back to the first so each points at its
    // successor; the final `downstream` is the head of the chain (fns[0]).
    for (auto it = fns.rbegin(); it != fns.rend(); ++it) {
        (*it)->reset();
        nodes.push_back(std::make_unique<FunctorNode<Key>>(**it, *downstream));
        downstream = nodes.back().get();
    }

    VectorSource<Key> src(static_cast<const Key*>(kinfo.ptr),
                          static_cast<const double*>(vinfo.ptr), n);
    run_batch<Key>(src, *downstream);

    for (auto* f : fns) f->reset();
    return py::make_tuple(out_k, out_v);
}

template <class Key>
static py::tuple merge_batch(py::list key_arrays, py::list value_arrays) {
    std::size_t n_children = key_arrays.size();
    if (value_arrays.size() != n_children) {
        throw std::runtime_error("merge: keys/values list length mismatch");
    }

    // Materialize child VectorSources and total length.
    std::vector<py::array_t<Key>> keys;
    std::vector<py::array_t<double>> vals;
    keys.reserve(n_children);
    vals.reserve(n_children);
    std::vector<std::unique_ptr<VectorSource<Key>>> sources;
    std::vector<Source<Key>*> child_ptrs;
    sources.reserve(n_children);
    child_ptrs.reserve(n_children);
    std::size_t total = 0;

    for (std::size_t i = 0; i < n_children; ++i) {
        keys.push_back(py::cast<py::array_t<Key>>(key_arrays[i]));
        vals.push_back(py::cast<py::array_t<double>>(value_arrays[i]));
        auto kinfo = keys[i].request();
        auto vinfo = vals[i].request();
        if (kinfo.shape[0] != vinfo.shape[0]) {
            throw std::runtime_error("merge: a child's keys/values length differ");
        }
        std::size_t n = static_cast<std::size_t>(kinfo.shape[0]);
        total += n;
        sources.push_back(std::make_unique<VectorSource<Key>>(
            static_cast<const Key*>(kinfo.ptr),
            static_cast<const double*>(vinfo.ptr), n));
        child_ptrs.push_back(sources.back().get());
    }

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
        std::size_t n_children = key_arrays.size();
        if (value_arrays.size() != n_children) {
            throw std::runtime_error("merge: keys/values list length mismatch");
        }
        for (std::size_t i = 0; i < n_children; ++i) {
            // Keep owning copies so the buffers outlive iteration.
            keys_.push_back(py::cast<py::array_t<Key>>(key_arrays[i]));
            vals_.push_back(py::cast<py::array_t<double>>(value_arrays[i]));
        }
        std::vector<Source<Key>*> child_ptrs;
        for (std::size_t i = 0; i < n_children; ++i) {
            auto kinfo = keys_[i].request();
            auto vinfo = vals_[i].request();
            if (kinfo.shape[0] != vinfo.shape[0]) {
                throw std::runtime_error("merge: a child's keys/values length differ");
            }
            std::size_t n = static_cast<std::size_t>(vinfo.shape[0]);
            sources_.push_back(std::make_unique<VectorSource<Key>>(
                static_cast<const Key*>(kinfo.ptr),
                static_cast<const double*>(vinfo.ptr), n));
            child_ptrs.push_back(sources_.back().get());
        }
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

void init_bindings_streams(py::module& m) {
    m.def("_run_chain_i64", &run_chain<std::int64_t>,
          py::arg("functors"), py::arg("keys"), py::arg("values"));
    m.def("_run_chain_f64", &run_chain<double>,
          py::arg("functors"), py::arg("keys"), py::arg("values"));
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
}
