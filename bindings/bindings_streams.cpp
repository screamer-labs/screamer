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

void init_bindings_streams(py::module& m) {
    m.def("_run_chain_i64", &run_chain<std::int64_t>,
          py::arg("functors"), py::arg("keys"), py::arg("values"));
    m.def("_run_chain_f64", &run_chain<double>,
          py::arg("functors"), py::arg("keys"), py::arg("values"));
}
