#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "screamer/common/eval_op.h"
#include "screamer/arithmetic.h"
#include "screamer/dag/frame.h"
#include "screamer/dag/functor_node.h"
#include "screamer/dag/collector.h"
#include "screamer/dag/driver.h"
#include "screamer/dag/combine_latest_node.h"
#include "screamer/dag/broadcast.h"
#include "screamer/streams/vector_source.h"
#include "screamer/streams/merge_source.h"

namespace py = pybind11;
using namespace screamer;

// Hand-wire source -> FunctorNode(op) -> collector and run it in batch.
// `values` is (T,) [width 1] or (T, W) [width W]; returns (T, op.n_out()).
static py::array_t<double> run_functor_batch(
        EvalOp& op,
        py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> keys,
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
        static_cast<const std::int64_t*>(keys.request().ptr),
        static_cast<const double*>(vinfo.ptr), T, width, node);
    return out;
}

// Shared helper: build a MergeSource over key_arrays/value_arrays and drive
// width-1 frames into node's ports. Returns total number of input events.
static std::size_t drive_ports(py::list key_arrays,
                               py::list value_arrays,
                               dag::CombineLatestNode<std::int64_t>& node) {
    std::size_t n = key_arrays.size();
    std::vector<py::array_t<std::int64_t>> keys;
    std::vector<py::array_t<double>> vals;
    std::vector<std::unique_ptr<streams::VectorSource<std::int64_t>>> srcs;
    std::vector<streams::Source<std::int64_t>*> child_ptrs;
    std::size_t total = 0;
    for (std::size_t i = 0; i < n; ++i) {
        keys.push_back(py::cast<py::array_t<std::int64_t>>(key_arrays[i]));
        vals.push_back(py::cast<py::array_t<double>>(value_arrays[i]));
        auto ki = keys[i].request(); auto vi = vals[i].request();
        std::size_t len = static_cast<std::size_t>(ki.shape[0]);
        total += len;
        srcs.push_back(std::make_unique<streams::VectorSource<std::int64_t>>(
            static_cast<const std::int64_t*>(ki.ptr),
            static_cast<const double*>(vi.ptr), len));
        child_ptrs.push_back(srcs.back().get());
    }
    streams::MergeSource<std::int64_t> merge(child_ptrs);

    double one;
    while (auto e = merge.next()) {
        one = e->value;
        dag::Frame<std::int64_t> f{e->key, &one, 1};
        node.port(e->source).push(f);
    }
    return total;
}

// Marshal a gathered key/value buffer into a Python tuple (keys_1d, values_2d).
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

static py::tuple run_combine_latest_batch(py::list key_arrays,
                                          py::list value_arrays,
                                          bool when_all) {
    std::size_t n = key_arrays.size();

    std::vector<std::int64_t> out_k;
    std::vector<double> out_v;

    struct Gather : dag::Sink<std::int64_t> {
        std::vector<std::int64_t>& k; std::vector<double>& v;
        Gather(std::vector<std::int64_t>& kk, std::vector<double>& vv) : k(kk), v(vv) {}
        void push(const dag::Frame<std::int64_t>& f) override {
            k.push_back(f.key);
            v.insert(v.end(), f.values, f.values + f.width);
        }
    } gather(out_k, out_v);

    dag::CombineLatestNode<std::int64_t> node(n, when_all, gather);
    drive_ports(key_arrays, value_arrays, node);

    return marshal_gather(out_k, out_v, n);
}

// CombineLatestNode(2) -> FunctorNode(Sub) -> collector.
static py::tuple run_combine_then_sub_batch(py::list key_arrays,
                                            py::list value_arrays,
                                            bool when_all) {
    std::size_t n = key_arrays.size();

    std::vector<std::int64_t> out_k;
    std::vector<double> out_v;

    struct Gather1 : dag::Sink<std::int64_t> {
        std::vector<std::int64_t>& k; std::vector<double>& v;
        Gather1(std::vector<std::int64_t>& kk, std::vector<double>& vv) : k(kk), v(vv) {}
        void push(const dag::Frame<std::int64_t>& f) override {
            k.push_back(f.key); v.push_back(f.values[0]);
        }
    } gather(out_k, out_v);

    screamer::Sub sub;                                        // 2->1 EvalOp
    dag::FunctorNode<std::int64_t> sub_node(sub, gather);
    dag::CombineLatestNode<std::int64_t> node(n, when_all, sub_node);
    drive_ports(key_arrays, value_arrays, node);

    return marshal_gather(out_k, out_v, 1);
}

// CombineLatestNode -> Broadcast -> [GatherA, GatherB]; returns both gathers.
static py::tuple run_combine_latest_fanout(py::list key_arrays,
                                           py::list value_arrays,
                                           bool when_all) {
    std::size_t n = key_arrays.size();

    std::vector<std::int64_t> out_k1, out_k2;
    std::vector<double> out_v1, out_v2;

    struct GatherN : dag::Sink<std::int64_t> {
        std::vector<std::int64_t>& k; std::vector<double>& v;
        GatherN(std::vector<std::int64_t>& kk, std::vector<double>& vv) : k(kk), v(vv) {}
        void push(const dag::Frame<std::int64_t>& f) override {
            k.push_back(f.key);
            v.insert(v.end(), f.values, f.values + f.width);
        }
    } gatherA(out_k1, out_v1), gatherB(out_k2, out_v2);

    dag::Broadcast<std::int64_t> bcast;
    bcast.add(gatherA);
    bcast.add(gatherB);

    dag::CombineLatestNode<std::int64_t> node(n, when_all, bcast);
    drive_ports(key_arrays, value_arrays, node);

    return py::make_tuple(marshal_gather(out_k1, out_v1, n),
                          marshal_gather(out_k2, out_v2, n));
}

void init_bindings_dag(py::module& m) {
    m.def("_run_functor_batch", &run_functor_batch,
          py::arg("op"), py::arg("keys"), py::arg("values"));
    m.def("_run_combine_latest_batch", &run_combine_latest_batch,
          py::arg("key_arrays"), py::arg("value_arrays"), py::arg("when_all"));
    m.def("_run_combine_then_sub_batch", &run_combine_then_sub_batch,
          py::arg("key_arrays"), py::arg("value_arrays"), py::arg("when_all"));
    m.def("_run_combine_latest_fanout", &run_combine_latest_fanout,
          py::arg("key_arrays"), py::arg("value_arrays"), py::arg("when_all"));
}
