// Shape-contract entry points for JavaScript: N-D and broadcast operator calls
// and layout plans, all decided by include/screamer/common/shape.h.
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <cstdint>
#include <string>
#include <vector>
#include "screamer/common/eval_op.h"
#include "screamer/common/shape.h"

namespace {

using screamer::shape::Shape;

// A call spec on the WASM heap, as doubles: [nArgs, then per argument: data
// pointer, ndim, dims...]. Each argument is a C-contiguous block of doubles.
struct Spec {
    std::vector<const double*> data;
    std::vector<Shape> shapes;
};

Spec read_spec(std::uintptr_t p) {
    const double* s = reinterpret_cast<const double*>(p);
    Spec r;
    std::size_t n = static_cast<std::size_t>(s[0]), k = 1;
    for (std::size_t i = 0; i < n; ++i) {
        r.data.push_back(reinterpret_cast<const double*>(static_cast<std::uintptr_t>(s[k++])));
        std::size_t nd = static_cast<std::size_t>(s[k++]);
        Shape sh;
        for (std::size_t d = 0; d < nd; ++d) sh.push_back(static_cast<std::size_t>(s[k++]));
        r.shapes.push_back(std::move(sh));
    }
    return r;
}

template <class V>
emscripten::val to_js(const std::vector<V>& v) {
    emscripten::val a = emscripten::val::array();
    for (const V& x : v) a.call<void>("push", static_cast<double>(x));
    return a;
}

// The (T, *A) result shape of an nIn-input call on these arguments.
emscripten::val shapedResult(std::uintptr_t spec, std::size_t n_in) {
    Spec s = read_spec(spec);
    return to_js(screamer::shape::broadcast(screamer::shape::input_views(s.shapes, n_in)).shape);
}

// Run op over every broadcast column, resetting it per column like a batch call;
// out receives (T, *A, n_out) doubles in row-major order.
void evalShaped(screamer::EvalOp& op, std::uintptr_t spec, std::uintptr_t outPtr) {
    Spec s = read_spec(spec);
    const std::size_t ni = op.n_in(), no = op.n_out();
    const screamer::shape::Plan plan =
        screamer::shape::broadcast(screamer::shape::input_views(s.shapes, ni));
    std::vector<const double*> data(ni);
    for (std::size_t i = 0; i < ni; ++i) data[i] = s.data[s.data.size() == 1 ? 0 : i];
    double* out = reinterpret_cast<double*>(outPtr);
    std::vector<double> in(ni);
    const std::size_t cols = plan.columns;
    for (std::size_t c = 0; c < cols; ++c) {
        op.reset();
        for (std::size_t t = 0; t < plan.T; ++t) {
            for (std::size_t i = 0; i < ni; ++i)
                in[i] = data[i][plan.offset[i * cols + c] + t * plan.time_stride[i]];
            op.eval(in.data(), out + (t * cols + c) * no);
        }
    }
    op.reset();
}

// The broadcast plan of one event of an nIn-input stream. `shapes` holds one
// shape per argument, each with a leading 1 for the time step. The result has
// the event's asset shape (`shape`, without T), the number of asset columns,
// and `offset[i * columns + c]`, where column c of input i starts in that
// input's data. The JS stream runs column c with its own op copy.
emscripten::val eventPlan(emscripten::val shapes, std::size_t n_in) {
    std::vector<Shape> args;
    const unsigned n = shapes["length"].as<unsigned>();
    for (unsigned i = 0; i < n; ++i) {
        emscripten::val s = shapes[i];
        const unsigned nd = s["length"].as<unsigned>();
        Shape sh;
        for (unsigned d = 0; d < nd; ++d) sh.push_back(static_cast<std::size_t>(s[d].as<double>()));
        args.push_back(std::move(sh));
    }
    const screamer::shape::Plan p = screamer::shape::broadcast(screamer::shape::input_views(args, n_in));
    emscripten::val r = emscripten::val::object();
    r.set("shape", to_js(Shape(p.shape.begin() + 1, p.shape.end())));
    r.set("columns", static_cast<double>(p.columns));
    r.set("offset", to_js(p.offset));
    return r;
}

emscripten::val layoutPlan(std::string layout, emscripten::val shape) {
    Shape in;
    const unsigned n = shape["length"].as<unsigned>();
    for (unsigned i = 0; i < n; ++i) in.push_back(static_cast<std::size_t>(shape[i].as<double>()));
    const auto p = screamer::shape::plan_layout(layout, in);
    emscripten::val r = emscripten::val::object();
    r.set("out", p.out);
    r.set("source", to_js(p.source));
    r.set("dropped", to_js(p.dropped));
    r.set("shape", to_js(p.shape));
    return r;
}

}  // namespace

EMSCRIPTEN_BINDINGS(screamer_shape) {
    emscripten::function("shapedResult", &shapedResult);
    emscripten::function("evalShaped", &evalShaped, emscripten::allow_raw_pointers());
    emscripten::function("layoutPlan", &layoutPlan);
    emscripten::function("eventPlan", &eventPlan);
}
