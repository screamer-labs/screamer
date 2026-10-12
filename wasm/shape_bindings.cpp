// Shape-contract entry points for JavaScript: N-D and broadcast operator calls
// and layout plans, all decided by include/screamer/common/shape.h.
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>
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

// A batch call: plan the broadcast of the arguments once, then run op over
// every asset column with one eval_column each. Returns a malloc'd block of
// (T, *A, n_out) doubles in row-major order (the caller frees it), and writes
// the (T, *A) result shape to shapeOut as [ndim, dims...]; shapeOut holds room
// for 1 + the largest argument rank.
std::uintptr_t evalShaped(screamer::EvalOp& op, std::uintptr_t spec, std::uintptr_t shapeOut) {
    Spec s = read_spec(spec);
    const std::size_t ni = op.n_in(), no = op.n_out();
    const screamer::shape::Plan plan =
        screamer::shape::broadcast(screamer::shape::input_views(s.shapes, ni));
    const std::size_t cols = plan.columns;
    std::unique_ptr<double, decltype(&std::free)> out(
        static_cast<double*>(std::malloc(std::max<std::size_t>(1, plan.T * cols * no) * sizeof(double))),
        &std::free);
    if (!out) throw std::bad_alloc();
    std::vector<const double*> ptr(ni);
    std::vector<std::ptrdiff_t> stride(ni);
    for (std::size_t i = 0; i < ni; ++i) stride[i] = static_cast<std::ptrdiff_t>(plan.time_stride[i]);
    const std::ptrdiff_t out_stride = static_cast<std::ptrdiff_t>(cols * no);
    for (std::size_t c = 0; c < cols; ++c) {
        for (std::size_t i = 0; i < ni; ++i)
            ptr[i] = s.data[s.data.size() == 1 ? 0 : i] + plan.offset[i * cols + c];
        op.eval_column(ptr.data(), stride.data(), plan.T, out.get() + c * no, out_stride);
    }
    op.reset();
    double* sh = reinterpret_cast<double*>(shapeOut);
    sh[0] = static_cast<double>(plan.shape.size());
    for (std::size_t d = 0; d < plan.shape.size(); ++d) sh[1 + d] = static_cast<double>(plan.shape[d]);
    return reinterpret_cast<std::uintptr_t>(out.release());
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

// `layout` is a val, not a std::string: Embind frees a std::string argument's
// wire buffer only on a normal return, so a rejected layout would leak it.
emscripten::val layoutPlan(emscripten::val layout_js, emscripten::val shape) {
    const std::string layout = layout_js.as<std::string>();
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
    emscripten::function("evalShaped", &evalShaped, emscripten::allow_raw_pointers());
    emscripten::function("layoutPlan", &layoutPlan);
    emscripten::function("eventPlan", &eventPlan);
}
