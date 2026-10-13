// Embind binding for JavaScript's Pipeline serialization (js/src/
// pipeline_io.ts). The schema, the JSON reader and the writer are
// include/screamer/dag/graph_json.h, which Python's to_json and from_json
// use too. JavaScript passes its description as JSON text (JSON.stringify,
// with the non-finite numbers spelled NaN, Infinity and -Infinity) and builds
// its nodes from the checked description this returns.
//
// None of these throws for a bad description: each returns { value } or
// { error } and JavaScript throws the RangeError. An exception thrown out of
// an Embind function leaks its string arguments' wire buffers, so a service
// that checks untrusted descriptions would grow its heap without bound.
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include "screamer/dag/graph_json.h"

namespace {

using screamer::dag::json::Value;

emscripten::val to_js(const Value& v) {
    using K = Value::Kind;
    switch (v.kind) {
    case K::Null: return emscripten::val::null();
    case K::Bool: return emscripten::val(v.b);
    case K::Int: return emscripten::val(static_cast<double>(v.i));
    case K::Float: return emscripten::val(v.d);
    case K::String: return emscripten::val(v.s);
    case K::Array: {
        emscripten::val a = emscripten::val::array();
        for (std::size_t k = 0; k < v.a.size(); ++k) a.set(k, to_js(v.a[k]));
        return a;
    }
    case K::Object: {
        emscripten::val o = emscripten::val::object();
        for (const auto& kv : v.o) o.set(kv.first, to_js(kv.second));
        return o;
    }
    }
    return emscripten::val::null();
}

emscripten::val failed(const std::exception& e) {
    emscripten::val r = emscripten::val::object();
    r.set("error", std::string(e.what()));
    return r;
}

template <class T>
emscripten::val ok(const T& v) {
    emscripten::val r = emscripten::val::object();
    r.set("value", v);
    return r;
}

// { value: the checked description of a JSON text, in its normal form }, or
// { error }.
emscripten::val graphFromJson(std::string text) {
    try {
        return ok(to_js(screamer::dag::check_graph(screamer::dag::json::parse(text))));
    } catch (const std::invalid_argument& e) {
        return failed(e);
    }
}

// { value: the text Python's to_json writes for a description given as JSON
// text }, or { error }; `indent` < 0 writes one line.
emscripten::val graphToJson(std::string text, int indent) {
    try {
        return ok(screamer::dag::json::dump(
            screamer::dag::check_graph(screamer::dag::json::parse(text)), indent));
    } catch (const std::invalid_argument& e) {
        return failed(e);
    }
}

// The message for a functor class the binding does not know.
std::string unknownFunctorMessage(std::string name) {
    try {
        screamer::dag::throw_unknown_functor(name);
    } catch (const std::invalid_argument& e) {
        return e.what();
    }
    return std::string();
}

// The message for a functor node fed another number of inputs, or "".
std::string functorInputsMessage(double node, std::string cls, double nIn, double got) {
    try {
        screamer::dag::check_functor_inputs(static_cast<std::size_t>(node), cls,
                                            static_cast<std::size_t>(nIn), static_cast<std::size_t>(got));
    } catch (const std::invalid_argument& e) {
        return e.what();
    }
    return std::string();
}

}  // namespace

EMSCRIPTEN_BINDINGS(screamer_graph_json) {
    emscripten::function("graphFromJson", &graphFromJson);
    emscripten::function("graphToJson", &graphToJson);
    emscripten::function("unknownFunctorMessage", &unknownFunctorMessage);
    emscripten::function("functorInputsMessage", &functorInputsMessage);
}
