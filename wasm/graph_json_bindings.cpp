// Embind binding for JavaScript's Pipeline.toJSON and Pipeline.fromJSON
// (js/src/pipeline.ts). The schema, the JSON reader and the writer are
// include/screamer/dag/graph_json.h, which Python's to_json and from_json
// use too. JavaScript passes its description as JSON text (JSON.stringify,
// with the non-finite numbers spelled NaN, Infinity and -Infinity) and builds
// its nodes from the checked description this returns.
#include <cmath>
#include <cstddef>
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

// The checked description of a JSON text, in its normal form.
emscripten::val graphFromJson(std::string text) {
    return to_js(screamer::dag::check_graph(screamer::dag::json::parse(text)));
}

// The text Python's to_json writes for a description given as JSON text;
// `indent` < 0 writes one line.
std::string graphToJson(std::string text, int indent) {
    return screamer::dag::json::dump(screamer::dag::check_graph(screamer::dag::json::parse(text)),
                                     indent);
}

void throwUnknownFunctor(std::string name) { screamer::dag::throw_unknown_functor(name); }

}  // namespace

EMSCRIPTEN_BINDINGS(screamer_graph_json) {
    emscripten::function("graphFromJson", &graphFromJson);
    emscripten::function("graphToJson", &graphToJson);
    emscripten::function("throwUnknownFunctor", &throwUnknownFunctor);
}
