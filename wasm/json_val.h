#pragma once
// A json::Value tree (include/screamer/common/json.h) as a JavaScript value:
// null, a boolean, a number (an integer too, so NaN and the infinities keep
// their value), a string, an Array or a plain object with its keys in order.
#include <cstddef>
#include <emscripten/val.h>
#include "screamer/common/json.h"

namespace screamer_wasm {

inline emscripten::val to_js(const screamer::json::Value& v) {
    using K = screamer::json::Value::Kind;
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

}  // namespace screamer_wasm
