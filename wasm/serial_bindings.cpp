// Embind binding of save and load (js/src/serial.ts, js/src/runtime.ts), on
// EvalOp once for every operator: the same C++ as Python's (bindings_core.cpp)
// writes and reads the files, include/screamer/serial. Trees reach JavaScript
// as plain values (json_val.h) and come back as JSON text, which JavaScript
// writes with its non-finite numbers spelled out (js/src/json_text.ts).
//
// A failure throws the C++ error (serial/errors.h), which js/src/errors.ts
// turns into the JavaScript class of the same name. Text arguments arrive as
// vals: an exception thrown out of an Embind function leaks the wire buffer
// of a std::string argument.
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include "screamer/common/eval_op.h"
#include "screamer/common/json.h"
#include "screamer/serial/errors.h"
#include "screamer/serial/file.h"
#include "screamer/serial/op_registry.h"
#include "screamer/serial/settings_compat.h"
#include "json_val.h"

namespace {

namespace json = screamer::json;
namespace serial = screamer::serial;
using screamer::EvalOp;

json::Value parse(const emscripten::val& text, const char* label) {
    try {
        return json::parse(text.as<std::string>(), label);
    } catch (const std::invalid_argument& e) {
        throw serial::SaveError(e.what());
    }
}

bool saveable(const EvalOp& op) { return op.saveable(); }

emscripten::val typeName(const EvalOp& op) {
    const char* name = op.type_name();
    return name ? emscripten::val(std::string(name)) : emscripten::val::null();
}

emscripten::val settings(const EvalOp& op) {
    json::Value v;
    op.write_settings(v);
    return screamer_wasm::to_js(v);
}

emscripten::val state(const EvalOp& op) {
    json::Value v;
    op.write_state(v);
    return screamer_wasm::to_js(v);
}

// Loads a state given as JSON text; `path` names the operator in errors.
void setState(EvalOp& op, emscripten::val text, emscripten::val path) {
    op.read_state(parse(text, "state JSON"), path.as<std::string>());
}

// The save file: JSON text, or a Uint8Array of CBOR.
emscripten::val dumps(const EvalOp& op, bool cbor) {
    if (!cbor) return emscripten::val(serial::dumps_json(op));
    const std::vector<std::uint8_t> b = serial::dumps_cbor(op);
    return emscripten::val::global("Uint8Array").new_(emscripten::typed_memory_view(b.size(), b.data()));
}

// The node of a save file (a string, or a Uint8Array of either encoding) as
// { type, settings (JSON text), state (JSON text or null) }, its shape checked.
emscripten::val loadNode(emscripten::val data) {
    std::string bytes;
    if (data.isString()) {
        bytes = data.as<std::string>();
    } else {
        const std::vector<std::uint8_t> b = emscripten::convertJSArrayToNumberVector<std::uint8_t>(data);
        bytes.assign(b.begin(), b.end());
    }
    const json::Value file =
        serial::parse_file(reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size());
    const serial::NodeParts parts = serial::node_parts(serial::unwrap(file), "object");
    emscripten::val r = emscripten::val::object();
    r.set("type", parts.type);
    r.set("settings", json::dump(*parts.settings, -1));
    r.set("state", parts.state ? emscripten::val(json::dump(*parts.state, -1)) : emscripten::val::null());
    return r;
}

// A fresh operator from its settings (serial::make_op); JavaScript owns it.
EvalOp* makeOp(emscripten::val type, emscripten::val text) {
    return serial::make_op(type.as<std::string>(), parse(text, "settings JSON")).release();
}

// The settings with the renamed parameters applied (serial/settings_compat.h),
// as a plain value.
emscripten::val settingsCompat(emscripten::val type, emscripten::val text) {
    json::Value s = parse(text, "settings JSON");
    const std::string t = type.as<std::string>();
    serial::apply_settings_compat(t, s, t);
    return screamer_wasm::to_js(s);
}

void testRename(emscripten::val type, emscripten::val old_name, emscripten::val new_name) {
    serial::push_test_rename(type.as<std::string>(), old_name.as<std::string>(), new_name.as<std::string>());
}

}  // namespace

EMSCRIPTEN_BINDINGS(screamer_serial) {
    emscripten::function("serialSaveable", &saveable);
    emscripten::function("serialTypeName", &typeName);
    emscripten::function("serialSettings", &settings);
    emscripten::function("serialState", &state);
    emscripten::function("serialSetState", &setState);
    emscripten::function("serialDumps", &dumps);
    emscripten::function("serialLoadNode", &loadNode);
    emscripten::function("serialMakeOp", &makeOp, emscripten::allow_raw_pointers());
    emscripten::function("serialSettingsCompat", &settingsCompat);
    emscripten::function("_serialTestRename", &testRename);
    emscripten::function("_serialTestRenameClear", &serial::clear_test_renames);
}
