#ifndef SCREAMER_SERIAL_FILE_H
#define SCREAMER_SERIAL_FILE_H

// The save file of an operator:
//
//   {"screamer": "save/1", "library_version": "...",
//    "object": {"type": T, "settings": {...}, "state": {...} | null}}
//
// in JSON text (json::dump_save) or CBOR. A settings-only file has
// "state": null.

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "screamer/common/eval_op.h"
#include "screamer/common/json.h"
#include "screamer/serial/cbor.h"
#include "screamer/serial/envelope.h"
#include "screamer/serial/errors.h"

namespace screamer { namespace serial {

// The node {"type", "settings", "state"} of `op`; `with_state` false writes
// "state": null (a settings-only file). Raises NotSaveable for an operator
// without save support.
inline json::Value save_node(const EvalOp& op, bool with_state = true) {
    if (!op.saveable()) throw NotSaveable("operator", "the operator has no save and load support yet");
    json::Value settings, state;
    op.write_settings(settings);
    if (with_state) op.write_state(state);
    json::Value v = json::Value::object();
    v.o.reserve(3);
    v.o.emplace_back("type", json::Value::string(op.type_name()));
    v.o.emplace_back("settings", std::move(settings));
    v.o.emplace_back("state", std::move(state));
    return v;
}

// The save file of `op` as JSON text (json::dump_save).
inline std::string dumps_json(const EvalOp& op) { return json::dump_save(wrap(save_node(op))); }

// The save file of `op` as CBOR.
inline std::vector<std::uint8_t> dumps_cbor(const EvalOp& op) { return cbor::encode(wrap(save_node(op))); }

// The tree of a save file in either encoding: CBOR when the first byte is not
// the '{' (or white space) that starts JSON text. A malformed file raises
// SaveError.
inline json::Value parse_file(const std::uint8_t* data, std::size_t size) {
    std::size_t k = 0;
    while (k < size && (data[k] == ' ' || data[k] == '\t' || data[k] == '\n' || data[k] == '\r')) ++k;
    if (k < size && data[k] == '{') {
        try {
            return json::parse(std::string(reinterpret_cast<const char*>(data), size), "save JSON");
        } catch (const std::invalid_argument& e) {
            throw SaveError(e.what());
        }
    }
    return cbor::decode(data, size);
}

// The parts of a node of a save file, checked for shape: the type name, the
// settings object and the state tree (null for a settings-only node).
struct NodeParts {
    std::string type;
    const json::Value* settings = nullptr;
    const json::Value* state = nullptr;
};

inline NodeParts node_parts(const json::Value& node, const std::string& path) {
    if (node.kind != json::Value::Kind::Object)
        throw SaveError(path + ": expected an object, got " + detail::kind_text(node));
    for (const auto& kv : node.o)
        if (kv.first != "type" && kv.first != "settings" && kv.first != "state")
            throw SaveError(path + "." + kv.first + ": unknown field");
    const json::Value* type = node.get("type");
    if (!type) throw SaveError(path + ".type: missing field");
    if (type->kind != json::Value::Kind::String)
        throw SaveError(path + ".type: expected a string, got " + detail::kind_text(*type));
    const json::Value* settings = node.get("settings");
    if (!settings) throw SaveError(path + ".settings: missing field");
    if (settings->kind != json::Value::Kind::Object)
        throw SaveError(path + ".settings: expected an object, got " + detail::kind_text(*settings));
    const json::Value* state = node.get("state");
    NodeParts parts;
    parts.type = type->s;
    parts.settings = settings;
    parts.state = state && state->kind != json::Value::Kind::Null ? state : nullptr;
    return parts;
}

}}  // namespace screamer::serial

#endif  // SCREAMER_SERIAL_FILE_H
