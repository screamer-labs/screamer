#ifndef SCREAMER_SERIAL_ENVELOPE_H
#define SCREAMER_SERIAL_ENVELOPE_H

// The outer object of a save file:
//
//   {"screamer": "save/1", "library_version": "3.10.1", "object": <node>}
//
// "screamer" names the file schema, which a reader must know; the library
// version is a note for people and is not checked. Any other field is
// refused as unknown, as it is in a node.

#include <string>

#include "screamer/common/json.h"
#include "screamer/serial/errors.h"

// The build defines the library version from CMake's project(VERSION).
#ifndef SCREAMER_VERSION_STRING
#error "SCREAMER_VERSION_STRING is not defined (the build passes PROJECT_VERSION)"
#endif

namespace screamer { namespace serial {

inline constexpr const char* kSaveSchema = "save/1";

// `node` in a save file's envelope.
inline json::Value wrap(json::Value node) {
    json::Value v = json::Value::object();
    v.o.reserve(3);
    v.o.emplace_back("screamer", json::Value::string(kSaveSchema));
    v.o.emplace_back("library_version", json::Value::string(SCREAMER_VERSION_STRING));
    v.o.emplace_back("object", std::move(node));
    return v;
}

// The node in a save file's envelope; raises SaveError for another schema or
// a tree that is no save file.
inline const json::Value& unwrap(const json::Value& file) {
    if (file.kind != json::Value::Kind::Object)
        throw SaveError("not a screamer save file: expected an object, got " + detail::kind_text(file));
    const json::Value* schema = file.get("screamer");
    if (!schema) throw SaveError("not a screamer save file: no \"screamer\" field");
    if (schema->kind != json::Value::Kind::String)
        throw SaveError("not a screamer save file: \"screamer\" is " + detail::kind_text(*schema));
    if (schema->s != kSaveSchema)
        throw SaveError("the file is a \"" + schema->s + "\" save file; this library reads \"" +
                        kSaveSchema + "\"");
    for (const auto& kv : file.o)
        if (kv.first != "screamer" && kv.first != "library_version" && kv.first != "object")
            throw SaveError(kv.first + ": unknown field");
    const json::Value* node = file.get("object");
    if (!node) throw SaveError("not a screamer save file: no \"object\" field");
    return *node;
}

}}  // namespace screamer::serial

#endif  // SCREAMER_SERIAL_ENVELOPE_H
