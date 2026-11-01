#ifndef SCREAMER_SERIAL_ERRORS_H
#define SCREAMER_SERIAL_ERRORS_H

// The errors of save and load. Each message starts with the path of the
// object it is about ("RollingMean.buffer", "Pipeline.nodes.3"), so a user
// can find the place in the file. Python raises them as screamer.SaveError
// and its subclasses (SaveError is a ValueError).

#include <stdexcept>
#include <string>

#include "screamer/common/json.h"

namespace screamer { namespace serial {

// A file or tree that cannot be saved or loaded: the base of the others, and
// itself the error of a malformed file (bad CBOR, another envelope).
class SaveError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// A stored state version that is not the one this library writes.
class StateVersionError : public SaveError {
public:
    StateVersionError(const std::string& path, long long stored, long long expected)
        : SaveError(path + ": state version " + std::to_string(stored) +
                    " in the file, this library reads version " + std::to_string(expected)) {}
};

// An object that has no save and load support (yet), or that cannot be saved
// in its current condition (a lazy call in progress, a poisoned session).
class NotSaveable : public SaveError {
public:
    NotSaveable(const std::string& path, const std::string& why) : SaveError(path + ": " + why) {}
};

// A state that does not fit the object: an unknown or missing field, a value
// of the wrong type or length, or a broken invariant.
class StateMismatch : public SaveError {
public:
    StateMismatch(const std::string& path, const std::string& what) : SaveError(path + ": " + what) {}
};

namespace detail {

// The kind of a value in a message: "a number", "an array", ...
inline std::string kind_text(const json::Value& v) {
    using K = json::Value::Kind;
    switch (v.kind) {
    case K::Null: return "null";
    case K::Bool: return "a boolean";
    case K::Int: case K::Float: return "a number";
    case K::String: return "a string";
    case K::Array: return "an array";
    case K::Object: return "an object";
    }
    return "a value";
}

}  // namespace detail

}}  // namespace screamer::serial

#endif  // SCREAMER_SERIAL_ERRORS_H
