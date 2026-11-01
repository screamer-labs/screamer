#ifndef SCREAMER_SERIAL_SAVEABLE_H
#define SCREAMER_SERIAL_SAVEABLE_H

// Saveable<Derived, Base>: the save virtuals of EvalOp, implemented once for
// an operator that describes itself with
//
//   static constexpr const char* kTypeName = "RollingMean";  // registry name
//   static constexpr int kStateVersion = 1;
//   template <class Ar> void settings(Ar& ar);  // constructor arguments
//   template <class Ar> void state(Ar& ar);     // what reset() touches
//
// (serial/archive.h has the field calls). FunctorBase and VarArityOp derive
// from Saveable<Derived, EvalOp>, so their operators only add the members; a
// ScreamerBase or RowOp operator derives from Saveable<Op, ScreamerBase> or
// Saveable<Op, RowOp>. An operator without kStateVersion stays not saveable.
//
// The state of an operator is the tree {"version": N, "values": {...}}; a
// version other than kStateVersion raises StateVersionError. serial/file.h
// builds and reads the save file.

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "screamer/common/eval_op.h"
#include "screamer/common/json.h"
#include "screamer/serial/archive.h"
#include "screamer/serial/errors.h"

namespace screamer { namespace serial {

namespace detail {

template <class D, class = void>
struct describes_state : std::false_type {};
template <class D>
struct describes_state<D, std::void_t<decltype(D::kStateVersion)>> : std::true_type {};

inline json::Value state_node(int version, json::Value values) {
    json::Value v = json::Value::object();
    v.o.reserve(2);
    v.o.emplace_back("version", json::Value::integer(version));
    v.o.emplace_back("values", std::move(values));
    return v;
}

// The values of a state tree {"version": N, "values": {...}} of the given
// version.
inline const json::Value& state_values(const json::Value& in, int version, const std::string& path) {
    if (in.kind != json::Value::Kind::Object)
        throw StateMismatch(path, "expected a state {\"version\": N, \"values\": {...}}, got " +
                                      kind_text(in));
    for (const auto& kv : in.o)
        if (kv.first != "version" && kv.first != "values")
            throw StateMismatch(path + "." + kv.first, "unknown field");
    const json::Value* stored = in.get("version");
    if (!stored) throw StateMismatch(path + ".version", "missing field");
    if (stored->kind != json::Value::Kind::Int)
        throw StateMismatch(path + ".version", "expected an integer, got " + kind_text(*stored));
    if (stored->i != version) throw StateVersionError(path, stored->i, version);
    const json::Value* values = in.get("values");
    if (!values) throw StateMismatch(path + ".values", "missing field");
    return *values;
}

}  // namespace detail

template <class Derived, class Base>
class Saveable : public Base {
public:
    using Base::Base;

    bool saveable() const override { return detail::describes_state<Derived>::value; }

    const char* type_name() const override {
        if constexpr (detail::describes_state<Derived>::value) return Derived::kTypeName;
        else return nullptr;
    }

    int state_version() const override {
        if constexpr (detail::describes_state<Derived>::value) return Derived::kStateVersion;
        else return 0;
    }

    void write_settings(json::Value& out) const override {
        if constexpr (detail::describes_state<Derived>::value) {
            Writer w(Derived::kTypeName, Writer::Mode::Settings);
            const_cast<Derived&>(derived()).settings(w);
            out = w.take();
        } else {
            Base::write_settings(out);
        }
    }

    void write_state(json::Value& out) const override {
        if constexpr (detail::describes_state<Derived>::value) {
            Writer w(Derived::kTypeName);
            const_cast<Derived&>(derived()).state(w);
            out = detail::state_node(Derived::kStateVersion, w.take());
        } else {
            Base::write_state(out);
        }
    }

    void read_state(const json::Value& in, const std::string& path) override {
        if constexpr (detail::describes_state<Derived>::value) {
            load_state(derived(), detail::state_values(in, Derived::kStateVersion, path), path);
        } else {
            Base::read_state(in, path);
        }
    }

private:
    const Derived& derived() const { return static_cast<const Derived&>(*this); }
    Derived& derived() { return static_cast<Derived&>(*this); }
};

}}  // namespace screamer::serial

#endif  // SCREAMER_SERIAL_SAVEABLE_H
