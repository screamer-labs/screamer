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
// The state of an operator is the tree {"version": N, "values": {...}}, or
// {"version": N, "columns": [{...}, ...]} for one holding per-column state (a
// panel, common/eval_op.h): one values object per column. A version other
// than kStateVersion raises StateVersionError. serial/file.h builds and reads
// the save file.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "screamer/common/eval_op.h"
#include "screamer/common/json.h"
#include "screamer/serial/archive.h"
#include "screamer/serial/errors.h"
#include "screamer/serial/panel.h"

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

// The body of a state tree of the given version: its "values" object, or its
// "columns" array when `columns` is set to true.
inline const json::Value& state_body(const json::Value& in, int version, const std::string& path,
                                     bool& columns) {
    if (in.kind != json::Value::Kind::Object)
        throw StateMismatch(path, "expected a state {\"version\": N, \"values\": {...}}, got " +
                                      kind_text(in));
    for (const auto& kv : in.o)
        if (kv.first != "version" && kv.first != "values" && kv.first != "columns")
            throw StateMismatch(path + "." + kv.first, "unknown field");
    const json::Value* stored = in.get("version");
    if (!stored) throw StateMismatch(path + ".version", "missing field");
    if (stored->kind != json::Value::Kind::Int)
        throw StateMismatch(path + ".version", "expected an integer, got " + kind_text(*stored));
    if (stored->i != version) throw StateVersionError(path, stored->i, version);
    const json::Value* values = in.get("values");
    const json::Value* cols = in.get("columns");
    if (values && cols) throw StateMismatch(path + ".columns", "a state holds values or columns, not both");
    if (values) {
        columns = false;
        return *values;
    }
    if (!cols) throw StateMismatch(path + ".values", "missing field");
    if (cols->kind != json::Value::Kind::Array)
        throw StateMismatch(path + ".columns", "expected an array, got " + kind_text(*cols));
    if (cols->a.empty()) throw StateMismatch(path + ".columns", "expected at least one column");
    columns = true;
    return *cols;
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
            if (this->panel_) {
                json::Value cols = json::Value::array();
                cols.a.reserve(this->panel_->cols.size());
                for (const auto& c : this->panel_->cols)
                    cols.a.push_back(values_of(static_cast<const Derived&>(*c)));
                out = json::Value::object();
                out.o.reserve(2);
                out.o.emplace_back("version", json::Value::integer(Derived::kStateVersion));
                out.o.emplace_back("columns", std::move(cols));
                return;
            }
            out = detail::state_node(Derived::kStateVersion, values_of(derived()));
        } else {
            Base::write_state(out);
        }
    }

    // A "values" state loads into the operator and drops a panel it holds; a
    // "columns" state loads into fresh copies (clone_fresh) that become the
    // panel, and resets the operator. Nothing changes when any part fails.
    void read_state(const json::Value& in, const std::string& path) override {
        if constexpr (detail::describes_state<Derived>::value) {
            bool columns = false;
            const json::Value& body = detail::state_body(in, Derived::kStateVersion, path, columns);
            if (!columns) {
                load_state(derived(), body, path);
                this->panel_.reset();
                return;
            }
            auto panel = std::make_unique<Panel>();
            panel->cols.reserve(body.a.size());
            for (std::size_t c = 0; c < body.a.size(); ++c) {
                std::unique_ptr<EvalOp> col = this->clone_fresh();
                if (!col)
                    throw StateMismatch(path + ".columns", "the operator cannot hold per-column state");
                load_state(static_cast<Derived&>(*col), body.a[c],
                           path + ".columns[" + std::to_string(c) + "]");
                panel->cols.push_back(std::move(col));
            }
            this->reset();
            this->panel_ = std::move(panel);
        } else {
            Base::read_state(in, path);
        }
    }

    // A copy in its reset state, one column of a panel. The copy is a
    // Derived: by the CRTP contract an operator derived from Derived only
    // renames it, or overrides this too (detail::SwapTwoInputs).
    std::unique_ptr<EvalOp> clone_fresh() const override {
        if constexpr (std::is_copy_constructible_v<Derived>) {
            auto p = std::make_unique<Derived>(derived());
            p->reset();
            return p;
        } else {
            return Base::clone_fresh();
        }
    }

private:
    static json::Value values_of(const Derived& op) {
        Writer w(Derived::kTypeName);
        const_cast<Derived&>(op).state(w);
        return w.take();
    }

    const Derived& derived() const { return static_cast<const Derived&>(*this); }
    Derived& derived() { return static_cast<Derived&>(*this); }
};

// Renamed<Derived, Base>: an operator that only renames a saveable Base
// (ExpandingSum is CumSum). It keeps Base's settings and state and saves under
// its own Derived::kTypeName; its panel columns are Derived copies.
template <class Derived, class Base>
class Renamed : public Base {
public:
    using Base::Base;

    const char* type_name() const override { return Derived::kTypeName; }

    std::unique_ptr<EvalOp> clone_fresh() const override {
        auto p = std::make_unique<Derived>(static_cast<const Derived&>(*this));
        p->reset();
        return p;
    }
};

}}  // namespace screamer::serial

#endif  // SCREAMER_SERIAL_SAVEABLE_H
