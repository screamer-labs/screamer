#ifndef SCREAMER_SERIAL_ARCHIVE_H
#define SCREAMER_SERIAL_ARCHIVE_H

// The archives that save and load an object's members as a json::Value tree.
// An object lists its members once, in a member template run by every
// archive:
//
//   template <class Ar> void state(Ar& ar) {
//       ar("count", count_);                    // a scalar, string or optional
//       ar("sums", sums_);                      // a std::vector of those
//       ar.fixed("buffer", buf_, window_size_); // a vector of a given length
//       ar.object("deque", deque_);             // a member with its own state()
//       ar.op("inner", *op_);                   // a nested operator
//       ar.check(count_ <= window_size_, "count must be at most window_size");
//   }
//
// Writer builds the tree (an object, keys in the order listed). Validator
// checks a tree against the object and leaves the object as it was; Reader
// assigns the tree's values. load_state() runs both, so a tree that does not
// fit leaves the object unchanged: every assignment is logged and undone when
// a later field fails.
//
// A field of the tree that the object does not list ("unknown field"), a
// listed field the tree lacks ("missing field"), a value of another type or
// length, and a failed check raise StateMismatch with the path of the field:
// "<path>.<field>: expected <what>, got <what>".
//
// ar.op recurses into a nested operator through its write_state(json::Value&)
// and read_state(const json::Value&, const std::string& path); a validation
// pass loads it and restores it from a snapshot.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "screamer/common/json.h"
#include "screamer/serial/errors.h"

namespace screamer { namespace serial {

namespace detail {

// ---- Values to the tree ------------------------------------------------------

inline json::Value to_value(bool v) { return json::Value::boolean(v); }
inline json::Value to_value(const std::string& v) { return json::Value::string(v); }

template <class T, std::enable_if_t<std::is_integral<T>::value, int> = 0>
json::Value to_value(T v) {
    if (std::is_unsigned<T>::value &&
        static_cast<std::uint64_t>(v) > static_cast<std::uint64_t>(INT64_MAX)) {
        const auto u = static_cast<std::uint64_t>(v);
        return json::Value::big_integer(static_cast<double>(u), json::detail::uint_text(u));
    }
    return json::Value::integer(static_cast<std::int64_t>(v));
}

template <class T, std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
json::Value to_value(T v) {
    return json::Value::number(static_cast<double>(v));
}

template <class T>
json::Value to_value(const std::optional<T>& v) {
    return v ? to_value(*v) : json::Value::null();
}

template <class T, class A>
json::Value to_value(const std::vector<T, A>& v) {
    json::Value a = json::Value::array();
    a.a.reserve(v.size());
    for (const auto& e : v) a.a.push_back(to_value(static_cast<const T&>(e)));
    return a;
}

// ---- Values from the tree ----------------------------------------------------

// A value as a message names it: a number by its text, anything else by its
// kind.
inline std::string got_text(const json::Value& v) {
    return v.is_number() ? json::dump(v, -1) : kind_text(v);
}

[[noreturn]] inline void mismatch(const std::string& where, const std::string& expected,
                                  const json::Value& got) {
    throw StateMismatch(where, "expected " + expected + ", got " + got_text(got));
}

inline void from_value(const json::Value& v, bool& out, const std::string& where) {
    if (v.kind != json::Value::Kind::Bool) mismatch(where, "a boolean", v);
    out = v.b;
}

inline void from_value(const json::Value& v, std::string& out, const std::string& where) {
    if (v.kind != json::Value::Kind::String) mismatch(where, "a string", v);
    out = v.s;
}

template <class T, std::enable_if_t<std::is_integral<T>::value, int> = 0>
void from_value(const json::Value& v, T& out, const std::string& where) {
    using L = std::numeric_limits<T>;
    const auto range = [&] {
        mismatch(where, "an integer from " + json::detail::int_text(static_cast<std::int64_t>(L::min())) +
                            " to " + json::detail::uint_text(static_cast<std::uint64_t>(L::max())),
                 v);
    };
    if (v.kind == json::Value::Kind::Int) {
        if (std::is_signed<T>::value) {
            if (v.i < static_cast<std::int64_t>(L::min()) || v.i > static_cast<std::int64_t>(L::max())) range();
        } else if (v.i < 0 || static_cast<std::uint64_t>(v.i) > static_cast<std::uint64_t>(L::max())) {
            range();
        }
        out = static_cast<T>(v.i);
        return;
    }
    // An unsigned value past int64 is a whole number that keeps its digits.
    std::uint64_t u = 0;
    if (v.kind == json::Value::Kind::Float && v.whole && json::detail::parse_uint(v.s, u)) {
        if (!std::is_unsigned<T>::value || u > static_cast<std::uint64_t>(L::max())) range();
        out = static_cast<T>(u);
        return;
    }
    if (v.kind == json::Value::Kind::Float && v.whole) range();
    mismatch(where, "an integer", v);
}

template <class T, std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
void from_value(const json::Value& v, T& out, const std::string& where) {
    if (!v.is_number()) mismatch(where, "a number", v);
    out = static_cast<T>(v.as_double());
}

template <class T>
void from_value(const json::Value& v, std::optional<T>& out, const std::string& where) {
    if (v.kind == json::Value::Kind::Null) {
        out.reset();
        return;
    }
    T x{};
    from_value(v, x, where);
    out = std::move(x);
}

template <class T, class A>
void from_value(const json::Value& v, std::vector<T, A>& out, const std::string& where) {
    if (v.kind != json::Value::Kind::Array) mismatch(where, "an array", v);
    std::vector<T, A> r;
    r.reserve(v.a.size());
    for (std::size_t k = 0; k < v.a.size(); ++k) {
        T x{};
        from_value(v.a[k], x, where + "[" + std::to_string(k) + "]");
        r.push_back(std::move(x));
    }
    out = std::move(r);
}

// The assignments of a load, undone in reverse order when the load fails or
// only validates.
class UndoLog {
public:
    template <class T>
    void assign(T& field, T value) {
        steps_.emplace_back([&field, old = std::move(field)]() mutable { field = std::move(old); });
        field = std::move(value);
    }
    void add(std::function<void()> undo) { steps_.push_back(std::move(undo)); }
    void undo() {
        while (!steps_.empty()) {
            steps_.back()();
            steps_.pop_back();
        }
    }
    void clear() { steps_.clear(); }

private:
    std::vector<std::function<void()>> steps_;
};

// Validator (Commit false) and Reader (Commit true): the same walk over the
// tree; a Validator undoes its assignments at the end.
template <bool Commit>
class Loader {
public:
    static constexpr bool kSaving = false;
    static constexpr bool kLoading = true;

    // `tree` is the object holding the fields; `path` names it in messages.
    Loader(const json::Value& tree, std::string path) : Loader(tree, std::move(path), nullptr) {}

    Loader(const Loader&) = delete;
    Loader& operator=(const Loader&) = delete;

    template <class T>
    void operator()(const char* name, T& field) {
        T x{};
        from_value(take(name), x, at(name));
        undo_->assign(field, std::move(x));
    }

    template <class V>
    void fixed(const char* name, V& field, std::size_t expected_len) {
        V x{};
        from_value(take(name), x, at(name));
        if (x.size() != expected_len)
            throw StateMismatch(at(name), "expected " + std::to_string(expected_len) + " values, got " +
                                              std::to_string(x.size()));
        undo_->assign(field, std::move(x));
    }

    template <class B>
    void object(const char* name, B& block) {
        Loader sub(take(name), at(name), undo_);
        block.state(sub);
        sub.finish();
    }

    template <class Op>
    void op(const char* name, Op& nested) {
        const json::Value& tree = take(name);
        json::Value snapshot;
        nested.write_state(snapshot);
        std::string where = at(name);
        nested.read_state(tree, where);
        undo_->add([&nested, snapshot = std::move(snapshot), where = std::move(where)] {
            nested.read_state(snapshot, where);
        });
    }

    void check(bool cond, const char* message) const {
        if (!cond) throw StateMismatch(path_, message);
    }

    // After the object's state(): a field of the tree that was not read is
    // unknown.
    void finish() const {
        for (std::size_t k = 0; k < used_.size(); ++k)
            if (!used_[k]) throw StateMismatch(path_ + "." + tree_.o[k].first, "unknown field");
    }

    // Ends the load: a Validator restores the object, a Reader keeps it.
    void done() {
        if (Commit) own_.clear();
        else own_.undo();
    }
    // Restores the object after a failed load.
    void abandon() { own_.undo(); }

private:
    Loader(const json::Value& tree, std::string path, UndoLog* undo)
        : tree_(tree), path_(std::move(path)), undo_(undo ? undo : &own_) {
        if (tree_.kind != json::Value::Kind::Object) mismatch(path_, "an object", tree_);
        used_.assign(tree_.o.size(), false);
    }

    std::string at(const char* name) const { return path_ + "." + name; }

    const json::Value& take(const char* name) {
        for (std::size_t k = 0; k < tree_.o.size(); ++k) {
            if (tree_.o[k].first == name) {
                used_[k] = true;
                return tree_.o[k].second;
            }
        }
        throw StateMismatch(at(name), "missing field");
    }

    const json::Value& tree_;
    std::string path_;
    UndoLog own_;
    UndoLog* undo_;
    std::vector<bool> used_;
};

}  // namespace detail

// Builds the tree of an object's fields.
class Writer {
public:
    static constexpr bool kSaving = true;
    static constexpr bool kLoading = false;

    Writer() : tree_(json::Value::object()) {}

    template <class T>
    void operator()(const char* name, const T& field) {
        tree_.o.emplace_back(name, detail::to_value(field));
    }

    template <class V>
    void fixed(const char* name, const V& field, std::size_t) {
        (*this)(name, field);
    }

    template <class B>
    void object(const char* name, B& block) {
        Writer sub;
        block.state(sub);
        tree_.o.emplace_back(name, sub.take());
    }

    template <class Op>
    void op(const char* name, const Op& nested) {
        json::Value v;
        nested.write_state(v);
        tree_.o.emplace_back(name, std::move(v));
    }

    void check(bool, const char*) const {}

    json::Value take() { return std::move(tree_); }

private:
    json::Value tree_;
};

using Validator = detail::Loader<false>;
using Reader = detail::Loader<true>;

// The tree of `obj`'s state(). The Writer only reads the members, but state()
// is one non-const template for every archive.
template <class T>
json::Value save_state(const T& obj) {
    Writer w;
    const_cast<T&>(obj).state(w);
    return w.take();
}

// Loads `tree` into `obj` through its state(): a Validator pass, then a Reader
// pass. Raises StateMismatch (or the error of a nested operator) naming the
// path of the field, and then leaves `obj` unchanged.
template <class T>
void load_state(T& obj, const json::Value& tree, const std::string& path) {
    {
        Validator v(tree, path);
        try {
            obj.state(v);
            v.finish();
        } catch (...) {
            v.abandon();
            throw;
        }
        v.done();
    }
    Reader r(tree, path);
    try {
        obj.state(r);
        r.finish();
    } catch (...) {
        r.abandon();
        throw;
    }
    r.done();
}

}}  // namespace screamer::serial

#endif  // SCREAMER_SERIAL_ARCHIVE_H
