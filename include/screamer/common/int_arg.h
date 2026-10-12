#ifndef SCREAMER_COMMON_INT_ARG_H
#define SCREAMER_COMMON_INT_ARG_H

// An integer operator setting (a window, a lag, a count). It converts like the
// built-in nanobind integer caster (an int, a NumPy integer, anything with
// __index__ or __int__ that is not a float; a bool, as before), except that a
// str, bytes or bytearray is a wrong kind: a string is never a number (R-a),
// so RollingMean("3") raises TypeError instead of parsing it. Bindings use
// IntArg<T> in place of T in nb::init<...>; it converts to T implicitly.

#include <Python.h>
#include <nanobind/nanobind.h>
#include <string>

namespace screamer {

template <class T>
struct IntArg {
    T value{};
    const char* text = nullptr;     // the type name of a refused string, or null
    // The conversion runs inside nb::init's call, so this TypeError reaches
    // Python with its own message (a caster cannot raise: it is noexcept).
    operator T() const {
        if (text)
            throw nanobind::type_error(
                (std::string("an integer setting must be a number, got ") + text).c_str());
        return value;
    }
};

}  // namespace screamer

namespace nanobind { namespace detail {

template <class T>
struct type_caster<screamer::IntArg<T>> {
    NB_TYPE_CASTER(screamer::IntArg<T>, const_name("int"))

    bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) noexcept {
        PyObject* o = src.ptr();
        if (PyUnicode_Check(o) || PyBytes_Check(o) || PyByteArray_Check(o)) {
            // Accepted here and refused by the conversion, with a clear message.
            value.text = PyUnicode_Check(o) ? (PyUnicode_CheckExact(o) ? "str" : "a str subclass")
                       : PyBytes_Check(o) ? "bytes" : "bytearray";
            return true;
        }
        make_caster<T> inner;
        if (!inner.from_python(src, flags, cleanup)) return false;
        value.value = inner.value;
        return true;
    }

    static handle from_cpp(const screamer::IntArg<T>& v, rv_policy policy,
                           cleanup_list* cleanup) noexcept {
        return make_caster<T>::from_cpp(v.value, policy, cleanup);
    }
};

}}  // namespace nanobind::detail
#endif
