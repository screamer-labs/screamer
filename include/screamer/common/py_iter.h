#ifndef SCREAMER_COMMON_PY_ITER_H
#define SCREAMER_COMMON_PY_ITER_H

#include <nanobind/nanobind.h>

namespace screamer {

namespace nb = nanobind;

// The next item of iterator `it`, or an invalid (null) object at its end. An
// iterator's tp_iternext is called through PyIter_Next: no attribute lookup,
// no bound method and no StopIteration exception per item. An object that is
// not an iterator but has __next__ is called through the attribute.
inline nb::object next_item(nb::handle it) {
    if (PyIter_Check(it.ptr())) {
        PyObject* r = PyIter_Next(it.ptr());
        if (r) return nb::steal(r);
        if (PyErr_Occurred()) throw nb::python_error();
        return nb::object();
    }
    try {
        return it.attr("__next__")();
    } catch (nb::python_error& e) {
        if (e.matches(PyExc_StopIteration)) return nb::object();
        throw;
    }
}

}  // namespace screamer
#endif
