#ifndef SCREAMER_GC_H
#define SCREAMER_GC_H

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace screamer {

// Garbage-collector support for bound classes that hold Python objects.
//
// A stream stored on the object that feeds it (src.it = RollingMean(2)(src))
// is a reference cycle. The collector frees a cycle only when every object in
// it reports its references (tp_traverse) and at least one can drop them
// (tp_clear). A class T opts in with
//
//     nb::class_<T>(m, "T", nb::type_slots(screamer::gc_slots<T>))
//
// and two members: `int traverse(visitproc, void*) const`, which reports every
// Python object T holds, and `void clear()`, which drops them and leaves T
// safe to call (its iterator ends). A class whose held objects back raw
// pointers it keeps using uses gc_traverse_slots<T> and needs only traverse():
// another object in the cycle must then clear it, as a Python subclass's
// __dict__ does.

// Reports one held reference (a null handle is skipped).
inline int gc_visit(nb::handle h, visitproc visit, void* arg) {
    Py_VISIT(h.ptr());
    return 0;
}

// Reports every object of a container of nb::object.
template <class C>
int gc_visit_all(const C& objects, visitproc visit, void* arg) {
    for (const auto& o : objects)
        if (int r = gc_visit(o, visit, arg)) return r;
    return 0;
}

template <class T>
int gc_traverse(PyObject* self, visitproc visit, void* arg) {
    Py_VISIT(Py_TYPE(self));               // a heap type is referenced by its instances
    if (!nb::inst_ready(self)) return 0;   // the C++ object is not constructed yet
    return nb::inst_ptr<T>(self)->traverse(visit, arg);
}

template <class T>
int gc_clear(PyObject* self) {
    if (nb::inst_ready(self)) nb::inst_ptr<T>(self)->clear();
    return 0;
}

template <class T>
inline PyType_Slot gc_slots[] = {
    {Py_tp_traverse, (void*) gc_traverse<T>},
    {Py_tp_clear, (void*) gc_clear<T>},
    {0, nullptr}};

template <class T>
inline PyType_Slot gc_traverse_slots[] = {
    {Py_tp_traverse, (void*) gc_traverse<T>},
    {0, nullptr}};

}  // namespace screamer

#endif  // SCREAMER_GC_H
