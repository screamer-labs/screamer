#include "screamer/common/async_generator.h"
#include <string>
#include <utility>

namespace screamer {

bool is_async_iterable(nb::handle obj) {
    // The type slot, not an attribute lookup: this runs on every operator call,
    // and a failed lookup on a number would raise and clear an AttributeError.
    return PyType_GetSlot(Py_TYPE(obj.ptr()), Py_am_aiter) != nullptr;
}

nb::object route_async(nb::object op_owner, nb::handle args, std::size_t n_in) {
    std::size_t n = 0, n_async = 0;
    for (nb::handle a : args) {
        ++n;
        if (is_async_iterable(a)) ++n_async;
    }
    if (n_async == 0) return nb::object();
    if (n_async != n)
        throw nb::type_error("async and synchronous inputs cannot be mixed: pass every input "
                             "as an async iterator, or none");
    if (n != 1 && n != n_in)
        throw nb::type_error(("this operator takes " + std::to_string(n_in) +
                              " async inputs or one async input of events holding every "
                              "field, got " + std::to_string(n)).c_str());
    std::vector<nb::object> sources;
    for (nb::handle a : args) sources.push_back(nb::borrow<nb::object>(a));
    return nb::cast(LazyAsyncIterator(std::move(op_owner), std::move(sources)));
}

LazyAsyncIterator::LazyAsyncIterator(nb::object op_owner, std::vector<nb::object> sources)
    : stepper_(std::move(op_owner), sources.size()) {
    // nanobind has no py::exec, so the coroutine ships as a real importable
    // module instead of a code string.
    anext_event_ = nb::module_::import_("screamer._async").attr("anext_event");
    for (auto& s : sources) sources_.append(s.attr("__aiter__")());
}

nb::object LazyAsyncIterator::__anext__() {
    // The coroutine holds this iterator, which holds the operator, so a
    // transient operator (RollingMean(5)(agen())) stays alive while awaited.
    return anext_event_(sources_, nb::find(this));
}

nb::object LazyAsyncIterator::step(nb::handle items) {
    items_.clear();
    for (nb::handle h : items) items_.push_back(nb::borrow<nb::object>(h));
    return stepper_.step(items_);
}

} // namespace screamer
