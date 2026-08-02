#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include "screamer/common/base.h"
#include "screamer/common/dispatch.h"
#include "screamer/ffill.h"
#include "screamer/fillna.h"
#include "screamer/clip.h"

namespace nb = nanobind;
using namespace nb::literals;

void init_bindings_preprocessing(nb::module_& m) {

    nb::class_<screamer::Ffill, screamer::ScreamerBase>(m, "Ffill")
        .def(nb::init<>())
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::Ffill::reset, "Reset to the initial state.");

    nb::class_<screamer::FillNa, screamer::ScreamerBase>(m, "FillNa")
        .def(nb::init<double>(), "fill"_a = 0.0)
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::FillNa::reset, "Reset to the initial state.");


     nb::class_<screamer::Clip, screamer::ScreamerBase>(m, "Clip")
        .def(
          nb::init<
               std::optional<double>,
               std::optional<double>
          >(),
          "lower"_a = nb::none(),
          "upper"_a = nb::none()
        )
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::Clip::reset, "Reset to the initial state.");

}
