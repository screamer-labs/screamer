#include <pybind11/pybind11.h>
#include "screamer/common/base.h"
#include "screamer/common/transform.h"
#include "screamer/transform_functions.h"
#include "screamer/linear.h"
#include "screamer/linear2.h"
#include "screamer/power.h"
#include "screamer/geometry.h"

namespace py = pybind11;

void init_bindings_math(py::module& m) {

     py::class_<screamer::Transform<(double (*)(double)) std::abs>, screamer::ScreamerBase>(m, "Abs")
        .def(py::init<>())
        .def("__call__", &screamer::Transform<(double (*)(double)) std::abs>::operator(), py::arg("value"))
        .def("reset", &screamer::Transform<(double (*)(double)) std::abs>::reset, "Reset to the initial state.");

     py::class_<screamer::Transform<(double (*)(double)) std::log>, screamer::ScreamerBase>(m, "Log")
        .def(py::init<>())
        .def("__call__", &screamer::Transform<(double (*)(double)) std::log>::operator(), py::arg("value"))
        .def("reset", &screamer::Transform<(double (*)(double)) std::log>::reset, "Reset to the initial state.");

     py::class_<screamer::Transform<(double (*)(double)) std::exp>, screamer::ScreamerBase>(m, "Exp")
        .def(py::init<>())
        .def("__call__", &screamer::Transform<(double (*)(double)) std::exp>::operator(), py::arg("value"))
        .def("reset", &screamer::Transform<(double (*)(double)) std::exp>::reset, "Reset to the initial state.");

     py::class_<screamer::Transform<(double (*)(double)) std::sqrt>, screamer::ScreamerBase>(m, "Sqrt")
        .def(py::init<>())
        .def("__call__", &screamer::Transform<(double (*)(double)) std::sqrt>::operator(), py::arg("value"))
        .def("reset", &screamer::Transform<(double (*)(double)) std::sqrt>::reset, "Reset to the initial state.");

     py::class_<screamer::Transform<(double (*)(double)) std::erf>, screamer::ScreamerBase>(m, "Erf")
        .def(py::init<>())
        .def("__call__", &screamer::Transform<(double (*)(double)) std::erf>::operator(), py::arg("value"))
        .def("reset", &screamer::Transform<(double (*)(double)) std::erf>::reset, "Reset to the initial state.");

     py::class_<screamer::Transform<(double (*)(double)) std::erfc>, screamer::ScreamerBase>(m, "Erfc")
        .def(py::init<>())
        .def("__call__", &screamer::Transform<(double (*)(double)) std::erfc>::operator(), py::arg("value"))
        .def("reset", &screamer::Transform<(double (*)(double)) std::erfc>::reset, "Reset to the initial state.");

     py::class_<screamer::Transform<(double (*)(double))screamer::signum<double> >, screamer::ScreamerBase>(m, "Sign")
        .def(py::init<>())
        .def("__call__", &screamer::Transform<(double (*)(double)) screamer::signum<double>>::operator(), py::arg("value"))
        .def("reset", &screamer::Transform<(double (*)(double)) screamer::signum<double>>::reset, "Reset to the initial state.");

     py::class_<screamer::Transform<(double (*)(double)) std::tanh>, screamer::ScreamerBase>(m, "Tanh")
        .def(py::init<>())
        .def("__call__", &screamer::Transform<(double (*)(double)) std::tanh>::operator(), py::arg("value"))
        .def("reset", &screamer::Transform<(double (*)(double)) std::tanh>::reset, "Reset to the initial state.");

     py::class_<screamer::Transform<(double (*)(double)) screamer::relu>, screamer::ScreamerBase>(m, "Relu")
        .def(py::init<>())
        .def("__call__", &screamer::Transform<(double (*)(double)) screamer::relu>::operator(), py::arg("value"))
        .def("reset", &screamer::Transform<(double (*)(double)) screamer::relu>::reset, "Reset to the initial state.");

     py::class_<screamer::Transform<(double (*)(double)) screamer::selu>, screamer::ScreamerBase>(m, "Selu")
        .def(py::init<>())
        .def("__call__", &screamer::Transform<(double (*)(double)) screamer::selu>::operator(), py::arg("value"))
        .def("reset", &screamer::Transform<(double (*)(double)) screamer::selu>::reset, "Reset to the initial state.");

     py::class_<screamer::Transform<(double (*)(double)) screamer::elu>, screamer::ScreamerBase>(m, "Elu")
        .def(py::init<>())
        .def("__call__", &screamer::Transform<(double (*)(double)) screamer::elu>::operator(), py::arg("value"))
        .def("reset", &screamer::Transform<(double (*)(double)) screamer::elu>::reset, "Reset to the initial state.");

     py::class_<screamer::Transform<(double (*)(double)) screamer::softsign>, screamer::ScreamerBase>(m, "Softsign")
        .def(py::init<>())
        .def("__call__", &screamer::Transform<(double (*)(double)) screamer::softsign>::operator(), py::arg("value"))
        .def("reset", &screamer::Transform<(double (*)(double)) screamer::softsign>::reset, "Reset to the initial state.");

     py::class_<screamer::Transform<(double (*)(double)) screamer::sigmoid>, screamer::ScreamerBase>(m, "Sigmoid")
        .def(py::init<>())
        .def("__call__", &screamer::Transform<(double (*)(double)) screamer::sigmoid>::operator(), py::arg("value"))
        .def("reset", &screamer::Transform<(double (*)(double)) screamer::sigmoid>::reset, "Reset to the initial state.");

     py::class_<screamer::Linear, screamer::ScreamerBase>(m, "Linear")
        .def(py::init<double, double>(),
             py::arg("scale") = 1.0, py::arg("shift") = 0.0)
        .def("__call__", &screamer::Linear::operator(), py::arg("value"))
        .def("reset", &screamer::Linear::reset, "Reset to the initial state.");

     // Linear2: two-input affine combination f(x, y) = a*x + b*y + c.
     // Stateless 2->1; pairs well with Sign / Relu / Sigmoid for
     // compact one-shot expressions (e.g. Sign . Linear2(1,-1,0) is
     // "is x > y").
     py::class_<screamer::Linear2>(m, "Linear2")
        .def(py::init<double, double, double>(),
             py::arg("a") = 1.0, py::arg("b") = 1.0, py::arg("c") = 0.0)
        .def("__call__", &screamer::Linear2::handle_input)
        .def("reset", &screamer::Linear2::reset, "Reset to the initial state.");

     py::class_<screamer::Power, screamer::ScreamerBase>(m, "Power")
        .def(py::init<double>(), py::arg("p") = 2.0)
        .def("__call__", &screamer::Power::operator(), py::arg("value"))
        .def("reset", &screamer::Power::reset, "Reset to the initial state.");

     // Element-wise transforms wired through the Transform<...> template.
     // Floor / Ceil round toward negative / positive infinity respectively.
     py::class_<screamer::Transform<(double (*)(double)) std::floor>, screamer::ScreamerBase>(m, "Floor")
        .def(py::init<>())
        .def("__call__", &screamer::Transform<(double (*)(double)) std::floor>::operator(), py::arg("value"))
        .def("reset", &screamer::Transform<(double (*)(double)) std::floor>::reset, "Reset to the initial state.");

     py::class_<screamer::Transform<(double (*)(double)) std::ceil>, screamer::ScreamerBase>(m, "Ceil")
        .def(py::init<>())
        .def("__call__", &screamer::Transform<(double (*)(double)) std::ceil>::operator(), py::arg("value"))
        .def("reset", &screamer::Transform<(double (*)(double)) std::ceil>::reset, "Reset to the initial state.");

     // Square (x*x) and Cube (x*x*x): faster than Power(2) / Power(3) since
     // they skip the std::pow logarithm.
     py::class_<screamer::Transform<(double (*)(double)) screamer::square>, screamer::ScreamerBase>(m, "Square")
        .def(py::init<>())
        .def("__call__", &screamer::Transform<(double (*)(double)) screamer::square>::operator(), py::arg("value"))
        .def("reset", &screamer::Transform<(double (*)(double)) screamer::square>::reset, "Reset to the initial state.");

     py::class_<screamer::Transform<(double (*)(double)) screamer::cube>, screamer::ScreamerBase>(m, "Cube")
        .def(py::init<>())
        .def("__call__", &screamer::Transform<(double (*)(double)) screamer::cube>::operator(), py::arg("value"))
        .def("reset", &screamer::Transform<(double (*)(double)) screamer::cube>::reset, "Reset to the initial state.");

     // Trig: useful for cyclical features (time-of-day encoded as sin/cos
     // of a fraction-of-day angle, etc.).
     py::class_<screamer::Transform<(double (*)(double)) std::sin>, screamer::ScreamerBase>(m, "Sin")
        .def(py::init<>())
        .def("__call__", &screamer::Transform<(double (*)(double)) std::sin>::operator(), py::arg("value"))
        .def("reset", &screamer::Transform<(double (*)(double)) std::sin>::reset, "Reset to the initial state.");

     py::class_<screamer::Transform<(double (*)(double)) std::cos>, screamer::ScreamerBase>(m, "Cos")
        .def(py::init<>())
        .def("__call__", &screamer::Transform<(double (*)(double)) std::cos>::operator(), py::arg("value"))
        .def("reset", &screamer::Transform<(double (*)(double)) std::cos>::reset, "Reset to the initial state.");

     py::class_<screamer::Transform<(double (*)(double)) std::atan>, screamer::ScreamerBase>(m, "Atan")
        .def(py::init<>())
        .def("__call__", &screamer::Transform<(double (*)(double)) std::atan>::operator(), py::arg("value"))
        .def("reset", &screamer::Transform<(double (*)(double)) std::atan>::reset, "Reset to the initial state.");

     // Inverse trig: outputs NaN for inputs outside [-1, 1] (matches numpy).
     py::class_<screamer::Transform<(double (*)(double)) std::asin>, screamer::ScreamerBase>(m, "Asin")
        .def(py::init<>())
        .def("__call__", &screamer::Transform<(double (*)(double)) std::asin>::operator(), py::arg("value"))
        .def("reset", &screamer::Transform<(double (*)(double)) std::asin>::reset, "Reset to the initial state.");

     py::class_<screamer::Transform<(double (*)(double)) std::acos>, screamer::ScreamerBase>(m, "Acos")
        .def(py::init<>())
        .def("__call__", &screamer::Transform<(double (*)(double)) std::acos>::operator(), py::arg("value"))
        .def("reset", &screamer::Transform<(double (*)(double)) std::acos>::reset, "Reset to the initial state.");

     // Round: nearest integer with half-to-even (banker's) rounding,
     // matching numpy.round and Python's built-in round. std::round
     // would round half-away-from-zero, which numpy does NOT do.
     py::class_<screamer::Transform<(double (*)(double)) std::nearbyint>, screamer::ScreamerBase>(m, "Round")
        .def(py::init<>())
        .def("__call__", &screamer::Transform<(double (*)(double)) std::nearbyint>::operator(), py::arg("value"))
        .def("reset", &screamer::Transform<(double (*)(double)) std::nearbyint>::reset, "Reset to the initial state.");

     // Identity: pass-through, useful as a no-op pipeline node.
     py::class_<screamer::Transform<(double (*)(double)) screamer::identity>, screamer::ScreamerBase>(m, "Identity")
        .def(py::init<>())
        .def("__call__", &screamer::Transform<(double (*)(double)) screamer::identity>::operator(), py::arg("value"))
        .def("reset", &screamer::Transform<(double (*)(double)) screamer::identity>::reset, "Reset to the initial state.");

     // 2D coordinate / vector math. Hypot and Atan2 are 2->1 and exist
     // partly as primitives, partly as validation references for the
     // 2->2 polar conversions: Hypot(x, y) == Cart2Polar(x, y)[0],
     // Atan2(y, x) == Cart2Polar(x, y)[1].
     py::class_<screamer::Hypot>(m, "Hypot")
        .def(py::init<>())
        .def("__call__", &screamer::Hypot::handle_input)
        .def("reset", &screamer::Hypot::reset, "Reset to the initial state.");

     py::class_<screamer::Atan2>(m, "Atan2")
        .def(py::init<>())
        .def("__call__", &screamer::Atan2::handle_input)
        .def("reset", &screamer::Atan2::reset, "Reset to the initial state.");

     py::class_<screamer::Cart2Polar, screamer::EvalOp>(m, "Cart2Polar")
        .def(py::init<>())
        .def("__call__", &screamer::Cart2Polar::handle_input)
        .def("reset", &screamer::Cart2Polar::reset, "Reset to the initial state.");

     py::class_<screamer::Polar2Cart>(m, "Polar2Cart")
        .def(py::init<>())
        .def("__call__", &screamer::Polar2Cart::handle_input)
        .def("reset", &screamer::Polar2Cart::reset, "Reset to the initial state.");

}