#include <nanobind/nanobind.h>
#include "screamer/common/base.h"
#include "screamer/common/dispatch.h"
#include "screamer/common/transform.h"
#include "screamer/common/eval_op.h"
#include "screamer/transform_functions.h"
#include "screamer/linear.h"
#include "screamer/linear2.h"
#include "screamer/power.h"
#include "screamer/geometry.h"
#include "screamer/arithmetic.h"
#include "screamer/logic.h"
#include <cmath>

namespace nb = nanobind;
using namespace nb::literals;

void init_bindings_math(nb::module_& m) {

     nb::class_<screamer::Transform<(double (*)(double)) std::abs>, screamer::ScreamerBase>(m, "Abs")
        .def(nb::init<>())
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::Transform<(double (*)(double)) std::abs>::reset, "Reset to the initial state.");

     nb::class_<screamer::Transform<(double (*)(double)) std::log>, screamer::ScreamerBase>(m, "Log")
        .def(nb::init<>())
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::Transform<(double (*)(double)) std::log>::reset, "Reset to the initial state.");

     nb::class_<screamer::Transform<(double (*)(double)) std::exp>, screamer::ScreamerBase>(m, "Exp")
        .def(nb::init<>())
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::Transform<(double (*)(double)) std::exp>::reset, "Reset to the initial state.");

     nb::class_<screamer::Transform<(double (*)(double)) std::sqrt>, screamer::ScreamerBase>(m, "Sqrt")
        .def(nb::init<>())
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::Transform<(double (*)(double)) std::sqrt>::reset, "Reset to the initial state.");

     nb::class_<screamer::Transform<(double (*)(double)) std::erf>, screamer::ScreamerBase>(m, "Erf")
        .def(nb::init<>())
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::Transform<(double (*)(double)) std::erf>::reset, "Reset to the initial state.");

     nb::class_<screamer::Transform<(double (*)(double)) std::erfc>, screamer::ScreamerBase>(m, "Erfc")
        .def(nb::init<>())
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::Transform<(double (*)(double)) std::erfc>::reset, "Reset to the initial state.");

     nb::class_<screamer::Transform<(double (*)(double))screamer::signum<double> >, screamer::ScreamerBase>(m, "Sign")
        .def(nb::init<>())
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::Transform<(double (*)(double)) screamer::signum<double>>::reset, "Reset to the initial state.");

     nb::class_<screamer::Transform<(double (*)(double)) std::tanh>, screamer::ScreamerBase>(m, "Tanh")
        .def(nb::init<>())
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::Transform<(double (*)(double)) std::tanh>::reset, "Reset to the initial state.");

     nb::class_<screamer::Transform<(double (*)(double)) screamer::relu>, screamer::ScreamerBase>(m, "Relu")
        .def(nb::init<>())
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::Transform<(double (*)(double)) screamer::relu>::reset, "Reset to the initial state.");

     nb::class_<screamer::Transform<(double (*)(double)) screamer::pos_part>, screamer::ScreamerBase>(m, "PosPart")
        .def(nb::init<>())
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::Transform<(double (*)(double)) screamer::pos_part>::reset, "Reset to the initial state.");

     nb::class_<screamer::Transform<(double (*)(double)) screamer::neg_part>, screamer::ScreamerBase>(m, "NegPart")
        .def(nb::init<>())
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::Transform<(double (*)(double)) screamer::neg_part>::reset, "Reset to the initial state.");

     nb::class_<screamer::Transform<(double (*)(double)) screamer::selu>, screamer::ScreamerBase>(m, "Selu")
        .def(nb::init<>())
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::Transform<(double (*)(double)) screamer::selu>::reset, "Reset to the initial state.");

     nb::class_<screamer::Transform<(double (*)(double)) screamer::elu>, screamer::ScreamerBase>(m, "Elu")
        .def(nb::init<>())
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::Transform<(double (*)(double)) screamer::elu>::reset, "Reset to the initial state.");

     nb::class_<screamer::Transform<(double (*)(double)) screamer::softsign>, screamer::ScreamerBase>(m, "Softsign")
        .def(nb::init<>())
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::Transform<(double (*)(double)) screamer::softsign>::reset, "Reset to the initial state.");

     nb::class_<screamer::Transform<(double (*)(double)) screamer::sigmoid>, screamer::ScreamerBase>(m, "Sigmoid")
        .def(nb::init<>())
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::Transform<(double (*)(double)) screamer::sigmoid>::reset, "Reset to the initial state.");

     nb::class_<screamer::Linear, screamer::ScreamerBase>(m, "Linear")
        .def(nb::init<double, double>(),
             "scale"_a = 1.0, "shift"_a = 0.0)
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::Linear::reset, "Reset to the initial state.");

     // Linear2: two-input affine combination f(x, y) = a*x + b*y + c.
     // Stateless 2->1; pairs well with Sign / Relu / Sigmoid for
     // compact one-shot expressions (e.g. Sign . Linear2(1,-1,0) is
     // "is x > y").
     nb::class_<screamer::Linear2, screamer::EvalOp>(m, "Linear2")
        .def(nb::init<double, double, double>(),
             "a"_a = 1.0, "b"_a = 1.0, "c"_a = 0.0)
        .def("__call__", &screamer::Linear2::handle_input)
        .def("reset", &screamer::Linear2::reset, "Reset to the initial state.");

     nb::class_<screamer::Power, screamer::ScreamerBase>(m, "Power")
        .def(nb::init<double>(), "p"_a = 2.0)
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::Power::reset, "Reset to the initial state.");

     // Element-wise transforms wired through the Transform<...> template.
     // Floor / Ceil round toward negative / positive infinity respectively.
     nb::class_<screamer::Transform<(double (*)(double)) std::floor>, screamer::ScreamerBase>(m, "Floor")
        .def(nb::init<>())
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::Transform<(double (*)(double)) std::floor>::reset, "Reset to the initial state.");

     nb::class_<screamer::Transform<(double (*)(double)) std::ceil>, screamer::ScreamerBase>(m, "Ceil")
        .def(nb::init<>())
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::Transform<(double (*)(double)) std::ceil>::reset, "Reset to the initial state.");

     // Square (x*x) and Cube (x*x*x): faster than Power(2) / Power(3) since
     // they skip the std::pow logarithm.
     nb::class_<screamer::Transform<(double (*)(double)) screamer::square>, screamer::ScreamerBase>(m, "Square")
        .def(nb::init<>())
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::Transform<(double (*)(double)) screamer::square>::reset, "Reset to the initial state.");

     nb::class_<screamer::Transform<(double (*)(double)) screamer::cube>, screamer::ScreamerBase>(m, "Cube")
        .def(nb::init<>())
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::Transform<(double (*)(double)) screamer::cube>::reset, "Reset to the initial state.");

     // Trig: useful for cyclical features (time-of-day encoded as sin/cos
     // of a fraction-of-day angle, etc.).
     nb::class_<screamer::Transform<(double (*)(double)) std::sin>, screamer::ScreamerBase>(m, "Sin")
        .def(nb::init<>())
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::Transform<(double (*)(double)) std::sin>::reset, "Reset to the initial state.");

     nb::class_<screamer::Transform<(double (*)(double)) std::cos>, screamer::ScreamerBase>(m, "Cos")
        .def(nb::init<>())
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::Transform<(double (*)(double)) std::cos>::reset, "Reset to the initial state.");

     nb::class_<screamer::Transform<(double (*)(double)) std::atan>, screamer::ScreamerBase>(m, "Atan")
        .def(nb::init<>())
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::Transform<(double (*)(double)) std::atan>::reset, "Reset to the initial state.");

     // Inverse trig: outputs NaN for inputs outside [-1, 1] (matches numpy).
     nb::class_<screamer::Transform<(double (*)(double)) std::asin>, screamer::ScreamerBase>(m, "Asin")
        .def(nb::init<>())
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::Transform<(double (*)(double)) std::asin>::reset, "Reset to the initial state.");

     nb::class_<screamer::Transform<(double (*)(double)) std::acos>, screamer::ScreamerBase>(m, "Acos")
        .def(nb::init<>())
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::Transform<(double (*)(double)) std::acos>::reset, "Reset to the initial state.");

     // Round: nearest integer with half-to-even (banker's) rounding,
     // matching numpy.round and Python's built-in round. std::round
     // would round half-away-from-zero, which numpy does NOT do.
     nb::class_<screamer::Transform<(double (*)(double)) std::nearbyint>, screamer::ScreamerBase>(m, "Round")
        .def(nb::init<>())
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::Transform<(double (*)(double)) std::nearbyint>::reset, "Reset to the initial state.");

     // Identity: pass-through, useful as a no-op pipeline node.
     nb::class_<screamer::Transform<(double (*)(double)) screamer::identity>, screamer::ScreamerBase>(m, "Identity")
        .def(nb::init<>())
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::Transform<(double (*)(double)) screamer::identity>::reset, "Reset to the initial state.");

     // 2D coordinate / vector math. Hypot and Atan2 are 2->1 and exist
     // partly as primitives, partly as validation references for the
     // 2->2 polar conversions: Hypot(x, y) == Cart2Polar(x, y)[0],
     // Atan2(y, x) == Cart2Polar(x, y)[1].
     nb::class_<screamer::Hypot, screamer::EvalOp>(m, "Hypot")
        .def(nb::init<>())
        .def("__call__", &screamer::Hypot::handle_input)
        .def("reset", &screamer::Hypot::reset, "Reset to the initial state.");

     nb::class_<screamer::Atan2, screamer::EvalOp>(m, "Atan2")
        .def(nb::init<>())
        .def("__call__", &screamer::Atan2::handle_input)
        .def("reset", &screamer::Atan2::reset, "Reset to the initial state.");

     nb::class_<screamer::Cart2Polar, screamer::EvalOp>(m, "Cart2Polar")
        .def(nb::init<>())
        .def("__call__", &screamer::Cart2Polar::handle_input)
        .def("reset", &screamer::Cart2Polar::reset, "Reset to the initial state.");

     nb::class_<screamer::Polar2Cart, screamer::EvalOp>(m, "Polar2Cart")
        .def(nb::init<>())
        .def("__call__", &screamer::Polar2Cart::handle_input)
        .def("reset", &screamer::Polar2Cart::reset, "Reset to the initial state.");

     nb::class_<screamer::Add, screamer::EvalOp>(m, "Add")
        .def(nb::init<>())
        .def("__call__", &screamer::Add::handle_input)
        .def("reset", &screamer::Add::reset, "Reset to the initial state.");

     nb::class_<screamer::Sub, screamer::EvalOp>(m, "Sub")
        .def(nb::init<>())
        .def("__call__", &screamer::Sub::handle_input)
        .def("reset", &screamer::Sub::reset, "Reset to the initial state.");

     nb::class_<screamer::Mul, screamer::EvalOp>(m, "Mul")
        .def(nb::init<>())
        .def("__call__", &screamer::Mul::handle_input)
        .def("reset", &screamer::Mul::reset, "Reset to the initial state.");

     nb::class_<screamer::Div, screamer::EvalOp>(m, "Div")
        .def(nb::init<>())
        .def("__call__", &screamer::Div::handle_input)
        .def("reset", &screamer::Div::reset, "Reset to the initial state.");

     // -----------------------------------------------------------------------
     // Comparison operators: 2 inputs -> 1.0/0.0 mask.  NaN in -> NaN out.
     // -----------------------------------------------------------------------

     nb::class_<screamer::GreaterThan, screamer::EvalOp>(m, "GreaterThan")
        .def(nb::init<>())
        .def("__call__", &screamer::GreaterThan::handle_input)
        .def("reset", &screamer::GreaterThan::reset, "Reset to the initial state.");

     nb::class_<screamer::LessThan, screamer::EvalOp>(m, "LessThan")
        .def(nb::init<>())
        .def("__call__", &screamer::LessThan::handle_input)
        .def("reset", &screamer::LessThan::reset, "Reset to the initial state.");

     nb::class_<screamer::GreaterEqual, screamer::EvalOp>(m, "GreaterEqual")
        .def(nb::init<>())
        .def("__call__", &screamer::GreaterEqual::handle_input)
        .def("reset", &screamer::GreaterEqual::reset, "Reset to the initial state.");

     nb::class_<screamer::LessEqual, screamer::EvalOp>(m, "LessEqual")
        .def(nb::init<>())
        .def("__call__", &screamer::LessEqual::handle_input)
        .def("reset", &screamer::LessEqual::reset, "Reset to the initial state.");

     nb::class_<screamer::Equal, screamer::EvalOp>(m, "Equal")
        .def(nb::init<>())
        .def("__call__", &screamer::Equal::handle_input)
        .def("reset", &screamer::Equal::reset, "Reset to the initial state.");

     nb::class_<screamer::NotEqual, screamer::EvalOp>(m, "NotEqual")
        .def(nb::init<>())
        .def("__call__", &screamer::NotEqual::handle_input)
        .def("reset", &screamer::NotEqual::reset, "Reset to the initial state.");

     // -----------------------------------------------------------------------
     // Logical operators: nonzero test, 2 inputs.  NaN propagates.
     // -----------------------------------------------------------------------

     nb::class_<screamer::And, screamer::EvalOp>(m, "And")
        .def(nb::init<>())
        .def("__call__", &screamer::And::handle_input)
        .def("reset", &screamer::And::reset, "Reset to the initial state.");

     nb::class_<screamer::Or, screamer::EvalOp>(m, "Or")
        .def(nb::init<>())
        .def("__call__", &screamer::Or::handle_input)
        .def("reset", &screamer::Or::reset, "Reset to the initial state.");

     // -----------------------------------------------------------------------
     // Where: 3-input conditional select.  NaN mask -> NaN output.
     // -----------------------------------------------------------------------

     nb::class_<screamer::Where, screamer::EvalOp>(m, "Where")
        .def(nb::init<>())
        .def("__call__", &screamer::Where::handle_input)
        .def("reset", &screamer::Where::reset, "Reset to the initial state.");

     // -----------------------------------------------------------------------
     // Unary logic: Not, IsNan, IsFinite.
     // -----------------------------------------------------------------------

     nb::class_<screamer::Not, screamer::EvalOp>(m, "Not")
        .def(nb::init<>())
        .def("__call__", &screamer::Not::handle_input)
        .def("reset", &screamer::Not::reset, "Reset to the initial state.");

     nb::class_<screamer::IsNan, screamer::EvalOp>(m, "IsNan")
        .def(nb::init<>())
        .def("__call__", &screamer::IsNan::handle_input)
        .def("reset", &screamer::IsNan::reset, "Reset to the initial state.");

     nb::class_<screamer::IsFinite, screamer::EvalOp>(m, "IsFinite")
        .def(nb::init<>())
        .def("__call__", &screamer::IsFinite::handle_input)
        .def("reset", &screamer::IsFinite::reset, "Reset to the initial state.");

}
