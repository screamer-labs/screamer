#ifndef SCREAMER_TRANSFORM_H
#define SCREAMER_TRANSFORM_H

#include <optional>
#include "screamer/common/base.h"
#include "screamer/transform_functions.h"
#include <algorithm>
#include <execution>
#include <cmath>
#include <functional>
#include <cstddef>

namespace screamer {

    // The registry name of each Transform, its type in a save file.
    template <double (*F)(double)>
    inline constexpr const char* transform_type_name = nullptr;

#define SCREAMER_TRANSFORM_NAME(func, name) \
    template <> inline constexpr const char* transform_type_name<(double (*)(double)) func> = name;
    SCREAMER_TRANSFORM_NAME(std::abs, "Abs")
    SCREAMER_TRANSFORM_NAME(std::log, "Log")
    SCREAMER_TRANSFORM_NAME(std::exp, "Exp")
    SCREAMER_TRANSFORM_NAME(std::sqrt, "Sqrt")
    SCREAMER_TRANSFORM_NAME(std::erf, "Erf")
    SCREAMER_TRANSFORM_NAME(std::erfc, "Erfc")
    SCREAMER_TRANSFORM_NAME(screamer::signum<double>, "Sign")
    SCREAMER_TRANSFORM_NAME(std::tanh, "Tanh")
    SCREAMER_TRANSFORM_NAME(screamer::relu, "Relu")
    SCREAMER_TRANSFORM_NAME(screamer::pos_part, "PosPart")
    SCREAMER_TRANSFORM_NAME(screamer::neg_part, "NegPart")
    SCREAMER_TRANSFORM_NAME(screamer::selu, "Selu")
    SCREAMER_TRANSFORM_NAME(screamer::elu, "Elu")
    SCREAMER_TRANSFORM_NAME(screamer::softsign, "Softsign")
    SCREAMER_TRANSFORM_NAME(screamer::sigmoid, "Sigmoid")
    SCREAMER_TRANSFORM_NAME(std::floor, "Floor")
    SCREAMER_TRANSFORM_NAME(std::ceil, "Ceil")
    SCREAMER_TRANSFORM_NAME(screamer::square, "Square")
    SCREAMER_TRANSFORM_NAME(screamer::cube, "Cube")
    SCREAMER_TRANSFORM_NAME(std::sin, "Sin")
    SCREAMER_TRANSFORM_NAME(std::cos, "Cos")
    SCREAMER_TRANSFORM_NAME(std::atan, "Atan")
    SCREAMER_TRANSFORM_NAME(std::asin, "Asin")
    SCREAMER_TRANSFORM_NAME(std::acos, "Acos")
    SCREAMER_TRANSFORM_NAME(std::nearbyint, "Round")
    SCREAMER_TRANSFORM_NAME(screamer::identity, "Identity")
#undef SCREAMER_TRANSFORM_NAME

    // A stateless element-wise function; it saves no state.
    template <double (*TransformFunc)(double)>
    class Transform : public serial::Saveable<Transform<TransformFunc>, ScreamerBase> {
    public:
        static constexpr const char* kTypeName = transform_type_name<TransformFunc>;
        static_assert(kTypeName != nullptr, "a Transform needs a name in transform_type_name");
        static constexpr int kStateVersion = 1;

        Transform() {}

        template <class Ar>
        void settings(Ar&) {}

        template <class Ar>
        void state(Ar&) {}

        double process_scalar(double newValue) override {
            return TransformFunc(newValue);
        }

        void process_array_no_stride(double* y, const double* x, size_t size) override {
            size_t i = 0;

            for (size_t i=0; i < size; i ++) {
                y[i] = TransformFunc(x[i]);
            }

        }

        void process_array_stride(double* y, size_t dyi, const double* x, size_t dxi, size_t size) override {
            size_t yi = 0;
            size_t xi = 0;

            for (size_t i = 0; i < size; i++) {
                y[yi] = TransformFunc(x[xi]);
                xi += dxi;
                yi += dyi;
            }
        }

    };

} // namespace screamer

#endif // SCREAMER_TRANSFORM_H
