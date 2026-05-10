#ifndef SCREAMER_GEOMETRY_H
#define SCREAMER_GEOMETRY_H

// Stateless 2D coordinate / vector math.
//
//   Hypot(x, y)           = sqrt(x^2 + y^2)        (2 -> 1)
//   Atan2(y, x)           = atan2(y, x)             (2 -> 1)
//   Cart2Polar(x, y)      = (r, theta)              (2 -> 2)
//   Polar2Cart(r, theta)  = (x, y)                  (2 -> 2)
//
// Argument order matches numpy: arctan2 takes (y, x); hypot takes (x, y).
// The polar pair is inverse: Polar2Cart(Cart2Polar(x, y)) == (x, y) up to
// floating-point precision.
//
// All four are stateless, so reset() is a no-op (FunctorBase default).

#include <cmath>
#include "screamer/common/functor_base.h"

namespace screamer {

class Hypot : public FunctorBase<Hypot, 2, 1> {
public:
    ResultTuple call(const InputArray& inputs) override {
        return std::hypot(inputs[0], inputs[1]);
    }
};

class Atan2 : public FunctorBase<Atan2, 2, 1> {
public:
    ResultTuple call(const InputArray& inputs) override {
        // numpy.arctan2(y, x): first arg is y, second is x.
        return std::atan2(inputs[0], inputs[1]);
    }
};

class Cart2Polar : public FunctorBase<Cart2Polar, 2, 2> {
public:
    ResultTuple call(const InputArray& inputs) override {
        const double x = inputs[0];
        const double y = inputs[1];
        return {std::hypot(x, y), std::atan2(y, x)};
    }
};

class Polar2Cart : public FunctorBase<Polar2Cart, 2, 2> {
public:
    ResultTuple call(const InputArray& inputs) override {
        const double r = inputs[0];
        const double theta = inputs[1];
        return {r * std::cos(theta), r * std::sin(theta)};
    }
};

}  // namespace screamer

#endif  // SCREAMER_GEOMETRY_H
