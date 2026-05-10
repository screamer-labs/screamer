#ifndef SCREAMER_LINEAR2_H
#define SCREAMER_LINEAR2_H

// Linear2: two-input affine combination,
//
//     f(x, y) = a*x + b*y + c
//
// Stateless 2->1 functor. Useful as a building block: chained with
// Sign / Relu / Sigmoid etc. it produces compact one-shot expressions
// for things like "is x > y" (Sign(Linear2(1, -1, 0)(x, y))),
// "positive excess" (Relu(Linear2(1, -1, 0)(x, y))), or weighted
// blends.

#include "screamer/common/functor_base.h"

namespace screamer {

class Linear2 : public FunctorBase<Linear2, 2, 1> {
public:
    Linear2(double a, double b, double c = 0.0) : a_(a), b_(b), c_(c) {}

    ResultTuple call(const InputArray& inputs) override {
        return a_ * inputs[0] + b_ * inputs[1] + c_;
    }

private:
    const double a_;
    const double b_;
    const double c_;
};

}  // namespace screamer

#endif
