#ifndef SCREAMER_ARITHMETIC_H
#define SCREAMER_ARITHMETIC_H

#include "screamer/common/functor_base.h"

namespace screamer {

// Binary elementwise arithmetic: 2 inputs -> 1 output. Stateless. These are the
// C++-only reduction vocabulary for computation DAGs (e.g. a price spread is
// Sub()(combine_latest(a, b))).
class Add : public FunctorBase<Add, 2, 1> {
public:
    ResultTuple call(const InputArray& in) override { return in[0] + in[1]; }
};

class Sub : public FunctorBase<Sub, 2, 1> {
public:
    ResultTuple call(const InputArray& in) override { return in[0] - in[1]; }
};

class Mul : public FunctorBase<Mul, 2, 1> {
public:
    ResultTuple call(const InputArray& in) override { return in[0] * in[1]; }
};

class Div : public FunctorBase<Div, 2, 1> {
public:
    ResultTuple call(const InputArray& in) override { return in[0] / in[1]; }
};

} // namespace screamer
#endif
