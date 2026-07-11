#ifndef SCREAMER_LOGIC_H
#define SCREAMER_LOGIC_H

#include <cmath>
#include <limits>
#include "screamer/common/functor_base.h"

namespace screamer {

// --------------------------------------------------------------------------
// Binary comparison operators: 2 inputs -> 1.0/0.0 mask.
// NaN policy: if either input is NaN the output is NaN (unknown comparison).
// --------------------------------------------------------------------------

class GreaterThan : public FunctorBase<GreaterThan, 2, 1> {
public:
    ResultTuple call(const InputArray& in) override {
        if (std::isnan(in[0]) || std::isnan(in[1])) return std::numeric_limits<double>::quiet_NaN();
        return in[0] > in[1] ? 1.0 : 0.0;
    }
};

class LessThan : public FunctorBase<LessThan, 2, 1> {
public:
    ResultTuple call(const InputArray& in) override {
        if (std::isnan(in[0]) || std::isnan(in[1])) return std::numeric_limits<double>::quiet_NaN();
        return in[0] < in[1] ? 1.0 : 0.0;
    }
};

class GreaterEqual : public FunctorBase<GreaterEqual, 2, 1> {
public:
    ResultTuple call(const InputArray& in) override {
        if (std::isnan(in[0]) || std::isnan(in[1])) return std::numeric_limits<double>::quiet_NaN();
        return in[0] >= in[1] ? 1.0 : 0.0;
    }
};

class LessEqual : public FunctorBase<LessEqual, 2, 1> {
public:
    ResultTuple call(const InputArray& in) override {
        if (std::isnan(in[0]) || std::isnan(in[1])) return std::numeric_limits<double>::quiet_NaN();
        return in[0] <= in[1] ? 1.0 : 0.0;
    }
};

class Equal : public FunctorBase<Equal, 2, 1> {
public:
    ResultTuple call(const InputArray& in) override {
        if (std::isnan(in[0]) || std::isnan(in[1])) return std::numeric_limits<double>::quiet_NaN();
        return in[0] == in[1] ? 1.0 : 0.0;
    }
};

class NotEqual : public FunctorBase<NotEqual, 2, 1> {
public:
    ResultTuple call(const InputArray& in) override {
        if (std::isnan(in[0]) || std::isnan(in[1])) return std::numeric_limits<double>::quiet_NaN();
        return in[0] != in[1] ? 1.0 : 0.0;
    }
};

// --------------------------------------------------------------------------
// Binary logical operators: treat any nonzero value as true.
// NaN policy: NaN input -> NaN output.
// --------------------------------------------------------------------------

class And : public FunctorBase<And, 2, 1> {
public:
    ResultTuple call(const InputArray& in) override {
        if (std::isnan(in[0]) || std::isnan(in[1])) return std::numeric_limits<double>::quiet_NaN();
        return (in[0] != 0.0 && in[1] != 0.0) ? 1.0 : 0.0;
    }
};

class Or : public FunctorBase<Or, 2, 1> {
public:
    ResultTuple call(const InputArray& in) override {
        if (std::isnan(in[0]) || std::isnan(in[1])) return std::numeric_limits<double>::quiet_NaN();
        return (in[0] != 0.0 || in[1] != 0.0) ? 1.0 : 0.0;
    }
};

// --------------------------------------------------------------------------
// Where: 3 inputs -> in[0] nonzero ? in[1] : in[2].
// NaN policy: NaN mask -> NaN output.  NaN in the selected branch passes
// through unchanged (this is expected pass-through, not comparison NaN).
// --------------------------------------------------------------------------

class Where : public FunctorBase<Where, 3, 1> {
public:
    ResultTuple call(const InputArray& in) override {
        if (std::isnan(in[0])) return std::numeric_limits<double>::quiet_NaN();
        return in[0] != 0.0 ? in[1] : in[2];
    }
};

// --------------------------------------------------------------------------
// Unary logic operators.
// --------------------------------------------------------------------------

// Not: nonzero -> 0.0, zero -> 1.0.  NaN propagates.
class Not : public FunctorBase<Not, 1, 1> {
public:
    ResultTuple call(const InputArray& in) override {
        if (std::isnan(in[0])) return std::numeric_limits<double>::quiet_NaN();
        return in[0] != 0.0 ? 0.0 : 1.0;
    }
};

// IsNan: classifies NaN - does NOT propagate it.
// NaN -> 1.0, finite or inf -> 0.0.
class IsNan : public FunctorBase<IsNan, 1, 1> {
public:
    ResultTuple call(const InputArray& in) override {
        return std::isnan(in[0]) ? 1.0 : 0.0;
    }
};

// IsFinite: classifies finiteness - does NOT propagate NaN.
// finite -> 1.0, NaN or inf -> 0.0.
class IsFinite : public FunctorBase<IsFinite, 1, 1> {
public:
    ResultTuple call(const InputArray& in) override {
        return std::isfinite(in[0]) ? 1.0 : 0.0;
    }
};

} // namespace screamer
#endif
