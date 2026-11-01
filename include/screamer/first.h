#ifndef SCREAMER_FIRST_H
#define SCREAMER_FIRST_H

// NaN policy "ignore": a NaN input is skipped --
// output is NaN at that index, the latched first value is unchanged.

#include <cmath>
#include <limits>
#include "screamer/common/base.h"

namespace screamer {

class First : public serial::Saveable<First, ScreamerBase> {
public:
    static constexpr const char* kTypeName = "First";
    static constexpr int kStateVersion = 1;

    template <class Ar>
    void settings(Ar&) {}

    template <class Ar>
    void state(Ar& ar) {
        ar("has_value", has_value_);
        ar("first_value", first_value_);
        ar.check(has_value_ != std::isnan(first_value_),
                 "first_value must be set exactly when has_value");
    }

    First() { reset(); }

    void reset() override {
        has_value_ = false;
        first_value_ = std::numeric_limits<double>::quiet_NaN();
    }

    double process_scalar(double x) override {
        if (std::isnan(x)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (!has_value_) {
            first_value_ = x;
            has_value_ = true;
        }
        return first_value_;
    }

private:
    bool has_value_ = false;
    double first_value_ = std::numeric_limits<double>::quiet_NaN();
};

}  // namespace screamer
#endif
