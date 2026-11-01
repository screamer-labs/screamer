#ifndef SCREAMER_MOMENTUM_H
#define SCREAMER_MOMENTUM_H

// Momentum(k): x[t] - x[t-k]. Mathematically identical to Diff(k);
// provided as a separately-named alias because TA-Lib calls this
// indicator MOM, and traders look for "momentum" in the API.
//
// Implemented as a thin subclass of Diff -- the underlying logic
// (delay buffer + subtraction) is shared, no duplication.

#include "screamer/diff.h"

namespace screamer {

class Momentum : public serial::Renamed<Momentum, Diff> {
public:
    static constexpr const char* kTypeName = "Momentum";
    using Renamed::Renamed;
};

}  // namespace screamer

#endif
