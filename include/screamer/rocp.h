#ifndef SCREAMER_ROCP_H
#define SCREAMER_ROCP_H

// ROCP(k): rate of change as a fraction,
//
//     ROCP[t] = (x[t] - x[t-k]) / x[t-k]
//
// Mathematically identical to Return(k); separately named because
// TA-Lib's ROCP is widely referenced. Implemented as a subclass of
// Return so the implementation is shared.

#include "screamer/return.h"

namespace screamer {

class ROCP : public serial::Renamed<ROCP, Return> {
public:
    static constexpr const char* kTypeName = "ROCP";
    using Renamed::Renamed;
};

}  // namespace screamer

#endif
