#ifndef SCREAMER_EXPANDING_MIN_H
#define SCREAMER_EXPANDING_MIN_H

// ExpandingMin: whole-history running minimum. Thin alias of CumMin, exposed
// under the Expanding* name. Same O(1) memory and "ignore" NaN policy.

#include "screamer/cum_min.h"

namespace screamer {

class ExpandingMin : public CumMin {};

}  // namespace screamer

#endif  // SCREAMER_EXPANDING_MIN_H
