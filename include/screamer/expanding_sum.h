#ifndef SCREAMER_EXPANDING_SUM_H
#define SCREAMER_EXPANDING_SUM_H

// ExpandingSum: whole-history running sum. Thin alias of CumSum, exposed under
// the Expanding* name so the expanding family is complete. Same O(1) memory
// and same "ignore" NaN policy as CumSum.

#include "screamer/cum_sum.h"

namespace screamer {

class ExpandingSum : public CumSum {};

}  // namespace screamer

#endif  // SCREAMER_EXPANDING_SUM_H
