#ifndef SCREAMER_EXPANDING_MAX_H
#define SCREAMER_EXPANDING_MAX_H

// ExpandingMax: whole-history running maximum. Thin alias of CumMax, exposed
// under the Expanding* name. Same O(1) memory and "ignore" NaN policy.

#include "screamer/cum_max.h"

namespace screamer {

class ExpandingMax : public CumMax {};

}  // namespace screamer

#endif  // SCREAMER_EXPANDING_MAX_H
