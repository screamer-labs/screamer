#ifndef SCREAMER_EXPANDING_PROD_H
#define SCREAMER_EXPANDING_PROD_H

// ExpandingProd: whole-history running product. Thin alias of CumProd, exposed
// under the Expanding* name. Same O(1) memory and "ignore" NaN policy.

#include "screamer/cum_prod.h"

namespace screamer {

class ExpandingProd : public CumProd {};

}  // namespace screamer

#endif  // SCREAMER_EXPANDING_PROD_H
