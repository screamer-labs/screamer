#ifndef SCREAMER_DETAIL_EW_ARGS_H
#define SCREAMER_DETAIL_EW_ARGS_H

// The smoothing arguments of an exponentially weighted operator as given:
// exactly one of com, span, halflife and alpha, the others empty. The
// operator keeps them for its settings (a save file names the one given);
// its constructor derives alpha from them.

#include <optional>

namespace screamer { namespace detail {

struct EwArgs {
    std::optional<double> com, span, halflife, alpha;

    template <class Ar>
    void settings(Ar& ar) {
        ar("com", com);
        ar("span", span);
        ar("halflife", halflife);
        ar("alpha", alpha);
    }
};

}}  // namespace screamer::detail

#endif  // SCREAMER_DETAIL_EW_ARGS_H
