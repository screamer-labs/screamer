#ifndef SCREAMER_PER_ASSET_H
#define SCREAMER_PER_ASSET_H

#include <initializer_list>
#include <vector>

namespace screamer {

// PerAsset: a parameter given once for all assets or once per asset, such as
// a portfolio engine's fees or contract multipliers. It holds one value or K
// values and converts to std::vector<double>; the bindings accept a single
// number for the one-value form.
struct PerAsset {
    std::vector<double> values;
    PerAsset() = default;
    PerAsset(std::initializer_list<double> v) : values(v) {}
    PerAsset(std::vector<double> v) : values(std::move(v)) {}
    operator std::vector<double>() const { return values; }
};

} // namespace screamer
#endif
