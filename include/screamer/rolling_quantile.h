#ifndef SCREAMER_ROLLING_QUANTILE_H
#define SCREAMER_ROLLING_QUANTILE_H

#include "screamer/common/finite_settings.h"
#include "screamer/common/buffer.h"
#include "screamer/common/base.h"
#include "common/order_statistic_tree.h"
#include "screamer/common/float_info.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
namespace screamer {

    class RollingQuantile : public serial::Saveable<RollingQuantile, ScreamerBase> {
    public:
        static constexpr const char* kTypeName = "RollingQuantile";
        static constexpr int kStateVersion = 1;

        // The buffer starts as reset() leaves it, NaN for an empty slot.
        RollingQuantile(int window_size, double quantile) :
            window_size(window_size),
            quantile(quantile),
            buffer(window_size, std::numeric_limits<double>::quiet_NaN()),
            ost(window_size)
        {
            require_finite("quantile", quantile);
            if (window_size <= 0) {
                throw std::invalid_argument("Window size must be positive.");
            }
            if (quantile < 0.0 || quantile > 1.0) {
                throw std::invalid_argument("Quantile must be between 0 and 1.");
            }
        }

        void reset() override
        {
            buffer.reset(std::numeric_limits<double>::quiet_NaN());
            ost.clear();
        }

        template <class Ar>
        void settings(Ar& ar) {
            ar("window_size", window_size);
            ar("quantile", quantile);
        }

        // The window oldest first (NaN for an empty slot) and the order
        // statistic tree of its values.
        template <class Ar>
        void state(Ar& ar) {
            std::vector<double> window;
            if constexpr (Ar::kSaving) window = buffer.oldest_first();
            ar.local("window", window, buffer.size());
            ar.object("tree", ost);
            if constexpr (Ar::kLoading) {
                std::vector<double> old = buffer.oldest_first();
                ar.apply([&] { buffer.assign_oldest_first(window); },
                         [this, old = std::move(old)] { buffer.assign_oldest_first(old); });
                ar.check(ost.holds(window), "tree must hold the values of the window");
            }
        }

    private:

        double process_scalar(double newValue) override
        {
            // NaN policy "ignore": leave the buffer and OST untouched, emit NaN.
            if (isnan2(newValue)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            double oldValue = buffer.append(newValue);

            if (!isnan2(oldValue)) {
                remove(oldValue);
            }

            add(newValue);

            if (ost.size() < window_size) {
                return std::numeric_limits<double>::quiet_NaN();
            } else {
                return getQuantile();
            }
        }

    private:
        int window_size;
        double quantile;
        FixedSizeBuffer buffer;
        OrderStatisticTree ost;

        void add(double x)
        {
            ost.insert(x);
        }

        void remove(double x)
        {
            ost.erase(x);
        }

        double getQuantile()
        {
            if (ost.size() == 0) {
                return -3; //std::numeric_limits<double>::quiet_NaN();
            }

            int n = ost.size();
            double pos = quantile * (n - 1);
            int index = static_cast<int>(std::floor(pos));
            double fraction = pos - index;

            double lower = ost.kth_element(index);
            double upper = lower;

            if (fraction > 0.0 && index + 1 < n) {
                upper = ost.kth_element(index + 1);
            }

            return lower + fraction * (upper - lower);
        }

    }; // end of class

} // end of namespace

#endif // SCREAMER_ROLLING_QUANTILE_H
