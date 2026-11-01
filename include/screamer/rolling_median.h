#ifndef SCREAMER_ROLLING_MEDIAN_H
#define SCREAMER_ROLLING_MEDIAN_H

#include <screamer/common/buffer.h>
#include "screamer/common/base.h"
#include "screamer/common/float_info.h"
#include <algorithm>
#include <cstddef>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>
namespace screamer {

    class RollingMedian : public serial::Saveable<RollingMedian, ScreamerBase> {
    public:
        static constexpr const char* kTypeName = "RollingMedian";
        static constexpr int kStateVersion = 1;

        template <class Ar>
        void settings(Ar& ar) {
            ar("window_size", window_size);
        }

        template <class Ar>
        void state(Ar& ar) {
            // The window oldest first (NaN for an empty slot) and the two halves in
            // increasing order.
            std::vector<double> window, lo, hi;
            if constexpr (Ar::kSaving) {
                window = buffer.oldest_first();
                lo.assign(low.begin(), low.end());
                hi.assign(high.begin(), high.end());
            }
            ar.local("window", window, buffer.size());
            ar.local("low", lo);
            ar.local("high", hi);
            if constexpr (Ar::kLoading) {
                ar.check(lo.size() == hi.size() || lo.size() == hi.size() + 1,
                         "low must hold as many values as high, or one more");
                std::vector<double> both(lo);
                both.insert(both.end(), hi.begin(), hi.end());
                for (std::size_t k = 1; k < both.size(); ++k)
                    ar.check(both[k - 1] <= both[k], "low then high must not decrease");
                std::vector<double> values;
                for (double v : window)
                    if (v == v) values.push_back(v);
                std::sort(values.begin(), values.end());
                bool same = values.size() == both.size();
                for (std::size_t k = 0; same && k < both.size(); ++k) same = values[k] == both[k];
                ar.check(same, "low and high must hold the values of the window");
                std::vector<double> old = buffer.oldest_first();
                ar.apply([&] { buffer.assign_oldest_first(window); },
                         [this, old = std::move(old)] { buffer.assign_oldest_first(old); });
                ar.set(low, std::multiset<double>(lo.begin(), lo.end()));
                ar.set(high, std::multiset<double>(hi.begin(), hi.end()));
            }
        }

        RollingMedian(int window_size) : 
            window_size(window_size), 
            buffer(window_size, std::numeric_limits<double>::quiet_NaN()) 
        {
            if (window_size <= 0) {
                throw std::invalid_argument("Window size must be positive.");
            }
        }

        void reset() override
        {
            buffer.reset(std::numeric_limits<double>::quiet_NaN());
            low.clear();
            high.clear();
        }
        
    private:

        double process_scalar(double newValue) override
        {
            // NaN policy "ignore": leave the buffer and the multisets
            // untouched and emit NaN.
            if (isnan2(newValue)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            double oldValue = buffer.append(newValue);

            if (!isnan2(oldValue)) {
                remove(oldValue);
            }

            add(newValue);

            if (low.empty() && high.empty()) {
                return std::numeric_limits<double>::quiet_NaN();
            } else {
                return getMedian();
            }
        }

    private:
        int window_size;
        FixedSizeBuffer buffer;
        std::multiset<double> low;  // Max heap (lower half)
        std::multiset<double> high; // Min heap (upper half)

        void add(double x)
        {
            // Insert into appropriate half
            if (low.empty() || x <= *low.rbegin()) {
                low.insert(x);
            } else {
                high.insert(x);
            }

            // Rebalance the two halves
            rebalance();
        }

        void remove(double x)
        {
            // Remove from the appropriate half
            auto it = low.find(x);
            if (it != low.end()) {
                low.erase(it);
            } else {
                it = high.find(x);
                if (it != high.end()) {
                    high.erase(it);
                }
            }

            // Rebalance after removal
            rebalance();
        }

        void rebalance()
        {
            // Ensure size properties: low.size() >= high.size()
            if (low.size() > high.size() + 1) {
                high.insert(*low.rbegin());
                low.erase(--low.end());
            } else if (high.size() > low.size()) {
                low.insert(*high.begin());
                high.erase(high.begin());
            }
        }

        double getMedian()
        {
            if (low.empty()) {
                return std::numeric_limits<double>::quiet_NaN();
            }

            if (low.size() == high.size()) {
                // Median is the average of max of low and min of high
                return (*low.rbegin() + *high.begin()) / 2.0;
            } else {
                // Median is max of low
                return *low.rbegin();
            }
        }
    
    }; // end of class

} // end of namespace

#endif // end of include guards