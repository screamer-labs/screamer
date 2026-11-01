#ifndef SCREAMER_DETAIL_START_POLICY_H
#define SCREAMER_DETAIL_START_POLICY_H

#include <string>

namespace screamer {
namespace detail {

enum class StartPolicy {
    Strict,
    Expanding,
    Zero
};

// FOrward declarations
StartPolicy parse_start_policy(const std::string& policy);

// The setting's name of a policy, the inverse of parse_start_policy.
inline const char* start_policy_name(StartPolicy policy) {
    switch (policy) {
    case StartPolicy::Expanding: return "expanding";
    case StartPolicy::Zero: return "zero";
    case StartPolicy::Strict: break;
    }
    return "strict";
}

} // namespace detail
} // namespace screamer
#endif