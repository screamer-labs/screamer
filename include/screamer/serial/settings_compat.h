#ifndef SCREAMER_SERIAL_SETTINGS_COMPAT_H
#define SCREAMER_SERIAL_SETTINGS_COMPAT_H

// The renamed constructor parameters, so a file written before a rename still
// loads. Every loader applies the table before it checks the settings: the C++
// registry (serial::make_op), Python's screamer.loads and JavaScript's loads.
//
// A row (type, old, new) renames the setting `old` of operator `type` to
// `new`, in place. A file that gives both names is refused. When a parameter
// of an operator is renamed, add its row here; never remove a row.

#include <cstddef>
#include <string>
#include <vector>

#include "screamer/common/json.h"
#include "screamer/serial/errors.h"

namespace screamer { namespace serial {

struct SettingRename {
    std::string type;
    std::string old_name;
    std::string new_name;
};

namespace detail {

// The table.
inline const std::vector<SettingRename>& setting_renames() {
    static const std::vector<SettingRename> table = {
        // {"Type", "old_name", "new_name"},
    };
    return table;
}

// The rows a test adds (push_test_rename), applied after the table.
inline std::vector<SettingRename>& test_renames() {
    static std::vector<SettingRename> rows;
    return rows;
}

inline void apply_rename(const SettingRename& r, json::Value& settings, const std::string& path) {
    std::size_t at = settings.o.size();
    bool has_new = false;
    for (std::size_t k = 0; k < settings.o.size(); ++k) {
        if (settings.o[k].first == r.old_name) at = k;
        if (settings.o[k].first == r.new_name) has_new = true;
    }
    if (at == settings.o.size()) return;
    if (has_new)
        throw SaveError(path + ".settings." + r.old_name + ": renamed to " + r.new_name +
                        ", and the file gives both");
    settings.o[at].first = r.new_name;
}

}  // namespace detail

// Applies the renames of `type` to its settings object; `path` names the
// operator in errors ("RollingMean"). Anything but an object is left to the
// caller's check.
inline void apply_settings_compat(const std::string& type, json::Value& settings, const std::string& path) {
    if (settings.kind != json::Value::Kind::Object) return;
    for (const SettingRename& r : detail::setting_renames())
        if (r.type == type) detail::apply_rename(r, settings, path);
    for (const SettingRename& r : detail::test_renames())
        if (r.type == type) detail::apply_rename(r, settings, path);
}

// Test hooks (tests/test_save_restore.py, js/test/save_restore.test.ts): add
// a row for the length of a test, and remove the rows added.
inline void push_test_rename(const std::string& type, const std::string& old_name,
                             const std::string& new_name) {
    detail::test_renames().push_back({type, old_name, new_name});
}

inline void clear_test_renames() { detail::test_renames().clear(); }

}}  // namespace screamer::serial

#endif  // SCREAMER_SERIAL_SETTINGS_COMPAT_H
