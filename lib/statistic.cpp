#include <map>
#include <sstream>

#include "hook.h"

namespace hook {

namespace {
std::string shortLibName(const std::string& full_lib_name) {
    auto pos = full_lib_name.find_last_of('/');
    if (pos != std::string::npos) {
        return full_lib_name.substr(pos + 1);
    }
    return full_lib_name;
}
}  // namespace

void HookRuntimeContext::dump() {
    std::stringstream ss;
    ss << "statistic info:\n";
    std::map<std::string, std::stringstream> statistic_map;
    auto alignWith = [](const std::string& str) {
        size_t size = str.size() >= 32 ? 1 : 32 - str.size();
        return std::string(size, ' ');
    };
    for (const auto& func_info : func_infos_) {
        auto iter =
            statistic_map
                .insert(std::make_pair(shortLibName(func_info.first.lib_name),
                                       std::stringstream()))
                .first;
        iter->second << func_info.first.sym_name
                     << alignWith(func_info.first.sym_name) << func_info.second
                     << "\n";
    }

    for (const auto& it : statistic_map) {
        ss << "library name:" << it.first << "\n" << it.second.str();
    }
    LOG(WARN) << "dump context map:\n" << ss.str();
}

std::ostream& operator<<(std::ostream& os,
                         const HookRuntimeContext::Statistic& s) {
    os << "total call:" << s.counter_ << " times";
    return os;
}

}  // namespace hook
