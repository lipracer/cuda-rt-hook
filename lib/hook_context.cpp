#include "hook_context.h"

namespace hook {
HookRuntimeContext& HookRuntimeContext::instance() {
    static HookRuntimeContext __instance;
    return __instance;
}

void HookRuntimeContext::Statistic::increase() { ++counter_; }
void HookRuntimeContext::Statistic::increase_cost(size_t time) {
    cost_ += time;
}

HookRuntimeContext::~HookRuntimeContext() {
    MLOG(PROFILE, INFO) << "dump context map:\n" << summary_string();
}

void HookRuntimeContext::registCompilerGen(WrapGeneratorBase* gen) {
    compiler_gen_.emplace_back(gen);
}

size_t HookRuntimeContext::getCallCount(const std::string& libName,
                                        const std::string& symName) {
    auto iter = std::find_if(
        func_infos_.begin(), func_infos_.end(), [&](const auto& it) {
            return it.first.lib_name == libName && it.first.sym_name == symName;
        });
    if (iter == func_infos_.end()) {
        return -1;
    }
    return iter->second.count();
}

void HookRuntimeContext::clear() {
    std::vector<std::pair<StringPair, StatisticPair>> func_infos{};
    func_infos_.swap(func_infos);

    std::vector<std::unique_ptr<WrapGeneratorBase>> compiler_gen{};
    compiler_gen_.swap(compiler_gen);

    std::vector<std::pair<std::string, std::vector<std::string>>>
        last_index_map{};
    last_index_map_.swap(last_index_map);

    std::unordered_map<const std::type_info*, std::vector<size_t>, TypeInfoHash>
        id_map{};
    id_map_.swap(id_map);
}

std::string HookRuntimeContext::summary_string() {
    std::stringstream ss;
    ss << "statistic info:\n";
    std::map<std::string, std::tuple<size_t, size_t>> total_statistic_map;
    std::map<std::string, std::stringstream> statistic_map;
    for (const auto& func_info : func_infos_) {
        auto total_iter = total_statistic_map.find(func_info.first.sym_name);
        if (total_iter != total_statistic_map.end()) {
            std::get<0>(total_iter->second) += func_info.second.count();
            std::get<1>(total_iter->second) += func_info.second.cost();
        } else {
            total_statistic_map.insert(std::make_pair(
                func_info.first.sym_name,
                std::tuple<size_t, size_t>(func_info.second.count(),
                                           func_info.second.cost())));
        }
        auto iter =
            statistic_map
                .insert(std::make_pair(shortLibName(func_info.first.lib_name),
                                       std::stringstream()))
                .first;
        iter->second << func_info.first.sym_name
                     << alignWith(func_info.first.sym_name) << func_info.second
                     << "\n";
    }

    for (const auto& it : total_statistic_map) {
        ss << "symbol name:" << it.first << " count:" << std::get<0>(it.second)
           << " cost:" << std::get<1>(it.second) << "\n";
    }

    for (const auto& it : statistic_map) {
        ss << "library name:" << it.first << "\n" << it.second.str() << "\n";
    }
    return std::move(ss.str());
}

std::ostream& operator<<(std::ostream& os,
                         const HookRuntimeContext::Statistic& s) {
    os << "total call:" << s.counter_ << alignWith(s.counter_, 5) << " times"
       << " total cost:" << s.cost_ << "ns(" << s.cost_ / 1e9 << "s)";
    return os;
}

}  // namespace hook