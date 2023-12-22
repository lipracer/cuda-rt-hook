#include "GlobalVarMgr.h"

#include <map>
#include <mutex>

#include "logger/logger.h"

namespace hook {

class GlobalValRegistry {
   public:
    struct FuncDesc {
        registry_func rfunc;
        destroy_func dfunc;
        void* obj;
    };
    static GlobalValRegistry& instance() {
        static GlobalValRegistry registry_func_list;
        return registry_func_list;
    }
    auto begin() { return map_.begin(); }
    auto end() { return map_.end(); }

    void insert(int priority, const FuncDesc& func) {
        std::lock_guard<std::mutex> lg(map_mtx_);
        map_.insert(std::make_pair(priority, func));
    }

    ~GlobalValRegistry() {
        for (auto iter = map_.rbegin(); iter != map_.rend(); ++iter) {
            (iter->second.dfunc)(iter->second.obj);
        }
    }

   private:
    std::multimap<int, FuncDesc, std::greater<int>> map_;
    std::mutex map_mtx_;
};

void init_all_global_variables() {
    LOG(INFO) << "call:" << __func__;
    for (auto& func : GlobalValRegistry::instance()) {
        auto obj = func.second.rfunc();
        func.second.obj = obj;
    }
}

void register_global_variable(int priority, const registry_func& rfunc,
                              const destroy_func& dfunc) {
    LOG(INFO) << "call:" << __func__;
    GlobalValRegistry::instance().insert(
        priority, GlobalValRegistry::FuncDesc{.rfunc = rfunc, .dfunc = dfunc});
}

}  // namespace hook
