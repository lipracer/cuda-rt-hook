#pragma once

#include <algorithm>
#include <functional>
#include <iosfwd>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "GlobalVarMgr.h"
#include "logger/logger.h"

namespace hook {
/**
 * @brief install all hook function to replace cuda runtime api
 */
void install_hook();

struct OriginalInfo {
    const char* libName = nullptr;
    const void* basePtr = nullptr;
    void* relaPtr = nullptr;
    void* oldFuncPtr = nullptr;
    void** pltTablePtr = nullptr;
    OriginalInfo() = default;
    OriginalInfo(const OriginalInfo& info) { *this = info; }
    OriginalInfo& operator=(const OriginalInfo& info) {
        libName = info.libName;
        basePtr = info.basePtr;
        relaPtr = info.relaPtr;
        oldFuncPtr = info.oldFuncPtr;
        pltTablePtr = info.pltTablePtr;
        return *this;
    }
};

std::ostream& operator<<(std::ostream& os, const OriginalInfo& info);

struct HookInstaller {
    std::function<bool(const char*)> isTargetLib;
    std::function<bool(const char*)> isTargetSymbol;
    std::function<void*(const OriginalInfo&)> newFuncPtr;
    std::function<void(void)> onSuccess;
};

void install_hook(const HookInstaller& installer);

void uninstall_hook(const OriginalInfo& info);

template <typename DerivedT, typename T = void>
struct MemberDetector : std::false_type {
    static bool targetSym(DerivedT* self, const char* name) {
        return static_cast<DerivedT*>(self)->targetSym(name);
    }
    static void* newFuncPtr(DerivedT* self, const hook::OriginalInfo& info) {
        return static_cast<DerivedT*>(self)->newFuncPtr(info);
    }
};

class HookRuntimeContext {
   public:
    struct StringPair {
        StringPair(const std::string& lib, const std::string& sym)
            : lib_name(lib), sym_name(sym) {}
        std::string lib_name;
        std::string sym_name;

        bool operator==(const StringPair& other) const {
            return lib_name == other.lib_name && sym_name == other.sym_name;
        }
        bool operator!=(const StringPair& other) const {
            return !(*this == other);
        }

        bool operator<(const StringPair& other) const {
            return (lib_name + sym_name) < (other.lib_name + other.sym_name);
        }
    };
    struct SPhash {
        size_t operator()(const StringPair& names) const {
            return std::hash<std::string>()(names.lib_name + names.sym_name);
        }
    };

    using map_type = std::map<StringPair, std::pair<void*, void*>>;
    //    using map_type = std::unordered_map<StringPair, std::pair<void*,
    //    void*>, SPhash>;

    HookRuntimeContext() = default;
    static HookRuntimeContext& instance() {
        static HookRuntimeContext __instance;
        return __instance;
    }

    template <typename... Args>
    auto insert(Args&&... args) {
        return map_.insert(std::forward<Args>(args)...);
    }
    map_type& map() { return map_; }

    map_type::const_iterator setCurrentState(size_t UniqueId) {
        cur_iter_ = map_.begin();
        std::advance(cur_iter_, UniqueId);
        return cur_iter_;
    }

    const std::string& curLibName() { return cur_iter_->first.lib_name; }
    const std::string& curSymName() { return cur_iter_->first.sym_name; }

    void dump() {
        LOG(WARN) << "dum context map:";
        for (auto iter = map_.begin(); iter != map_.end(); ++iter) {
            LOG(WARN) << "offset:" << std::distance(map_.begin(), iter)
                      << " lib name:" << iter->first.lib_name
                      << " sym name:" << iter->first.sym_name;
        }
    }

   private:
    map_type map_;
    map_type::const_iterator cur_iter_;
};

struct StringLiteral {
    template <size_t N>
    constexpr StringLiteral(const char (&str)[N]) : str_(str), size_(N) {}
    constexpr char operator[](size_t index) const { return str_[index]; }
    constexpr size_t size() const { return size_; }

    operator const char*() const { return str_; }

   private:
    const char* str_;
    const size_t size_;
};

template <size_t N>
constexpr size_t hash(const char (&str)[N]) {
    size_t v = 0;
    for (size_t i = 3; i < N; ++i) {
        size_t vv = str[i];
        v |= (vv << (sizeof(size_t) - i - 3 - 1));
    }
    return v;
}

template <size_t UniqueId, typename R, typename... Args>
struct MapedFunc {
    static R func(Args... args) {
        auto iter = HookRuntimeContext::instance().setCurrentState(UniqueId);
        return reinterpret_cast<R (*)(Args...)>(iter->second.second)(args...);
    }
};

template <size_t UniqueId, typename... Args>
struct MapedFunc<UniqueId, void, Args...> {
    static void func(Args... args) {
        auto iter = HookRuntimeContext::instance().setCurrentState(UniqueId);
        reinterpret_cast<void (*)(Args...)>(iter->second.second)(args...);
    }
};

template <size_t N, typename R, typename... Args>
constexpr void* GetMapedFuncImpl(
    size_t UniqueId, R (*new_func)(Args...),
    typename std::enable_if_t<N == 256, void*>* = nullptr) {
    return nullptr;
}

template <size_t N, typename R, typename... Args>
constexpr void* GetMapedFuncImpl(size_t UniqueId, R (*new_func)(Args...),
                                 typename std::enable_if_t <
                                     N<256, void*>* = nullptr) {
    if (N == UniqueId) {
        return reinterpret_cast<void*>(&MapedFunc<N, R, Args...>::func);
    }
    return GetMapedFuncImpl<N + 1>(UniqueId, new_func);
}

template <typename R, typename... Args>
constexpr void* GetMapedFunc(size_t UniqueId, R (*new_func)(Args...)) {
    return GetMapedFuncImpl<0>(UniqueId, new_func);
}

struct HookFeature {
    template <size_t N, typename R, typename... Args, typename T>
    constexpr HookFeature(const char (&sym_name)[N], R (*new_func)(Args...),
                          T** old_func)
        : symName(sym_name),
          newFunc(reinterpret_cast<void*>(new_func)),
          oldFunc(reinterpret_cast<void**>(old_func)) {
        findUniqueFunc = [=](size_t uniqueId) {
            return GetMapedFunc(uniqueId, new_func);
        };
    }

    void* getNewFunc(const char* libName = nullptr) {
        if (libName) {
            HookRuntimeContext::StringPair pair_str(libName, symName);
            auto iter =
                HookRuntimeContext::instance()
                    .insert(std::make_pair(
                        pair_str, std::pair<void*, void*>(nullptr, nullptr)))
                    .first;
            auto uniqueId = std::distance(
                HookRuntimeContext::instance().map().begin(), iter);
            auto wrapFunc = findUniqueFunc(uniqueId);
            iter->second.first = wrapFunc;
            iter->second.second = newFunc;
            return wrapFunc;
        }
        return newFunc;
    }

    const char* symName;
    void* newFunc;
    void** oldFunc;
    std::function<void*(size_t)> findUniqueFunc;
};

// TODO(lipracer): mix predefine mode and dynamic mode
template <typename DerivedT>
struct MemberDetector<DerivedT,
                      std::void_t<decltype(std::declval<DerivedT>().symbols)>>
    : std::true_type {
    static bool targetSym(DerivedT* self, const char* name) {
        for (auto& sym : static_cast<DerivedT*>(self)->symbols) {
            if (!strcmp(name, sym.symName)) {
                return true;
            }
        }
        return false;
    }
    static void* newFuncPtr(DerivedT* self, const hook::OriginalInfo& info) {
        auto iter = std::find_if(
            std::begin(static_cast<DerivedT*>(self)->symbols),
            std::end(static_cast<DerivedT*>(self)->symbols), [self](auto& sym) {
                return !strcmp(static_cast<DerivedT*>(self)->curSymName(),
                               sym.symName);
            });
        // TODO: if std::get<2>(*iter) is a pointer and it's point to
        // std::get<1>(*iter) then there will return nullptr
        *iter->oldFunc = info.oldFuncPtr;
        return iter->getNewFunc(info.libName);
    }
};

template <typename DerivedT>
struct PreDefineInterface {
    bool targetSym(const char* name) {
        return MemberDetector<DerivedT>::targetSym(static_cast<DerivedT*>(this),
                                                   name);
    }

    void* newFuncPtr(const hook::OriginalInfo& info) {
        return MemberDetector<DerivedT>::newFuncPtr(
            static_cast<DerivedT*>(this), info);
    }
};

template <typename DerivedT>
struct HookInstallerWrap
    : public std::enable_shared_from_this<HookInstallerWrap<DerivedT>>,
      PreDefineInterface<DerivedT> {
    // NB: c++ std::shared_ptr and enable_shared_from_this
    // shared_from_this can't call in ctor
    void install() { install_hook(buildInstaller()); }

    ~HookInstallerWrap() = default;

    bool targetLib(const char* name) {
        libName = name;
        isTarget = static_cast<DerivedT*>(this)->targetLib(name);
        return isTarget;
    }
    bool targetSym(const char* name) {
        symName = name;
        return static_cast<PreDefineInterface<DerivedT>*>(this)->targetSym(
            name);
    }

    void* newFuncPtr(const hook::OriginalInfo& info) {
        orgInfos.emplace_back(
            std::unique_ptr<OriginalInfo, std::function<void(OriginalInfo*)>>(
                new OriginalInfo(info), [](OriginalInfo* info) {
                    uninstall_hook(*info);
                    delete info;
                }));
        return static_cast<PreDefineInterface<DerivedT>*>(this)->newFuncPtr(
            info);
    }

    void onSuccess() { static_cast<DerivedT*>(this)->onSuccess(); }

    const char* curSymName() const { return symName; }
    const char* curLibName() const { return libName; }

    HookInstaller buildInstaller() {
        return HookInstaller{
            .isTargetLib =
                [self = this->shared_from_this()](const char* name) {
                    return self->targetLib(name);
                },
            .isTargetSymbol =
                [self = this->shared_from_this()](const char* name) {
                    return self->targetSym(name);
                },
            .newFuncPtr =
                [self =
                     this->shared_from_this()](const hook::OriginalInfo& info) {
                    return self->newFuncPtr(info);
                },
            .onSuccess =
                [self = this->shared_from_this()]() { self->onSuccess(); }};
    }

    bool isTarget{false};
    bool isSymbol{false};
    const char* libName{nullptr};
    const char* symName{nullptr};
    std::vector<
        std::unique_ptr<OriginalInfo, std::function<void(OriginalInfo*)>>>
        orgInfos;
};

}  // namespace hook
