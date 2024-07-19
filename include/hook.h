#pragma once

#include <dlfcn.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <iosfwd>
#include <map>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "GlobalVarMgr.h"
#include "backtrace.h"
#include "hook_context.h"
#include "logger/logger.h"
#include "macro.h"
#include "support.h"

namespace hook {
/**
 * @brief install all hook function to replace cuda runtime api
 */
void install_hook();

struct OriginalInfo {
    const char* libName = nullptr;
    const void* basePtr = nullptr;
    /// @brief elf header load pointer at runtime, symbol address = baseHeadPtr
    /// + offset(in elf)
    const void* baseHeadPtr = nullptr;
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
        baseHeadPtr = info.baseHeadPtr;
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

template <size_t N>
constexpr size_t hash(const char (&str)[N]) {
    size_t v = 0;
    for (size_t i = 3; i < N; ++i) {
        size_t vv = str[i];
        v |= (vv << (sizeof(size_t) - i - 3 - 1));
    }
    return v;
}

template <typename T, typename = void>
struct DetectedToString : public std::false_type {
    static void apply(T t, std::ostream& os) {
        os << "unknown"
           << "(" << __support__demangle(typeid(t).name()) << ")";
    }
};

template <typename T>
struct DetectedToString<T,
                        std::void_t<decltype(operator<<(
                            std::declval<std::ostream>(), std::declval<T>()))>>
    : public std::true_type {
    static void apply(T t, std::ostream& os) {
        os << t << "(" << __support__demangle(typeid(t).name()) << ")";
    }
};

template <int Idx>
void args_to_string_impl(std::ostream& os) {}

template <int Idx, typename Args>
void args_to_string_impl(Args args, std::ostream& os) {
    os << "arg" << Idx << ":";
    DetectedToString<Args>::apply(args, os);
}

template <int Idx, typename First, typename... Args>
typename std::enable_if_t<(sizeof...(Args) > 0), void> args_to_string_impl(
    First f, Args... args, std::ostream& os) {
    os << "arg" << Idx << ":";
    DetectedToString<First>::apply(f, os);
    os << " ";
    args_to_string_impl<Idx + 1, Args...>(args..., os);
}

template <typename... Args>
std::string args_to_string(Args... args) {
    std::stringstream ss;
    args_to_string_impl<0, Args...>(args..., ss);
    return ss.str();
}

#define IF_ENABLE_LOG_TRACE_AND_ARGS(func)                                   \
    do {                                                                     \
        int ctrl = enable_log_backtrace((func));                             \
        if (ctrl) {                                                          \
            if (ctrl & 0b10) {                                               \
                auto parser_func =                                           \
                    HookRuntimeContext::instance().lookUpArgsParser((func)); \
                MLOG(TRACE, WARN)                                            \
                    << func << ": "                                          \
                    << (parser_func                                          \
                            ? reinterpret_cast<std::string (*)(Args...)>(    \
                                  parser_func)(args...)                      \
                            : args_to_string(args...));                      \
            }                                                                \
            if (ctrl & 0b01) {                                               \
                trace::CallFrames callFrames;                                \
                callFrames.CollectNative();                                  \
                callFrames.CollectPython();                                  \
                MLOG(TRACE, WARN) << func << " with frame:\n" << callFrames; \
            }                                                                \
        }                                                                    \
    } while (0)

template <typename StrT, size_t UniqueId, typename R, typename... Args>
struct MapedFunc {
    static R func(Args... args) {
        auto id = HookRuntimeContext::instance().getCUniqueId<StrT>(UniqueId);
        auto iter = wrapCurrentIter(id);
        IF_ENABLE_LOG_TRACE_AND_ARGS(
            HookRuntimeContext::instance().curSymName().c_str());
        iter.reset();
        return reinterpret_cast<R (*)(Args...)>(iter->second.second)(args...);
    }
};

template <typename StrT, size_t UniqueId, typename... Args>
struct MapedFunc<StrT, UniqueId, void, Args...> {
    static void func(Args... args) {
        auto id = HookRuntimeContext::instance().getCUniqueId<StrT>(UniqueId);
        auto iter = wrapCurrentIter(id);
        IF_ENABLE_LOG_TRACE_AND_ARGS(
            HookRuntimeContext::instance().curSymName().c_str());
        iter.reset();
        reinterpret_cast<void (*)(Args...)>(iter->second.second)(args...);
    }
};

template <size_t UniqueId, typename R, typename... Args>
struct MapedFunc<void, UniqueId, R, Args...> {
    static R func(Args... args) {
        auto iter = wrapCurrentIter<UniqueId>();
        IF_ENABLE_LOG_TRACE_AND_ARGS(
            HookRuntimeContext::instance().curSymName().c_str());
        iter.reset();
        return reinterpret_cast<R (*)(Args...)>(iter->second.second)(args...);
    }
};

template <size_t UniqueId, typename... Args>
struct MapedFunc<void, UniqueId, void, Args...> {
    static void func(Args... args) {
        auto iter = wrapCurrentIter<UniqueId>();
        IF_ENABLE_LOG_TRACE_AND_ARGS(
            HookRuntimeContext::instance().curSymName().c_str());
        iter.reset();
        reinterpret_cast<void (*)(Args...)>(iter->second.second)(args...);
    }
};

template <size_t N, typename R, typename... Args>
constexpr void* GetMapedFuncImpl(
    size_t UniqueId,
    typename std::enable_if_t<N == kMaxEachSignatureFuncSize, void*>* =
        nullptr) {
    return nullptr;
}

template <size_t N, typename R, typename... Args>
constexpr void* GetMapedFuncImpl(
    size_t UniqueId, typename std::enable_if_t <
                         N<kMaxEachSignatureFuncSize, void*>* = nullptr) {
    if (N == UniqueId) {
        return reinterpret_cast<void*>(&MapedFunc<void, N, R, Args...>::func);
    }
    return GetMapedFuncImpl<N + 1, R, Args...>(UniqueId);
}

template <typename R, typename... Args>
constexpr void* GetMapedFunc(size_t UniqueId) {
    return GetMapedFuncImpl<0, R, Args...>(UniqueId);
}

// TODO
// use func name as template arguments, then generate functions map:
// name0: {func0, func1, func2...}
// name1: {func0, func1, func2...}
// ...
// reduce the template functions count and then reduce complitation time
template <typename StrT, typename R, typename... Args>
class CompilerWrapGenerator : public WrapGeneratorBase {
   public:
    typedef R (*type)(Args...);
    // forbidden construct on stack
    CompilerWrapGenerator() {
        gen();
        HookRuntimeContext::instance().registCompilerGen(this);
    }

    template <size_t... Idxs>
    void gen_impl(std::index_sequence<Idxs...>) {
        funcs_ = {reinterpret_cast<void*>(
            &MapedFunc<StrT, Idxs, R, Args...>::func)...};
    }
    void gen() { gen_impl(std::make_index_sequence<kMaxLibrarySize>()); }

    void* getFunction(size_t uniqueId,
                      const char* libName = nullptr) const override {
        size_t offset =
            HookRuntimeContext::instance().caclOffset<StrT>(libName, uniqueId);
        CHECK_LT(offset, kMaxLibrarySize);
        return funcs_[offset];
    }

    std::string symName() const override { return std::string(StrT()); }

   private:
    std::vector<void*> funcs_;
};

template <typename R, typename... Args>
class RuntimeWrapGenerator : public WrapGeneratorBase {
   public:
    void* getFunction(size_t index, const char* = nullptr) const override {
        return GetMapedFunc<R, Args...>(index);
    }

    std::string symName() const override { return ""; }
};

struct HookFeatureBase {
    template <size_t N, typename R, typename... Args, typename T>
    constexpr HookFeatureBase(const char (&sym_name)[N], R (*new_func)(Args...),
                              T** old_func,
                              const std::function<bool(void)>& filter = {})
        : symName(sym_name),
          newFunc(reinterpret_cast<void*>(new_func)),
          oldFunc(reinterpret_cast<void**>(old_func)),
          filter_(filter) {}

    template <typename R, typename... Args, typename T>
    HookFeatureBase(const std::string& sym_name, R (*new_func)(Args...),
                    T** old_func, const std::function<bool(void)>& filter = {})
        : symName(sym_name),
          newFunc(reinterpret_cast<void*>(new_func)),
          oldFunc(reinterpret_cast<void**>(old_func)),
          filter_(filter) {}

    void* getNewFunc(const char* libName = nullptr) { return newFunc; }

    std::string symName;
    void* newFunc;
    void** oldFunc;
    std::function<bool(void)> filter_;
    std::function<void(const OriginalInfo&)> getNewCallback_;
};

using WrapFuncGenerator = std::function<void*(const char*, const char*, void*)>;

template <typename GenT>
struct GeneratorFunctor {
    static void* func(const char* libName, const char* symName, void* newFunc) {
        HookRuntimeContext::StringPair pair_str(libName, symName);
        auto iter = HookRuntimeContext::instance().insert(
            std::make_pair(pair_str, HookRuntimeContext::StatisticPair()));
        auto uniqueId = HookRuntimeContext::instance().getUniqueId(iter);
        auto wrapGen = new GenT();
        auto wrapFunc = wrapGen->getFunction(uniqueId, libName);
        iter->second.first = wrapFunc;
        iter->second.second = newFunc;
        return wrapFunc;
    }
};

struct SimpleGenerator {
    template <typename R, typename... Args>
    struct type {
        WrapFuncGenerator getGenerator() {
            return &GeneratorFunctor<RuntimeWrapGenerator<R, Args...>>::func;
        }
    };
};

struct FastGenerator {
    template <typename StrT, typename R, typename... Args>
    struct type {
        WrapFuncGenerator getGenerator() {
            return &GeneratorFunctor<
                CompilerWrapGenerator<StrT, R, Args...>>::func;
        }
    };
};

template <typename GeneratorT>
struct __HookFeature : public HookFeatureBase {
    template <size_t N, typename R, typename... Args, typename T>
    constexpr __HookFeature(const char (&sym_name)[N], R (*new_func)(Args...),
                            T** old_func,
                            const std::function<bool(void)>& filter = {})
        : HookFeatureBase(sym_name, new_func, old_func, filter) {
        typename GeneratorT::template type<R, Args...> generator;
        newFuncGenerator = generator.getGenerator();
    }

    template <typename StrT, typename R, typename... Args, typename T>
    __HookFeature(StrT strT, R (*new_func)(Args...), T** old_func,
                  const std::function<bool(void)>& filter = {})
        : HookFeatureBase(std::string(strT), new_func, old_func, filter) {
        typename GeneratorT::template type<StrT, R, Args...> generator;
        newFuncGenerator = generator.getGenerator();
    }

    void* getNewFunc(const char* libName = nullptr) {
        if (!libName) {
            return newFunc;
        }
        return newFuncGenerator(libName, symName.c_str(), newFunc);
    }

    __HookFeature& setGetNewCallback(
        const std::function<void(const OriginalInfo&)>& getNewCallback) {
        getNewCallback_ = getNewCallback;
        return *this;
    }

    template <typename... Args>
    __HookFeature& setArgsParser(std::string (*parser)(Args...)) {
        HookRuntimeContext::instance().argsParserMap().emplace(
            symName, reinterpret_cast<void*>(parser));
        return *this;
    }

    std::function<void*(size_t)> findUniqueFunc;
    WrapFuncGenerator newFuncGenerator;
};

using HookFeature = __HookFeature<SimpleGenerator>;
using FHookFeature = __HookFeature<FastGenerator>;

template <typename DerivedT>
struct MemberDetector<DerivedT,
                      std::void_t<decltype(std::declval<DerivedT>().symbols)>>
    : std::true_type {
    static bool targetSym(DerivedT* self, const char* name) {
        for (auto& sym : static_cast<DerivedT*>(self)->symbols) {
            if (name == sym.symName) {
                if (sym.filter_ && !sym.filter_()) {
                    return false;
                }
                return true;
            }
        }
        return false;
    }
    static void* newFuncPtr(DerivedT* self, const hook::OriginalInfo& info) {
        auto iter = std::find_if(
            std::begin(static_cast<DerivedT*>(self)->symbols),
            std::end(static_cast<DerivedT*>(self)->symbols), [self](auto& sym) {
                return static_cast<DerivedT*>(self)->curSymName() ==
                       sym.symName;
            });
        // TODO: if std::get<2>(*iter) is a pointer and it's point to
        // std::get<1>(*iter) then there will return nullptr
        *iter->oldFunc = info.oldFuncPtr;
        if (iter->getNewCallback_) {
            iter->getNewCallback_(info);
        }
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
    void install() {
        MLOG(HOOK, INFO) << "install hooker:" << typeid(DerivedT).name();
        install_hook(buildInstaller());
    }

    ~HookInstallerWrap() {
        for (auto& handle : handle_map_) {
            MLOG(HOOK, INFO) << "close lib:" << handle.first;
            dlclose(handle.second);
        }
        MLOG(HOOK, INFO) << "HookInstallerWrap<" << typeid(DerivedT).name()
                         << "> complete!";
    }

    bool targetLib(const char* name) {
        if (!src_lib_name_.empty() && strstr(name, src_lib_name_.c_str())) {
            full_src_lib_name_ = name;
        }
        libName = name;
        isTarget = static_cast<DerivedT*>(this)->targetLib(name);
        MLOG(HOOK, INFO) << name << " isTarget:" << isTarget;
        return isTarget;
    }

    bool targetSym(const char* name) {
        symName = name;
        bool isSymbol =
            static_cast<PreDefineInterface<DerivedT>*>(this)->targetSym(name);
        return isSymbol;
    }

    void* newFuncPtr(const hook::OriginalInfo& info) {
        orgInfos.emplace_back(
            std::unique_ptr<OriginalInfo, std::function<void(OriginalInfo*)>>(
                new OriginalInfo(info), [](OriginalInfo* info) {
                    uninstall_hook(*info);
                    delete info;
                }));
        MLOG(HOOK, INFO) << " replace symbol:" << curSymName() << " in "
                         << curLibName();
        return static_cast<PreDefineInterface<DerivedT>*>(this)->newFuncPtr(
            info);
    }

    void onSuccess() {
        increase_lib_ref();
        static_cast<DerivedT*>(this)->onSuccess();
    }

    void increase_lib_ref() {
        if (full_src_lib_name_.empty()) {
            return;
        }
        auto lib_handle = dlopen(full_src_lib_name_.c_str(), RTLD_LAZY);
        if (!lib_handle) {
            MLOG(HOOK, INFO) << "can't open lib:" << full_src_lib_name_;
        } else {
            handle_map_.insert(
                std::pair<std::string, void*>(full_src_lib_name_, lib_handle));
        }
    }

    void set_src_lib(const std::string& lib) { src_lib_name_ = lib; }

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
    std::string src_lib_name_;
    std::string full_src_lib_name_;
    std::vector<
        std::unique_ptr<OriginalInfo, std::function<void(OriginalInfo*)>>>
        orgInfos;
    std::unordered_map<std::string, void*> handle_map_;
};

}  // namespace hook
