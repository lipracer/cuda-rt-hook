#pragma once

#include <algorithm>
#include <functional>
#include <iosfwd>
#include <memory>
#include <vector>
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

// TODO(lipracer): mix predefine mode and dynamic mode
template <typename DerivedT>
struct MemberDetector<DerivedT,
                      std::void_t<decltype(std::declval<DerivedT>().symbols)>>
    : std::true_type {
    static bool targetSym(DerivedT* self, const char* name) {
        for (auto& sym : static_cast<DerivedT*>(self)->symbols) {
            if (!strcmp(name, std::get<0>(sym))) {
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
                               std::get<0>(sym));
            });
        *std::get<2>(*iter) = info.oldFuncPtr;
        return std::get<1>(*iter);
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