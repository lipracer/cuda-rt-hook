#pragma once

#include <functional>
#include <iosfwd>
#include <memory>

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

template <typename DerivedT>
struct HookInstallerWrap
    : public std::enable_shared_from_this<HookInstallerWrap<DerivedT>> {
    // NB: c++ std::shared_ptr and enable_shared_from_this
    // shared_from_this can't call in ctor
    void install() { install_hook(buildInstaller()); }

    ~HookInstallerWrap() { uninstall_hook(orgInfo); }

    bool targetLib(const char* name) {
        libName = name;
        isTarget = static_cast<DerivedT*>(this)->targetLib(name);
        return isTarget;
    }
    bool targetSym(const char* name) {
        symName = name;
        isSymbol = static_cast<DerivedT*>(this)->targetSym(name);
        return isSymbol;
    }

    void* newFuncPtr(const hook::OriginalInfo& info) {
        orgInfo = info;
        return static_cast<DerivedT*>(this)->newFuncPtr(info);
    }

    void onSuccess() { 
        static_cast<DerivedT*>(this)->onSuccess(); }

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
    OriginalInfo orgInfo;
};

}  // namespace hook