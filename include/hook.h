#pragma once

#include <functional>

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
};

struct HookInstaller {
    std::function<bool(const char*)> isTargetLib;
    std::function<bool(const char*)> isTargetSymbol;
    std::function<void*(const OriginalInfo&)> newFuncPtr;
    std::function<void(void)> onSuccess;
};

void install_hook(const HookInstaller& installer);

void uninstall_hook(const HookInstaller& installer);

}  // namespace hook