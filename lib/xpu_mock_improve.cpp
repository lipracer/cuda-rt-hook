#include "hook.h"
#include "logger/StringRef.h"
#include "logger/logger.h"

struct SkipKernelRunHook : public hook::HookInstallerWrap<SkipKernelRunHook> {
    static int any_aunch(...) {
        LOG(INFO) << "any_aunch";
        return 0;
    }

    bool targetLib(const char* name) {
        return !adt::StringRef(name).contain("libcudart.so") &&
               !adt::StringRef(name).contain("libxpurt.so");
    }

    bool targetSym(const char* name) {
        return adt::StringRef(name) == "cudaLaunchKernel" ||
               adt::StringRef(name) == "xpu_launch_async";
    }

    void* newFuncPtr(const hook::OriginalInfo& info) {
        return reinterpret_cast<void*>(&any_aunch);
    }
    void onSuccess() {}

    static SkipKernelRunHook* instance() {
        static auto install_wrap = std::make_shared<SkipKernelRunHook>();
        return install_wrap.get();
    }
};

extern "C" {
HOOK_API bool use_xpu_mock_improve() {
    SkipKernelRunHook::instance()->install();
    return true;
}
}
