#include <cstdarg>

#include "hook.h"
#include "logger/StringRef.h"
#include "logger/logger.h"

// not in used
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

#pragma push_macro("DEF_DUMMY_FUNC")

#define DEF_DUMMY_FUNC(index) \
    int __dummy_any_function__##index() { return 0; }

extern "C" {
DEF_DUMMY_FUNC(0)
DEF_DUMMY_FUNC(1)
DEF_DUMMY_FUNC(2)
DEF_DUMMY_FUNC(3)
DEF_DUMMY_FUNC(4)
DEF_DUMMY_FUNC(5)
DEF_DUMMY_FUNC(6)
DEF_DUMMY_FUNC(7)
DEF_DUMMY_FUNC(8)
DEF_DUMMY_FUNC(9)
DEF_DUMMY_FUNC(10)
DEF_DUMMY_FUNC(11)
DEF_DUMMY_FUNC(12)
DEF_DUMMY_FUNC(13)
DEF_DUMMY_FUNC(14)
DEF_DUMMY_FUNC(15)
DEF_DUMMY_FUNC(16)
DEF_DUMMY_FUNC(17)
DEF_DUMMY_FUNC(18)
DEF_DUMMY_FUNC(19)
DEF_DUMMY_FUNC(20)
}

#pragma pop_macro("DEF_DUMMY_FUNC")
