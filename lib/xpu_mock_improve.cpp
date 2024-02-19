#include <cstdarg>

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

#define __internal_printf()                       \
    va_list args;                                 \
    constexpr size_t kMax = 4096;                 \
    char buf[kMax] = {0};                         \
    va_start(args, fmt);                          \
    vsprintf(buf, fmt, args);                     \
    va_end(args);                                 \
    XpuRuntimePrintfHook::instance()->ss_ << buf; \
    printf("%s", buf);

struct XpuRuntimePrintfHook
    : public hook::HookInstallerWrap<XpuRuntimePrintfHook> {
    static int builtin_printf_chk(int flag, const char* fmt, ...) {
        __internal_printf();
        return 0;
    }

    static int builtin_printf(const char* fmt, ...) {
        __internal_printf();
        return 0;
    }

    static int builtin_fprintf(void* stdcout, const char* fmt, ...) {
        __internal_printf();
        return 0;
    }

    bool targetLib(const char* name) {
        return adt::StringRef(name).contain("libcuda.so");
    }

    bool targetSym(const char* name) {
        return adt::StringRef(name) == "__printf_chk" ||
               adt::StringRef(name) == "printf" ||
               adt::StringRef(name) == "fprintf" ||
               adt::StringRef(name) == "__fprintf" ||
               adt::StringRef(name) == "vfprintf";
    }

    void* newFuncPtr(const hook::OriginalInfo& info) {
        if (adt::StringRef("__printf_chk") == curSymName()) {
            return reinterpret_cast<void*>(&builtin_printf_chk);
        } else if (adt::StringRef("printf") == curSymName()) {
            return reinterpret_cast<void*>(&builtin_printf);
        } else if (adt::StringRef("fprintf") == curSymName() ||
                   adt::StringRef("__fprintf") == curSymName() ||
                   adt::StringRef("vfprintf") == curSymName()) {
            return reinterpret_cast<void*>(&builtin_fprintf);
        }
        return nullptr;
    }
    void onSuccess() {}

    static XpuRuntimePrintfHook* instance() {
        static auto install_wrap = std::make_shared<XpuRuntimePrintfHook>();
        return install_wrap.get();
    }

    void start_capture() { start_capture_ = true; }
    std::string end_capture() {
        auto str = ss_.str();
        ss_.str("");
        return str;
    }
    bool start_capture_{false};
    std::stringstream ss_;
    void* py_obj_{nullptr};
};

extern "C" {

void dh_start_capture_rt_print() {
    XpuRuntimePrintfHook::instance()->start_capture();
}

std::string dh_end_capture_rt_print() {
    return XpuRuntimePrintfHook::instance()->end_capture();
}

HOOK_API bool use_xpu_mock_improve() {
    // SkipKernelRunHook::instance()->install();
    XpuRuntimePrintfHook::instance()->install();
    return true;
}
}
