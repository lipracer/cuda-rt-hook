#include "cuda_mock.h"

#include <dlfcn.h>
#include <csetjmp>
#include <string.h>

#include "backtrace.h"
#include "cuda_op_tracer.h"
#include "hook.h"
#include "logger/logger.h"

std::jmp_buf log_jump_buffer = {{}};

#ifdef __cplusplus

extern "C" {

void dh_initialize() {
    LOG(INFO) << "initialize";
    hook::HookInstaller hookInstaller = trace::getHookInstaller();
    hook::install_hook(hookInstaller);
    // hook::install_hook();
}

static void* oldFuncAddr = nullptr;

void log_router() {
    LOG(INFO) << __func__ << ":" << oldFuncAddr;
    // sometime will crash
    // trace::BackTraceCollection::CallStackInfo tracer({});
    // tracer.snapshot();
    // LOG(WARN) << tracer;
    longjmp(log_jump_buffer, 1);
}

void __any_mock_func__() {
// Conditional code for x86_64 architecture
#if defined(__x86_64__)
    asm volatile("pop %rbp");
    asm volatile("push %rax");
    asm volatile("push %rdi");
    asm volatile("push %rdi");
    if (!setjmp(log_jump_buffer)) {
        log_router();
    }
    asm volatile("pop %rdi");
    asm volatile("pop %rdi");
    asm volatile("pop %rax");
    asm volatile("add    $0x8,%rsp");
    asm volatile("jmp *%0" : : "r"(oldFuncAddr));

// Conditional code for aarch64 architecture
#else
    // asm volatile("push {r0}");
    // if (!setjmp(log_jump_buffer)) {
    //     log_router();
    // }
    // asm volatile("pop {r0}");
#endif
}

int builtin_printf(int flag, const char* fmt, va_list argp) {
    constexpr size_t kMax = 1024;
    char buf[kMax] = {"myprintf "};
    snprintf(buf + strlen(buf), kMax - strlen(buf), fmt, argp);
    LOG(INFO) << buf;
    return 0;
}

static std::unordered_map<std::string, void*> gBuiltinFuncs = {
    {"__printf_chk", reinterpret_cast<void*>(&builtin_printf)},
};

void dh_internal_install_hook(const char* srcLib, const char* targetLib,
                              const char* symbolName, const char* hookerLibPath,
                              const char* hookerSymbolName) {
    LOG(INFO) << "initialize srcLib:" << srcLib << " targetLib:" << targetLib
              << " symbolName:" << symbolName;
    auto iter = gBuiltinFuncs.find(symbolName);
    auto hookerAddr = iter == gBuiltinFuncs.end()
                          ? reinterpret_cast<void*>(&__any_mock_func__)
                          : iter->second;
    if (hookerLibPath) {
        auto handle = dlopen(hookerLibPath, RTLD_LAZY);
        CHECK(handle, "can't not dlopen {}", hookerLibPath);
        hookerAddr =
            dlsym(handle, hookerSymbolName ? hookerSymbolName : symbolName);
    }
    CHECK(hookerAddr, "hookerAddr can't be empty!");
    hook::HookInstaller hookInstaller =
        trace::getHookInstaller(trace::HookerInfo{.srcLib = srcLib,
                                                  .targeLib = targetLib,
                                                  .symbolName = symbolName,
                                                  .newFuncPtr = hookerAddr});
    hookInstaller.onSuccess = [&]() {
        oldFuncAddr =
            trace::CudaInfoCollection::instance().getSymbolAddr(symbolName);
        LOG(INFO) << __func__ << ":" << oldFuncAddr;
    };
    hook::install_hook(hookInstaller);
}
}

#endif
