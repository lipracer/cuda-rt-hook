#include "cuda_mock.h"

#include <dlfcn.h>

#include <csetjmp>

#include "backtrace.h"
#include "cuda_op_tracer.h"
#include "hook.h"
#include "logger.h"

std::jmp_buf log_jump_buffer = {{0}};

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

void __attribute__((optimize("O0"))) __any_mock_func__() {
    // why need not pop rbp, inline??
    // asm volatile("pop %rbp");
    asm volatile("push %rax");
    asm volatile("push %rdi");
    if (!setjmp(log_jump_buffer)) {
        log_router();
    }
    asm volatile("pop %rdi");
    asm volatile("pop %rax");
    asm volatile("add    $0x8,%rsp");
    asm volatile("jmp *%0" : : "r"(oldFuncAddr));
}

void dh_internal_install_hook(const char* srcLib, const char* targetLib,
                              const char* symbolName, const char* hookerLibPath,
                              const char* hookerSymbolName) {
    LOG(INFO) << "initialize srcLib:" << srcLib << " targetLib:" << targetLib
              << " symbolName:" << symbolName;
    auto hookerAddr = reinterpret_cast<void*>(&__any_mock_func__);
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
