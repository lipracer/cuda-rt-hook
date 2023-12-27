#include "cuda_mock.h"

#include <dlfcn.h>
#include <string.h>

#include <csetjmp>

#include "GlobalVarMgr.h"
#include "backtrace.h"
#include "cuda_op_tracer.h"
#include "hook.h"
#include "logger/logger.h"

std::jmp_buf log_jump_buffer = {{}};

#ifdef __cplusplus

extern "C" {

void dh_initialize() {
    hook::init_all_global_variables();
    LOG(INFO) << "initialize";
    hook::HookInstaller hookInstaller = trace::getHookInstaller();
    hook::install_hook(hookInstaller);
    // hook::install_hook();
}

static hook::GlobalVarMgr<void*> oldFuncAddr(nullptr);

void log_router() {
    LOG(INFO) << __func__ << ":" << *oldFuncAddr;
    // sometime will crash
    // trace::BackTraceCollection::CallStackInfo tracer({});
    // tracer.snapshot();
    // LOG(WARN) << tracer;
    longjmp(log_jump_buffer, 1);
}

void __any_mock_func__() {
// Conditional code for x86_64 architecture
#if defined(__x86_64__)
    // avoid inline asm, use varidic c func and parse func result
    // asm volatile("pop %rbp");
    // asm volatile("push %rax");
    // asm volatile("push %rdi");
    // if (!setjmp(log_jump_buffer)) {
    //     log_router();
    // }
    // asm volatile("pop %rdi");
    // asm volatile("pop %rax");
    // asm volatile("add    $0x8,%rsp");
    // asm volatile("jmp *%0" : : "r"(oldFuncAddr));

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

static hook::GlobalVarMgr<std::unordered_map<std::string, void*>> gBuiltinFuncs(
    std::unordered_map<std::string, void*>{
        std::make_pair("__printf_chk",
                       reinterpret_cast<void*>(&builtin_printf))});

void dh_internal_install_hook(const char* srcLib, const char* targetLib,
                              const char* symbolName, const char* hookerLibPath,
                              const char* hookerSymbolName) {
    LOG(INFO) << "initialize srcLib:" << srcLib << " targetLib:" << targetLib
              << " symbolName:" << symbolName;
    auto iter = gBuiltinFuncs->find(symbolName);
    auto hookerAddr = iter == gBuiltinFuncs->end()
                          ? reinterpret_cast<void*>(&__any_mock_func__)
                          : iter->second;
    if (hookerLibPath) {
        auto handle = dlopen(hookerLibPath, RTLD_LAZY);
        CHECK(handle, "can't not dlopen:{0} error info:{1}", hookerLibPath,
              dlerror());
        hookerAddr =
            dlsym(handle, hookerSymbolName ? hookerSymbolName : symbolName);
    }
    CHECK(hookerAddr, "hookerAddr can't be empty!");
    hook::HookInstaller hookInstaller =
        trace::getHookInstaller(trace::HookerInfo{.srcLib = srcLib,
                                                  .targeLib = targetLib,
                                                  .symbolName = symbolName,
                                                  .newFuncPtr = hookerAddr});
    auto newFuncPtr = hookInstaller.newFuncPtr;
    hookInstaller.newFuncPtr = [=](const hook::OriginalInfo& orgInfo) -> void* {
        auto handle = dlopen(hookerLibPath, RTLD_LAZY);
        CHECK(handle, "can't not dlopen {0} error info:{1}", hookerLibPath,
              dlerror());
        std::string org_symbol = "__origin_" + std::string(symbolName);
        auto org_addr = dlsym(handle, org_symbol.c_str());
        if (org_addr) {
            *reinterpret_cast<void**>(org_addr) = orgInfo.oldFuncPtr;
        } else {
            LOG(WARN) << "can't find symbol:" << org_symbol;
        }
        return newFuncPtr(orgInfo);
    };
    hookInstaller.onSuccess = [&]() {
        *oldFuncAddr =
            trace::CudaInfoCollection::instance().getSymbolAddr(symbolName);
        LOG(INFO) << __func__ << ":" << *oldFuncAddr;
    };
    hook::install_hook(hookInstaller);
}
}

#endif
