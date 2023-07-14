#include <csetjmp>
#include <dlfcn.h>

#include "cuda_op_tracer.h"
#include "hook.h"
#include "logger.h"
#include "cuda_op_tracer.h"


std::jmp_buf log_jump_buffer = {{0}};


namespace cuda_mock {

void initialize() {
    LOG(0) << "initialize";
    hook::HookInstaller hookInstaller = tracer::getHookInstaller();
    hook::install_hook(hookInstaller);
    // hook::install_hook();
}

static void* oldFuncAddr = nullptr;

void log_router(const char* name) {
    LOG(0) << name << ":" << oldFuncAddr;
    longjmp(log_jump_buffer, 1);
}

void __any_mock_func__() {
    // why need not pop rbp, inline??
    // asm volatile("pop %rbp");
    asm volatile("jmp *%0" : : "r"(oldFuncAddr));
    // asm volatile("ret");
    // if (!setjmp(log_jump_buffer)) {
    //     log_router(__func__);
    // }
}

void internal_install_hook(const char* srcLib, const char* targetLib,
                           const char* symbolName, const char* hookerLibPath,
                           const char* hookerSymbolName) {
    LOG(0) << "initialize srcLib:" << srcLib << " targetLib:" << targetLib
           << " symbolName:" << symbolName;
    auto hookerAddr = reinterpret_cast<void*>(&__any_mock_func__);
    if (hookerLibPath) {
        auto handle = dlopen(hookerLibPath, RTLD_LAZY);
        CHECK(handle, std::string("can't not dlopen ") + hookerLibPath);
        hookerAddr =
            dlsym(handle, hookerSymbolName ? hookerSymbolName : symbolName);
    }
    CHECK(hookerAddr, "hookerAddr can't be empty!");
    hook::HookInstaller hookInstaller =
        tracer::getHookInstaller(tracer::HookerInfo{.srcLib = srcLib,
                                                    .targeLib = targetLib,
                                                    .symbolName = symbolName,
                                                    .newFuncPtr = hookerAddr});
    hookInstaller.onSuccess = [&]() {
        oldFuncAddr =
            tracer::CudaInfoCollection::instance().getSymbolAddr(symbolName);
        LOG(0) << __func__ << ":" << oldFuncAddr;
    };
    hook::install_hook(hookInstaller);
}

}  // namespace cuda_mock