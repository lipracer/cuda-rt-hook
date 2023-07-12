#include "cuda_op_tracer.h"
#include "hook.h"
#include "logger.h"

namespace cuda_mock {

void initialize() {
    LOG(0) << "initialize";
    hook::HookInstaller hookInstaller = tracer::getHookInstaller();
    hook::install_hook(hookInstaller);
    // hook::install_hook();
}

void __any_mock_func__(...) {
    LOG(0) << __func__;
}

void internal_install_hook(const char* srcLib, const char* targetLib,
                const char* symbolName) {
    LOG(0) << "initialize srcLib:" << srcLib << " targetLib:" << targetLib
           << " symbolName:" << symbolName;
    hook::HookInstaller hookInstaller =
        tracer::getHookInstaller(tracer::HookerInfo{
            .srcLib = srcLib,
            .targeLib = targetLib,
            .symbolName = symbolName,
            .newFuncPtr = reinterpret_cast<void*>(&__any_mock_func__)});
    hook::install_hook(hookInstaller);
}

}  // namespace cuda_mock