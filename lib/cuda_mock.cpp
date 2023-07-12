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
}  // namespace cuda_mock