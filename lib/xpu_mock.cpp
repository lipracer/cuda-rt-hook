#include "xpu_mock.h"

#include <Python.h>
#include <dlfcn.h>  // dladdr
#include <execinfo.h>
#include <frameobject.h>

#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <vector>

#include "backtrace.h"
#include "hook.h"
#include "hooks/print_hook.h"
#include "logger/StringRef.h"
#include "logger/logger.h"
#include "statistic.h"
#include "support.h"

// NB: don't use same name with original function, this will result in the
// replacement not taking effect

#define DEF_FUNCTION_IMPL(name, RetT, ...) \
    typedef RetT (*name##_t)(__VA_ARGS__); \
    name##_t origin_##name = nullptr;      \
    RetT name(__VA_ARGS__)

#define DEF_FUNCTION_INT(name, ...) DEF_FUNCTION_IMPL(name, int, __VA_ARGS__)

#define DEF_FUNCTION_VOID(name, ...) DEF_FUNCTION_IMPL(name, void, __VA_ARGS__)

namespace {

//-------------------------- xpu api --------------------------//

DEF_FUNCTION_INT(xpu_current_device, int* devid) {
    return origin_xpu_current_device(devid);
}

DEF_FUNCTION_INT(xpu_malloc, void** pdevptr, uint64_t size, int kind) {
    int r = 0;
    int devId = 0;

    CHECK(origin_xpu_current_device, "xpu_current_device not binded");
    r = origin_xpu_current_device(&devId);
    if (r != 0) {
        return r;
    }

    r = origin_xpu_malloc(pdevptr, size, kind);
    if (r != 0) {
        LOG(WARN) << "xpu malloc device memory failed!\n"
                  << hook::MemoryStatisticCollection::instance();
        return r;
    }

    hook::MemoryStatisticCollection::instance().record_alloc(
        hook::HookRuntimeContext::instance().curLibName(), devId, *pdevptr,
        size, kind);

    return r;
}

DEF_FUNCTION_INT(xpu_free, void* devptr) {
    int r = 0;
    int devId = 0;

    CHECK(origin_xpu_current_device, "xpu_current_device not binded");
    r = origin_xpu_current_device(&devId);
    if (r != 0) {
        return r;
    }

    r = origin_xpu_free(devptr);

    hook::MemoryStatisticCollection::instance().record_free(
        hook::HookRuntimeContext::instance().curLibName(), devId, devptr);

    return r;
}

DEF_FUNCTION_INT(xpu_wait, void* stream) { return origin_xpu_wait(stream); }

DEF_FUNCTION_INT(xpu_memcpy, void* dst, const void* src, uint64_t size,
                 int kind) {
    return origin_xpu_memcpy(dst, src, size, kind);
}

DEF_FUNCTION_INT(xpu_set_device, int devid) {
    return origin_xpu_set_device(devid);
}

DEF_FUNCTION_INT(xpu_launch_async, void* func) {
    // TODO: get symbol name from symbol table
    return origin_xpu_launch_async(func);
}

DEF_FUNCTION_INT(xpu_stream_create, void** pstream) {
    return origin_xpu_stream_create(pstream);
}

DEF_FUNCTION_INT(xpu_stream_destroy, void* stream) {
    return origin_xpu_stream_destroy(stream);
}

//-------------------------- cuda api --------------------------//

DEF_FUNCTION_INT(cudaSetDevice, int device) {
    return origin_cudaSetDevice(device);
}

DEF_FUNCTION_INT(cudaGetDevice, int* device) {
    return origin_cudaGetDevice(device);
}

DEF_FUNCTION_INT(cudaMalloc, void** devPtr, size_t size) {
    int r = 0;
    int devId = 0;

    CHECK(origin_cudaGetDevice, "cudaGetDevice not binded");
    r = origin_cudaGetDevice(&devId);
    if (r != 0) {
        return r;
    }

    r = origin_cudaMalloc(devPtr, size);
    if (r != 0) {
        LOG(WARN) << "xpu cudaMalloc device memory failed!\n"
                  << hook::MemoryStatisticCollection::instance();
        return r;
    }

    hook::MemoryStatisticCollection::instance().record_alloc(
        hook::HookRuntimeContext::instance().curLibName(), devId, *devPtr, size,
        /*kind=GLOBAL_MEM*/ 0);

    return r;
}

DEF_FUNCTION_INT(cudaFree, void* devPtr) {
    int r = 0;
    int devId = 0;

    CHECK(origin_cudaGetDevice, "cudaGetDevice not binded");
    r = origin_cudaGetDevice(&devId);
    if (r != 0) {
        return r;
    }

    r = origin_cudaFree(devPtr);

    hook::MemoryStatisticCollection::instance().record_free(
        hook::HookRuntimeContext::instance().curLibName(), devId, devPtr);

    return r;
}

DEF_FUNCTION_INT(cudaMemcpy, void* dst, const void* src, size_t count,
                 int kind) {
    return origin_cudaMemcpy(dst, src, count, kind);
}

#define BUILD_FEATURE(name) \
    hook::FHookFeature(STR_TO_TYPE(#name), &name, &origin_##name)

class XpuRuntimeApiHook : public hook::HookInstallerWrap<XpuRuntimeApiHook> {
   public:
    bool targetLib(const char* name) {
        return !adt::StringRef(name).contain("libxpurt.so") &&
               !adt::StringRef(name).contain("libcudart.so");
    }

    hook::FHookFeature symbols[14] = {
        BUILD_FEATURE(xpu_malloc),         BUILD_FEATURE(xpu_free),
        BUILD_FEATURE(xpu_current_device), BUILD_FEATURE(xpu_set_device),
        BUILD_FEATURE(xpu_wait),           BUILD_FEATURE(xpu_memcpy),
        BUILD_FEATURE(xpu_launch_async),   BUILD_FEATURE(xpu_stream_create),
        BUILD_FEATURE(xpu_stream_destroy),

        BUILD_FEATURE(cudaMalloc),         BUILD_FEATURE(cudaFree),
        BUILD_FEATURE(cudaMemcpy),         BUILD_FEATURE(cudaSetDevice),
        BUILD_FEATURE(cudaGetDevice),
    };

    void onSuccess() { LOG(WARN) << "install " << curSymName() << " success"; }
};

struct PatchRuntimeHook : public hook::HookInstallerWrap<PatchRuntimeHook> {
    static int unifySetDevice(int devId) {
        LOG(INFO) << "devId:" << devId;
        auto ret = PatchRuntimeHook::instance()->xpu_set_device_(devId);
        CHECK_EQ(ret, 0, "xpu_set_device fail with result:{}", ret);
        return PatchRuntimeHook::instance()->cuda_set_device_(devId);
    }

    using SetDevFuncType_t = decltype(&unifySetDevice);

    bool targetLib(const char* name) {
        return !adt::StringRef(name).contain("libcudart.so") &&
               !adt::StringRef(name).contain("libxpurt.so");
    }

    bool targetSym(const char* name) {
        return adt::StringRef(name) == "cudaSetDevice" ||
               adt::StringRef(name) == "xpu_set_device";
    }

    void* newFuncPtr(const hook::OriginalInfo& info) {
        if (adt::StringRef(curSymName()) == "xpu_set_device") {
            xpu_set_device_ =
                reinterpret_cast<SetDevFuncType_t>(info.oldFuncPtr);
            return reinterpret_cast<void*>(&unifySetDevice);
        }
        cuda_set_device_ = reinterpret_cast<SetDevFuncType_t>(info.oldFuncPtr);
        return reinterpret_cast<void*>(&unifySetDevice);
    }
    void onSuccess() {}

    static PatchRuntimeHook* instance() {
        static auto install_wrap = std::make_shared<PatchRuntimeHook>();
        return install_wrap.get();
    }

    SetDevFuncType_t cuda_set_device_{nullptr};
    SetDevFuncType_t xpu_set_device_{nullptr};
};

}  // namespace

void __runtimeapi_hook_initialize() {
    static auto install_wrap = std::make_shared<XpuRuntimeApiHook>();
    install_wrap->install();
}

void dh_patch_runtime() { PatchRuntimeHook::instance()->install(); }
