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

namespace {

//-------------------------- xpu api --------------------------//

typedef int (*xpu_malloc_t)(void** pdevptr, uint64_t size, int kind);
typedef int (*xpu_free_t)(void* devptr);
typedef int (*xpu_wait_t)(void* stream);
typedef int (*xpu_memcpy_t)(void* dst, const void* src, uint64_t size,
                            int kind);
typedef int (*xpu_set_device_t)(int devid);
typedef int (*xpu_current_device_t)(int* devid);
typedef int (*xpu_launch_async_t)(void*);
typedef int (*xpu_stream_create_t)(void** pstream);
typedef int (*xpu_stream_destroy_t)(void* stream);

xpu_malloc_t origin_xpu_malloc = nullptr;
xpu_free_t origin_xpu_free = nullptr;
xpu_wait_t origin_xpu_wait = nullptr;
xpu_memcpy_t origin_xpu_memcpy = nullptr;
xpu_set_device_t origin_xpu_set_device = nullptr;
xpu_current_device_t origin_xpu_current_device = nullptr;
xpu_launch_async_t origin_xpu_launch_async = nullptr;
xpu_stream_create_t origin_xpu_stream_create = nullptr;
xpu_stream_destroy_t origin_xpu_stream_destroy = nullptr;

int xpu_malloc(void** pdevptr, uint64_t size, int kind) {
    int r = 0;
    int devId = 0;

    CHECK(origin_xpu_current_device, "xpu_current_device not binded");
    CHECK(origin_xpu_malloc, "xpu_malloc not binded");

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

int xpu_free(void* devptr) {
    int r = 0;
    int devId = 0;

    CHECK(origin_xpu_current_device, "xpu_current_device not binded");
    CHECK(origin_xpu_free, "xpu_free not binded");

    r = origin_xpu_current_device(&devId);
    if (r != 0) {
        return r;
    }

    r = origin_xpu_free(devptr);

    hook::MemoryStatisticCollection::instance().record_free(
        hook::HookRuntimeContext::instance().curLibName(), devId, devptr);

    return r;
}

int xpu_wait(void* stream) { return origin_xpu_wait(stream); }

int xpu_memcpy(void* dst, const void* src, uint64_t size, int kind) {
    return origin_xpu_memcpy(dst, src, size, kind);
}

int xpu_set_device(int devid) {
    return origin_xpu_set_device(devid);
}

int xpu_current_device(int* devid) {
    int ret = origin_xpu_current_device(devid);
    return ret;
}

int xpu_launch_async(void* func) {
    // TODO: get symbol name from symbol table
    return origin_xpu_launch_async(func);
}

int xpu_stream_create(void** pstream) {
    return origin_xpu_stream_create(pstream);
}

int xpu_stream_destroy(void* stream) {
    return origin_xpu_stream_destroy(stream);
}

//-------------------------- cuda api --------------------------//

typedef int (*CudaMalloc_t)(void** devPtr, size_t size);
typedef int (*CudaFree_t)(void* devPtr);
typedef int (*CudaMemcpy_t)(void* dst, const void* src, size_t count, int kind);
typedef int (*CudaSetDevice_t)(int);
typedef int (*CudaGetDevice_t)(int*);

CudaMalloc_t origin_cudaMalloc = nullptr;
CudaFree_t origin_cudaFree = nullptr;
CudaMemcpy_t origin_cudaMemcpy = nullptr;
CudaSetDevice_t origin_cudaSetDevice = nullptr;
CudaGetDevice_t origin_cudaGetDevice = nullptr;

int cudaMalloc(void** devPtr, size_t size) {
    return origin_cudaMalloc(devPtr, size);
}

int cudaFree(void* devPtr) {
    return origin_cudaFree(devPtr);
}

int cudaMemcpy(void* dst, const void* src, size_t count, int kind) {
    return origin_cudaMemcpy(dst, src, count, kind);
}

int cudaSetDevice(int device) {
    return origin_cudaSetDevice(device);
}

int cudaGetDevice(int* device) {
    return origin_cudaGetDevice(device);
}

#define BUILD_FEATURE(name) hook::HookFeature(#name, &name, &origin_##name)

class XpuRuntimeApiHook : public hook::HookInstallerWrap<XpuRuntimeApiHook> {
   public:
    bool targetLib(const char* name) {
        return !adt::StringRef(name).contain("libxpurt.so") &&
               !adt::StringRef(name).contain("libcudart.so");
    }

    hook::HookFeature symbols[14] = {
        BUILD_FEATURE(xpu_malloc),
        BUILD_FEATURE(xpu_free),
        BUILD_FEATURE(xpu_current_device),
        BUILD_FEATURE(xpu_set_device),
        BUILD_FEATURE(xpu_wait),
        BUILD_FEATURE(xpu_memcpy),
        BUILD_FEATURE(xpu_launch_async),
        BUILD_FEATURE(xpu_stream_create),
        BUILD_FEATURE(xpu_stream_destroy),

        BUILD_FEATURE(cudaMalloc),
        BUILD_FEATURE(cudaFree),
        BUILD_FEATURE(cudaMemcpy),
        BUILD_FEATURE(cudaSetDevice),
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