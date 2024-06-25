import ctypes
from .dynamic_obj import *

src_code = '''
#include <dlfcn.h>  // dladdr
#include <execinfo.h>

#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <vector>

#include "backtrace.h"
#include "cuda_types.h"
#include "hook.h"
#include "logger/StringRef.h"
#include "logger/logger.h"

// NB: don't use same name with original function, this will result in the
// replacement not taking effect

#define DEF_FUNCTION_INT(name, ...)       \
    typedef int (*name##_t)(__VA_ARGS__); \
    name##_t origin_##name = nullptr;     \
    int name(__VA_ARGS__)

#define DEF_FUNCTION_VOID(name, ...)      \
    typedef int (*name##_t)(__VA_ARGS__); \
    name##_t origin_##name = nullptr;     \
    void name(__VA_ARGS__)

DEF_FUNCTION_INT(cudaMalloc, void** devPtr, size_t size) {
    LOG(INFO) << __func__;
    *devPtr = malloc(10);
    return 0;
}

DEF_FUNCTION_INT(cudaFree, void* devPtr) {
    LOG(INFO) << __func__;
    free(devPtr);
    return 0;
}

DEF_FUNCTION_INT(cudaMemcpy, void* dst, const void* src, size_t count,
                 int kind) {
    LOG(INFO) << __func__;
    return 0;
}

DEF_FUNCTION_INT(cudaSetDevice, int device) {
    LOG(INFO) << __func__;
    return 0;
}

DEF_FUNCTION_INT(cudaGetDevice, int* device) {
    LOG(INFO) << __func__;
    *device = 0;
    return 0;
}

DEF_FUNCTION_INT(cudaGetDeviceCount, int* count) {
    LOG(INFO) << __func__;
    *count = 1;
    return 0;
}

DEF_FUNCTION_INT(cuDevicePrimaryCtxGetState, int, unsigned int* flag,
                 int* active) {
    LOG(INFO) << __func__;
    *flag = 0;
    *active = 1;
    return 0;
}

DEF_FUNCTION_INT(cudaGetDeviceProperties_v2, struct cudaDeviceProp* prop,
                 int device) {
    LOG(INFO) << __func__;
    prop->major = 8;
    prop->minor = 0;
    prop->multiProcessorCount = 108;
    prop->regsPerMultiprocessor = 65536;
    prop->sharedMemPerMultiprocessor = 163840;
    prop->warpSize = 32;
    return 0;
}

DEF_FUNCTION_INT(cudaDeviceGetStreamPriorityRange, int* leastPriority,
                 int* greatestPriority) {
    LOG(INFO) << __func__;
    *leastPriority = 0;
    *greatestPriority = 10;
    return 0;
}

DEF_FUNCTION_INT(cudaGetLastError, void) {
    LOG(INFO) << __func__;
    return 0;
}

DEF_FUNCTION_INT(cudaDriverGetVersion, int* version) {
    LOG(INFO) << __func__;
    *version = 5;
    return 0;
}

DEF_FUNCTION_VOID(_ZN2at14DynamicLibraryC1EPKcS2_b, char const*, char const*,
                  bool) {}

void* _ZN2at14DynamicLibrary3symEPKc(const char* sym) {
    LOG(INFO) << "symbol:" << sym;
    return reinterpret_cast<void*>(&cuDevicePrimaryCtxGetState);
}

void (*origin__ZN2at14DynamicLibrary3symEPKc)(char const*, char const*, bool);

DEF_FUNCTION_VOID(_ZN3c106detail14torchCheckFailEPKcS2_jRKSs, const char* func,
                  const char* file, uint32_t line, const std::string& msg) {
    // maybe api is not compatible
    LOG(INFO) << __func__ << " file:" << file << " line:" << line;
}
DEF_FUNCTION_VOID(_ZN3c106detail14torchCheckFailEPKcS2_jS2_, const char* func,
                  const char* file, uint32_t line, const char* msg) {
    LOG(INFO) << __func__ << " file:" << file << " line:" << line
              << " msg:" << msg;
}

DEF_FUNCTION_VOID(_ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib,
                  int err, char const* filename, char const* func_name,
                  int lineno, bool) {
    LOG(INFO) << "error:" << err << " filename:" << filename
              << " func_name:" << func_name << " lineno:" << lineno;
}

DEF_FUNCTION_INT(cudaStreamIsCapturing, void*, int* is_c) {
    LOG(INFO) << __func__;
    *is_c = 0;
    return 0;
}

// NB: don't use same name with original function, this will result in the
// replacement not taking effect
#define BUILD_FEATURE(name) hook::HookFeatureBase(#name, &name, &origin_##name)

namespace {

class CudaRuntimeApiHook : public hook::HookInstallerWrap<CudaRuntimeApiHook> {
   public:
    bool targetLib(const char* name) {
        return !adt::StringRef(name).contain("libcudart.so") &&
               !adt::StringRef(name).contain("libcuda.so");
    }

    hook::HookFeatureBase symbols[15] = {
        BUILD_FEATURE(cudaMalloc),
        BUILD_FEATURE(cudaFree),
        BUILD_FEATURE(cudaMemcpy),
        BUILD_FEATURE(cudaSetDevice),
        BUILD_FEATURE(cudaGetDevice),
        BUILD_FEATURE(cudaGetDeviceCount),
        BUILD_FEATURE(cuDevicePrimaryCtxGetState),
        BUILD_FEATURE(cudaGetDeviceProperties_v2),
        BUILD_FEATURE(cudaDeviceGetStreamPriorityRange),
        BUILD_FEATURE(cudaGetLastError),
        BUILD_FEATURE(cudaDriverGetVersion),
        BUILD_FEATURE(cudaStreamIsCapturing),
        hook::HookFeature("_ZN2at14DynamicLibraryC1EPKcS2_b",
                          &_ZN2at14DynamicLibraryC1EPKcS2_b,
                          &origin__ZN2at14DynamicLibraryC1EPKcS2_b),
        hook::HookFeature("_ZN2at14DynamicLibrary3symEPKc",
                          &_ZN2at14DynamicLibrary3symEPKc,
                          &origin__ZN2at14DynamicLibrary3symEPKc),
        /*
                hook::HookFeature(
                    "_ZN3c106detail14torchCheckFailEPKcS2_jRKSs",
                    &_ZN3c106detail14torchCheckFailEPKcS2_jRKSs,
                    &origin__ZN3c106detail14torchCheckFailEPKcS2_jRKSs,
                    [this]() -> bool {
                        if
           (!adt::StringRef(this->curLibName()).contain("libc10.so")) { return
           true;
                        }
                        return false;
                    }),
                hook::HookFeature(
                    "_ZN3c106detail14torchCheckFailEPKcS2_jS2_",
                    &_ZN3c106detail14torchCheckFailEPKcS2_jS2_,
                    &origin__ZN3c106detail14torchCheckFailEPKcS2_jS2_,
                    [this]() -> bool {
                        if
           (!adt::StringRef(this->curLibName()).contain("libc10.so")) { return
           true;
                        }
                        return false;
                    }),

        */
        hook::HookFeature(
            "_ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib",
            &_ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib,
            &origin__ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib,
            [this]() -> bool {
                if (!adt::StringRef(this->curLibName())
                         .contain("libc10_cuda.so")) {
                    return true;
                }
                return false;
            }),
    };

    void onSuccess() {
        LOG(INFO) << "install " << curLibName() << " function:" << curSymName()
                  << " success";
    }
};

}  // namespace

extern "C" {

void install_triton_hook() {
    static auto install_wrap = std::make_shared<CudaRuntimeApiHook>();
    install_wrap->install();
}

}
'''

def triton_mock():
    lib = dynamic_obj(src_code, True).appen_compile_opts('-lpthread').compile().get_lib()
    lib = ctypes.CDLL(lib)
    lib.install_triton_hook()