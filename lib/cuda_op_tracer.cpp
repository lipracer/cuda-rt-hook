#include "cuda_op_tracer.h"

#include <dlfcn.h>
#include <elf.h>
#include <errno.h>
#include <execinfo.h>
#include <limits.h>
#include <link.h>
#include <stdarg.h>
#include <stdio.h>

#include <atomic>
#include <fstream>
#include <iosfwd>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "backtrace.h"
#include "cuda_types.h"
#include "logger/logger.h"

namespace trace {

static const char* kPytorchCudaLibName = "libtorch_cuda.so";
static const char* kCudaRTLibName = "libcudart.so";

CudaInfoCollection& CudaInfoCollection::instance() {
    static CudaInfoCollection self;
    return self;
}

void CudaInfoCollection::collectRtLib(const std::string& lib) {
    if (libcudart_.empty() && lib.find(kCudaRTLibName) != std::string::npos) {
        libcudart_ = lib;
    }
}

void* CudaInfoCollection::getSymbolAddr(const std::string& name) {
    if (libcudart_.empty()) {
        return nullptr;
    }
    handle_ = dlopen(libcudart_.c_str(), RTLD_LAZY);
    CHECK(handle_, "can't open {}", libcudart_);
    return dlsym(handle_, name.c_str());
}

CudaInfoCollection::~CudaInfoCollection() {
    if (!!handle_) dlclose(handle_);
}

extern "C" CUresult cudaLaunchKernel_wrapper(const void* func, dim3 gridDim,
                                             dim3 blockDim, void** args,
                                             size_t sharedMem,
                                             cudaStream_t stream) {
    static std::mutex mtx;
    {
        std::lock_guard<std::mutex> guard(mtx);
        BackTraceCollection::instance().collect_backtrace(func);
    }

    static void* org_addr =
        CudaInfoCollection::instance().getSymbolAddr("cudaLaunchKernel");
    // org_addr =
    // function_map[reinterpret_cast<void*>(&cudaLaunchKernel_wrapper)];
    CHECK(org_addr, "empty cudaLaunchKernel addr!");
    return reinterpret_cast<decltype(&cudaLaunchKernel_wrapper)>(org_addr)(
        func, gridDim, blockDim, args, sharedMem, stream);
}

hook::HookInstaller getHookInstaller(const HookerInfo& info) {
    static const char* symbolName = "cudaLaunchKernel";
    static void* newFuncAddr =
        reinterpret_cast<void*>(&cudaLaunchKernel_wrapper);
    if (info.srcLib && info.targeLib && info.symbolName && info.newFuncPtr) {
        kCudaRTLibName = info.srcLib;
        kPytorchCudaLibName = info.targeLib;
        symbolName = info.symbolName;
        newFuncAddr = info.newFuncPtr;
    }
    hook::HookInstaller installer;
    installer.isTargetLib = [](const char* libName) -> bool {
        CudaInfoCollection::instance().collectRtLib(libName);
        // TODO 为啥这行打印不生效？
        MLOG(HOOK, INFO) << "[installer.isTargetLib] libName:"
                         << kPytorchCudaLibName
                         << " targetlibName: " << libName;

        /*
            模糊匹配而不是精确匹配，因为：
            1. libname 可能包含版本号，例如 libtorch_cuda.so.1
            2. libname包含了完整的路径，例如
           /usr/local/cuda-10.2/lib64/libcudart.so
        */
        if (std::string(libName).find(kPytorchCudaLibName) !=
            std::string::npos) {
            return true;
        }
        return false;
    };
    installer.isTargetSymbol = [=](const char* symbol) -> bool {
        MLOG(HOOK, INFO) << "[installer.isTargetSymbol] symbol:" << symbol
                         << " targetSymbolName:" << symbolName;
        if (std::string(symbol) == symbolName) {
            return true;
        }
        return false;
    };
    installer.newFuncPtr = [](const hook::OriginalInfo& info) -> void* {
        BackTraceCollection::instance().setBaseAddr(info.libName, info.basePtr);
        return reinterpret_cast<void*>(newFuncAddr);
    };
    return installer;
}

}  // namespace trace