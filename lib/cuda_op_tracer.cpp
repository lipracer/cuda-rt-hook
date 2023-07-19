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
#include "logger.h"

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
    CHECK(!libcudart_.empty(), "libcudart empty!");
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

    auto func_name = reinterpret_cast<const char*>(func);
    // LOG(INFO) << __func__ << ":" << std::string(func_name, 16);
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
    static void* newFuncAddr = reinterpret_cast<void*>(&cudaLaunchKernel_wrapper);
    if (info.srcLib && info.targeLib && info.symbolName && info.newFuncPtr) {
        kCudaRTLibName = info.srcLib;
        kPytorchCudaLibName = info.targeLib;
        symbolName = info.symbolName;
        newFuncAddr = info.newFuncPtr;
    }
    hook::HookInstaller installer;
    installer.isTargetLib = [](const char* libName) -> bool {
        LOG(INFO) << "visit lib:" << libName;
        CudaInfoCollection::instance().collectRtLib(libName);
        if (std::string(libName).find(kPytorchCudaLibName) !=
            std::string::npos) {
            return true;
        }
        return false;
    };
    installer.isTargetSymbol = [=](const char* symbol) -> bool {
        // LOG(INFO) << "visit symbol:" << symbol;
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