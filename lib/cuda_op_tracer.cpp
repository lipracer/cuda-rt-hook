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

#include "cuda_types.h"
#include "logger.h"

namespace tracer {

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
    auto handle = dlopen(libcudart_.c_str(), RTLD_LAZY);
    CHECK(handle, std::string("can't open ") + libcudart_);
    return dlsym(handle, name.c_str());
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
    // LOG(0) << __func__ << ":" << std::string(func_name, 16);
    static void* org_addr =
        CudaInfoCollection::instance().getSymbolAddr("cudaLaunchKernel");
    // org_addr =
    // function_map[reinterpret_cast<void*>(&cudaLaunchKernel_wrapper)];
    CHECK(org_addr, "empty cudaLaunchKernel addr!");
    return reinterpret_cast<decltype(&cudaLaunchKernel_wrapper)>(org_addr)(
        func, gridDim, blockDim, args, sharedMem, stream);
}

bool BackTraceCollection::CallStackInfo::snapshot() {
    void* buffer[kMaxStackDeep] = {0};
    char** symbols = nullptr;
    int num = backtrace(buffer, kMaxStackDeep);
    CHECK(num > 0, "Expect frams num {" + std::to_string(num) + "} > 0!");
    symbols = backtrace_symbols(buffer, num);
    if (symbols == nullptr) {
        return false;
    }
    LOG(0) << "get stack deep num:" << num;
    for (int j = 0; j < num; j++) {
        LOG(0) << "current frame " << j << " addr:" << buffer[j]
               << " symbol:" << symbols[j];
        backtrace_addrs_.push_back(buffer[j]);
        backtrace_.emplace_back(symbols[j]);
    }
    free(symbols);
    return true;
}

BackTraceCollection& BackTraceCollection::instance() {
    static BackTraceCollection self;
    return self;
}

void BackTraceCollection::collect_backtrace(const void* func_ptr) {
    auto iter = cached_map_.find(func_ptr);

    if (iter != cached_map_.end()) {
        ++std::get<1>(backtraces_[iter->second]);
        return;
    }
    cached_map_.insert(std::make_pair(func_ptr, backtraces_.size()));

    backtraces_.emplace_back();
    if (!std::get<0>(backtraces_.back()).snapshot()) {
        LOG(2) << "can't get backtrace symbol!";
    }
}

void BackTraceCollection::dump() {
    std::ofstream ofs("./backtrace.log");
    ofs << "ignore:[base address]:" << base_addr_ << "\n";
    for (const auto& stack_info : backtraces_) {
        ofs << "ignore:[call " << std::get<1>(stack_info) << " times"
            << "]\n";
        ofs << std::get<0>(stack_info);
    }
    ofs.flush();
    ofs.close();

    // ofs.open("./backtrace_addrs.log");
    // for(const auto& backtrace : backtrace_addrs_) {
    //     ofs << "[call]" << "\n";
    //     for(const auto& line : backtrace) {
    //         ofs << line << "\n";
    //     }
    // }
    // ofs.flush();
}

std::ostream& operator<<(std::ostream& os,
                         const BackTraceCollection::CallStackInfo& info) {
    for (size_t i = 0; i < info.backtrace_.size(); ++i) {
        os << info.backtrace_[i] << "GlobalAddress:" << info.backtrace_addrs_[i]
           << "\n";
    }
    return os;
}

hook::HookInstaller getHookInstaller(const HookerInfo& info) {
    static const char* symbolName = "cudaLaunchKernel";
    static void* newFuncAddr = reinterpret_cast<void*>(&cudaLaunchKernel_wrapper);
    if (info.srcLib && info.targeLib && info.symbolName && info.newFuncPtr) {
        kPytorchCudaLibName = info.srcLib;
        kCudaRTLibName = info.targeLib;
        symbolName = info.symbolName;
        newFuncAddr = info.newFuncPtr;
    }
    hook::HookInstaller installer;
    installer.isTargetLib = [](const char* libName) -> bool {
        LOG(0) << "visit lib:" << libName;
        CudaInfoCollection::instance().collectRtLib(libName);
        if (std::string(libName).find(kPytorchCudaLibName) !=
            std::string::npos) {
            return true;
        }
        return false;
    };
    installer.isTargetSymbol = [=](const char* symbol) -> bool {
        // LOG(0) << "visit symbol:" << symbol;
        if (std::string(symbol) == symbolName) {
            return true;
        }
        return false;
    };
    installer.newFuncPtr = [](const hook::OriginalInfo& info) -> void* {
        BackTraceCollection::instance().setBaseAddr(info.basePtr);
        return reinterpret_cast<void*>(newFuncAddr);
    };
    return installer;
}

}  // namespace tracer