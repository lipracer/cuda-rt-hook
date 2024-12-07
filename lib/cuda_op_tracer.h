#pragma once

#include <atomic>
#include <string>
#include <unordered_map>
#include <vector>

#include "hook.h"

namespace trace {

class CudaInfoCollection {
   public:
    static CudaInfoCollection& instance();
    // collect the lib which the symbol defined, then we can call dlopen and
    // find the symbol address, when we need call this in hook function
    void collectRtLib(const std::string& lib);
    void* getSymbolAddr(const std::string& name);
    ~CudaInfoCollection();

   private:
    std::string libcudart_;
    std::atomic<void*> handle_{nullptr};
};

struct HookerInfo {
    // the dynamic lib which the target symbol defined
    const char* srcLib = nullptr;
    // the dynamic lib which the target symbol will be replace
    const char* targeLib = nullptr;
    // the symbol which will be replace
    const char* symbolName = nullptr;
    void* newFuncPtr = nullptr;
};

hook::HookInstaller getHookInstaller(const HookerInfo& info = HookerInfo{});

}  // namespace trace