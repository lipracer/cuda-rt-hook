#pragma once

#include <atomic>
#include <string>
#include <unordered_map>
#include <vector>

#include "hook.h"

namespace tracer {

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

class BackTraceCollection {
   public:
    class CallStackInfo {
       public:
        static constexpr size_t kMaxStackDeep = 64;

        CallStackInfo() {
            backtrace_addrs_.reserve(kMaxStackDeep);
            backtrace_.reserve(kMaxStackDeep);
        }

        bool snapshot();

        friend std::ostream& operator<<(std::ostream&, const CallStackInfo&);

       private:
        std::vector<const void*> backtrace_addrs_;
        std::vector<std::string> backtrace_;
    };

    static BackTraceCollection& instance();

    void collect_backtrace(const void* func_ptr);
    void dump();

    void setBaseAddr(void* addr) { base_addr_ = addr; }

    ~BackTraceCollection() { dump(); }

   private:
    std::vector<std::tuple<CallStackInfo, size_t>> backtraces_;
    std::unordered_map<const void*, size_t> cached_map_;
    void* base_addr_{nullptr};
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

}  // namespace tracer