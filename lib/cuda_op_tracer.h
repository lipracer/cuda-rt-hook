#include <atomic>
#include <string>
#include <unordered_map>
#include <vector>

#include "hook.h"

namespace tracer {

class CudaInfoCollection {
   public:
    static CudaInfoCollection& instance();
    void collectRtLib(const std::string& lib);
    void* getSymbolAddr(const std::string& name);

   private:
    const std::string cudaRTLibPath = "libcudart.so";
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

hook::HookInstaller getHookInstaller();

}  // namespace tracer