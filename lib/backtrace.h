#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace trace {

class BackTraceCollection {
   public:
    class CallStackInfo {
       public:
        static constexpr size_t kMaxStackDeep = 256;

        explicit CallStackInfo(
            const std::function<void*(const std::string&)>& getBaseAddr)
            : getBaseAddr_(getBaseAddr) {
            backtrace_addrs_.reserve(kMaxStackDeep);
            backtrace_.reserve(kMaxStackDeep);
            // test_feed_and_parse();
        }

        bool snapshot();

        bool parse();

        void test_feed_and_parse();

        friend std::ostream& operator<<(std::ostream&, const CallStackInfo&);

       private:
        // every call function's address, we need not this, just check
        std::vector<const void*> backtrace_addrs_;
        std::vector<std::string> backtrace_;
        std::function<void*(const std::string&)> getBaseAddr_;
    };

    static BackTraceCollection& instance();

    void collect_backtrace(const void* func_ptr);
    void dump();

    void setBaseAddr(const char* libName, void* addr) {
        base_addrs_.emplace(libName, addr);
    }

    ~BackTraceCollection() { dump(); }

   private:
    std::vector<std::tuple<CallStackInfo, size_t>> backtraces_;
    std::unordered_map<const void*, size_t> cached_map_;
    std::unordered_map<std::string, void*> base_addrs_;
};

}  // namespace trace
