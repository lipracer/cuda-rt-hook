#pragma once

#include <stddef.h>

#include <iosfwd>
#include <set>
#include <string>
#include <unordered_map>

namespace hook {

class MemoryStatistic {
   public:
    struct DevPtr {
        size_t size_{0};
        void* ptr_{nullptr};
        DevPtr(void* ptr, size_t size) : size_(size), ptr_(ptr) {}
        bool operator<(const DevPtr& other) const { return ptr_ < other.ptr_; }
    };

    size_t total_size() const { return total_size_; }
    size_t peak_size() const { return peak_size_; }

    void record_alloc(void* ptr, size_t size);
    void record_free(void* ptr);

    friend std::ostream& operator<<(std::ostream& os, const MemoryStatistic& s);

   private:
    std::string identity_;
    size_t total_size_{0};
    size_t peak_size_{0};
    std::set<DevPtr> ptr_map_;
};

class MemoryStatisticCollection {
   public:
    struct PtrIdentity {
        std::string lib;
        size_t devId;
        int kind;
        PtrIdentity(std::string lib, size_t devId, int kind)
            : lib(lib), devId(devId), kind(kind) {}
        bool operator==(const PtrIdentity& other) const {
            return lib == other.lib && devId == other.devId &&
                   kind == other.kind;
        }
        friend std::ostream& operator<<(std::ostream& os,
                                        const PtrIdentity& id);
    };
    struct PtrIdentityHash {
        size_t operator()(const PtrIdentity& id) const {
            std::string ret;
            ret += id.lib;
            ret += std::to_string(id.kind);
            ret += std::to_string(id.devId);
            return std::hash<std::string>()(ret);
        }
    };

    ~MemoryStatisticCollection();

    void record_alloc(const std::string& libName, size_t devId, void* ptr,
                      size_t size, int kind);
    void record_free(const std::string& libName, size_t devId, void* ptr,
                     int kind);

    void record_free(const std::string& libName, size_t devId, void* ptr);

    static MemoryStatisticCollection& instance();

    friend std::ostream& operator<<(std::ostream& os,
                                    const MemoryStatisticCollection& s);

   private:
    std::unordered_map<PtrIdentity, MemoryStatistic, PtrIdentityHash>
        statistics_;
    struct KindDevPtr {
        size_t devId;
        void* ptr;
        bool operator==(const KindDevPtr& other) const {
            return devId == other.devId && ptr == other.ptr;
        }
    };
    struct KindDevPtrHash {
        size_t operator()(const KindDevPtr& other) const {
            size_t value = other.devId << 16 |
                           ((reinterpret_cast<size_t>(other.ptr) << 16) >> 16);
            return std::hash<size_t>()(value);
        }
    };
    std::unordered_map<KindDevPtr, int, KindDevPtrHash> kind_map_;
};

}  // namespace hook
