#include "statistic.h"

#include <iomanip>
#include <map>
#include <sstream>

#include "hook.h"
#include "hook_context.h"

namespace hook {

namespace {

std::string prettyFormatSize(size_t bytes) {
    const char* sizes[] = {"B", "KB", "MB", "GB"};
    size_t order = 0;
    double size = static_cast<double>(bytes);

    while (size >= 1024 && order < sizeof(sizes) / sizeof(sizes[0]) - 1) {
        order++;
        size /= 1024;
    }

    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << size << " " << sizes[order];
    return out.str();
}

}  // namespace

std::string shortLibName(const std::string& full_lib_name) {
#if 1
    auto pos = full_lib_name.find_last_of('/');
    if (pos != std::string::npos) {
        return full_lib_name.substr(pos + 1);
    }
#endif
    return full_lib_name;
}

std::ostream& operator<<(std::ostream& os,
                         const MemoryStatisticCollection::PtrIdentity& id) {
    os << "id(" << id.lib << "),devId:" << id.devId
       << ",kind:" << id.kind << ")";
    return os;
}

void MemoryStatistic::record_alloc(void* ptr, size_t size) {
    ptr_map_.insert(MemoryStatistic::DevPtr(ptr, size));
    total_size_ += size;
    if (total_size_ > peak_size_) {
        peak_size_ = total_size_;
    }
}

void MemoryStatistic::record_free(void* ptr) {
    auto iter = ptr_map_.find(MemoryStatistic::DevPtr(ptr, 0));
    if (iter == ptr_map_.end()) {
        LOG(WARN) << "free dangling pointer ptr didn't record in map";
        return;
    }
    total_size_ -= iter->size_;
    ptr_map_.erase(iter);
}

std::ostream& operator<<(std::ostream& os, const MemoryStatistic& s) {
    std::stringstream ss;
    ss << "total alloc:" << s.total_size_;
    ss << alignWith(ss.str()) << "peak size:" << s.peak_size_;
    os << ss.str();
    return os;
}

MemoryStatisticCollection::~MemoryStatisticCollection() {
    LOG(WARN) << "memory statistic info:\n" << *this;
}

MemoryStatisticCollection& MemoryStatisticCollection::instance() {
    static MemoryStatisticCollection instance;
    return instance;
}

void MemoryStatisticCollection::record_alloc(const std::string& libName,
                                             size_t devId, void* ptr,
                                             size_t size, int kind) {
    kind_map_.insert(
        std::make_pair(KindDevPtr{.devId = devId, .ptr = ptr}, kind));
    auto& statistic = statistics_[PtrIdentity(libName, devId, kind)];
    statistic.record_alloc(ptr, size);

    MLOG(MEMORY, INFO) << "[" << shortLibName(libName)
                       << "][ALLOC] devId: " << devId << ", ptr: " << ptr
                       << ", size: " << prettyFormatSize(statistic.total_size())
                       << ", kind: " << kind << ", allocated: "
                       << prettyFormatSize(statistic.total_size())
                       << ", peak: " << prettyFormatSize(statistic.peak_size());
}

void MemoryStatisticCollection::record_free(const std::string& libName,
                                            size_t devId, void* ptr, int kind) {
    MLOG(MEMORY, INFO) << "[" << shortLibName(libName)
                       << "][FREE] devId: " << devId << ", ptr: " << ptr;
    statistics_[PtrIdentity(libName, devId, kind)].record_free(ptr);
}

void MemoryStatisticCollection::record_free(const std::string& libName,
                                            size_t devId, void* ptr) {
    auto iter = kind_map_.find(KindDevPtr{.devId = devId, .ptr = ptr});
    if (iter == kind_map_.end()) {
        LOG(WARN) << "free dangling pointer can't found ptr kind!";
        return;
    }
    record_free(libName, devId, ptr, iter->second);
}

std::ostream& operator<<(std::ostream& os, const MemoryStatisticCollection& s) {
    using ptr_type = decltype(s.statistics_)::const_pointer;
    std::vector<ptr_type> ptrs;
    ptrs.reserve(s.statistics_.size());

    for (auto& ms : s.statistics_) {
        ptrs.push_back(&ms);
    }

    std::sort(ptrs.begin(), ptrs.end(), [](ptr_type lhs, ptr_type rhs) {
        if (lhs->first.lib < rhs->first.lib) {
            return true;
        } else if (lhs->first.lib == rhs->first.lib) {
            return lhs->first.devId < rhs->first.devId;
        }
        return false;
    });

    for (auto ptr : ptrs) {
        os << ptr->first << "\n" << ptr->second << "\n";
    }
    return os;
}

}  // namespace hook
