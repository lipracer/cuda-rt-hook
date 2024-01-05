#include "statistic.h"

#include <map>
#include <sstream>

#include "hook.h"

namespace hook {

namespace {
std::string shortLibName(const std::string& full_lib_name) {
    auto pos = full_lib_name.find_last_of('/');
    if (pos != std::string::npos) {
        return full_lib_name.substr(pos + 1);
    }
    return full_lib_name;
}
}  // namespace

void HookRuntimeContext::dump() {
    std::stringstream ss;
    ss << "statistic info:\n";
    std::map<std::string, std::stringstream> statistic_map;
    for (const auto& func_info : func_infos_) {
        auto iter =
            statistic_map
                .insert(std::make_pair(shortLibName(func_info.first.lib_name),
                                       std::stringstream()))
                .first;
        iter->second << func_info.first.sym_name
                     << alignWith(func_info.first.sym_name) << func_info.second
                     << "\n";
    }

    for (const auto& it : statistic_map) {
        ss << "library name:" << it.first << "\n" << it.second.str();
    }
    LOG(WARN) << "dump context map:\n" << ss.str();
}

std::ostream& operator<<(std::ostream& os,
                         const HookRuntimeContext::Statistic& s) {
    os << "total call:" << s.counter_ << " times";
    return os;
}

std::ostream& operator<<(std::ostream& os,
                         const MemoryStatisticCollection::PtrIdentity& id) {
    os << "id(" << id.lib << ",devId:" << id.devId << ",kind:" << id.kind
       << ")";
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
        LOG(WARN) << "free dangling pointer";
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
    LOG(WARN) << *this;
}

MemoryStatisticCollection& MemoryStatisticCollection::instance() {
    static MemoryStatisticCollection instance;
    return instance;
}

void MemoryStatisticCollection::record_alloc(const std::string& libName,
                                             size_t devId, void* ptr,
                                             size_t size, int kind) {
    kind_map_.insert(std::make_pair(ptr, kind));
    statistics_[PtrIdentity(libName, devId, kind)].record_alloc(ptr, size);
}

void MemoryStatisticCollection::record_free(const std::string& libName,
                                            size_t devId, void* ptr, int kind) {
    statistics_[PtrIdentity(libName, devId, kind)].record_free(ptr);
}

void MemoryStatisticCollection::record_free(const std::string& libName,
                                            size_t devId, void* ptr) {
    auto iter = kind_map_.find(ptr);
    if (iter == kind_map_.end()) {
        LOG(WARN) << "free dangling pointer";
    }
    record_free(libName, devId, ptr, iter->second);
}

std::ostream& operator<<(std::ostream& os, const MemoryStatisticCollection& s) {
    for (auto& ms : s.statistics_) {
        os << ms.first << "\n" << ms.second << "\n";
    }
    return os;
}

}  // namespace hook
