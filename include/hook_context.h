#pragma once

#include <atomic>
#include <chrono>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "logger/logger.h"

namespace hook {

std::string shortLibName(const std::string& full_lib_name);

inline std::string alignWith(const std::string& str, size_t size = 32) {
    size_t align_size = str.size() >= size ? 1 : size - str.size();
    return std::string(align_size, ' ');
}

inline std::string alignWith(std::string&& str, size_t size = 32) {
    size_t align_size = str.size() >= size ? 1 : size - str.size();
    return std::string(align_size, ' ');
}

template <typename T>
inline std::string alignWith(T&& t, size_t size = 32) {
    return alignWith(std::to_string(std::forward<T>(t)), size);
}

static constexpr size_t kMaxLibrarySize = 64;
static constexpr size_t kMaxEachSignatureFuncSize = 256;

class WrapGeneratorBase {
   public:
    WrapGeneratorBase() {}
    virtual ~WrapGeneratorBase() {}
    virtual std::string symName() const = 0;
    virtual void* getFunction(size_t index,
                              const char* libName = nullptr) const = 0;
};

class HookRuntimeContext {
   public:
    struct StringPair {
        StringPair(const std::string& lib, const std::string& sym)
            : lib_name(lib), sym_name(sym) {}
        std::string lib_name;
        std::string sym_name;

        bool operator==(const StringPair& other) const {
            return lib_name == other.lib_name && sym_name == other.sym_name;
        }
        bool operator!=(const StringPair& other) const {
            return !(*this == other);
        }

        bool operator<(const StringPair& other) const {
            return (lib_name + sym_name) < (other.lib_name + other.sym_name);
        }
    };
    struct SPhash {
        size_t operator()(const StringPair& names) const {
            return std::hash<std::string>()(names.lib_name + names.sym_name);
        }
    };

    struct Statistic {
        void increase();
        void increase_cost(size_t time);
        Statistic() = default;
        Statistic(const Statistic& other) { *this = other; }
        Statistic(Statistic&& other) { *this = std::move(other); }
        Statistic& operator=(const Statistic& other) {
            counter_.store(other.counter_);
            cost_.store(other.cost_);
            return *this;
        }
        Statistic& operator=(Statistic&& other) {
            counter_.store(other.counter_);
            cost_.store(other.cost_);
            return *this;
        }

        size_t count() const { return counter_; }

        size_t cost() const { return cost_; }

        friend std::ostream& operator<<(std::ostream& os, const Statistic& s);

       private:
        mutable std::atomic<size_t> counter_{0};
        mutable std::atomic<size_t> cost_{0};
    };

    struct StatisticPair : public std::pair<void*, void*>, public Statistic {
        using std::pair<void*, void*>::pair;
        StatisticPair()
            : std::pair<void*, void*>(nullptr, nullptr), Statistic() {}
    };
    using vec_type = std::vector<std::pair<StringPair, StatisticPair>>;
    using map_type = std::map<StringPair, size_t>;
    // unordered_map insert will cause iter position change
    //    using map_type = std::unordered_map<StringPair, std::pair<void*,
    //    void*>, SPhash>;

    HookRuntimeContext() = default;
    ~HookRuntimeContext();
    static HookRuntimeContext& instance();

    template <typename... Args>
    vec_type::iterator insert(
        const std::pair<StringPair, StatisticPair>& feature) {
        std::lock_guard<std::mutex> lg(mtx_);
        auto iter = std::find_if(
            func_infos_.begin(), func_infos_.end(),
            [&](const auto& it) { return it.first == feature.first; });
        if (iter != func_infos_.end()) {
            return iter;
        }
        func_infos_.emplace_back(feature);
        // map_.insert(std::make_pair(feature.first, func_infos_.size() - 1));

        return std::prev(func_infos_.end());
    }

    size_t getUniqueId(vec_type::iterator iter) {
        return std::distance(func_infos_.begin(), iter);
    }

    vec_type::iterator& current_iter() {
        thread_local static vec_type::iterator iter;
        return iter;
    }

    vec_type::iterator setCurrentState(size_t UniqueId) {
        current_iter() = func_infos_.begin();
        std::advance(current_iter(), UniqueId);
        current_iter()->second.increase();
        return current_iter();
    }

    const std::string& curLibName() { return current_iter()->first.lib_name; }
    const std::string& curSymName() { return current_iter()->first.sym_name; }

    std::string summary_string();

    void registCompilerGen(WrapGeneratorBase* gen);

    template <typename T>
    size_t caclOffset(const char* libName, size_t uniqueId) {
        std::lock_guard<std::mutex> lg(mtx_);
        auto iter =
            std::find_if(last_index_map_.begin(), last_index_map_.end(),
                         [&](const auto& it) { return it.first == libName; });
        size_t offset = std::distance(last_index_map_.begin(), iter);
        if (iter != last_index_map_.end()) {
            auto sym_iter = std::find_if(
                iter->second.begin(), iter->second.end(),
                [&](const auto& it) { return it == std::string(T()); });
            if (sym_iter == iter->second.end()) {
                iter->second.emplace_back(std::string(T()));
            }
        } else {
            last_index_map_.emplace_back(decltype(last_index_map_)::value_type(
                libName, {std::string(T())}));
        }
        id_map_.insert(std::make_pair(
            &typeid(T), std::vector<size_t>(kMaxLibrarySize, -1)));
        id_map_[&typeid(T)][offset] = uniqueId;
        return offset;
    }

    template <typename T>
    size_t getCUniqueId(size_t index) {
        return id_map_[&typeid(T)][index];
    }

    size_t getCallCount(const std::string& libName, const std::string& symName);

    void* lookUpArgsParser(const std::string& name) {
        auto iter = args_parser_map_.find(name);
        return iter == args_parser_map_.end() ? nullptr : iter->second;
    }

    std::unordered_map<std::string, void*>& argsParserMap() {
        return args_parser_map_;
    }

    struct TypeInfoHash {
        size_t operator()(const std::type_info* ti) const {
            return ti->hash_code();
        }
    };

    void clear();

   private:
    map_type map_;
    std::vector<std::pair<StringPair, StatisticPair>> func_infos_;
    std::mutex mtx_;

    std::vector<std::unique_ptr<WrapGeneratorBase>> compiler_gen_;
    std::vector<std::pair<std::string, std::vector<std::string>>>
        last_index_map_;
    std::unordered_map<const std::type_info*, std::vector<size_t>, TypeInfoHash>
        id_map_;
    std::unordered_map<std::string, void*> args_parser_map_;
};

template <typename IterT>
class TimeStatisticWrapIter {
   public:
    using Deleter = std::function<void(std::chrono::nanoseconds)>;

    TimeStatisticWrapIter(IterT iter, Deleter deleter = {})
        : iter_(iter), deleter_(deleter) {
        reset();
    }

    TimeStatisticWrapIter(const TimeStatisticWrapIter&) = delete;
    TimeStatisticWrapIter& operator=(const TimeStatisticWrapIter&) = delete;

    TimeStatisticWrapIter(TimeStatisticWrapIter&& other) {
        *this = std::move(other);
    }
    TimeStatisticWrapIter& operator=(TimeStatisticWrapIter&& other) {
        iter_ = other.iter_;
        deleter_ = other.deleter_;
        sp_ = other.sp_;
        return *this;
    }

    ~TimeStatisticWrapIter() {
        auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - sp_);
        deleter_(dur);
    }

    void reset() { sp_ = std::chrono::steady_clock::now(); }

    auto operator->() { return &*iter_; }

   private:
    IterT iter_;
    Deleter deleter_;
    decltype(std::chrono::steady_clock::now()) sp_;
};

template <size_t UniqueId>
auto wrapCurrentIter() {
    auto iter = HookRuntimeContext::instance().setCurrentState(UniqueId);
    return TimeStatisticWrapIter<HookRuntimeContext::vec_type::iterator>(
        iter, [=](std::chrono::nanoseconds dur) {
            iter->second.increase_cost(dur.count());
            MLOG(PROFILE, INFO)
                << iter->first.sym_name << " costs " << dur.count() << "ns";
        });
}

inline auto wrapCurrentIter(size_t UniqueId) {
    auto iter = HookRuntimeContext::instance().setCurrentState(UniqueId);
    return TimeStatisticWrapIter<HookRuntimeContext::vec_type::iterator>(
        iter, [=](std::chrono::nanoseconds dur) {
            iter->second.increase_cost(dur.count());
            MLOG(PROFILE, INFO)
                << iter->first.sym_name << " costs " << dur.count() << "ns";
        });
}

}  // namespace hook
