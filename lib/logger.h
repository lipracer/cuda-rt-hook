#pragma once

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <memory>
#include <sstream>
#ifndef HAS_NOT_FMT_LIB
#include "fmt/core.h"
#endif

#ifdef HAS_NOT_FMT_LIB
#define __FMT_FUNC__(...) "unsupport fmt!"
#else
#define __FMT_FUNC__(...) fmt::format(__VA_ARGS__)
#endif

#ifndef LOGGER_LIKELY
#define LOGGER_LIKELY(x) __builtin_expect(!!(x), 1)
#endif /* LOGGER_LIKELY */
#ifndef LOGGER_UNLIKELY
#define LOGGER_UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif /* LOGGER_UNLIKELY */

namespace logger {

enum class LogLevel { info = 0, debug, warning, error };

struct StringLiteralBase {
    constexpr StringLiteralBase(size_t N) {
        // MN = MN > N ? MN : N;
    }
    static size_t MN;
};

template <size_t N>
struct StringLiteral : public StringLiteralBase {
    constexpr StringLiteral(const char (&str)[N]) : StringLiteralBase(N) {
        for (size_t i = 0; i < N; ++i) {
            str_[i] = str[i];
        }
    }

    char str_[N];

    constexpr auto simpleFile() const {
        for (int i = N - 1; i > 0; --i) {
            if (str_[i] == '/') {
                return str_ + i + 1;
            }
        }
        return str_;
    }
};

template <size_t N>
constexpr static auto makeStringLiteral(const char (&str)[N]) {
    return StringLiteral<N>(str);
}

template <size_t N>
constexpr std::ostream& operator<<(std::ostream& os,
                                   const StringLiteral<N>& str) {
    os << str.str_;
    return os;
}

class LogConsumer;

class LogStream {
   public:
    static LogStream& instance();

    LogStream(std::shared_ptr<LogConsumer>& logConsumer);
    ~LogStream();

    void flush();

    LogLevel getLevel() const { return level_; }

    const char* getStrLevel(LogLevel level) {
        constexpr const char* str[] = {"INFO", "DEBUG", "WARN", "ERROR"};
        return str[static_cast<int>(level)];
    }

    template <typename T>
    friend LogStream& operator<<(LogStream& s, T&& t);

   private:
    LogLevel level_ = LogLevel::warning;
    std::stringstream ss_;
    std::shared_ptr<LogConsumer> logConsumer_;
};

#define LOG_CONDITATION(level)    \
    (static_cast<int>((level)) >= \
     static_cast<int>(logger::LogStream::instance().getLevel()))

template <typename T>
LogStream& operator<<(LogStream& s, T&& t) {
    s.ss_ << std::forward<T>(t);
    return s;
}

struct LogWrapper {
    explicit LogWrapper(LogLevel level) : level_(level) {
        st_ = std::chrono::high_resolution_clock::now();
        LogStream::instance()
            << "[" << LogStream::instance().getStrLevel(level) << "]";
    }
    explicit LogWrapper(int level) : level_(static_cast<LogLevel>(level)) {}
    ~LogWrapper() {
        if (LOG_CONDITATION(level_)) {
            LogStream::instance().flush();
        }
        totalDur += std::chrono::high_resolution_clock::now() - st_;
    }
    LogLevel level_;
    std::chrono::high_resolution_clock::time_point st_;
    static thread_local std::chrono::high_resolution_clock::duration totalDur;
};

template <typename T>
static const LogWrapper& operator<<(const LogWrapper& s, T&& t) {
    if (LOG_CONDITATION(s.level_)) {
        LogStream::instance() << std::forward<T>(t);
    }
    return s;
}

class StreamPlaceHolder {};

static void operator<(const StreamPlaceHolder&, const LogWrapper&) {}

class StringPool;

struct SimpleStringRef {
    size_t size_;
    SimpleStringRef(const char* str, size_t size);

    size_t objSize() const;
    size_t invalidSize() const;
    SimpleStringRef* next() {
        assert(objSize() > 0 && "alloc size equal 0!");
        assert(objSize() % sizeof(size_t) == 0 && "alloc size unalign!");
        return reinterpret_cast<SimpleStringRef*>(
            reinterpret_cast<char*>(this) + objSize());
    }
    const char* c_str() const {
        return reinterpret_cast<const char*>(this) + sizeof(size_t);
    }
    size_t size() const { return size_; }
    static SimpleStringRef* create(StringPool& pool, const char* str, size_t size);
    // operator bool() const { return !!this; }
    // bool operator!() const { return static_cast<bool>(*this); }
};

inline std::ostream& operator<<(std::ostream& os,
                                const SimpleStringRef& strRef) {
    os << "size:" << strRef.size() << " str:" << strRef.c_str();
    return os;
}

class StringPool {
   public:
    class StringRefIterator {
       public:
        StringRefIterator(SimpleStringRef* ptr) : ptr_(ptr) {}
        StringRefIterator() : StringRefIterator(nullptr) {}
        StringRefIterator operator++() {
            ptr_ = ptr_->next();
            return ptr_;
        }
        StringRefIterator operator++(int) {
            auto cur = ptr_;
            ptr_ = ptr_->next();
            return cur;
        }

        const SimpleStringRef& operator*() const { return *ptr_; }
        const SimpleStringRef* operator->() const { return ptr_; }

        SimpleStringRef& operator*() { return *ptr_; }
        SimpleStringRef* operator->() { return ptr_; }

        bool operator==(const StringRefIterator& other) {
            return ptr_ == other.ptr_;
        }
        bool operator!=(const StringRefIterator& other) {
            return ptr_ != other.ptr_;
        }

       private:
        SimpleStringRef* ptr_{nullptr};
    };

    using FlushFunc = std::function<void(const char*, size_t)>;
    using interator = StringRefIterator;
    using constan_interator = const StringRefIterator;

    // c++11 didn't support inline constexpr static
    constexpr size_t kPoolSize() { return 1024 * 1024; }
    StringPool(const FlushFunc& flush);
    ~StringPool();

    char* allocStringBuf(size_t size);
    void flushPool();

    const char* pool() const { return pool_; }

    interator begin() { return reinterpret_cast<SimpleStringRef*>(pool_); }
    interator end() {
        return reinterpret_cast<SimpleStringRef*>(currentPoolBegin_);
    }

    size_t size() const { return size_; }

    constan_interator begin() const { return begin_; }
    constan_interator end() const {
        return reinterpret_cast<SimpleStringRef*>(currentPoolBegin_);
    }

    auto pool_begin() { return reinterpret_cast<char*>(pool_); }
    auto pool_end() { return reinterpret_cast<char*>(currentPoolEnd_); }

    void pop_front() { begin_++; }

   private:
    FlushFunc flushFunc_;
    char* pool_{nullptr};
    char* currentPoolBegin_{nullptr};
    char* currentPoolEnd_{nullptr};
    size_t size_{0};
    interator begin_;
    interator end_;
};

}  // namespace logger

#define INFO logger::LogLevel::info
#define DEBUG logger::LogLevel::debug
#define WARN logger::LogLevel::warning
#define ERROR logger::LogLevel::error

#define LOG(level)                                                           \
    !LOG_CONDITATION(level)                                                  \
        ? void(0)                                                            \
        : logger::StreamPlaceHolder() <                                      \
              logger::LogWrapper(level)                                      \
                  << "[" << logger::makeStringLiteral(__FILE__).simpleFile() \
                  << ":" << std::dec << __LINE__ << "]"

#define CHECK(p, ...)                                            \
    do {                                                         \
        if (!(p)) {                                              \
            LOG(ERROR) << __FILE__ << ":" << __LINE__;           \
            throw std::runtime_error(__FMT_FUNC__(__VA_ARGS__)); \
        }                                                        \
    } while (0)

#define CHECK_LT(l, r, ...) CHECK(((l) < (r)), __VA_ARGS__)
#define CHECK_LE(l, r, ...) CHECK(((l) <= (r)), __VA_ARGS__)
#define CHECK_GT(l, r, ...) CHECK(((l) > (r)), __VA_ARGS__)
#define CHECK_GE(l, r, ...) CHECK(((l) >= (r)), __VA_ARGS__)
#define CHECK_EQ(l, r, ...) CHECK(((l) == (r)), __VA_ARGS__)