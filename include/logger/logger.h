#pragma once

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <sstream>
#include <thread>
#include <unordered_map>

#include "StringRef.h"

// #define DEBUG_LOGER
#ifdef DEBUG_LOGER
#include <iostream>
#endif

#include "fmt/core.h"

#ifndef LOGGER_LIKELY
#define LOGGER_LIKELY(x) __builtin_expect(!!(x), 1)
#endif /* LOGGER_LIKELY */
#ifndef LOGGER_UNLIKELY
#define LOGGER_UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif /* LOGGER_UNLIKELY */

namespace logger {
enum class LogLevel { info = 0, warning, error, fatal, last };
enum class LogModule { profile, trace, hook, python, last };

class LogModuleHelper {
   public:
    static auto& enum_strs() {
        static std::array<const char*, 5> strs = {"PROFILE", "TRACE", "HOOK",
                                                         "PYTHON", "LAST"};
        return strs;
    }
    static auto begin() { return enum_strs().begin(); }
    static auto end() { return enum_strs().end(); }

    template <size_t N>
    static int strToEnum(const char(name)[N]) {
        for (size_t i = 0; i < enum_strs().size(); ++i) {
            if (!strcmp(enum_strs()[i], name)) {
                return i;
            }
        }
        return -1;
    }

    static int strToEnum(const char* name) {
        for (size_t i = 0; i < enum_strs().size(); ++i) {
            if (!strcmp(enum_strs()[i], name)) {
                return i;
            }
        }
        return -1;
    }
};

}  // namespace logger

namespace std {
inline std::string to_string(logger::LogLevel e) {
    const char* name[] = {"info", "warning", "error", "fatal", "last"};
    return name[static_cast<int>(e)];
}
inline std::string to_string(logger::LogModule e) {
    return logger::LogModuleHelper::enum_strs()[static_cast<int>(e)];
}
}  // namespace std

using std::to_string;

namespace logger {

#define PROFILE logger::LogModule::profile
#define TRACE logger::LogModule::trace
#define HOOK logger::LogModule::hook
#define PYTHON logger::LogModule::python

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

    constexpr StringLiteral() : StringLiteralBase(N) {}

    constexpr size_t size() const { return N; }

    char str_[N] = {0};

    constexpr operator const char*() const { return str_; }

    constexpr auto simpleFileSize() const {
        for (int i = N - 1; i > 0; --i) {
            if (str_[i] == '/') {
                return N - i - 1;
            }
        }
        return N;
    }
};

template <size_t size, size_t N>
constexpr auto simpleFile(StringLiteral<N> f) {
    StringLiteral<size> result;
    for (size_t i = 0, j = N - size; j != N;) {
        result.str_[i++] = f.str_[j++];
    }
    return result;
}

template <size_t N, size_t M>
inline constexpr auto operator+(StringLiteral<N> lhs, StringLiteral<M> rhs) {
    StringLiteral<M + N - 1> result;
    size_t i = 0;
    for (; i < N - 1; ++i) {
        result.str_[i] = lhs.str_[i];
    }
    for (; i < N + M - 1; ++i) {
        result.str_[i] = rhs.str_[i + 1 - N];
    }
    return result;
}

template <size_t N>
constexpr static auto makeStringLiteral(const char (&str)[N]) {
    return StringLiteral<N>(str);
}

namespace {

template <size_t N>
constexpr inline size_t getIntToStringSize() {
    size_t nByte = 0;
    size_t tmpN = N;
    do {
        ++nByte;
    } while (tmpN /= 10);
    return nByte;
}
}  // namespace

template <size_t N>
constexpr static auto makeStringLiteral() {
    constexpr size_t nByte = getIntToStringSize<N>();
    StringLiteral<nByte + 1> result;
    size_t tmpN = N;
    for (size_t i = 0; i < nByte; ++i) {
        result.str_[nByte - 1 - i] = tmpN % 10 + '0';
        tmpN /= 10;
    }
    return result;
}

template <size_t N>
constexpr std::ostream& operator<<(std::ostream& os,
                                   const StringLiteral<N>& str) {
    os << str.str_;
    return os;
}

struct LogConfig {
    enum LoggerMode {
        kSync,
        kAsync,
    };
    // NB: some time log string over 4K
    // TODO: need check too long string
    size_t pageSize{4 * 1024 * 1024};
    LoggerMode mode{kAsync};
    std::FILE* stream{stdout};
};

class LogConsumer;

class LogStream {
   public:
    static LogStream& instance(const LogConfig& cfg = {});
    static std::thread::id threadId();

    LogStream(std::shared_ptr<LogConsumer>& logConsumer,
              const std::shared_ptr<LogConfig>& cfg);
    ~LogStream();

    void flush();

    LogLevel getLevel() const { return level_; }
    void setLevel(LogLevel level) { level_ = level; }
    void setModuleLevel(LogModule m, LogLevel level) {
        module_set_[static_cast<int>(m)] = level;
    }

    bool IsModuleEnable(size_t m, size_t l) {
        return l >= static_cast<size_t>(module_set_[m]);
    }

    const char* getStrLevel(LogLevel level) {
        constexpr const char* console_str[] = {
            "\033[0;32m[INFO]\033[0m", "\033[0;33m[WARN]\033[0m",
            "\033[0;31m[ERROR]\033[0m", "\033[0;31m[FATAL]\033[0m"};
        constexpr const char* file_str[] = {"[INFO]", "[WARN]", "ERROR]",
                                            "[FATAL]"};
        return cfg_->stream == stdout ? console_str[static_cast<int>(level)]
                                      : file_str[static_cast<int>(level)];
    }

    LogConsumer* logConsumer() { return logConsumer_.get(); }

    std::ostream& getStream() { return ss_; }

    void log_fatal();

    auto time_duration() {
        return std::to_string(
                   std::chrono::duration_cast<std::chrono::microseconds>(
                       std::chrono::steady_clock::now() - start_point_)
                       .count()) +
               "us";
    }

    const std::string& logHeader() { return logHeader_; }

   private:
    LogLevel level_ = LogLevel::warning;
    std::stringstream ss_;
    std::shared_ptr<LogConsumer> logConsumer_;
    std::shared_ptr<LogConfig> cfg_;
    LogLevel module_set_[static_cast<size_t>(LogModule::last) + 1] = {
        LogLevel::info};
    std::chrono::steady_clock::time_point start_point_{
        std::chrono::steady_clock::now()};
    std::string logHeader_;
};

inline bool LOG_CONDITATION(LogLevel level) {
    return (static_cast<int>((level)) >=
            static_cast<int>(logger::LogStream::instance().getLevel()));
}

inline bool MLOG_CONDITATION(const char* m, LogLevel l) {
    return logger::LogStream::instance().IsModuleEnable(
        static_cast<size_t>(LogModuleHelper::strToEnum(m)),
        static_cast<size_t>(l));
}

inline bool MLOG_CONDITATION(LogModule m, LogLevel l) {
    return logger::LogStream::instance().IsModuleEnable(static_cast<size_t>(m),
                                                        static_cast<size_t>(l));
}

template <typename T>
using VoidType = void;

template <typename T, typename ST = void>
struct HasToStringFunc : public std::false_type {};

template <typename T>
struct HasToStringFunc<
    T, VoidType<decltype(std::stringstream() << std::declval<T>())>>
    : public std::true_type {};

template <typename T>
inline LogStream& WriteToLoggerStream(
    LogStream& s, T&& t,
    std::enable_if_t<HasToStringFunc<T>::value>* = nullptr) {
    s.getStream() << std::forward<T>(t);
    return s;
}

template <typename T>
LogStream& operator<<(LogStream& s, T&& t) {
    return WriteToLoggerStream(s, std::forward<T>(t));
}

struct LogWrapper {
    explicit LogWrapper(LogLevel level) : level_(level) {}
    explicit LogWrapper(int level, const char* str)
        : LogWrapper(static_cast<LogLevel>(level), str) {}

    explicit LogWrapper(LogLevel level, const char* str) : LogWrapper(level) {
        LogStream::instance()
            << LogStream::instance().getStrLevel(level) << str
            << LogStream::instance().logHeader()
            << "[TS:" << logger::LogStream::instance().time_duration() << "]";
    }

    ~LogWrapper() {
        LogStream::instance().flush();
        totalDur += std::chrono::high_resolution_clock::now() - st_;
        // crash here
        if (LOGGER_UNLIKELY(level_ == LogLevel::fatal)) {
            LogStream::instance().log_fatal();
        }
    }
    LogLevel level_;
    std::chrono::high_resolution_clock::time_point st_;
    static thread_local std::chrono::high_resolution_clock::duration totalDur;
};

struct MLogWrapper : public LogWrapper {
    explicit MLogWrapper(const char* module, LogLevel level, const char* str)
        : LogWrapper(level, str), module_(module) {}


    const char* module_;
};

template <typename T>
static const LogWrapper& operator<<(const LogWrapper& s, T&& t) {
    LogStream::instance() << std::forward<T>(t);
    return s;
}

template <typename T>
static const MLogWrapper& operator<<(const MLogWrapper& s, T&& t) {
    LogStream::instance() << std::forward<T>(t);
    return s;
}

class StreamPlaceHolder {};

static void operator<(const StreamPlaceHolder&, const LogWrapper&) {}
static void operator<(const StreamPlaceHolder&, const MLogWrapper&) {}

class StringPool;

struct SimpleStringRef {
    size_t size_;
    char c_str_[1];

    // forbidden move copy ctor
    SimpleStringRef(const SimpleStringRef&) = delete;

    static constexpr size_t alignSize() { return sizeof(size_t); }
    static constexpr size_t headerSize() { return sizeof(size_); }

    size_t objSize() const;
    size_t invalidSize() const;
    SimpleStringRef* next() {
        assert(objSize() > 0 && "alloc size equal 0!");
        assert(objSize() % alignSize() == 0 && "alloc size unalign!");
        return reinterpret_cast<SimpleStringRef*>(
            reinterpret_cast<char*>(this) + objSize());
    }
    const char* c_str() const { return c_str_; }
    size_t size() const { return size_; }
    static SimpleStringRef* create(StringPool& pool, const char* str,
                                   size_t size);

   private:
    SimpleStringRef(const char* str, size_t size);
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
        StringRefIterator operator++();
        StringRefIterator operator++(int);

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

    StringPool(
        size_t pageSize, const FlushFunc& flush = [](const char*, size_t) {});
    ~StringPool();

    char* allocStringBuf();
    void flushPool();

    const char* pool() const { return pool_; }

    interator begin() { return begin_; }
    interator end() {
        return reinterpret_cast<SimpleStringRef*>(currentPoolBegin_);
    }

    size_t size() const { return size_; }

    constan_interator begin() const { return begin_; }
    constan_interator end() const {
        return reinterpret_cast<SimpleStringRef*>(currentPoolBegin_);
    }

    SimpleStringRef* front() { return &*begin_; }

    auto pool_begin() { return reinterpret_cast<char*>(pool_); }
    auto pool_end() { return reinterpret_cast<char*>(currentPoolEnd_); }

    void push_back(const std::string& str);
    void pop_front();

    bool empty() { return !size_; }

    size_t debugSize();

    void setFlushFunc(const FlushFunc& flushFunc) { flushFunc_ = flushFunc; }

    bool hasEnoughSpace(size_t size);

    void setPageSize(size_t size) { pageSize_ = size; }

   private:
    FlushFunc flushFunc_;
    char* pool_{nullptr};
    // point the last string's end
    char* currentPoolBegin_{nullptr};
    char* currentPoolEnd_{nullptr};
    size_t size_{0};
    interator begin_;
    interator end_;
    size_t pageSize_;
};

void initLogger(const LogConfig& = LogConfig{});

void destroy_logger();

}  // namespace logger

#define INFO logger::LogLevel::info
#define WARN logger::LogLevel::warning
#define ERROR logger::LogLevel::error
#define FATAL logger::LogLevel::fatal

static constexpr char __LOGGER_HEADER__[] = "[{}:{}]";
static constexpr char __MLOGGER_HEADER__[] = "[{}][{}:{}]";

#define GetFixedLogerHeader(f, L)                                          \
    logger::makeStringLiteral("[") +                                       \
        logger::simpleFile<logger::makeStringLiteral(f).simpleFileSize()>( \
            logger::makeStringLiteral(f)) +                                \
        logger::makeStringLiteral(":") + logger::makeStringLiteral<L>() +  \
        logger::makeStringLiteral("]")

#define MGetFixedLogerHeader(m, f, L)                                      \
    logger::makeStringLiteral("[") + logger::makeStringLiteral(m) +        \
        logger::makeStringLiteral("]") + logger::makeStringLiteral("[") +  \
        logger::simpleFile<logger::makeStringLiteral(f).simpleFileSize()>( \
            logger::makeStringLiteral(f)) +                                \
        logger::makeStringLiteral(":") + logger::makeStringLiteral<L>() +  \
        logger::makeStringLiteral("]")

#define LOG_IMPL(level)                 \
    !logger::LOG_CONDITATION(level)     \
        ? void(0)                       \
        : logger::StreamPlaceHolder() < \
              logger::LogWrapper(level, \
                                 GetFixedLogerHeader(__FILE__, __LINE__))

#define MLOG_IMPL(m, str_m, level)      \
    !logger::MLOG_CONDITATION(m, level) \
        ? void(0)                       \
        : logger::StreamPlaceHolder() < \
              logger::MLogWrapper(      \
                  str_m, level,         \
                  MGetFixedLogerHeader(str_m, __FILE__, __LINE__))

#define LOG(level) LOG_IMPL(level)

#define INTERNAL_CHECK_IMPL(p, msg)                                  \
    do {                                                             \
        if (!(p)) {                                                  \
            LOG(FATAL) << __FILE__ << ":" << __LINE__ << ":" << msg; \
        }                                                            \
    } while (0)

#define CHECK(p, ...) INTERNAL_CHECK_IMPL((p), fmt::format(__VA_ARGS__))

#define CHECK_LT(l, r, ...)          \
    INTERNAL_CHECK_IMPL(((l) < (r)), \
                        fmt::format("expect lhs:{} < rhs:{}", l, r))
#define CHECK_LE(l, r, ...)           \
    INTERNAL_CHECK_IMPL(((l) <= (r)), \
                        fmt::format("expect lhs:{} <= rhs:{}", l, r))
#define CHECK_GT(l, r, ...)          \
    INTERNAL_CHECK_IMPL(((l) > (r)), \
                        fmt::format("expect lhs:{} > rhs:{}", l, r))
#define CHECK_GE(l, r, ...)           \
    INTERNAL_CHECK_IMPL(((l) >= (r)), \
                        fmt::format("expect lhs:{} >= rhs:{}", l, r))
#define CHECK_EQ(l, r, ...)           \
    INTERNAL_CHECK_IMPL(((l) == (r)), \
                        fmt::format("expect lhs:{} == rhs:{}", l, r))

#define be_unreachable(...) INTERNAL_CHECK_IMPL(false, __VA_ARGS__)

#define MLOG(m, level) MLOG_IMPL(m, #m, level)
