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

#define LOG_MODULE_LIST(_)                                                  \
    _(profile, PROFILE), _(trace, TRACE), _(hook, HOOK), _(python, PYTHON), \
        _(memory, MEMORY), _(debug, DEBUG), _(last, LAST)

#define LOG_GREEN_STR_WRAP(v) "\033[0;32m[" #v "]\033[0m"
#define LOG_YELLOW_STR_WRAP(v) "\033[0;33m[" #v "]\033[0m"
#define LOG_RED_STR_WRAP(v) "\033[0;31m[" #v "]\033[0m"

#define LOG_LEVEL_LIST(_)                                                     \
    _(info, INFO, LOG_GREEN_STR_WRAP), _(warning, WARN, LOG_YELLOW_STR_WRAP), \
        _(error, ERROR, LOG_RED_STR_WRAP), _(fatal, FATAL, LOG_RED_STR_WRAP), \
        _(last, LAST, LOG_RED_STR_WRAP)

#define GET_LIST_INDEX_0_ITEM(v0, ...) v0
#define GET_LIST_INDEX_0_STR_ITEM(v0, v1, ...) #v0
#define GET_LIST_INDEX_1_STR_ITEM(v0, v1, ...) #v1
#define GET_LIST_INDEX_0_LOG_STR_ITEM(v0, v1, ...) "[" #v1 "]"
#define GET_LEVEL_COLOR_STR(v0, v1, v2, ...) v2(v1)

namespace logger {
enum class LogLevel { LOG_LEVEL_LIST(GET_LIST_INDEX_0_ITEM) };
enum class LogModule { LOG_MODULE_LIST(GET_LIST_INDEX_0_ITEM) };

// log need prefix avoid conflict
#define __LOG_PROFILE__ logger::LogModule::profile
#define __LOG_TRACE__ logger::LogModule::trace
#define __LOG_HOOK__ logger::LogModule::hook
#define __LOG_PYTHON__ logger::LogModule::python
#define __LOG_MEMORY__ logger::LogModule::memory
#define __LOG_DEBUG__ logger::LogModule::debug

#define __LOG_INFO__ logger::LogLevel::info
#define __LOG_WARN__ logger::LogLevel::warning
#define __LOG_ERROR__ logger::LogLevel::error
#define __LOG_FATAL__ logger::LogLevel::fatal

class LogModuleHelper {
   public:
    static auto& enum_strs() {
        static std::array<const char*, 7> strs = {
            LOG_MODULE_LIST(GET_LIST_INDEX_1_STR_ITEM)};
        static_assert(sizeof(strs) / sizeof(const char*) ==
                      static_cast<size_t>(LogModule::last) + 1);
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
    const char* name[] = {LOG_LEVEL_LIST(GET_LIST_INDEX_0_STR_ITEM)};
    return name[static_cast<int>(e)];
}
inline std::string to_string(logger::LogModule e) {
    return logger::LogModuleHelper::enum_strs()[static_cast<int>(e)];
}
}  // namespace std

using std::to_string;

namespace logger {

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
    constexpr char operator[](size_t index) const { return str_[index]; }

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

template <char... chr>
struct TypeStr {
    operator std::string() {
        std::string str = {chr...};
        return str;
    }
};

template <size_t N, const char (&str)[N]>
struct TypeStrGenerator {
    template <size_t... idx>
    static constexpr auto gen(std::index_sequence<idx...>) {
        return TypeStr<str[idx]...>();
    }

    using type = decltype(gen(std::make_index_sequence<N - 1>()));
};

template <size_t M>
static constexpr size_t constexpr_strlen(const char (&)[M]) {
    return M;
}

#define STR_TO_TYPE(str)                                     \
    []() -> auto {                                           \
        static constexpr const char ls[] = str;              \
        constexpr size_t N = ::logger::constexpr_strlen(ls); \
        return ::logger::TypeStrGenerator<N, ls>::type();    \
    }                                                        \
    ()

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
    size_t pageSize{4 * 1024 * 1024};
    LoggerMode mode{kAsync};
    std::FILE* stream{stdout};
};

class LogConsumer;

void setLoggerLevel(
    std::array<LogLevel, static_cast<size_t>(LogModule::last) + 1>& module_set,
    LogLevel& mLevel);

class LogStream {
   public:
    static LogStream& instance(const LogConfig& cfg = {});
    static std::thread::id threadId();

    LogStream(std::shared_ptr<LogConsumer>& logConsumer,
              const std::shared_ptr<LogConfig>& cfg);
    ~LogStream();

    void flush();

    LogLevel getLevel() const { return level_; }
    LogLevel getModuleLevel(LogModule m) const {
        return module_set_[static_cast<int>(m)];
    }

    void setLevel(LogLevel level) { level_ = level; }
    void setModuleLevel(LogModule m, LogLevel level) {
        module_set_[static_cast<int>(m)] = level;
    }

    bool IsModuleEnable(size_t m, size_t l) {
        return l >= static_cast<size_t>(module_set_[m]);
    }

    const char* getStrLevel(LogLevel level) {
        constexpr const char* console_str[] = {
            LOG_LEVEL_LIST(GET_LEVEL_COLOR_STR)};
        constexpr const char* file_str[] = {
            LOG_LEVEL_LIST(GET_LIST_INDEX_0_LOG_STR_ITEM)};
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
    std::array<LogLevel, static_cast<size_t>(LogModule::last) + 1> module_set_ =
        {LogLevel::info};
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
    // reset format ctrl flag
    ~LogWrapper() {
        LogStream::instance().flush();
#ifdef NDEBUG
        totalDur += std::chrono::high_resolution_clock::now() - st_;
#endif
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

void regist_on_exit(const std::function<void(void)>& OnExit = {});

}  // namespace logger

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

#define LOG_PREFIX_DEFINE(v) __LOG_##v##__

#define LOG_IMPL(level)                                    \
    !logger::LOG_CONDITATION(LOG_PREFIX_DEFINE(level))     \
        ? void(0)                                          \
        : logger::StreamPlaceHolder() <                    \
              logger::LogWrapper(LOG_PREFIX_DEFINE(level), \
                                 GetFixedLogerHeader(__FILE__, __LINE__))

#define MLOG_IMPL(m, str_m, level)                                            \
    !logger::MLOG_CONDITATION(LOG_PREFIX_DEFINE(m), LOG_PREFIX_DEFINE(level)) \
        ? void(0)                                                             \
        : logger::StreamPlaceHolder() <                                       \
              logger::MLogWrapper(                                            \
                  str_m, LOG_PREFIX_DEFINE(level),                            \
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
                        fmt::format("expect " #l ":{} < " #r ":{}", l, r))
#define CHECK_LE(l, r, ...)           \
    INTERNAL_CHECK_IMPL(((l) <= (r)), \
                        fmt::format("expect " #l ":{} <= " #r ":{}", l, r))
#define CHECK_GT(l, r, ...)          \
    INTERNAL_CHECK_IMPL(((l) > (r)), \
                        fmt::format("expect " #l ":{} > " #r ":{}", l, r))
#define CHECK_GE(l, r, ...)           \
    INTERNAL_CHECK_IMPL(((l) >= (r)), \
                        fmt::format("expect " #l ":{} >= " #r ":{}", l, r))
#define CHECK_EQ(l, r, ...)           \
    INTERNAL_CHECK_IMPL(((l) == (r)), \
                        fmt::format("expect " #l ":{} == " #r ":{}", l, r))

#define CHECK_NE(l, r, ...)           \
    INTERNAL_CHECK_IMPL(((l) != (r)), \
                        fmt::format("expect " #l ":{} != " #r ":{}", l, r))

#define hook_unreachable(...) INTERNAL_CHECK_IMPL(false, __VA_ARGS__)

#define MLOG(m, level) MLOG_IMPL(m, #m, level)
